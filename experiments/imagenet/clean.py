#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from datasets import load_from_disk
from dawgz import job, schedule
from tqdm import trange
from typing import *

from utils import *


CONFIG = {
    # Architecture
    'hid_channels': (128, 256, 384, 512),
    'hid_blocks': (3, 3, 3, 3),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'attention': {2, 3},
    'heads': 16,
    'dropout': 0.1,
    # Training
    'objective': 'edm',
    'epochs': 1024,
    'batch_size': 256,
    'scheduler': 'constant',
    'lr_init': 2e-4,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
    'ema_decay': 0.999,
}


@job(cpus=4, gpus=4, ram='16GB', time='7-00:00:00', partition='a5000,quadro,tesla')
def train():
    run = wandb.init(project='priors-imagenet', dir=PATH, config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    config = run.config

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)
    sharding = jax.sharding.PositionalSharding(jax.devices())

    # RNG
    seed = hash(runpath) % 2**16
    rng = inox.random.PRNG(seed)

    latent = rng.normal((4, 4, 64 * 64 * 3))
    latent = jax.device_put(latent, sharding.reshape(-1, 1, 1))

    # Data
    dataset = load_from_disk(PATH / 'hf/imagenet-64')['train']

    # Model
    model = make_model(key=rng.split(), **config)
    model.train(True)

    static, params, others = model.partition(nn.Parameter)

    # Objective
    if config.objective == 'edm':
        objective = EDMLoss()
    elif config.objective == 'ddpm':
        objective = DDPMLoss()

    # Optimizer
    steps = config.epochs * len(dataset) // config.batch_size
    optimizer = Adam(steps=steps, **config)
    opt_state = optimizer.init(params)

    # EMA
    ema = EMA(decay=config.ema_decay)
    avg = params

    # Training
    avg, params, others, opt_state = jax.device_put((avg, params, others, opt_state), sharding.replicate())

    @jax.jit
    def sgd_step(avg, params, others, opt_state, x, key):
        keys = jax.random.split(key, 3)

        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])

        def ell(params):
            return objective(static(params, others), x, z, t, key=keys[2])

        loss, grads = jax.value_and_grad(ell)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avg = ema(avg, params)

        return loss, avg, params, opt_state

    for epoch in (bar := trange(config.epochs + 1, ncols=88)):
        loader = (
            dataset
            .to_iterable_dataset()
            .shuffle(seed=seed + epoch, buffer_size=16384)
            .iter(batch_size=config.batch_size, drop_last_batch=True)
        )

        losses = []

        for x in prefetch(map(transform, loader)):
            x = jax.device_put(x, sharding.reshape(-1, 1))
            loss, avg, params, opt_state = sgd_step(avg, params, others, opt_state, x, key=rng.split())
            losses.append(loss)

        losses = np.stack(losses)

        bar.set_postfix(loss=losses.mean(), loss_std=losses.std())

        ## Checkpoint
        model = static(avg, others)
        model.train(False)

        dump_module(model, runpath / 'checkpoint.pkl')

        ## Eval
        sampler = Euler(model)

        x = sampler(latent, steps=256)
        x = unflatten(x, 64, 64)

        run.log({
            'loss': losses.mean(),
            'loss_std': losses.std(),
            'samples': wandb.Image(to_pil(x)),
        })

    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='Clean training',
        backend='slurm',
        export='ALL',
        env=['export WANDB_SILENT=true'],
    )
