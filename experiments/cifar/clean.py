#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from datasets import load_dataset
from dawgz import job, schedule
from tqdm import trange
from typing import *

from utils import *


CONFIG = {
    # Architecture
    'hid_channels': (128, 256, 384),
    'hid_blocks': (3, 3, 3),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {2: 4},
    'dropout': 0.1,
    # Training
    'objective': 'edm',
    'epochs': 4096,
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


@job(cpus=4, gpus=1, ram='16GB', time='7-00:00:00', partition='a5000,quadro,tesla')
def train():
    run = wandb.init(project='priors-cifar', dir=PATH, config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    config = run.config

    # RNG
    seed = hash(runpath) % 2**16
    rng = inox.random.PRNG(seed)

    latent = rng.normal((4, 4, 32 * 32 * 3))

    # Data
    trainset = load_dataset('cifar10', split='train', keep_in_memory=True)
    testset = load_dataset('cifar10', split='test', keep_in_memory=True)

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
    steps = config.epochs * len(trainset) // config.batch_size
    optimizer = Adam(steps=steps, **config)
    opt_state = optimizer.init(params)

    # EMA
    avrg = params

    if config.ema_decay is None:
        ema = lambda x, y: y
    else:
        ema = EMA(decay=config.ema_decay)

    # Training
    @jax.jit
    def ell(params, others, x, key):
        keys = jax.random.split(key, 3)

        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])

        return objective(static(params, others), x, z, t, key=keys[2])

    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)

        return loss, avrg, params, opt_state

    for epoch in (bar := trange(config.epochs + 1, ncols=88)):
        loader = (
            trainset
            .shuffle(seed=seed + epoch)
            .iter(batch_size=config.batch_size, drop_last_batch=True)
        )

        losses = []

        for x in map(collate, loader):
            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, key=rng.split())
            losses.append(loss)

        loss_train = np.stack(losses).mean()

        ## Validation
        loader = (
            testset
            .iter(batch_size=config.batch_size, drop_last_batch=True)
        )

        losses = []

        for x in map(collate, loader):
            loss = ell(params, others, x, key=rng.split())
            losses.append(loss)

        loss_val = np.stack(losses).mean()

        bar.set_postfix(loss=loss_train, loss_val=loss_val)

        ## Eval
        if epoch % 64 == 0:
            model = static(avrg, others)
            model.train(False)

            dump_module(model, runpath / 'checkpoint.pkl')

            sampler = Euler(model)

            x = sampler(latent, steps=256)
            x = unflatten(x, 32, 32)

            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
                'samples': wandb.Image(to_pil(x, zoom=2)),
            })
        else:
           run.log({
                'loss': loss_train,
                'loss_val': loss_val,
            })

    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='Training from clean data',
        backend='slurm',
        export='ALL',
        env=['export WANDB_SILENT=true'],
    )
