#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from dawgz import job, schedule
from tqdm import trange
from typing import *

from priors.nn import *
from priors.score import *

from utils import *


CONFIG = {
    # Architecture
    'hid_channels': (128, 192, 256),
    'hid_blocks': (3, 3, 3),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'attention': {2},
    'dropout': 0.1,
    # Training
    'epochs': 256,
    'warmups': 16,
    'batch_size': 64,
    'scheduler': 'cosine',
    'init_lr': 1e-5,
    'peak_lr': 1e-3,
    'end_lr': 1e-6,
    'optimizer': 'adam',
    'clip': 1.0,
}


@job(cpus=4, gpus=1, ram='64GB', time='24:00:00')
def train():
    run = wandb.init(project='priors-cifar', config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    config = run.config

    # RNG
    seed = hash(runpath) % 2**16
    rng = inox.random.PRNG(seed)

    latent = rng.normal((16, 32 * 32 * 3))

    # Data
    dataset = cifar(split='train')

    # Model
    model = make_model(key=rng.split(), **CONFIG)
    model.train(True)

    static, params, others = model.partition(nn.Parameter)

    # Objective
    objective = MeasureLoss()

    # Optimizer
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.init_lr,
        peak_value=config.peak_lr,
        end_value=config.end_lr,
        warmup_steps=config.warmups * len(dataset) // config.batch_size,
        decay_steps=config.epochs * len(dataset) // config.batch_size,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=config.clip),
        optax.adam(scheduler),
    )
    opt_state = optimizer.init(params)

    # Training
    @jax.jit
    def sgd_step(params, others, opt_state, x, key):
        keys = jax.random.split(key, 3)

        z = jax.random.normal(keys[0], x.shape)
        t = jax.random.uniform(keys[1], x.shape[:1])

        def ell(params):
            return objective(static(params, others), x, z, t, key=keys[2])

        loss, grads = jax.value_and_grad(ell)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss, params, opt_state

    for epoch in (bar := trange(config.epochs, ncols=88)):
        loader = (
            dataset
            .shuffle(seed=seed + epoch)
            .iter(batch_size=config.batch_size, drop_last_batch=True)
        )

        losses = []

        for x in map(process, loader):
            loss, params, opt_state = sgd_step(params, others, opt_state, x, key=rng.split())
            losses.append(loss)

        losses = np.stack(losses)
        bar.set_postfix(loss=losses.mean())

        ## Checkpoint
        model = static(params, others)
        model.train(False)

        dump_model(model, runpath / 'checkpoint.pkl')

        ## Eval
        sampler = Euler(model)

        x = sampler(latent, steps=256)
        x = x.reshape(4, 4, -1)

        run.log({
            'loss': losses.mean(),
            'samples': wandb.Image(show(x, zoom=2)),
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
