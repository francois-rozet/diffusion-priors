#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from datasets import load_from_disk, Array3D, Features
from dawgz import job, schedule
from tqdm import trange
from typing import *

from utils import *


CONFIG = {
    # Architecture
    'hid_channels': (128, 256, 384, 512),
    'hid_blocks': (2, 3, 5, 7),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {3: 16},
    'dropout': 0.1,
    # Training
    'laps': 4,
    'epochs': 256,
    'batch_size': 1024,
    'scheduler': 'constant',
    'lr_init': 1e-4,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
    'ema_decay': 0.999,
}


def measure(A, x):
    return flatten(A * unflatten(x, 64, 64))

def sample(model, y, A, key):
    if isinstance(model, GaussianDenoiser):
        sigma_x = model.sigma_x
    else:
        sigma_x = None

    sampler = DDPM(
        PosteriorDenoiser(
            model=model,
            A=inox.Partial(measure, A),
            y=flatten(y),
            sigma_y=1e-2 ** 2,
            sigma_x=sigma_x,
        ),
    )

    z = jax.random.normal(key, flatten(y).shape)
    x = z * sampler.sde.sigma(1.0)

    x = sampler(x, steps=64, key=key)
    x = unflatten(x, 64, 64)
    x = np.asarray(x)

    return x

def generate(model, dataset, rng, batch_size, sharding=None):
    def transform(batch):
        y, A = batch['y'], batch['A']
        y, A = jax.device_put((y, A), sharding)
        x = sample(model, y, A, rng.split())

        return {'x': x}

    types = {'x': Array3D(shape=(64, 64, 3), dtype='float32')}

    return dataset.map(
        transform,
        features=Features(types),
        remove_columns=['y', 'A'],
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
    )


@job(cpus=4, gpus=4, ram='512GB', time='2-00:00:00', partition='ia')
def train():
    run = wandb.init(project='priors-imagenet-patch', dir=PATH, config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    config = run.config

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    sharding = jax.sharding.PositionalSharding(jax.devices())
    replicated = sharding.replicate()
    distributed = sharding.reshape(-1, 1, 1, 1)

    # RNG
    seed = hash(runpath) % 2**16
    rng = inox.random.PRNG(seed)

    # Data
    dataset = load_from_disk(PATH / 'hf/imagenet-patch')
    dataset.set_format('numpy')

    y_fit, A_fit = dataset[:16384]['y'], dataset[:16384]['A']
    y_eval, A_eval = dataset[:16]['y'], dataset[:16]['A']

    y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

    mu_x, sigma_x, _ = fit_moments(
        features=64 * 64 * 3,
        rank=64,
        A=inox.Partial(measure, A_fit),
        y=flatten(y_fit),
        sigma_y=1e-2 ** 2,
        key=rng.split(),
    )

    del y_fit, A_fit

    # Model
    model = make_model(key=rng.split(), **config)
    model.train(True)

    static, params, others = model.partition(nn.Parameter)
    start = params

    # Objective
    objective = DenoiserLoss()

    # Optimizer
    steps = config.epochs * len(dataset) // config.batch_size
    optimizer = Adam(steps=steps, **config)
    opt_state = optimizer.init(params)

    # EMA
    ema = EMA(decay=config.ema_decay)
    avrg = params

    # Training
    start, avrg, params, others, opt_state = jax.device_put((start, avrg, params, others, opt_state), replicated)

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

    for lap in range(config.laps):
        if lap > 0:
            del trainset
            trainset = generate(model, dataset, rng, config.batch_size, distributed)
        else:
            trainset = generate(GaussianDenoiser(mu_x, sigma_x), dataset, rng, config.batch_size, distributed)

        for epoch in (bar := trange(config.epochs, ncols=88)):
            loader = (
                trainset
                .shuffle(seed=seed + lap * config.epochs + epoch)
                .iter(batch_size=config.batch_size, drop_last_batch=True)
            )

            losses = []

            for batch in prefetch(loader):
                x = batch['x']
                x = jax.device_put(x, distributed)
                x = flatten(x)

                loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, key=rng.split())
                losses.append(loss)

            loss_train = np.stack(losses).mean()

            bar.set_postfix(loss=loss_train)

            ## Eval
            if (epoch + 1) % 4 == 0:
                model = static(avrg, others)
                model.train(False)

                x = sample(model, y_eval, A_eval, rng.split())
                x = x.reshape(4, 4, -1)

                run.log({
                    'loss': loss_train,
                    'samples': wandb.Image(show(x, zoom=2)),
                })
            else:
                run.log({
                    'loss': loss_train,
                })

        ## Checkpoint
        model = static(avrg, others)
        model.train(False)

        dump_module(model, runpath / f'checkpoint_{lap}.pkl')

        ## Refresh
        params = avrg = start
        opt_state = optimizer.init(params)

    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='Training from corrupted data',
        backend='slurm',
        export='ALL',
        account='ariacpg',
        env=['export WANDB_SILENT=true'],
    )
