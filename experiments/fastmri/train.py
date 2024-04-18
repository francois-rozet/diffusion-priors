#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from datasets import load_from_disk, concatenate_datasets, Array3D, Features
from dawgz import job, schedule
from functools import partial
from tqdm import trange
from typing import *

from utils import *


CONFIG = {
    # Architecture
    'hid_channels': (128, 256, 384, 512),
    'hid_blocks': (3, 3, 3, 3),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {3: 4},
    'dropout': 0.1,
    # Data
    'duplicate': 4,
    # Training
    'laps': 16,
    'epochs': 64,
    'batch_size': 256,
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
    return flatten(A * unflatten(x, 320, 320))


def sample(model, y, A, key):
    if isinstance(model, GaussianDenoiser):
        mu_x = model.mu_x
        sigma_x = model.sigma_x
    else:
        mu_x = model.mu_x
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
    x = mu_x + z * sampler.sde.sigma(1.0)
    x = sampler(x, steps=64, key=key)
    x = unflatten(x, 320, 320)

    return x


def generate(model, dataset, rng, batch_size, sharding=None):
    def transform(batch):
        y, A = batch['y'], batch['A']
        y, A = jax.device_put((y, A), sharding)
        x = sample(model, y, A, rng.split())
        x = np.asarray(x)

        return {'x': x}

    types = {'x': Array3D(shape=(320, 320, 2), dtype='float32')}

    return dataset.map(
        transform,
        features=Features(types),
        remove_columns=['y', 'A'],
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
    )


def train(runid: int, lap: int):
    run = wandb.init(
        project='priors-fastmri-kspace',
        id=runid,
        resume='allow',
        dir=PATH,
        config=CONFIG,
    )

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
    dataset = load_from_disk(PATH / 'hf/fastmri-kspace')
    dataset.set_format('numpy')
    dataset = concatenate_datasets([dataset] * config.duplicate)

    y_eval, A_eval = dataset[:1024:256]['y'], dataset[:1024:256]['A']
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

    # Previous
    if lap > 0:
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
    else:
        y_fit, A_fit = dataset[:16384:4]['y'], dataset[:16384:4]['A']
        y_fit, A_fit = jnp.array_split(y_fit, 4, axis=0), jnp.array_split(A_fit, 4, axis=0)
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)

        mu_x, sigma_x = fit_moments(
            features=320 * 320 * 2,
            rank=64,
            A=[inox.Partial(measure, A) for A in A_fit],
            y=[flatten(y) for y in y_fit],
            sigma_y=1e-2 ** 2,
            key=rng.split(),
        )

        del y_fit, A_fit

        previous = GaussianDenoiser(mu_x, sigma_x)

    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)

    # Model
    model = make_model(key=rng.split(), **config)
    model.mu_x = previous.mu_x
    model.train(True)

    static, params, others = model.partition(nn.Parameter)

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
    avrg, params, others, opt_state = jax.device_put((avrg, params, others, opt_state), replicated)

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

    trainset = generate(previous, dataset, rng, config.batch_size, distributed)

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
            x = x.reshape(2, 2, -1)

            run.log({
                'loss': loss_train,
                'samples': wandb.Image(show(x)),
            })
        else:
            run.log({
                'loss': loss_train,
            })

    ## Checkpoint
    model = static(avrg, others)
    model.train(False)

    dump_module(model, runpath / f'checkpoint_{lap}.pkl')


if __name__ == '__main__':
    runid = wandb.util.generate_id()

    jobs = []

    for lap in range(CONFIG.get('laps')):
        jobs.append(
            job(
                partial(train, runid=runid, lap=lap),
                name=f'train_{lap}',
                cpus=4,
                gpus=4,
                ram='192GB',
                time='2-00:00:00',
                partition='gpu',
            )
        )

        if lap > 0:
            jobs[lap].after(jobs[lap - 1])

    schedule(
        *jobs,
        name='Training from corrupted data',
        backend='slurm',
        export='ALL',
        account='ariacpg',
        env=['export WANDB_SILENT=true'],
    )
