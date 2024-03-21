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
    'hid_channels': (16, 32, 64, 128, 192, 256),
    'hid_blocks': (3, 3, 3, 3, 3, 3),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {5: 4},
    'dropout': 0.1,
    # Training
    'laps': 4,
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
    x = unflatten(x, 320, 320)
    y = fft2c(x)
    y = jnp.concatenate((y.real, y.imag), axis=-1)

    return flatten(A * y)

def sample(model, y, A, key):
    sampler = DDIM(
        PosteriorDenoiser(
            model=model,
            A=inox.Partial(measure, A),
            y=flatten(y),
            sigma_y=1e-2 ** 2,
        ),
    )

    z = jax.random.normal(key, (len(y), 320 * 320))
    x = sampler(z, steps=64, key=key)
    x = unflatten(x, 320, 320)
    x = np.asarray(x)

    return x

def generate(model, dataset, rng, batch_size, sharding=None):
    def transform(batch):
        y, A = batch['y'], batch['A']
        y, A = jax.device_put((y, A), sharding)
        x = sample(model, y, A, rng.split())

        return {'x': x}

    dtype = Array3D(shape=(320, 320, 1), dtype='float32')

    return dataset.map(
        transform,
        features=Features(**dataset.features, x=dtype),
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
    )


@job(cpus=4, gpus=4, ram='256GB', time='2-00:00:00', partition='gpu')
def train():
    run = wandb.init(project='priors-fastmri-kspace', dir=PATH, config=CONFIG)
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

    y_eval, A_eval = dataset[:4]['y'], dataset[:4]['A']
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

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

    def ell(params, others, x, A, y, key):
        keys = jax.random.split(key, 3)

        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=2, shape=x.shape[:1])

        return objective(static(params, others), x, z, t, inox.Partial(measure, A), y, key=keys[2])

    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, A, y, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, A, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)

        return loss, avrg, params, opt_state

    for lap in range(config.laps):
        if lap > 0:
            trainset = generate(model, dataset, rng, config.batch_size, distributed)
        else:
            trainset = generate(GaussianDenoiser(), dataset, rng, config.batch_size, distributed)

        for epoch in (bar := trange(config.epochs, ncols=88)):
            loader = (
                trainset
                .shuffle(seed=seed + lap * config.epochs + epoch)
                .iter(batch_size=config.batch_size, drop_last_batch=True)
            )

            losses = []

            for batch in prefetch(loader):
                x, A, y = batch['x'], batch['A'], batch['y']
                x, A, y = jax.device_put((x, A, y), distributed)
                x, y = flatten(x), flatten(y)

                loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, A, y, key=rng.split())
                losses.append(loss)

            loss_train = np.stack(losses).mean()

            bar.set_postfix(loss=loss_train)

            ## Eval
            model = static(avrg, others)
            model.train(False)

            x = sample(model, y_eval, A_eval, rng.split())
            x = x.reshape(2, 2, 320, 320, 1)

            run.log({
                'loss': loss_train,
                'samples': wandb.Image(to_pil(x)),
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
