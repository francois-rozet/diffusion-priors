#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from datasets import load_from_disk, concatenate_datasets, Array3D, Features
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
    # Data
    'duplicate': 4,
    # Training
    'laps': 4,
    'epochs': 256,
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


def measure(A, x):
    return flatten(A * unflatten(x, 32, 32))

def sample(model, y, A, key):
    sampler = Euler(
        PosteriorScoreModel(
            model=model,
            A=inox.Partial(measure, A),
            y=flatten(y),
            noise=1e-3,
            gamma=1e-1,
        ),
    )

    z = jax.random.normal(key, flatten(y).shape)
    x = sampler(z, steps=64)
    x = unflatten(x, 32, 32)
    x = np.asarray(x)

    return x

def generate(model, dataset, rng, batch_size):
    def transform(batch):
        y, A = batch['y'], batch['A']
        x = sample(model, y, A, rng.split())

        return {'x': x, 'A': A}

    types = {
        'x': Array3D(shape=(32, 32, 3), dtype='float32'),
        'A': Array3D(shape=(32, 32, 1), dtype='bool'),
    }

    return dataset.map(
        transform,
        remove_columns=['y'],
        features=Features(types),
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
    )


@job(cpus=4, gpus=1, ram='16GB', time='7-00:00:00', partition='a5000,quadro,tesla')
def train():
    run = wandb.init(project='priors-cifar-mask', dir=PATH, config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    config = run.config

    # RNG
    seed = hash(runpath) % 2**16
    rng = inox.random.PRNG(seed)

    # Data
    dataset = load_from_disk(PATH / 'hf/cifar-mask')
    dataset.set_format('numpy')

    trainset_yA = dataset['train']
    trainset_yA = concatenate_datasets([trainset_yA] * config.duplicate)

    testset_yA = dataset['test']
    y_eval, A_eval = testset_yA[:16]['y'], testset_yA[:16]['A']

    # Model
    model = make_model(key=rng.split(), **config)
    model.train(True)

    static, params, others = model.partition(nn.Parameter)
    start = params

    # Objective
    objective = EDMLoss()

    # Optimizer
    steps = config.epochs * len(trainset_yA) // config.batch_size
    optimizer = Adam(steps=steps, **config)
    opt_state = optimizer.init(params)

    # EMA
    ema = EMA(decay=config.ema_decay)
    avrg = params

    # Training
    @jax.jit
    def ell(params, others, x, A, key):
        keys = jax.random.split(key, 3)

        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])

        return objective(static(params, others), x, z, t, A=inox.Partial(measure, A), key=keys[2])

    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, A, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, A, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)

        return loss, avrg, params, opt_state

    for lap in range(config.laps):
        if lap > 0:
            trainset_xA = generate(model, trainset_yA, rng, config.batch_size)
            testset_xA = generate(model, testset_yA, rng, config.batch_size)
        else:
            trainset_xA = generate(StandardScoreModel(), trainset_yA, rng, config.batch_size)
            testset_xA = generate(StandardScoreModel(), testset_yA, rng, config.batch_size)

        for epoch in (bar := trange(config.epochs, ncols=88)):
            loader = (
                trainset_xA
                .shuffle(seed=seed + lap * config.epochs + epoch)
                .iter(batch_size=config.batch_size, drop_last_batch=True)
            )

            losses = []

            for batch in loader:
                x, A = batch['x'], batch['A']
                x = flatten(x)

                loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, A, key=rng.split())
                losses.append(loss)

            loss_train = np.stack(losses).mean()

            ## Validation
            loader = (
                testset_xA
                .iter(batch_size=config.batch_size, drop_last_batch=True)
            )

            losses = []

            for batch in loader:
                x, A = batch['x'], batch['A']
                x = flatten(x)

                loss = ell(params, others, x, A, key=rng.split())
                losses.append(loss)

            loss_val = np.stack(losses).mean()

            bar.set_postfix(loss=loss_train, loss_val=loss_val)

            ## Eval
            if (epoch + 1) % 16 == 0:
                model = static(avrg, others)
                model.train(False)

                x = sample(model, y_eval, A_eval, rng.split())
                x = x.reshape(4, 4, 32, 32, 3)

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

        ## Checkpoint
        model = static(avrg, others)
        model.train(False)

        dump_module(model, runpath / f'checkpoint_{lap}.pkl')

        ## Refresh
        if lap > 0:
            params = avrg
        else:
            params = avrg = start

        opt_state = optimizer.init(params)

    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='Training from corrupted data',
        backend='slurm',
        export='ALL',
        env=['export WANDB_SILENT=true'],
    )