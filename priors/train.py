r"""Training helpers"""

import inox
import inox.nn as nn
import jax
import jax.numpy as jnp
import optax
import pickle

from inox.random import PRNG, get_rng
from jax import Array
from pathlib import Path
from typing import *

from .linalg import *
from .score import *


def dump_module(module: nn.Module, file: Path):
    with open(file, 'wb') as f:
        pickle.dump(module, f)


def load_module(file: Path) -> nn.Module:
    with open(file, 'rb') as f:
        return pickle.load(f)


class Adam(inox.Namespace):
    def __init__(
        self,
        steps: int,
        scheduler: str = 'constant',
        lr_init: float = 1e-3,
        lr_end: float = 1e-6,
        lr_warmup: float = 0.0,
        weight_decay: float = None,
        clip: float = None,
        **absorb,
    ):
        super().__init__(
            steps=steps,
            scheduler=scheduler,
            lr_init=lr_init,
            lr_end=lr_end,
            lr_warmup=lr_warmup,
            weight_decay=weight_decay,
            clip=clip,
        )

    def learning_rate(self, step: int) -> float:
        progress = jnp.minimum((step + 1) / (self.steps + 1), 1)
        heat = jnp.minimum((step + 1) / (self.steps * self.lr_warmup + 1), 1)

        if self.scheduler == 'constant':
            lr = self.lr_init
        elif self.scheduler == 'linear':
            lr = self.lr_init + (self.lr_end - self.lr_init) * progress
        elif self.scheduler == 'exponential':
            lr = self.lr_init * (self.lr_end / self.lr_init) ** progress

        return lr * heat

    @property
    def transform(self) -> optax.GradientTransformation:
        if self.weight_decay is None:
            optimizer = optax.adam(self.learning_rate)
        else:
            optimizer = optax.adamw(self.learning_rate, weight_decay=self.weight_decay)

        if self.clip is None:
            return optimizer
        else:
            return optax.chain(
                optax.clip_by_global_norm(max_norm=self.clip),
                optimizer,
            )

    def init(self, *args, **kwargs) -> Any:
        return self.transform.init(*args, **kwargs)

    def update(self, *args, **kwargs) -> Any:
        return self.transform.update(*args, **kwargs)


class EMA(inox.Namespace):
    def __init__(self, decay: float = 0.999):
        self.alpha = 1.0 - decay

    def __call__(self, x: Any, y: Any) -> Any:
        return jax.tree_util.tree_map(self.average, x, y)

    def average(self, x: Array, y: Array) -> Array:
        return x + self.alpha * (y - x)


def fit_moments(
    features: int,
    rank: int,
    A: Callable[[Array], Array],
    y: Array,
    sigma_y: Array,
    epochs: int = 4096,
    learning_rate: float = 1e-3,
    epsilon: float = 1e-2,
    key: Array = None,
) -> Tuple[Array, DPLR]:
    r"""Fits :math:`\mu_x` and :math:`\Sigma_x` given pairs :math:`(A, y)`."""

    if key is None:
        rng = get_rng()
    else:
        rng = PRNG(key)

    At = transpose(A, jnp.zeros((len(y), features)))

    mu_x, _ = jax.scipy.sparse.linalg.cg(
        A=lambda x: jnp.mean(At(A(x)), axis=0),
        b=jnp.mean(At(y), axis=0),
    )

    bias = A(mu_x) - y

    def objective(params, z):
        d, U = params

        sigma_x = DPLR((d ** 2 + epsilon) * jnp.ones(features), U, U.T)
        sigma_Ax = lambda v: A(sigma_x @ At(v)) + sigma_y * v

        loss = jnp.sum(sigma_Ax(z) ** 2, axis=-1) - 2 * jnp.einsum('...i,...i', bias, sigma_Ax(bias))

        return jnp.mean(loss)

    params = (
        jnp.ones(()),
        rng.normal((features, rank)) * 1e-2,
    )

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    def update(state, key):
        params, opt_state = state

        loss, grads = jax.value_and_grad(objective)(params, jax.random.normal(key, y.shape))
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return (params, opt_state), loss

    (params, opt_state), losses = jax.lax.scan(
        f=update,
        init=(params, opt_state),
        xs=rng.split(epochs),
    )

    d, U = params
    sigma_x = DPLR((d ** 2 + epsilon) * jnp.ones(features), U, U.T)

    return mu_x, sigma_x, losses
