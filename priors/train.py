r"""Training helpers"""

import inox
import inox.nn as nn
import jax
import jax.experimental.sparse as jes
import jax.numpy as jnp
import optax
import pickle

from inox.random import PRNG, get_rng
from jax import Array
from pathlib import Path
from tqdm import tqdm
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


@inox.jit
def ppca(x: Array, key: Array, rank: int = 1) -> Tuple[Array, DPLR]:
    r"""Fits :math:`(\mu_x, \Sigma_x)` by probabilistic principal component analysis (PPCA).

    References:
        https://www.miketipping.com/papers/met-mppca.pdf
    """

    samples, features = x.shape

    mu_x = jnp.mean(x, axis=0)
    x = x - mu_x

    if samples < features:
        C = x @ x.T / samples
    else:
        C = x.T @ x / samples

    if rank < len(C) // 5:
        Q = jax.random.normal(key, (len(C), rank))
        L, Q, _ = jes.linalg.lobpcg_standard(C, Q)
    else:
        L, Q = jnp.linalg.eigh(C)
        L, Q = L[-rank:], Q[:, -rank:]

    if samples < features:
        Q = x.T @ Q
        Q = Q / jnp.linalg.norm(Q, axis=0)

    D = (jnp.trace(C) - jnp.sum(L)) / (features - rank)
    U = Q * jnp.sqrt(L - D)

    sigma_x = DPLR(D * jnp.ones(features), U, U.T)

    return mu_x, sigma_x


def fit_moments(
    features: int,
    rank: int,
    A: Callable[[Array], Array],
    y: Array,
    sigma_y: Tuple[Array, DPLR],
    iterations: int = 16,
    steps: int = 64,
    key: Array = None,
) -> Tuple[Array, DPLR]:
    r"""Fits :math:`(\mu_x, \Sigma_x)` by expectation maximization."""

    if key is None:
        rng = get_rng()
    else:
        rng = PRNG(key)

    def sample(mu_x, sigma_x, A, y, key):
        sampler = DDPM(
            PosteriorDenoiser(
                model=GaussianDenoiser(mu_x, sigma_x),
                A=A,
                y=y,
                sigma_y=sigma_y,
                sigma_x=sigma_x,
            )
        )

        z = jax.random.normal(key, (len(y), features))
        x = mu_x + z * sampler.sde.sigma(1.0)
        x = sampler(x, steps=steps, key=key)

        return x

    mu_x = jnp.zeros(features)
    sigma_x = DPLR(
        jnp.ones(features),
        jnp.zeros((features, rank)),
        jnp.zeros((rank, features)),
    )

    for _ in tqdm(range(iterations), ncols=88):
        # Expectation
        if isinstance(y, list):
            x = []

            for Ai, yi in zip(A, y):
                x.append(sample(mu_x, sigma_x, Ai, yi, rng.split()))

            x = jnp.concatenate(x, axis=0)
        else:
            x = sample(mu_x, sigma_x, A, y, rng.split())

        # Maximization
        mu_x, sigma_x = ppca(x, rank=rank, key=rng.split())

    return mu_x, sigma_x
