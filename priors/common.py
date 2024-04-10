r"""Common helpers"""

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
    key: Array = None,
    **kwargs,
) -> Tuple[Array, DPLR]:
    r"""Fits :math:`(\mu_x, \Sigma_x)` by expectation maximization."""

    if key is None:
        rng = get_rng()
    else:
        rng = PRNG(key)

    mu_x = jnp.zeros(features)
    sigma_x = DPLR(
        jnp.ones(features),
        jnp.zeros((features, rank)),
        jnp.zeros((rank, features)),
    )

    for _ in tqdm(range(iterations), ncols=88):
        # Expectation
        x = sample_any(
            model=GaussianDenoiser(mu_x, sigma_x),
            shape=(len(y), features),
            A=A,
            y=y,
            sigma_y=sigma_y,
            key=rng.split(),
            **kwargs,
        )

        # Maximization
        mu_x, sigma_x = ppca(x, rank=rank, key=rng.split())

    return mu_x, sigma_x


def sample_any(
    model: nn.Module,
    shape: Sequence[int],
    A: Callable[[Array], Array] = None,
    y: Array = None,
    sigma_y: Union[Array, DPLR] = None,
    key: Array = None,
    sampler: str = 'ddpm',
    steps: int = 64,
    rtol: float = 1e-3,
    maxiter: int = 5,
    **kwargs,
) -> Array:
    r"""Samples from :math:`q(x)` or :math:`q(x | A, y)`."""

    mu_x = getattr(model, 'mu_x', None)
    sigma_x = getattr(model, 'sigma_x', None)

    if A is None or y is None:
        pass
    else:
        model = PosteriorDenoiser(
            model=model,
            A=A,
            y=y,
            sigma_y=sigma_y,
            sigma_x=sigma_x,
            rtol=rtol,
            maxiter=maxiter,
        )

    if sampler == 'ddpm':
        sampler = DDPM(model, **kwargs)
    elif sampler == 'ddim':
        sampler = DDIM(model, **kwargs)
    elif sampler == 'pc':
        sampler = PredictorCorrector(model, **kwargs)

    z = jax.random.normal(key, shape)

    if mu_x is None:
        x1 = sampler.sde(0.0, z, 1.0)
    else:
        x1 = sampler.sde(mu_x, z, 1.0)

    x0 = sampler(x1, steps=steps, key=key)

    return x0
