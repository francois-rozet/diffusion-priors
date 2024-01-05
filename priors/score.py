r"""Score helpers"""

import inox
import inox.nn as nn
import jax
import jax.numpy as jnp
import math

from inox.random import get_rng
from jax import Array
from typing import *


class TimeEmbedding(nn.Sequential):
    def __init__(self, features: int, freqs: int = 8, key: Array = None):
        if key is None:
            keys = get_rng().split(2)
        else:
            keys = jax.random.split(key, 2)

        super().__init__(
            nn.Linear(2 * freqs, 256, key=keys[0]),
            nn.ReLU(),
            nn.Linear(256, features, key=keys[1]),
        )

        self.freqs = jnp.pi * jnp.arange(1, freqs + 1)

    @inox.jit
    def __call__(self, t: Array) -> Array:
        t = self.freqs * t[..., None]
        t = jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)

        return super().__call__(t)


class SDE(nn.Module):
    r"""Abstract stochastic differential equation (SDE)."""

    pass


class VPSDE(SDE):
    r"""Variance preserving (VP) SDE.

    .. math:: x(t) = \alpha(t) x + \sigma(t) z

    with

    .. math::
        \alpha(t) & = \cos(\arccos(\sqrt{\eta}) t)^2 \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \eta^2

    Arguments:
        eta: A numerical stability term.
    """

    def __init__(self, eta: float = 1e-4):
        self.eta = eta

    @inox.jit
    def alpha(self, t: Array) -> Array:
        return jnp.cos(math.acos(math.sqrt(self.eta)) * t) ** 2

    @inox.jit
    def sigma(self, t: Array) -> Array:
        return jnp.sqrt(1 - self.alpha(t) ** 2 + self.eta ** 2)

    @inox.jit
    def __call__(self, x: Array, z: Array, t: Array) -> Array:
        t = t[..., None]
        x = self.alpha(t) * x + self.sigma(t) * z

        return x


class ReverseSDE(nn.Module):
    r"""Predictor-corrector sampler for the reverse SDE.

    Arguments:
        model: A model of the score/noise :math:`\epsilon(x(t), t)`.
        sde: The forward SDE.
    """

    def __init__(self, model: nn.Module, sde: SDE = None):
        super().__init__()

        self.model = model  # epsilon(x(t), t) = -sigma(t) * score(x(t), t)

        if sde is None:
            self.sde = VPSDE()
        else:
            self.sde = sde

    def __call__(
        self,
        shape: Sequence[int],
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
        key: Array = None,
    ) -> Array:
        if key is None:
            key = get_rng().split()

        dt = 1 / steps
        time = jnp.linspace(1 - dt, 0, steps)
        keys = jax.random.split(key, steps)
        z = jax.random.normal(key, shape)

        def f(x, t_key):
            t, key = t_key

            # Predictor
            r = self.sde.alpha(t) / self.sde.alpha(t + dt)
            x = r * x + (self.sde.sigma(t) - r * self.sde.sigma(t + dt)) * self.model(x, t + dt)

            # Corrector
            if corrections > 0:
                for subkey in jax.random.split(key, corrections):
                    z = jax.random.normal(subkey, x.shape)
                    s = self.model(x, t)

                    x = x - (tau * s + math.sqrt(2 * tau) * z) * self.sde.sigma(t)

            return x, None

        x, _ = jax.lax.scan(f, z, (time, keys))

        return x

    def loss(self, key: Array, x: Array) -> Array:
        keys = jax.random.split(key, 2)

        t = jax.random.uniform(keys[0], x.shape[:-1])
        z = jax.random.uniform(keys[1], x.shape)

        x = self.sde(x, t)

        return jnp.mean(jnp.square(self.model(x, t) - z))


class StandardScoreModel(nn.Module):
    r"""Score model for a standard Gaussian random variable.

    .. math:: \epsilon(x(t), t) = -\sigma(t) x(t)

    Arguments:
        sde: The forward SDE.
    """

    def __init__(self, sde: SDE = None):
        super().__init__()

        if sde is None:
            self.sde = VPSDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, x: Array, t: Array) -> Array:
        return x * self.sde.sigma(t)[..., None]


class PosteriorScoreModel(nn.Module):
    r"""Posterior score model for a Gaussian observation

    .. math:: p(y | x) = N(y | A(x), \Sigma_y)

    Arguments:
        TODO
    """

    def __init__(
        self,
        model: nn.Module,
        y: Array,
        A: Callable[[Array], Array],
        noise: Union[float, Array],
        gamma: Union[float, Array] = 1.0,
        sde: SDE = None,
    ):
        super().__init__()

        self.model = model
        self.y = y
        self.A = A
        self.noise = noise
        self.gamma = gamma

        if sde is None:
            self.sde = VPSDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, x: Array, t: Array) -> Array:
        alpha, sigma = self.sde.alpha(t), self.sde.sigma(t)

        def log_prob(x):
            z = self.model(x, t)
            x_hat = (x - sigma * z) / alpha

            err = (self.y - self.A(x_hat)) ** 2
            var = self.noise ** 2 + self.gamma * (sigma / alpha) ** 2

            log_p = -jnp.sum(err / var) / 2

            return log_p, z

        s, z = jax.grad(log_prob, has_aux=True)(x)

        return z - sigma * s
