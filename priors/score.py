r"""Score helpers"""

import inox
import inox.nn as nn
import jax
import jax.numpy as jnp

from inox.random import get_rng
from jax import Array
from typing import *


class NoiseEmbedding(nn.Sequential):
    r"""Creates a noise embedding module.

    Arguments:
        features: The number of embedding features.
        key: A PRNG key for initialization.
    """

    def __init__(self, features: int, key: Array = None):
        if key is None:
            keys = get_rng().split(2)
        else:
            keys = jax.random.split(key, 2)

        super().__init__(
            nn.Linear(1, 256, key=keys[0]),
            nn.ReLU(),
            nn.Linear(256, features, key=keys[1]),
        )

    def __call__(self, sigma: Array) -> Array:
        return super().__call__(jnp.log(sigma))


class VESDE(nn.Module):
    r"""Variance exploding (VE) SDE.

    .. math:: x(t) = x + \sigma(t) z

    with

    .. math:: \sigma(t) = \tan(\arctan(a) (1 - t) + \arctan(b) t)

    Arguments:
        a: The noise lower bound.
        b: The noise upper bound.
    """

    def __init__(self, a: Array = 1e-3, b: Array = 1e2):
        self.a = jnp.arctan(a)
        self.b = jnp.arctan(b)

    @jax.jit
    def __call__(self, x: Array, z: Array, t: Array) -> Array:
        sigma = self.sigma(t)
        sigma = sigma[..., None]

        return x + sigma * z

    @jax.jit
    def sigma(self, t: Array) -> Array:
        return jnp.tan(self.a + (self.b - self.a) * t)


class ReverseSDE(nn.Module):
    r"""Predictor-corrector sampler for the reverse SDE.

    Arguments:
        model: A score/noise model :math:`z(x(t), t)`.
        sde: The forward SDE.
    """

    def __init__(self, model: nn.Module, sde: VESDE = None):
        super().__init__()

        self.model = model

        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    def __call__(
        self,
        shape: Sequence[int],
        steps: int = 64,
        corrections: int = 0,
        tau: Array = 0.5,
        key: Array = None,
    ) -> Array:
        tau = jnp.asarray(tau)

        if key is None:
            key = get_rng().split()

        dt = 1 / steps
        time = jnp.linspace(1 - dt, 0, steps)
        keys = jax.random.split(key, (steps, corrections))
        x = jax.random.normal(key, shape) * self.sde.sigma(1.0)

        def f(x, t_keys):
            t, keys = t_keys

            x = self.predict(x, t + dt, t)

            for key in keys:
                x = self.correct(x, t, tau, key)

            return x, None

        x, _ = jax.lax.scan(f, x, (time, keys))

        return x

    @inox.jit
    def predict(self, x: Array, s: Array, t: Array) -> Array:
        return x + (self.sde.sigma(t) - self.sde.sigma(s)) * self.model(x, s)

    @inox.jit
    def correct(self, x: Array, t: Array, tau: Array, key: Array) -> Array:
        z = self.model(x, t)
        eps = jax.random.normal(key, x.shape)
        norm = jnp.mean(z**2, axis=-1, keepdims=True)
        delta = tau / jnp.clip(norm, a_min=1.0)

        return x - self.sde.sigma(t) * (delta * z + jnp.sqrt(2 * delta) * eps)


class ScoreModel(nn.Module):
    r"""Score model.

    .. math:: z(x(t), t)
        & = -\sigma(t) * score(x(t), t) \\
        & = \sigma(t) x(t) / (\sigma(t)^2 + 1)
        + 1 / \sqrt{\sigma(t)^2 + 1} network(x(t) / \sqrt{\sigma(t)^2 + 1}, \log \sigma(t))

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
        | https://arxiv.org/abs/2206.00364

    Arguments:
        TODO
    """

    def __init__(
        self,
        network: nn.Module,
        embedding: int = 64,
        sde: VESDE = None,
        key: Array = None,
    ):
        self.network = network
        self.embed = NoiseEmbedding(embedding, key=key)

        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, xt: Array, t: Array) -> Array:
        sigma = self.sde.sigma(t)
        sigma = sigma[..., None]
        denum = jnp.sqrt(sigma ** 2 + 1)

        return xt / (sigma + 1 / sigma) + 1 / denum * self.network(xt / denum, self.embed(sigma))

    @inox.jit
    def loss(self, x: Array, z: Array, t: Array, A: Callable = None) -> Array:
        sigma = self.sde.sigma(t)
        sigma = sigma[..., None]

        err = z - self(x + sigma * z, t)

        if callable(A):
            err = A(err)

        return jnp.mean(err ** 2)


class StandardScoreModel(nn.Module):
    r"""Score model for a standard Gaussian random variable.

    .. math:: z(x(t), t) = \sigma(t) x(t) / (\sigma(t)^2 + 1)

    Arguments:
        sde: The forward SDE.
    """

    def __init__(self, sde: VESDE = None):
        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, xt: Array, t: Array) -> Array:
        sigma = self.sde.sigma(t)
        sigma = sigma[..., None]

        return xt / (sigma + 1 / sigma)


class PosteriorScoreModel(nn.Module):
    r"""Posterior score model for a Gaussian observation.

    .. math:: p(y | x) = N(y | A(x), \Sigma_y)

    Arguments:
        TODO
    """

    def __init__(
        self,
        model: nn.Module,
        A: Callable[[Array], Array],
        y: Array,
        noise: Union[float, Array],
        gamma: Union[float, Array] = 1.0,
        sde: VESDE = None,
    ):
        super().__init__()

        self.model = model
        self.A = A
        self.y = jnp.asarray(y)
        self.noise = jnp.asarray(noise)
        self.gamma = jnp.asarray(gamma)

        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, xt: Array, t: Array) -> Array:
        sigma = self.sde.sigma(t)
        sigma = sigma[..., None]

        def log_prob(xt):
            z = self.model(xt, t)
            x = xt - sigma * z

            err = (self.y - self.A(x)) ** 2
            var = self.noise ** 2 + self.gamma * sigma ** 2

            log_p = -jnp.sum(err / var) / 2

            return log_p, z

        s, z = jax.grad(log_prob, has_aux=True)(xt)

        return z - sigma * s
