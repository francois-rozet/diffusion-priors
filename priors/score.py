r"""Score helpers"""

import inox
import inox.nn as nn
import jax
import jax.numpy as jnp
import math

from jax import Array
from typing import *


class VESDE(nn.Module):
    r"""Variance exploding (VE) SDE.

    .. math:: x(t) = x + \sigma(t) z

    with

    .. math:: \sigma(t) = \exp(\log(a) (1 - t) + \log(b) t)

    Arguments:
        a: The noise lower bound.
        b: The noise upper bound.
    """

    def __init__(self, a: Array = 1e-3, b: Array = 1e2):
        self.a = jnp.log(a)
        self.b = jnp.log(b)

    @inox.jit
    def __call__(self, x: Array, z: Array, t: Array) -> Array:
        sigma = self.sigma(t)
        sigma = sigma[..., None]

        return x + sigma * z

    @inox.jit
    def sigma(self, t: Array) -> Array:
        return jnp.exp(self.a + (self.b - self.a) * t)


class PredictorCorrector(nn.Module):
    r"""Predictor-Corrector sampler for the reverse SDE.

    Arguments:
        model: A score/noise model :math:`z(x(t), t) \approx E[z | x(t)]`.
        sde: The forward SDE.
        tau: The Langevin Monte Carlo step size.
    """

    def __init__(self, model: nn.Module, sde: VESDE = None, tau: Array = 1e-2):
        super().__init__()

        self.model = model

        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

        self.tau = jnp.asarray(tau)

    @inox.jit
    def __call__(self, z: Array, t: Array = 1.0, steps: int = 64, corrections: int = 1, key: Array = None) -> Array:
        dt = jnp.asarray(t / steps)
        time = jnp.linspace(t, dt, steps)
        keys = jax.random.split(key, steps)

        def f(xt, t_key):
            t, key = t_key
            s = t - dt

            xs = self.predict(xt, t, s)

            for key in jax.random.split(key, corrections):
                xs = self.correct(xs, s, key)

            return xs, None

        x = z * self.sde.sigma(t)
        x, _ = jax.lax.scan(f, x, (time, keys))

        return x

    @inox.jit
    def predict(self, xt: Array, t: Array, s: Array) -> Array:
        return xt + (self.sde.sigma(s) - self.sde.sigma(t)) * self.model(xt, t)

    @inox.jit
    def correct(self, xt: Array, t: Array, key: Array) -> Array:
        zt = self.model(xt, t)
        eps = jax.random.normal(key, zt.shape)
        norm = jnp.sqrt(jnp.mean(zt ** 2, axis=-1, keepdims=True))
        tau = self.tau / jnp.clip(norm, a_min=1.0)

        return xt - self.sde.sigma(t) * (tau * zt + jnp.sqrt(2 * tau) * eps)


class Euler(nn.Module):
    r"""Euler (1st order) sampler for the reverse SDE.

    .. math:: x(s) = x(t) + (\sigma(s) - \sigma(t)) z(x(t), t)

    Arguments:
        model: A score/noise model :math:`z(x(t), t) \approx E[z | x(t)]`.
        sde: The forward SDE.
    """

    def __init__(self, model: nn.Module, sde: VESDE = None):
        super().__init__()

        self.model = model

        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, z: Array, t: Array = 1.0, steps: int = 64) -> Array:
        dt = jnp.asarray(t / steps)
        time = jnp.linspace(t, dt, steps)

        def f(xt, t):
            return self.step(xt, t, t - dt), None

        x = z * self.sde.sigma(t)
        x, _ = jax.lax.scan(f, x, time)

        return x

    @inox.jit
    def step(self, xt: Array, t: Array, s: Array) -> Array:
        return xt + (self.sde.sigma(s) - self.sde.sigma(t)) * self.model(xt, t)


class Heun(Euler):
    r"""Heun (2nd order) sampler for the reverse SDE.

    Arguments:
        model: A score/noise model :math:`z(x(t), t) \approx E[z | x(t)]`.
        sde: The forward SDE.
    """

    @inox.jit
    def step(self, xt: Array, t: Array, s: Array) -> Array:
        zt = self.model(xt, t)
        xs = xt + (self.sde.sigma(s) - self.sde.sigma(t)) * zt
        zs = self.model(xs, s)
        xs = xt + (self.sde.sigma(s) - self.sde.sigma(t)) * (zt + zs) / 2

        return xs


class NoiseEmbedding(nn.Module):
    r"""Creates a noise embedding module.

    Arguments:
        features: The number of embedding features.
    """

    def __init__(self, features: int):
        self.anchors = jnp.linspace(math.log(1e-3), math.log(1e2), features)
        self.scale = jnp.square(features / (math.log(1e2) - math.log(1e-3)))

    @inox.jit
    def __call__(self, sigma: Array) -> Array:
        x = jnp.log(sigma)
        x = -self.scale * (x - self.anchors) ** 2
        x = jax.nn.softmax(x, axis=-1)

        return x


class ScoreModel(nn.Module):
    r"""Score/noise model based on the EDM preconditioning.

    .. math:: z(x(t), t) \approx E[z | x(t)]

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
        | https://arxiv.org/abs/2206.00364

    Arguments:
        network: A noise conditional network.
        sde: The forward SDE.
    """

    def __init__(self, network: nn.Module, emb_features: int = 64, sde: VESDE = None):
        self.net = network
        self.emb = NoiseEmbedding(emb_features)

        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, xt: Array, t: Array, key: Array = None) -> Array:
        sigma = self.sde.sigma(t)
        sigma = sigma[..., None]
        denum = jnp.sqrt(sigma ** 2 + 1)

        return xt / (sigma + 1 / sigma) + 1 / denum * self.net(xt / denum, self.emb(sigma), key)


class DDPMLoss(nn.Module):
    r"""DDPM loss for a score/noise model.

    .. math:: || A(z - z(x(t), t)) ||^2

    References:
        | Denoising Diffusion Probabilistic Models (Ho et al., 2020)
        | https://arxiv.org/abs/2006.11239

    Arguments:
        sde: The forward SDE.
    """

    def __init__(self, sde: VESDE = None):
        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, model: nn.Module, x: Array, z: Array, t: Array, A: Callable = None, key: Array = None) -> Array:
        xt = self.sde(x, z, t)

        if A is None:
            A = lambda x: x
        else:
            _, A = jax.linearize(A, x)

        error = A(z - model(xt, t, key))

        return jnp.mean(error ** 2)


class EDMLoss(nn.Module):
    r"""EDM loss for a score/noise model.

    .. math:: \lambda(t) || A(x - x(t) + \sigma(t) z(x(t), t)) ||^2

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
        | https://arxiv.org/abs/2206.00364

    Arguments:
        sde: The forward SDE.
    """

    def __init__(self, sde: VESDE = None):
        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, model: nn.Module, x: Array, z: Array, t: Array, A: Callable = None, key: Array = None) -> Array:
        sigma = self.sde.sigma(t)
        sigma = sigma[..., None]
        lmbda = 1 / sigma ** 2 + 1

        xt = self.sde(x, z, t)
        x0 = xt - sigma * model(xt, t, key)

        if A is None:
            A = lambda x: x
        else:
            _, A = jax.linearize(A, x)

        error = A(x - x0)

        return jnp.mean(lmbda * jnp.mean(error ** 2, axis=-1, keepdims=True))


class StandardScoreModel(nn.Module):
    r"""Score model for a standard Gaussian random variable.

    .. math:: z(x(t), t) = \frac{\sigma(t) x(t)}{\sigma(t)^2 + 1}

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
        model: A score/noise model :math:`z(x(t), t) \approx E[z | x(t)]`.
        A: The forward model :math:`\mathcal{A}`.
        y: An observation.
        sigma_y: The observed standard deviation :math:`\sigma_y`.
        sigma_x: The hidden standard deviation :math:`\sigma_x`.
        sde: The forward SDE.
    """

    def __init__(
        self,
        model: nn.Module,
        A: Callable[[Array], Array],
        y: Array,
        sigma_y: Union[float, Array],
        sigma_x: Union[float, Array] = 1.0,
        sde: VESDE = None,
        rtol: float = 1e-5,
        maxiter: int = None,
    ):
        super().__init__()

        self.model = model
        self.A = A
        self.y = jnp.asarray(y)
        self.sigma_y = jnp.asarray(sigma_y)
        self.sigma_x = jnp.asarray(sigma_x)

        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

        self.rtol = rtol
        self.maxiter = maxiter

    @inox.jit
    def __call__(self, xt: Array, t: Array) -> Array:
        sigma = self.sde.sigma(t)
        sigma = sigma[..., None]

        var_y = self.sigma_y ** 2
        var_x = self.sigma_x ** 2 * sigma ** 2 / (self.sigma_x ** 2 + sigma ** 2)

        def denoise(xt):
            z = self.model(xt, t)
            x = xt - sigma * z
            return x, z

        x, vjp, z = jax.vjp(denoise, xt, has_aux=True)
        y, A = jax.linearize(self.A, x)
        At = jax.linear_transpose(self.A, x)

        def cov(y):
            return var_y * y + A(var_x * next(iter(At(y))))

        error = self.y - y
        error, _ = jax.scipy.sparse.linalg.cg(cov, error, tol=self.rtol, maxiter=self.maxiter)
        error, = At(error)
        score, = vjp(error)

        return z - sigma * score
