r"""Score helpers"""

from __future__ import annotations

import inox
import inox.nn as nn
import jax
import jax.numpy as jnp
import math

from jax import Array
from typing import *


class VESDE(nn.Module):
    r"""Variance exploding (VE) SDE.

    .. math:: x_t = x + \sigma_t z

    with

    .. math:: \sigma_t = \exp(\log(a) (1 - t) + \log(b) t)

    Arguments:
        a: The noise lower bound.
        b: The noise upper bound.
    """

    def __init__(self, a: Array = 1e-3, b: Array = 1e2):
        self.a = jnp.log(a)
        self.b = jnp.log(b)

    @inox.jit
    def __call__(self, x: Array, z: Array, t: Array) -> Array:
        sigma_t = self.sigma(t)
        sigma_t = sigma_t[..., None]

        return x + sigma_t * z

    @inox.jit
    def sigma(self, t: Array) -> Array:
        return jnp.exp(self.a + (self.b - self.a) * t)


class DDPM(nn.Module):
    r"""DDPM sampler for the reverse SDE.

    .. math:: x_s = x_t - \tau (x_t - f(x_t)) + \sigma_s \sqrt{\tau} \epsilon

    where :math:`\tau = 1 - \frac{\sigma_s^2}{\sigma_t^2}`.

    Arguments:
        model: A denoiser model :math:`f(x_t) \approx E[x | x_t]`.
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
    def __call__(self, z: Array, t: Array = 1.0, steps: int = 64, key: Array = None) -> Array:
        dt = jnp.asarray(t / steps)
        time = jnp.linspace(t, dt, steps)
        keys = jax.random.split(key, steps)

        def f(xt, t_key):
            t, key = t_key
            return self.step(xt, t, t - dt, key), None

        xt = z * self.sde.sigma(t)
        x0, _ = jax.lax.scan(f, xt, (time, keys))

        return self.model(x0, 0.0)

    @inox.jit
    def step(self, xt: Array, t: Array, s: Array, key: Array) -> Array:
        tau = 1 - (self.sde.sigma(s) / self.sde.sigma(t)) ** 2
        eps = jax.random.normal(key, xt.shape)

        return xt - tau * (xt - self.model(xt, t)) + self.sde.sigma(s) * jnp.sqrt(tau) * eps


class DDIM(DDPM):
    r"""DDIM sampler for the reverse SDE.

    .. math:: x_s = x_t - (1 - \frac{\sigma_s}{\sigma_t}) (x_t - f(x_t))

    Arguments:
        model: A denoiser model :math:`f(x_t) \approx E[x | x_t]`.
        sde: The forward SDE.
    """

    @inox.jit
    def step(self, xt: Array, t: Array, s: Array, key: Array = None) -> Array:
        return xt - (1 - self.sde.sigma(s) / self.sde.sigma(t)) * (xt - self.model(xt, t))


class PredictorCorrector(DDPM):
    r"""Predictor-Corrector sampler for the reverse SDE.

    Arguments:
        model: A denoiser model :math:`f(x_t) \approx E[x | x(t)]`.
        corrections: The number of Langevin Monte Carlo (LMC) corrections.
        tau: The LMC step size.
    """

    def __init__(self, model: nn.Module, corrections: int = 1, tau: Array = 1e-2, **kwargs):
        super().__init__(model, **kwargs)

        self.corrections = corrections
        self.tau = jnp.asarray(tau)

    @inox.jit
    def step(self, xt: Array, t: Array, s: Array, key: Array) -> Array:
        xs = self.predict(xt, t, s)

        for key in jax.random.split(key, self.corrections):
            xs = self.correct(xs, s, key)

        return xs

    @inox.jit
    def predict(self, xt: Array, t: Array, s: Array) -> Array:
        return xt - (1 - self.sde.sigma(s) / self.sde.sigma(t)) * (xt - self.model(xt, t))

    @inox.jit
    def correct(self, xt: Array, t: Array, key: Array) -> Array:
        eps = jax.random.normal(key, xt.shape)

        return xt - self.tau * (xt - self.model(xt, t)) + self.sde.sigma(t) * jnp.sqrt(2 * self.tau) * eps


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


class Denoiser(nn.Module):
    r"""Denoiser model with EDM-style preconditioning.

    .. math:: f(x_t) \approx E[x | x_t]

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
        sigma_t = self.sde.sigma(t)
        sigma_t = sigma_t[..., None]

        c_skip = 1 / (sigma_t ** 2 + 1)
        c_out = sigma_t / jnp.sqrt(sigma_t ** 2 + 1)
        c_in = 1 / jnp.sqrt(sigma_t ** 2 + 1)

        return c_skip * xt + c_out * self.net(c_in * xt, self.emb(sigma_t), key)


class DenoiserLoss(nn.Module):
    r"""Loss for a denoiser model.

    .. math:: \lambda_t || A f(x_t) - y ||^2

    Arguments:
        sde: The forward SDE.
    """

    def __init__(self, sde: VESDE = None):
        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(
        self,
        model: nn.Module,
        x: Array,
        z: Array,
        t: Array,
        A: Callable[[Array], Array] = None,  # /!\ linear
        y: Array = None,
        key: Array = None,
    ) -> Array:
        sigma_t = self.sde.sigma(t)
        lmbda_t = 1 / sigma_t ** 2 + 1

        xt = self.sde(x, z, t)
        ft = model(xt, t, key)

        if A is None:
            A = lambda x: x

        if y is None:
            y = A(x)

        error = A(ft) - y

        return jnp.mean(lmbda_t * jnp.mean(error ** 2, axis=-1))


class DPLR(NamedTuple):
    r"""Diagonal plus low-rank matrix."""

    D: Array
    U: Array = None
    V: Array = None

    def __add__(self, C: Array) -> DPLR:
        return DPLR(self.D + C, self.U, self.V)

    def __radd__(self, C: Array) -> DPLR:
        return DPLR(C + self.D, self.U, self.V)

    def __sub__(self, C: Array) -> DPLR:
        return DPLR(self.D - C, self.U, self.V)

    def __mul__(self, C: Array) -> DPLR:
        D = self.D * C

        if self.U is None:
            U, V = None, None
        else:
            U, V = self.U, self.V * C[..., None, :]

        return DPLR(D, U, V)

    def __rmul__(self, C: Array) -> DPLR:
        D = C * self.D

        if self.U is None:
            U, V = None, None
        else:
            U, V = C[..., None] * self.U, self.V

        return DPLR(D, U, V)

    def __matmul__(self, x: Array) -> Array:
        if self.U is None:
            return self.D * x
        else:
            return self.D * x + jnp.einsum('...ij,...jk,...k', self.U, self.V, x)

    def __call__(self, x: Array) -> Array:
        return self @ x

    @property
    def rank(self) -> int:
        if self.U is None:
            return 0
        else:
            return self.U.shape[-1]

    @property
    def inv(self) -> DPLR:
        D = 1 / self.D

        if self.U is None:
            U, V = None, None
        else:
            U = -D[..., None] * self.U
            V = jnp.linalg.solve(jnp.eye(self.rank) + jnp.einsum('...ik,...k,...kj', self.V, D, self.U), self.V) * D[..., None, :]

        return DPLR(D, U, V)

    def diag(self) -> Array:
        if self.U is None:
            return self.D
        else:
            return self.D + jnp.einsum('...ij,...ji->...i', self.U, self.V)

    def norm(self) -> Array:
        if self.U is None:
            return jnp.sum(self.D ** 2, axis=-1)
        else:
            return jnp.sum(self.D ** 2, axis=-1) + 2 * jnp.einsum('...i,...ij,...ji', self.D, self.U, self.V) + jnp.sum((self.V @ self.U) ** 2, axis=(-1, -2))


class GaussianDenoiser(nn.Module):
    r"""Denoiser model for a Gaussian random variable.

    .. math:: p(x) = N(x | \mu_x, \Sigma_x)

    Arguments:
        mu_x: The data mean :math:`\mu_x`.
        sigma_x: The data covariance :math:`\Sigma_x`.
        sde: The forward SDE.
    """

    def __init__(
        self,
        mu_x: Array = 0.0,
        sigma_x: Union[Array, DPLR] = 1.0,
        sde: VESDE = None,
    ):
        if not isinstance(sigma_x, DPLR):
            sigma_x = DPLR(sigma_x)

        self.mu_x = jnp.asarray(mu_x)
        self.sigma_x = jax.tree_util.tree_map(jnp.asarray, sigma_x)

        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, xt: Array, t: Array, key: Array = None) -> Array:
        sigma_t = self.sde.sigma(t)
        sigma_t = sigma_t[..., None] ** 2

        return xt - sigma_t * (self.sigma_x + sigma_t).inv(xt - self.mu_x)


class PosteriorDenoiser(nn.Module):
    r"""Posterior denoiser model for a Gaussian observation.

    .. math:: p(y | x) = N(y | Ax, \Sigma_y)

    Arguments:
        model: A denoiser model :math:`f(x_t) \approx E[x | x_t]`.
        A: The forward model :math:`A`.
        y: An observation.
        sigma_y: The observation covariance :math:`\Sigma_y`.
        sigma_x: The data covariance :math:`\Sigma_x`.
        sde: The forward SDE.
    """

    def __init__(
        self,
        model: nn.Module,
        A: Callable[[Array], Array],
        y: Array,
        sigma_y: Union[Array, DPLR],
        sigma_x: Union[Array, DPLR] = 1.0,
        sde: VESDE = None,
        rtol: float = 1e-3,
        maxiter: int = None,
    ):
        super().__init__()

        self.model = model
        self.A = A
        self.y = jnp.asarray(y)

        if not isinstance(sigma_y, DPLR):
            sigma_y = DPLR(sigma_y)

        if not isinstance(sigma_x, DPLR):
            sigma_x = DPLR(sigma_x)

        self.sigma_y = jax.tree_util.tree_map(jnp.asarray, sigma_y)
        self.sigma_x = jax.tree_util.tree_map(jnp.asarray, sigma_x)

        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

        self.rtol = rtol
        self.maxiter = maxiter

    @inox.jit
    def __call__(self, xt: Array, t: Array, key: Array = None) -> Array:
        sigma_t = self.sde.sigma(t)
        sigma_t = sigma_t[..., None] ** 2

        x, vjp = jax.vjp(lambda xt: self.model(xt, t, key), xt)
        y, A = jax.linearize(self.A, x)
        At = jax.linear_transpose(self.A, x)

        sigma_x_xt = sigma_t + (-sigma_t) * (self.sigma_x + sigma_t).inv * sigma_t
        sigma_y_xt = lambda vec: self.sigma_y(vec) + A(sigma_x_xt(*At(vec)))

        error = self.y - y
        error, _ = jax.scipy.sparse.linalg.cg(sigma_y_xt, error, tol=self.rtol, maxiter=self.maxiter)
        error, = At(error)
        score, = vjp(error)

        return x + sigma_t * score
