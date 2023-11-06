r"""Score helpers"""

import math
import torch
import torch.nn as nn

from torch import Tensor
from tqdm import tqdm
from typing import *


class TimeEmbedding(nn.Sequential):
    def __init__(self, features: int, freqs: int = 8):
        super().__init__(
            nn.Linear(2 * freqs, 256),
            nn.ReLU(),
            nn.Linear(256, features),
        )

        self.register_buffer('freqs', torch.pi * torch.arange(1, freqs + 1))

    def forward(self, t: Tensor) -> Tensor:
        t = self.freqs * t.unsqueeze(dim=-1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        return super().forward(t)


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
        super().__init__()

        self.eta = eta

    def alpha(self, t: Tensor) -> Tensor:
        return torch.cos(math.acos(math.sqrt(self.eta)) * t) ** 2

    def sigma(self, t: Tensor) -> Tensor:
        return torch.sqrt(1 - self.alpha(t) ** 2 + self.eta ** 2)

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        t = t[..., None]
        z = torch.randn_like(x)
        x = self.alpha(t) * x + self.sigma(t) * z

        return x, z


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

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,  # x(1)
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> Tensor:
        time = torch.linspace(1, 0, steps + 1).to(x)
        dt = 1 / steps

        for t in tqdm(time[:-1]):
            # Predictor
            r = self.sde.alpha(t - dt) / self.sde.alpha(t)
            x = r * x + (self.sde.sigma(t - dt) - r * self.sde.sigma(t)) * self.model(x, t)

            # Corrector
            for _ in range(corrections):
                z = torch.randn_like(x)
                s = -self.model(x, t - dt) / self.sde.sigma(t - dt)
                delta = tau / s.square().mean(dim=-1, keepdim=True)

                x = x + delta * s + torch.sqrt(2 * delta) * z

        return x

    def loss(self, x: Tensor) -> Tensor:
        t = torch.rand_like(x[..., 0])
        x, z = self.sde(x, t)

        return (self.model(x, t) - z).square().mean()


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

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return x * self.sde.sigma(t[..., None])


class PosteriorScoreModel(nn.Module):
    r"""Posterior score model for a Gaussian observation

    .. math:: p(y | x) = N(y | A(x), \Sigma_y)

    Arguments:
        TODO
    """

    def __init__(
        self,
        model: nn.Module,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        noise: Union[float, Tensor],
        gamma: Union[float, Tensor] = 1.0,
        sde: SDE = None,
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('noise', torch.as_tensor(noise))
        self.register_buffer('gamma', torch.as_tensor(gamma))

        self.model = model
        self.A = A

        if sde is None:
            self.sde = VPSDE()
        else:
            self.sde = sde

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        alpha, sigma = self.sde.alpha(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.clone().requires_grad_()
            z = self.model(x, t)
            x_hat = (x - sigma * z) / alpha

            err = (self.y - self.A(x_hat)).square()
            var = self.noise ** 2 + self.gamma * (sigma / alpha) ** 2

            log_p = -(err / var).sum() / 2

        s, = torch.autograd.grad(log_p, x)

        return z - sigma * s
