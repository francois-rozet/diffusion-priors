r"""Neural networks"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import *


class Residual(nn.Sequential):
    r"""Creates a residual block."""

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class LayerNorm(nn.Module):
    r"""Creates a normalization layer that standardizes features along a dimension.

    References:
       Layer Normalization (Lei Ba et al., 2016)
       https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) to standardize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Iterable[int]] = -1, eps: float = 1e-5):
        super().__init__()

        self.dim = dim if type(dim) is int else tuple(dim)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, unbiased=True, dim=self.dim, keepdim=True)

        return (x - mean) / (variance + self.eps).sqrt()


class MLP(nn.Sequential):
    r"""Creates a multi-layer perceptron (MLP).

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The number of hidden features.
        activation: The activation function constructor.
        normalize: Whether features are normalized between layers or not.
        kwargs: Keyword arguments passed to :class:`nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = nn.ReLU,
        normalize: bool = False,
        **kwargs,
    ):
        layers = []

        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            layers.extend([
                nn.Linear(before, after, **kwargs),
                activation(),
                LayerNorm() if normalize else None,
            ])

        layers = filter(lambda l: l is not None, layers[:-2])

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features
