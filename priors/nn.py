r"""Neural networks"""

import inox
import inox.nn as nn
import jax.numpy as jnp
import math

from einops import rearrange
from inox.random import PRNG, get_rng, set_rng
from jax import Array
from typing import *


class MLP(nn.Sequential):
    r"""Creates a multi-layer perceptron (MLP).

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hid_features: The number of hidden features.
        activation: The activation function constructor.
        normalize: Whether features are normalized between layers or not.
        key: A PRNG key for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hid_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = nn.ReLU,
        normalize: bool = False,
        key: Array = None,
    ):
        if key is None:
            rng = get_rng()
        else:
            rng = PRNG(key)

        layers = []

        for before, after in zip(
            (in_features, *hid_features),
            (*hid_features, out_features),
        ):
            layers.extend([
                nn.Linear(before, after, key=rng.split()),
                activation(),
                nn.LayerNorm() if normalize else None,
            ])

        layers = filter(lambda l: l is not None, layers[:-2])

        super().__init__(*layers)


class ResBlock(nn.Module):
    r"""Creates a residual block."""

    def __init__(
        self,
        channels: int,
        emb_features: int,
        dropout: float = None,
        **kwargs,
    ):
        self.project = nn.Sequential(
            nn.Linear(emb_features, 2 * channels, bias=False),
            nn.Rearrange('... C -> ... 1 1 C'),
        )

        self.residual = nn.Sequential(
            nn.LayerNorm(),
            nn.Conv(channels, channels, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.TrainingDropout(dropout),
            nn.Conv(channels, channels, bias=False, **kwargs),
        )

    def __call__(self, x: Array, t: Array) -> Array:
        ab = self.project(t)
        a, b = jnp.array_split(ab, 2, axis=-1)

        y = (a + 1) * x + b
        y = self.residual(y)

        return (x + y) / math.sqrt(2)


class SelfAttention2d(nn.Module):
    r"""Creates a 2-d self-attention layer."""

    def __init__(self, channels: int, heads: int = 1):
        self.through = nn.Linear(channels, channels, bias=False)
        self.attention = nn.MultiheadAttention(
            heads=heads,
            in_features=channels,
            out_features=channels,
            hid_features=channels // heads,
        )

    def __call__(self, x: Array) -> Array:
        *_, H, W, C = x.shape

        x = rearrange(x, '... H W C -> ... (H W) C')
        x = self.through(x) + self.attention(x)
        x = rearrange(x, '... (H W) C -> ... H W C', H=H, W=W)

        return x


class AttBlock(nn.Module):
    r"""Creates a residual self-attention block."""

    def __init__(self, channels: int, heads: int = 1):
        self.residual = nn.Sequential(
            nn.LayerNorm(),
            SelfAttention2d(channels, heads),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=False),
        )

    @inox.checkpoint
    def __call__(self, x: Array) -> Array:
        y = self.residual(x)

        return (x + y) / math.sqrt(2)


class UNet(nn.Module):
    r"""Creates a time (or noise) conditional U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Sequence[int] = (3, 3),
        emb_features: int = 64,
        attention: Container[int] = set(),
        heads: int = 1,
        dropout: float = None,
        key: Array = None,
    ):
        if key is None:
            key = get_rng().split()

        stride = [2 for k in kernel_size]
        kwargs = dict(
            kernel_size=kernel_size,
            padding=[(k // 2, k // 2) for k in kernel_size]
        )

        with set_rng(PRNG(key)):
            self.descent, self.ascent = [], []

            for i, blocks in enumerate(hid_blocks):
                do, up = [], []

                for _ in range(blocks):
                    do.append(ResBlock(hid_channels[i], emb_features, dropout=dropout, **kwargs))
                    up.append(ResBlock(hid_channels[i], emb_features, dropout=dropout, **kwargs))

                    if i in attention:
                        do.append(AttBlock(hid_channels[i], heads))
                        up.append(AttBlock(hid_channels[i], heads))

                if i > 0:
                    do.insert(0,
                        nn.Conv(
                            hid_channels[i - 1],
                            hid_channels[i],
                            bias=False,
                            stride=stride,
                            **kwargs,
                        )
                    )

                    up.append(
                        nn.Sequential(
                            nn.Resample(factor=stride, method='nearest'),
                            nn.Conv(hid_channels[i], hid_channels[i - 1], bias=False, **kwargs),
                        )
                    )
                else:
                    do.insert(0, nn.Conv(in_channels, hid_channels[i], **kwargs))
                    up.append(nn.Linear(hid_channels[i], out_channels))

                self.descent.append(do)
                self.ascent.insert(0, up)

    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        if key is None:
            rng = None
        else:
            rng = PRNG(key)

        with set_rng(rng):
            memory = []

            for blocks in self.descent:
                for block in blocks:
                    if isinstance(block, ResBlock):
                        x = block(x, t)
                    else:
                        x = block(x)

                memory.append(x)

            for blocks in self.ascent:
                y = memory.pop()

                if x is not y:
                    x = (x + y) / math.sqrt(2)

                for block in blocks:
                    if isinstance(block, ResBlock):
                        x = block(x, t)
                    else:
                        x = block(x)

            return x
