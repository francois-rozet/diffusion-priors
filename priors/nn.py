r"""Neural networks"""

import inox
import inox.nn as nn
import jax
import jax.numpy as jnp
import math

from einops import rearrange
from inox.random import *
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


class Checkpoint(nn.Module):
    r"""Gradient checkpointing module."""

    def __init__(self, fun: nn.Module):
        self.fun = fun

    @inox.checkpoint
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fun(*args, **kwargs)


class ResBlock(nn.Module):
    r"""Creates a residual block."""

    def __init__(
        self,
        channels: int,
        emb_features: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        self.project = nn.Sequential(
            nn.Linear(emb_features, channels),
            nn.Rearrange('... C -> ... 1 1 C'),
        )

        self.block = nn.Sequential(
            nn.GroupNorm(min(16, channels // 16)),
            nn.Conv(channels, channels, **kwargs),
            nn.SiLU(),
            nn.TrainingDropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(channels, channels),
        )

    def __call__(self, x: Array, t: Array) -> Array:
        y = x + self.project(t)
        y = self.block(y)

        return (x + y) / math.sqrt(2)


class SelfAttention2d(nn.MultiheadAttention):
    r"""Creates a 2-d self-attention layer."""

    def __init__(self, channels: int, heads: int):
        super().__init__(
            heads=heads,
            in_features=channels,
            out_features=channels,
            hid_features=channels // heads,
        )

    def __call__(self, x: Array) -> Array:
        r""""""

        *_, H, W, C = x.shape

        x = rearrange(x, '... H W C -> ... (H W) C')
        x = super().__call__(x)
        x = rearrange(x, '... (H W) C -> ... H W C', H=H, W=W)

        return x


class AttnBlock(nn.Module):
    r"""Creates a residual attention block."""

    def __init__(self, channels: int, heads: int = 8):
        self.block = nn.Sequential(
            nn.GroupNorm(min(16, channels // 16)),
            Checkpoint(SelfAttention2d(channels, heads)),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def __call__(self, x: Array) -> Array:
        y = self.block(x)

        return (x + y) / math.sqrt(2)


class UNet(nn.Module):
    r"""Creates a time-conditioned U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (128, 192, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Sequence[int] = (3, 3),
        emb_features: int = 64,
        attention: Container[int] = set(),
        dropout: float = 0.1,
        key: Array = None,
    ):
        if key is None:
            key = get_rng().split()

        kwargs = dict(
            kernel_size=kernel_size,
            padding=[(k // 2, k // 2) for k in kernel_size]
        )

        with set_rng(PRNG(key)):
            self.blocks_down, self.blocks_up = [], []

            for i, blocks in enumerate(hid_blocks):
                do, up = [], []

                for _ in range(blocks):
                    do.append(ResBlock(hid_channels[i], emb_features, dropout=dropout, **kwargs))
                    up.append(ResBlock(hid_channels[i], emb_features, dropout=dropout, **kwargs))

                    if i in attention:
                        do.append(AttnBlock(hid_channels[i]))
                        up.append(AttnBlock(hid_channels[i]))

                if i > 0:
                    do.insert(0,
                        nn.Conv(
                            hid_channels[i - 1],
                            hid_channels[i],
                            kernel_size=kernel_size,
                            padding=[(k // 2, k // 2) for k in kernel_size],
                            stride=2,
                        )
                    )

                    up.append(
                        nn.Sequential(
                            nn.GroupNorm(min(16, hid_channels[i] // 16)),
                            nn.ConvTransposed(
                                hid_channels[i],
                                hid_channels[i - 1],
                                kernel_size=kernel_size,
                                padding=[(k // 2, k // 2 - 1) for k in kernel_size],
                                stride=2,
                            ),
                        )
                    )
                else:
                    do.insert(0, nn.Conv(in_channels, hid_channels[i], **kwargs))
                    up.append(nn.Linear(hid_channels[i], out_channels))

                self.blocks_down.append(do)
                self.blocks_up.insert(0, up)

    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        if key is None:
            rng = None
        else:
            rng = PRNG(key)

        with set_rng(rng):
            memory = []

            for blocks in self.blocks_down:
                for block in blocks:
                    if isinstance(block, ResBlock):
                        x = block(x, t)
                    else:
                        x = block(x)

                memory.append(x)

            for blocks in self.blocks_up:
                y = memory.pop()

                if x is not y:
                    x = x + y

                for block in blocks:
                    if isinstance(block, ResBlock):
                        x = block(x, t)
                    else:
                        x = block(x)

            return x
