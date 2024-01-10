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


class Residual(nn.Sequential):
    r"""Creates a residual block."""

    def __call__(self, x: Array) -> Array:
        return x + super().__call__(x)


class MLP(nn.Sequential):
    r"""Creates a multi-layer perceptron (MLP).

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The number of hidden features.
        activation: The activation function constructor.
        normalize: Whether features are normalized between layers or not.
        key: A PRNG key for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
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
            (in_features, *hidden_features),
            (*hidden_features, out_features),
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


class IRBlock(nn.Module):
    r"""Creates an inverted residual (IR) block."""

    def __init__(
        self,
        channels: int,
        embedding: int,
        **kwargs,
    ):
        self.project = nn.Sequential(
            nn.Linear(embedding, channels, bias=False),
            nn.Rearrange('... C -> ... 1 1 C'),
        )

        self.block = nn.Sequential(
            nn.GroupNorm(4),
            nn.Conv(channels, 4 * channels, **kwargs),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
        )

    def __call__(self, x: Array, t: Array) -> Array:
        y = x + self.project(t)
        y = self.block(y)

        return (x + y) / math.sqrt(2)


class SelfAttention2d(nn.MultiheadAttention):
    r"""Creates a 2-d self-attention layer."""

    def __init__(
        self,
        heads: int,
        in_features: int,
        hid_features: int = 16,
    ):
        super().__init__(heads, in_features, hid_features)

    def __call__(self, x: Array) -> Array:
        r""""""

        *_, H, W, C = x.shape

        x = rearrange(x, '... H W C -> ... (H W) C')
        x = super().__call__(x)
        x = rearrange(x, '... (H W) C -> ... H W C', H=H, W=W)

        return x


class UNet(nn.Module):
    r"""Creates a time-conditioned U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (2, 2, 2),
        kernel_size: Sequence[int] = (3, 3),
        emb_features: int = 64,
        key: Array = None,
    ):
        if key is None:
            key = get_rng().split()

        with set_rng(PRNG(key)):
            # Layers
            self.blocks_down, self.blocks_up = [], []
            self.skips = []

            kwargs = dict(
                kernel_size=kernel_size,
                padding=[(k // 2, k // 2) for k in kernel_size]
            )

            for i, blocks in enumerate(hid_blocks):
                do, up = [], []

                if i > 0:
                    do.append(
                        nn.Conv(
                            hid_channels[i - 1],
                            hid_channels[i],
                            kernel_size=kernel_size,
                            padding=[(k // 2, k // 2) for k in kernel_size],
                            stride=2,
                        )
                    )

                    up.append(
                        nn.ConvTransposed(
                            hid_channels[i],
                            hid_channels[i - 1],
                            kernel_size=kernel_size,
                            padding=[(k // 2, k // 2 - 1) for k in kernel_size],
                            stride=2,
                        )
                    )
                else:
                    do.append(nn.Conv(in_channels, hid_channels[i], **kwargs))
                    up.append(nn.Linear(hid_channels[i], out_channels))

                for j in range(blocks):
                    do.append(IRBlock(hid_channels[i], emb_features, **kwargs))
                    up.append(IRBlock(hid_channels[i], emb_features, **kwargs))

                self.blocks_down.append(do)
                self.blocks_up.insert(0, list(reversed(up)))
                self.skips.insert(0, nn.Linear(2 * hid_channels[i], hid_channels[i]))

            self.attn = SelfAttention2d(
                heads=8,
                in_features=hid_channels[-1],
                hid_features=64,
            )

    def __call__(self, x: Array, t: Array) -> Array:
        memory = []

        for blocks in self.blocks_down:
            for block in blocks:
                if isinstance(block, IRBlock):
                    x = block(x, t)
                else:
                    x = block(x)

            memory.append(x)

        x = self.attn(x)

        for skip, blocks in zip(self.skips, self.blocks_up):
            x = jnp.concatenate((x, memory.pop()), axis=-1)
            x = skip(x)

            for block in blocks:
                if isinstance(block, IRBlock):
                    x = block(x, t)
                else:
                    x = block(x)

        return x
