r"""FastMRI experiment helpers"""

import numpy as np
import os

from jax import Array
from pathlib import Path
from typing import *

from priors.data import *
from priors.image import *
from priors.nn import *
from priors.score import *
from priors.train import *


if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'priors/fastmri'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def fft2c(x: Array, norm: str = 'ortho') -> Array:
    return jnp.fft.fftshift(
        jnp.fft.fft2(
            jnp.fft.ifftshift(
                x,
                axes=(-3, -2)
            ),
            axes=(-3, -2),
            norm=norm,
        ),
        axes=(-3, -2),
    )


def ifft2c(k: Array, norm: str = 'ortho') -> Array:
    return jnp.fft.fftshift(
        jnp.fft.ifft2(
            jnp.fft.ifftshift(
                k,
                axes=(-3, -2),
            ),
            axes=(-3, -2),
            norm=norm,
        ),
        axes=(-3, -2),
    )


def show(x: Array, zoom: int = 1) -> Image:
    return to_pil(unflatten(x, 320, 320), zoom=zoom)


def make_model(
    key: Array,
    hid_channels: Sequence[int] = (64, 128, 256),
    hid_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: Sequence[int] = (3, 3),
    emb_features: int = 256,
    heads: Dict[int, int] = {2: 1},
    dropout: float = None,
    **absorb,
) -> Denoiser:
    return Denoiser(
        network=FlatUNet(
            in_channels=1,
            out_channels=1,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            emb_features=emb_features,
            heads=heads,
            dropout=dropout,
            key=key,
        ),
        emb_features=emb_features,
    )


class FlatUNet(UNet):
    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        x = unflatten(x, width=320, height=320)
        x = super().__call__(x, t, key)
        x = flatten(x)

        return x
