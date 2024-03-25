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


def real2complex(x: Array) -> Array:
    return jax.lax.complex(*jnp.array_split(x, 2, axis=-1))


def complex2real(x: Array) -> Array:
    return jnp.concatenate((x.real, x.imag), axis=-1)


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
    x = unflatten(x, 320, 320)
    x = real2complex(x)
    x = ifft2c(x).real

    return to_pil(x, zoom=zoom)


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
        network=SpectralUNet(
            in_channels=32,
            out_channels=32,
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


class SpectralUNet(UNet):
    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        x = unflatten(x, width=320, height=320)
        x = complex2real(ifft2c(real2complex(x)))
        x = rearrange(x, '... (H h) (W w) C -> ... H W (h w C)', h=4, w=4)
        x = super().__call__(x, t, key)
        x = rearrange(x, '... H W (h w C) -> ... (H h) (W w) C', h=4, w=4)
        x = complex2real(fft2c(real2complex(x)))
        x = flatten(x)

        return x
