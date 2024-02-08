r"""Image helpers"""

import numpy as np

from einops import rearrange
from jax import Array
from PIL import Image
from typing import *


def flatten(x: Array) -> Array:
    return rearrange(x, '... H W C -> ... (H W C)')


def unflatten(x: Array, height: int, width: int) -> Array:
    return rearrange(x, '... (H W C) -> ... H W C', H=height, W=width)


def from_pil(img: Image) -> Array:
    x = np.asarray(img)
    x = x + np.random.uniform(size=x.shape)
    x = x * (4 / 256) - 2

    return x


def to_pil(x: Array, zoom: int = 1) -> Image:
    x = np.asarray(x)
    x = np.clip((x + 2) * (256 / 4), 0, 255)
    x = x.astype(np.uint8)
    x = np.tile(x, (1, 1, 1, 1, 1))
    x = rearrange(x, 'M N H W C -> (M H) (N W) C')

    if x.shape[-1] == 1:
        x = Image.fromarray(x.squeeze(-1), mode='L')
    else:
        x = Image.fromarray(x, mode='RGB')

    if zoom > 1:
        x = x.resize((zoom * x.width, zoom * x.height), Image.NEAREST)

    return x
