r"""Image helpers"""

import jax
import jax.numpy as jnp
import numpy as np

from einops import rearrange
from jax import Array
from PIL import Image
from typing import *


def flatten(x: Array) -> Array:
    return rearrange(x, '... H W C -> ... (H W C)')


def unflatten(x: Array, height: int, width: int) -> Array:
    return rearrange(x, '... (H W C) -> ... H W C', H=height, W=width)


def from_pil(img: Image.Image) -> Array:
    x = np.asarray(img)
    x = x * (4 / 256) - 2

    return x


def to_pil(x: Array, zoom: int = 1) -> Image.Image:
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


def collate(
    images: List[List[Image.Image]],
    pad: int = 4,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    M, N = len(images), max(map(len, images))

    for i in range(M):
        for j in range(N):
            try:
                W, H = images[i][j].size
            except:
                continue

    canvas = Image.new(
        'RGB',
        size=(
            N * (W + pad) + pad,
            M * (H + pad) + pad,
        ),
        color=background,
    )

    for i in range(M):
        for j in range(N):
            offset = (
                j * (W + pad) + pad,
                i * (H + pad) + pad,
            )

            try:
                canvas.paste(images[i][j], offset)
            except:
                continue

    return canvas


def rand_flip(x: Array, key: Array, axis: int = -2) -> Array:
    return jnp.where(
        jax.random.bernoulli(key),
        x,
        jnp.flip(x, axis=axis),
    )


def rand_shake(x: Array, key: Array, delta: int = 1, mode: str = 'reflect') -> Array:
    i = jax.random.randint(key, shape=(3,), minval=0, maxval=2 * delta + 1)
    i = i.at[-1].set(0)

    return jax.lax.dynamic_slice(
        jnp.pad(
            x,
            pad_width=((delta, delta), (delta, delta), (0, 0)),
            mode=mode,
        ),
        start_indices=i,
        slice_sizes=x.shape,
    )
