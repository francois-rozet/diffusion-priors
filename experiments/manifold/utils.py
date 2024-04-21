r"""Manifold experiment helpers"""

import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import os
import ot

from functools import partial
from jax import Array
from PIL import Image
from typing import *

from priors.common import *
from priors.image import *
from priors.nn import *
from priors.score import *


if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'priors/manifold'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def show(x: Array, zoom: int = 2, **kwargs) -> Image.Image:
    kwargs.setdefault('bins', 128)
    kwargs.setdefault('range', ((-3.0, 3.0), (-3.0, 3.0)))

    # Histogram
    p, _, _ = jnp.histogram2d(x[:, 0], x[:, 1], **kwargs)

    # Color map
    cmap = plt.get_cmap('magma')
    cmap = plt.cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=0)

    # Image
    p = np.asarray(p)
    if np.any(p > 0):
        p = p / np.sum(p)
    p = p ** 0.5

    img = cmap.to_rgba(p, bytes=True)
    img = Image.fromarray(img)

    if zoom > 1:
        img = img.resize((img.height * zoom, img.width * zoom), Image.NEAREST)

    return img


def corner(x: Array, pad: int = 4, **kwargs) -> Image.Image:
    m, n = x.shape

    images = [
        [
            show(x[:, i:j+1:j-i], **kwargs) if i < j else None
            for j in range(1, n)
        ]
        for i in range(n - 1)
    ]

    return collate(images, pad=pad)


def sinkhorn_divergence(
    u1: Array,
    u2: Array,
    v: Array,
    lmbda: float = 1e-3,
    maxiter: int = 1024,
    epsilon: float = 1e-3,
) -> Array:
    r"""Computes the Sinkhorn divergence between two samples.

    References:
        | Faster Wasserstein Distance Estimation with the Sinkhorn Divergence (Chizat et al., 2020)
        | https://arxiv.org/abs/2006.08172
    """

    def transport(u, v):
        return ot.sinkhorn2(
            a=jnp.asarray(()),
            b=jnp.asarray(()),
            M=ot.dist(u, v),
            reg=lmbda,
            numItermax=maxiter,
            stopThr=epsilon,
            method='sinkhorn_log',
        )

    return jnp.maximum(transport(u1, v) - transport(u1, u2), 1e-6)


def smooth_manifold(
    key: Array,
    shape: Sequence[int] = (),
    m: int = 1,
    n: int = 3,
    alpha: float = 3.0,
    epsilon: float = 1e-3,
) -> Array:
    r"""Samples points from a smooth random manifold.

    References:
        https://github.com/fzenke/randman

    Arguments:
        m: The manifold dimension.
        n: The space dimension.
        alpha: The smoothness coefficient.
    """

    key_params, key_z = jax.random.split(key, 2)

    cutoff = math.ceil(epsilon ** (-1 / alpha))
    k = jnp.arange(cutoff) + 1

    a, b, c = jax.random.uniform(key_params, (3, n, m, cutoff))

    z = jax.random.uniform(key_z, (*shape, 1, m, 1))
    x = a / k ** alpha * jnp.sin(2 * jnp.pi * (k * b * z + c))
    x = jnp.sum(x, axis=-1)
    x = jnp.prod(x, axis=-1)

    return x


def make_model(
    key: Array,
    features: int,
    hid_features: Sequence[int] = (256, 256, 256),
    emb_features: int = 256,
    normalize: bool = True,
    **absorb,
) -> Denoiser:
    return Denoiser(
        network=TimeMLP(
            features=features,
            hid_features=hid_features,
            emb_features=emb_features,
            normalize=normalize,
            key=key,
        ),
        emb_features=emb_features,
    )


class TimeMLP(MLP):
    def __init__(self, features: int, emb_features: int = 64, **kwargs):
        super().__init__(features + emb_features, features, **kwargs)

    @staticmethod
    @partial(jnp.vectorize, signature='(m),(n)->(p)')
    def cat(x: Array, y: Array) -> Array:
        return jnp.concatenate((x, y))

    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        return super().__call__(self.cat(x, t))
