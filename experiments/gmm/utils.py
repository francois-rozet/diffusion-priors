r"""GMM experiment helpers"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from functools import partial
from jax import Array
from PIL import Image
from typing import *

from priors.common import *
from priors.nn import *
from priors.score import *


def show(x: Array, zoom: int = 4, **kwargs) -> Image:
    kwargs.setdefault('bins', 64)
    kwargs.setdefault('density', True)
    kwargs.setdefault('range', ((-3.0, 3.0), (-3.0, 3.0)))

    hist, _, _ = np.histogram2d(x[:, 0], x[:, 1], **kwargs)
    hist = plt.cm.ScalarMappable().to_rgba(hist, bytes=True)
    hist = Image.fromarray(hist)

    if zoom > 1:
        hist = hist.resize((hist.height * zoom, hist.width * zoom), Image.NEAREST)

    return hist


def make_data(n: int, key: Array) -> Tuple[Array, Array, Array]:
    keys = jax.random.split(key, 5)

    # x
    modes = jax.random.uniform(keys[0], (8, 5), minval=-2.0, maxval=2.0)

    i = jax.random.randint(keys[1], (n,), minval=0, maxval=len(modes))
    x = jax.random.normal(keys[2], (n, 5)) / 8
    x = modes[i] + x

    # A
    A = jax.random.normal(keys[3], (n, 2, 5))
    A = A / jnp.linalg.norm(A, axis=-1, keepdims=True)

    # y
    y = measure(A, x) + 1e-3 * jax.random.normal(keys[4], (n, 2))

    return x, A, y


def measure(A: Array, x: Array) -> Array:
    return jnp.einsum('...ij,...j->...i', A, x)


def sample(model: nn.Module, A: Array, y: Array, key: Array, **kwargs) -> Array:
    return sample_any(
        model=model,
        shape=(len(y), 5),
        A=inox.Partial(measure, A),
        y=y,
        sigma_y=1e-3 ** 2,
        key=key,
        **kwargs,
    )


def make_model(
    key: Array,
    hid_features: Sequence[int] = (256, 256, 256),
    emb_features: int = 256,
    normalize: bool = True,
    **absorb,
) -> Denoiser:
    return Denoiser(
        network=TimeMLP(
            features=5,
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
