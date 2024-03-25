r"""GMM experiment helpers"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from functools import partial
from jax import Array
from PIL import Image
from typing import *

from priors.nn import *
from priors.score import *
from priors.train import *


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
