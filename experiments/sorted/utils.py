r"""Sorted experiment helpers"""

import inox
import inox.nn as nn
import jax
import jax.numpy as jnp
import optax

from functools import partial
from inox.random import PRNG
from jax import Array
from typing import *

from priors.nn import *
from priors.score import *
from priors.plots import corner


DOMAIN = -3.0 * jnp.ones(10), 3.0 * jnp.ones(10)


def generate(n: int, key: Array = None) -> Array:
    return jnp.sort(jax.random.normal(key, shape=(n, 10)), axis=-1)

def show(x: Array, **kwargs) -> object:
    return corner(x[..., ::3], domain=DOMAIN, smooth=1, **kwargs)

def make_model(
    key: Array,
    hid_features: Sequence[int] = (256, 256),
    emb_features: int = 64,
    normalize: bool = True,
    **absorb,
) -> ScoreModel:
    return ScoreModel(
        network=TimeMLP(
            features=10,
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
