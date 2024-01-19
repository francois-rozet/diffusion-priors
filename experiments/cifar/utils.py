r"""CIFAR experiment helpers"""

import inox
import jax
import numpy as np
import os
import pickle

from datasets import load_dataset
from einops import rearrange
from jax import Array
from pathlib import Path
from PIL import Image
from typing import *

from priors.nn import *
from priors.score import *


if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'priors/cifar'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def cifar(split: str = 'train'):
    return load_dataset('cifar10', split=split, keep_in_memory=True)

def flatten(x: Array) -> Array:
    return rearrange(x, '... H W C -> ... (H W C)')

def unflatten(x: Array) -> Array:
    return rearrange(x, '... (H W C) -> ... H W C', H=32, W=32)

def process(batch: Dict) -> Array:
    img = batch['img']

    if isinstance(img, list):
        x = np.stack(list(map(np.asarray, img)))
    else:
        x = np.asarray(img)

    x = x + np.random.uniform(size=x.shape)
    x = x * (4 / 256) - 2
    x = flatten(x)

    return x

def show(x: Array, zoom: int = 4) -> Image:
    x = np.asarray(x)
    x = np.clip((x + 2) * (256 / 4), 0, 255)
    x = x.astype(np.uint8)
    x = np.tile(x, (1, 1, 1))
    x = unflatten(x)
    x = rearrange(x, 'M N H W C -> (M H) (N W) C')

    x = Image.fromarray(x)

    if zoom > 1:
        x = x.resize((zoom * x.width, zoom * x.height), Image.NEAREST)

    return x

def dump_model(model: ScoreModel, file: Path):
    with open(file, 'wb') as f:
        pickle.dump(model, f)

def load_model(file: Path) -> ScoreModel:
    with open(file, 'rb') as f:
        return pickle.load(f)

def make_model(
    key: Array = None,
    hid_channels: Sequence[int] = (64, 128, 256),
    hid_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: Sequence[int] = (3, 3),
    emb_features: int = 256,
    attention: Container[int] = {2},
    dropout: float = 0.0,
    **absorb,
) -> ScoreModel:
    if key is None:
        rng = inox.random.get_rng()
    else:
        rng = inox.random.PRNG(key)

    with inox.random.set_rng(rng):
        return ScoreModel(
            network=FlatUNet(
                in_channels=3,
                out_channels=3,
                hid_channels=hid_channels,
                hid_blocks=hid_blocks,
                kernel_size=kernel_size,
                emb_features=emb_features,
                attention=attention,
                dropout=dropout,
            ),
            emb_features=emb_features,
        )


class FlatUNet(UNet):
    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        x = unflatten(x)
        x = super().__call__(x, t, key)
        x = flatten(x)

        return x
