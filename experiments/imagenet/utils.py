r"""ImageNet experiment helpers"""

import numpy as np
import os

from jax import Array
from pathlib import Path
from PIL import Image
from typing import *

from priors.data import *
from priors.image import *
from priors.nn import *
from priors.score import *
from priors.train import *


if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'priors/imagenet'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def resize(image: Image, res: int = 64) -> Image:
    image = image.convert('RGB')
    image = image.resize(
        size=(res, res),
        resample=Image.LANCZOS,
        box=(0, 0, min(image.size), min(image.size)),
    )

    return Image.fromarray(np.asarray(image))


def collate(batch: Dict) -> Array:
    img = batch['image']

    if isinstance(img, list):
        x = np.stack(list(map(from_pil, img)))
    else:
        x = from_pil(img)

    return flatten(x)


def show(x: Array, zoom: int = 1) -> Image:
    return to_pil(unflatten(x, 64, 64), zoom=zoom)


def make_model(
    key: Array,
    hid_channels: Sequence[int] = (64, 128, 256),
    hid_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: Sequence[int] = (3, 3),
    emb_features: int = 256,
    heads: Dict[int, int] = {2: 1},
    dropout: float = None,
    **absorb,
) -> ScoreModel:
    return ScoreModel(
        network=FlatUNet(
            in_channels=3,
            out_channels=3,
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
        x = unflatten(x, width=64, height=64)
        x = super().__call__(x, t, key)
        x = flatten(x)

        return x
