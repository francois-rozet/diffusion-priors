#!/usr/bin/env python

import inox
import io
import random
import zipfile

from datasets import load_from_disk
from dawgz import job, schedule
from functools import partial
from torch import Tensor
from torch.utils import data
from torchvision.transforms.functional import pil_to_tensor
from torch_fidelity.fidelity import calculate_metrics
from tqdm import tqdm
from typing import *

from utils import *
from train import sample


class ZipDataset(data.Dataset):
    r"""Zip image dataset."""

    def __init__(self, archive: Path):
        self.images = []

        with zipfile.ZipFile(archive, mode='r') as file:
            for name in file.namelist():
                with file.open(name) as data:
                    img = Image.open(data)
                    img = img.convert('RGB')

                self.images.append(img)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> Tensor:
        return pil_to_tensor(self.images[i])

    @staticmethod
    def zip(archive: Path, images: List):
        with zipfile.ZipFile(archive, mode='w') as file:
            for i, img in enumerate(images):
                buffer = io.BytesIO()
                img.save(buffer, 'png')
                file.writestr(f'IMG_{i}.png', buffer.getvalue())


def generate(checkpoint: Path, archive: Path, prior: bool = False, seed: int = None):
    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # RNG
    seed = hash((checkpoint, seed)) % 2**16
    rng = inox.random.PRNG(seed)

    # Data
    dataset = load_from_disk(PATH / 'hf/cifar-mask')
    dataset.set_format('numpy')

    # Model
    model = load_module(checkpoint)

    static, arrays = model.partition()
    arrays = jax.device_put(arrays, replicated)
    model = static(arrays)

    # Setting
    if getattr(model, 'sigma_x', None) is None:
        kwargs = {'steps': 64, 'maxiter': 5}
    else:
        kwargs = {'steps': 256, 'maxiter': None}

    # Generate
    images = []

    if prior:
        for _ in range(0, 50000, 256):
            x = sample_any(model, (256, 32 * 32 * 3), key=rng.split(), sampler='ddim', steps=64)
            x = unflatten(x, 32, 32)
            x = np.asarray(x)

            for img in map(to_pil, x):
                images.append(img)
    else:
        loader = dataset['train'].iter(batch_size=256)

        for batch in tqdm(loader):
            y, A = batch['y'], batch['A']
            y, A = jax.device_put((y, A), distributed)
            x = sample(model, y, A, rng.split(), **kwargs)
            x = np.asarray(x)

            for img in map(to_pil, x):
                images.append(img)

    # Archive
    ZipDataset.zip(archive, images)


def fid(archive: Path, run: str, lap: int, prior: bool, seed: int):
    stats = calculate_metrics(
        input1=ZipDataset(archive),
        input2='cifar10-train',
        fid=True,
        isc=True,
    )

    fid = stats['frechet_inception_distance']
    isc = stats['inception_score_mean']

    with open(PATH / f'statistics.csv', mode='a') as f:
        f.write(f'{run},{lap},{prior},{seed},{fid},{isc}\n')


if __name__ == '__main__':
    run = 'daily-dew-78_f0cmqr99'
    runpath = PATH / f'runs/{run}'
    prior = False
    seed = 0

    jobs = []

    for lap in range(16):
        checkpoint = runpath / f'checkpoint_{lap}.pkl'
        archive = runpath / f'archive_{lap}_{prior}_{seed}.zip'

        if not checkpoint.exists():
            break

        a = job(
            partial(generate, checkpoint, archive, prior, seed),
            name=f'generate_{lap}',
            cpus=4,
            gpus=4,
            ram='64GB',
            time='06:00:00',
            partition='gpu',
        )

        b = job(
            partial(fid, archive, run, lap, prior, seed),
            name=f'fid_{lap}',
            cpus=4,
            gpus=1,
            ram='64GB',
            time='01:00:00',
            partition='gpu',
        )

        if not archive.exists():
            b.after(a)

        jobs.append(b)

    schedule(
        *jobs,
        name=f'Eval {run} ({seed})',
        backend='slurm',
        export='ALL',
        account='ariacpg',
    )
