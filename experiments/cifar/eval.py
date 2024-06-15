#!/usr/bin/env python

import inox
import io
import zipfile

from dawgz import job, schedule
from functools import partial
from torch import Tensor
from torch.utils import data
from torch_fidelity.fidelity import calculate_metrics
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from typing import *

# isort: split
from utils import *


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


def generate(checkpoint: Path, archive: Path, seed: int = None):
    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # RNG
    seed = hash((checkpoint, seed)) % 2**16
    rng = inox.random.PRNG(seed)

    # Model
    model = load_module(checkpoint)

    static, arrays = model.partition()
    arrays = jax.device_put(arrays, replicated)
    model = static(arrays)

    # Generate
    images = []

    for _ in tqdm(range(0, 50000, 256), ncols=88):
        x = sample_any(model, (256, 32 * 32 * 3), key=rng.split(), sampler='ddim', steps=64)
        x = unflatten(x, 32, 32)
        x = np.asarray(x)

        for img in map(to_pil, x):
            images.append(img)

    # Archive
    ZipDataset.zip(archive, images)


def fid(archive: Path, run: str, lap: int, seed: int):
    stats = calculate_metrics(
        input1=ZipDataset(archive),
        input2='cifar10-train',
        fid=True,
        isc=True,
    )

    fid = stats['frechet_inception_distance']
    isc = stats['inception_score_mean']

    with open(PATH / 'statistics.csv', mode='a') as f:
        f.write(f'{run},{lap},{seed},{fid},{isc}\n')


if __name__ == '__main__':
    run = 'easy-deluge-92_wubsk6w2'
    runpath = PATH / f'runs/{run}'
    seed = 0

    jobs = []

    for lap in range(32):
        checkpoint = runpath / f'checkpoint_{lap}.pkl'
        archive = runpath / f'archive_{lap}_{seed}.zip'

        if not checkpoint.exists():
            break

        a = job(
            partial(generate, checkpoint, archive, seed),
            name=f'generate_{lap}',
            cpus=4,
            gpus=4,
            ram='64GB',
            time='06:00:00',
            partition='gpu',
        )

        b = job(
            partial(fid, archive, run, lap, seed),
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
