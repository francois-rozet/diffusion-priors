#!/usr/bin/env python

from datasets import load_dataset, Array3D, Features
from dawgz import job, after, schedule

from utils import *


@job(cpus=4, ram='64GB', time='06:00:00')
def download():
    load_dataset('cifar10', cache_dir=PATH / 'hf')


@after(download)
@job(cpus=4, ram='64GB', time='06:00:00')
def corrupt():
    def transform(row):
        x = from_pil(row['img'])
        x = x + 4 / 256 * np.random.uniform(size=x.shape)  # dequantize
        A = np.random.uniform(size=(32, 32, 1)) < 0.25
        y = np.random.normal(loc=A * x, scale=1e-3)

        return {'A': A, 'y': y}

    types = {
        'A': Array3D(shape=(32, 32, 1), dtype='bool'),
        'y': Array3D(shape=(32, 32, 3), dtype='float32'),
    }

    dataset = load_dataset('cifar10', cache_dir=PATH / 'hf')
    dataset = dataset.map(
        transform,
        features=Features(types),
        remove_columns=['img', 'label'],
        keep_in_memory=True,
        num_proc=4,
    )

    dataset.save_to_disk(PATH / 'hf/cifar-mask')


if __name__ == '__main__':
    schedule(
        corrupt,
        name='Data corruption',
        backend='slurm',
    )
