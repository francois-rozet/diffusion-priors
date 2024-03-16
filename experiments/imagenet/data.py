#!/usr/bin/env python

from datasets import load_dataset, load_from_disk, Array3D, Features
from dawgz import job, after, ensure, schedule

from utils import *


@job(cpus=4, ram='16GB', time='1-00:00:00')
def download():
    load_dataset('imagenet-1k', cache_dir=PATH / 'hf', trust_remote_code=True, num_proc=4)


@after(download)
@ensure(lambda: (PATH / 'hf/imagenet-64').is_dir())
@job(cpus=16, ram='64GB', time='1-00:00:00')
def downsample():
    def transform(row):
        image = row['image']
        row['image'] = resize(image, 64)

        return row

    dataset = load_dataset('imagenet-1k', cache_dir=PATH / 'hf', trust_remote_code=True)
    dataset = dataset.map(transform, num_proc=16)

    dataset.save_to_disk(PATH / 'hf/imagenet-64')
    dataset.cleanup_cache_files()


@after(downsample)
@job(cpus=16, ram='64GB', time='1-00:00:00')
def corrupt():
    def transform(row):
        x = from_pil(row['image'])
        x = x + 4 / 256 * np.random.uniform(size=x.shape)  # dequantize

        A = np.ones(shape=(64, 64, 1), dtype=bool)

        for _ in range(3):
            i, j = np.random.randint(0, 48, size=2)
            w, h = np.random.randint(16, 32, size=2)

            A[i:i+w, j:j+h] = False

        y = np.random.normal(loc=A * x, scale=1e-3)

        return {'A': A, 'y': y}

    types = {
        'A': Array3D(shape=(64, 64, 1), dtype='bool'),
        'y': Array3D(shape=(64, 64, 3), dtype='float32'),
    }

    dataset = load_from_disk(PATH / 'hf/imagenet-64')
    dataset = dataset['train']
    dataset = dataset.select(range(2**20))
    dataset = dataset.map(
        transform,
        features=Features(types),
        remove_columns=['image', 'label'],
        num_proc=16,
    )

    dataset.save_to_disk(PATH / 'hf/imagenet-patch')
    dataset.cleanup_cache_files()


if __name__ == '__main__':
    schedule(
        corrupt,
        name='Data corruption',
        backend='slurm',
        prune=True,
    )
