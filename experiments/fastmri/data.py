#!/usr/bin/env python

import h5py
import io
import tarfile

from datasets import load_from_disk, Array3D, Dataset, Features
from dawgz import job, after, ensure, schedule

from utils import *


@ensure(lambda: (PATH / 'hf/fastmri').is_dir())
@job(cpus=4, ram='64GB', time='06:00:00')
def export():
    def gen():
        with tarfile.open(PATH / 'knee_singlecoil_train.tar.xz', mode='r|xz') as tarball:
            for member in tarball:
                if not member.name.endswith('.h5'):
                    continue

                file = tarball.extractfile(member).read()
                file = io.BytesIO(file)

                with h5py.File(file) as mri:
                    slices = mri['reconstruction_rss'][10:40]
                    slices = slices / slices.max()  # in [0, 1]
                    slices = 4 * slices - 2         # in [-2, 2]
                    slices = slices[..., None]

                    for x in slices:
                        yield {'x': x}

    types = {'x': Array3D(shape=(320, 320, 1), dtype='float32')}

    dataset = Dataset.from_generator(
        gen,
        features=Features(types),
        cache_dir=PATH / 'hf/temp',
    )

    dataset.save_to_disk(PATH / 'hf/fastmri')
    dataset.cleanup_cache_files()


@after(export)
@job(cpus=4, ram='64GB', time='06:00:00')
def corrupt():
    def transform(row):
        jax.config.update('jax_platform_name', 'cpu')

        x = row['x']
        y = complex2real(fft2c(x))

        A = np.random.uniform(size=(1, 320, 1)) < 0.1
        A[:, 150:170] = True

        y = np.random.normal(loc=A * y, scale=1e-2)

        return {'A': A, 'y': y}

    types = {
        'A': Array3D(shape=(1, 320, 1), dtype='bool'),
        'y': Array3D(shape=(320, 320, 2), dtype='float32'),
    }

    dataset = load_from_disk(PATH / 'hf/fastmri')
    dataset.set_format('numpy')
    dataset = dataset.map(
        transform,
        features=Features(types),
        remove_columns=['x'],
        num_proc=4,
    )

    dataset.save_to_disk(PATH / 'hf/fastmri-kspace')
    dataset.cleanup_cache_files()


if __name__ == '__main__':
    schedule(
        corrupt,
        name='Data corruption',
        backend='slurm',
        prune=True,
        export='ALL',
        account='ariacpg',
    )
