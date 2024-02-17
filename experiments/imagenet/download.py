#!/usr/bin/env python

from datasets import load_dataset
from dawgz import job, after, schedule

from utils import *


@job(cpus=4, ram='16GB', time='1-00:00:00')
def download():
    load_dataset('imagenet-1k', cache_dir=PATH / 'hf', trust_remote_code=True, num_proc=4)


@after(download)
@job(cpus=16, ram='64GB', time='1-00:00:00')
def process():
    def transform(row):
        image = row['image']
        row['image'] = resize(image, 64)

        return row

    dataset = load_dataset('imagenet-1k', cache_dir=PATH / 'hf', trust_remote_code=True)
    dataset = dataset.map(transform, num_proc=16)
    dataset.save_to_disk(PATH / 'hf/imagenet-64', num_proc=16)


if __name__ == '__main__':
    schedule(
        process,
        name='Download & process',
        backend='slurm',
    )
