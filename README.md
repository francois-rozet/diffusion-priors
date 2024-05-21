# Learning Diffusion Priors from Observations by Expectation Maximization

<!-- TODO -->

## Code

The majority of the code is written in [Python](https://www.python.org). Neural networks are built and trained using the [JAX](https://github.com/google/jax) automatic differentiation framework. We also rely on [torch-fidelity](https://github.com/toshas/torch-fidelity) to compute image quality metrics and [POT](https://github.com/PythonOT/POT) to compute Sinkhorn divergences. All dependencies are provided as a [conda](https://conda.io) environment file.

```
conda env create -f environment.yml
conda activate priors
```

To run the experiments, it is necessary to have access to a [Slurm](https://slurm.schedmd.com/overview.html) cluster, to login to a [Weights & Biases](https://wandb.ai) account and to install the [priors](priors) module as a package.

```
pip install -e .
```

### Organization

The [priors](priors) directory contains the implementations of the [neural networks](priors/nn.py), the [diffusion models](priors/score.py) and various [helpers](priors/common.py).

The [manifold](experiments/manifold), [cifar](experiments/cifar) and [fastmri](experiments/fastmri) directories contain the scripts for the experiments (data generation, training and evaluation) as well as the notebooks that produced the figures of the paper. For example,

```
~/experiments/cifar $ python data.py
```

launches the CIFAR-10 data processing/generation jobs and

```
~/experiments/cifar $ python train.py
```

launches the training jobs with the default config (specified within the file).

For the accelerated MRI experiment, the `knee_singlecoil_{train|val}.tar.xz` archives should be obtained from the [NYU fastMRI Initiative database](https://fastmri.med.nyu.edu/).

## Acknowledgements

MRI data used in the preparation of this article were obtained from the NYU fastMRI Initiative database. As such, NYU fastMRI investigators provided data but did not participate in analysis or writing of this report. A listing of NYU fastMRI investigators, subject to updates, can be found at https://fastmri.med.nyu.edu. The primary goal of fastMRI is to test whether machine learning can aid in the reconstruction of medical images.

```bib
@misc{Zbontar2018fastMRI,
  author = {Zbontar, Jure and Knoll, Florian and Sriram, Anuroop and Murrell, Tullie and Huang, Zhengnan and Muckley, Matthew J. and Defazio, Aaron and Stern, Ruben and Johnson, Patricia and Bruno, Mary and Parente, Marc and Geras, Krzysztof J. and Katsnelson, Joe and Chandarana, Hersh and Zhang, Zizhao and Drozdzal, Michal and Romero, Adriana and Rabbat, Michael and Vincent, Pascal and Yakubova, Nafissa and Pinkerton, James and Wang, Duo and Owens, Erich and Zitnick, C. Lawrence and Recht, Michael P. and Sodickson, Daniel K. and Lui, Yvonne W.},
  title  = {{fastMRI}: {An} {Open} {Dataset} and {Benchmarks} for {Accelerated} {MRI}},
  year   = {2018},
  url    = {http://arxiv.org/abs/1811.08839},
}

@article{Knoll2020fastMRI,
  author  = {Knoll, Florian and Zbontar, Jure and Sriram, Anuroop and Muckley, Matthew J. and Bruno, Mary and Defazio, Aaron and Parente, Marc and Geras, Krzysztof J. and Katsnelson, Joe and Chandarana, Hersh and Zhang, Zizhao and Drozdzalv, Michal and Romero, Adriana and Rabbat, Michael and Vincent, Pascal and Pinkerton, James and Wang, Duo and Yakubova, Nafissa and Owens, Erich and Zitnick, C. Lawrence and Recht, Michael P. and Sodickson, Daniel K. and Lui, Yvonne W.},
  journal = {Radiology: Artificial Intelligence},
  title   = {{fastMRI}: {A} {Publicly} {Available} {Raw} k-{Space} and {DICOM} {Dataset} of {Knee} {Images} for {Accelerated} {MR} {Image} {Reconstruction} {Using} {Machine} {Learning}},
  year    = {2020},
  doi     = {10.1148/ryai.2020190007},
}
```
