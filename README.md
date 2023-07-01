# py2d: High-Performance 2D Navier-Stokes solver in Python

## Table of contents

* [Introduction](#Introduction)
* [Installation](#Installation)
* [Requirements](#Requirements)


## Introduction
Py2D is a Python solver for incompressible 2-dimensional (2D) Navier-stokes equations. 

Py2D leverages JAX, a high-performance numerical computing library, allowing for rapid execution on GPUs while also seamlessly supporting CPUs for users who do not have access to GPUs.

## Installation

Clone the [py2d git repository](https://github.com/envfluids/py2d.git) to use the latest development version.
```
git clone https://github.com/envfluids/py2d.git
```

Then install pyqg locally on your system:
```
cd py2d && pip install -e ./
```

## Requirements

- python 3.10
  - [jax](https://pypi.org/project/jax/)
  - [scipy](https://pypi.org/project/scipy/)
  - [numpy](https://pypi.org/project/numpy/)
  - [PyTorch](https://pypi.org/project/torch/) : If coupling LES solver with Neural Networks

### To use JAX and all requirements for the solver:

> **Warning**
> Please be aware that using both Torch and JAX on the same code can be challenging due to their use of CuDNN. Specifically, JAX requires the same version of CuDNN that Torch used during compilation, which can cause version mismatch issues.

> Currently, for Cuda11, JAX requires CuDNN version 8.6 or above. However, Torch with CUDA 11.8 only supports this specific version of CuDNN. Therefore, it is important to install the appropriate version of PyTorch that matches the required version of CuDNN for JAX.

> If you encounter version mismatch issues while using both Torch and JAX, consider installing the appropriate versions of the required libraries or seeking further guidance from the relevant documentation.

** Set up a new python environment and install the required libraries **
```
conda create -n jax python=3.10
conda activate jax
conda install numpy scipy scikit-learn prettytable 
conda install -c conda-forge cudnn=8.8.0.121
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
module load cuda
```
Add following lines in your SLURM file before running the script. This activates the environment with the required libraries and loads the cuda module 
```
conda activate jax
module load cuda
```
