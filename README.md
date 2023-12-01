# py2d: High-Performance 2D Navier-Stokes solver in Python

## Table of contents

* [Introduction](#Introduction)
* [Installation](#Installation)
* [Requirements](#Requirements)


## Introduction
Py2D is a Python solver for incompressible 2-dimensional (2D) Navier-stokes equations. 

Py2D leverages JAX, a high-performance numerical computing library, allowing for rapid execution on GPUs while also seamlessly supporting CPUs for users who do not have access to GPUs.

**Py2D features the following capabilities:**

- Direct numerical simulations (DNS) for 2-dimensional (2D) turbulence, catering to a variety of systems including decaying, forced homogeneous, and beta-plane turbulence.
- Large Eddy Simulations (LES) with Sub-Grid Scale (SGS) models. The compatible models include Smagorinsky (SMAG), Leith (LEITH), Dynamic Smagorinsky (DSMAG), Dynamic Leith (DLEITH), as well as gradient models - GM2, GM4, and GM6.
- Coupling Neural Networks-based eddy viscosity or Sub-Grid Scale (SGS) terms with the LES solver. 

## Requirements

- python 3.10
  - [jax](https://pypi.org/project/jax/)
  - [scipy](https://pypi.org/project/scipy/)
  - [numpy](https://pypi.org/project/numpy/)

## Installation

## Set up a new Python environment and install the required libraries:

Create a new Conda environment:
```
conda create -n py2d python=3.10
conda activate py2d
```

Clone the [py2d git repository](https://github.com/envfluids/py2d.git) to use the latest development version.
```
git clone https://github.com/envfluids/py2d.git
```
Then install py2d locally on your system
```
cd py2d
pip install -e ./
```
Install basic scientific libraries
```
pip install numpy scipy
```
### Install JAX 
JAX can be installed for either CPU-only or GPU-supported environments. Follow the instructions below based on your requirements:

#### For CPU-only usage
> This installation will allow you to use JAX with CPU support but without GPU acceleration.
```
pip install jax
```
#### For GPU support
[Installing JAX](https://jax.readthedocs.io/en/latest/installation.html)
> Note: Note: GPU support requires a compatible NVIDIA GPU and the correct CUDA and CuDNN versions installed on your system. If you're unsure about your CUDA and CuDNN versions, consult the documentation for your GPU and the JAX installation guide for further guidance.

### If using JAX with Pytorch:

> **Warning**
> Combining PyTorch and JAX in the same codebase can be challenging due to their dependencies on CuDNN. It's crucial to ensure that JAX and Torch are compatible with the same version of CuDNN. JAX requires CuDNN version 8.6 or above for CUDA 11. However, it's important to verify that the version of PyTorch you are using is compiled against a compatible version of CuDNN. Mismatched versions can lead to runtime issues.

> As of now, for CUDA 11, JAX works with CuDNN version 8.6 or newer. Ensure that the version of PyTorch you install is compatible with this CuDNN version. If you encounter version mismatch issues, you may need to adjust the versions of the libraries you install or consult the relevant documentation for guidance.

Install a specific version of CuDNN that is compatible with both JAX and PyTorch:
```
conda install -c conda-forge cudnn=8.8.0.121
```
Install PyTorch with the appropriate CUDA toolkit version
```
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
```
Install JAX with CUDA 11 support
```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
If your system uses environment modules, load the CUDA module (this step is system-dependent and may vary)
```
module load cuda
```
