# py2d: High-Performance 2D Navier-Stokes solver in Python

## Table of contents

* [Introduction](#Introduction)
* [Requirements](#Requirements)
* [Installation](#Installation)
* [Example](#Example)
* [A Note on JAX](#A-Note-on-JAX)
* [Citations](#Citations)

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
  - [jaxlib](https://pypi.org/project/jaxlib/)
  - [numpy](https://pypi.org/project/numpy/)

## Installation

Clone the [py2d git repository](https://github.com/envfluids/py2d.git) to use the latest development version.
```
git clone https://github.com/envfluids/py2d.git
```
Then install py2d locally on your system
```
cd py2d
pip install -e ./
```

## Example

```
from py2d.Py2D_solver import Py2D_solver

# Script to call the function with the given parameters
Py2D_solver(Re=20e3, # Reynolds number
               fkx=4, # Forcing wavenumber in x dimension
               fky=4, # Forcing wavenumber in y dimension
               alpha=0.1, # Rayleigh drag coefficient
               beta=0, # Coriolis parameter (Beta-plane turbulence)
               NX=32, # Number of grid points in x and y (Presuming a square domain) '32', '64', '128', '256', '512'
               SGSModel_string='NoSGS', # SGS closure model/parametrization to use. 'NoSGS' (no closure) for DNS simulations. Available SGS models: 'SMAG', 'DSMAG', 'LEITH', 'DLEITH', 'PiOmegaGM2', 'PiOmegaGM4', 'PiOmegaGM6'
               eddyViscosityCoeff=0, # Coefficient for eddy viscosity models: Only used for SMAG and LEITH SGS Models
               dt=5e-3, # Time step
               dealias=True, # Dealiasing
               saveData=True, # Save data: The saved data directory would be printed.
               tSAVE=0.1, # Time interval to save data
               tTotal=1, # Length (total time) of simulation
               readTrue=False, 
               ICnum=1, # Initial condition number: Choose between 1 to 20
               resumeSim=False, # start new simulation (False) or resume simulation (True) 
               )
```

## A Note on JAX 
JAX can be installed for either CPU-only or GPU-supported environments.

The default installation above will install JAX with CPU support but without GPU acceleration.

#### For GPU support
For instructions on installing with GPU support see [Installing JAX](https://jax.readthedocs.io/en/latest/installation.html).
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

## Citations
- [Jakhar, K.](https://scholar.google.com/citations?user=buVddBgAAAAJ&hl=en), [Guan, Y.](https://gyf135.github.io/), [Mojgani, R.](https://www.rmojgani.com), [Chattopadhyay, A.](https://scholar.google.com/citations?user=wtHkCRIAAAAJ&hl=en), and [Hassanzadeh, P.
](https://scholar.google.com/citations?user=o3_eO6EAAAAJ&hl=en), 
[**Learning Closed-form Equations for Subgrid-scale Closures from High-fidelity Data: Promises and Challenges**](https://arxiv.org/abs/2306.05014)
```bibtex
@article{jakhar2024learning,
  title={Learning closed-form equations for subgrid-scale closures from high-fidelity data: Promises and challenges},
  author={Jakhar, Karan and Guan, Yifei and Mojgani, Rambod and Chattopadhyay, Ashesh and Hassanzadeh, Pedram},
  journal={Journal of Advances in Modeling Earth Systems},
  volume={16},
  number={7},
  pages={e2023MS003874},
  year={2024},
  publisher={Wiley Online Library}
}
```
