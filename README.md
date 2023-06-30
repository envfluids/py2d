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
