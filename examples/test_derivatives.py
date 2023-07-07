#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:26:44 2023

@author: rm99
Test of derivatives
"""
import sys
sys.path
sys.path.append('../')
sys.path.append('/home/rm99/Mount/py2d')
# Import os module
import os
from pathlib import Path

# Import Python Libraries
from prettytable import PrettyTable
import jax
from jax import jit
import numpy as nnp
import jax.numpy as np
from scipy.io import loadmat, savemat
import time as runtime
from timeit import default_timer as timer
print(jax.default_backend())
print(jax.devices())

# Import Custom Module
from py2d.convection_conserved import *
from py2d.convert import *
from py2d.aposteriori_analysis import eddyTurnoverTime_2DFHIT
from py2d.SGSModel import *
# from py2d.uv2tau_CNN import *
from py2d.initialize import *




from py2d.derivative_2DFHIT import *



# Enable x64 Precision for Jax
jax.config.update('jax_enable_x64', True)
#%%
from matplotlib import pyplot as plt

#%%
NX = 128

# Domain length
Lx = 2 * np.pi

# Filter Width
Delta = 2 * Lx / NX

# Mesh size
dx = Lx / NX

# Mesh points in x direction
x = np.linspace(0, Lx - dx, num=NX)

for INDEXING in ['xy', 'ij']:
    # -------------- Create the meshgrid both in physical and spectral space --------------
    # Kx, Ky, Ksq = initialize_wavenumbers_2DFHIT(NX, NX, Lx, Lx)
    nx = NX
    ny = NX
    Lx, Ly = 2 * np.pi, 2 * np.pi
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)
    (Kx, Ky) = np.meshgrid(kx, ky, indexing=INDEXING)
    Ksq = Kx ** 2 + Ky ** 2
        #$$
    Kx = np.array(Kx)
    Ky = np.array(Ky)
    Ksq = np.array(Ksq)
    
    # Mesh points in x and y direction
    X, Y = nnp.meshgrid(x, x, indexing =INDEXING)
    
    X = np.array(X)
    Y = np.array(Y)
    #%%
    W = np.sin(X)+np.cos(Y)
    W_hat = np.fft.fft2(W)
    DWDX_exact = np.cos(X)
    DWDY_exact = -np.sin(Y)
    #%%
    #%%
    DWDX = np.real(np.fft.ifft2(derivative_2DFHIT(W_hat, [1,0], Kx, Ky)))
    DWDY = np.real(np.fft.ifft2(derivative_2DFHIT(W_hat, [0,1], Kx, Ky)))
    
    #%%
    aa = DWDY# DWDX
    aa_exact = DWDY_exact# DWDX_exact
    # #%%
    # plt.figure(figsize=(4,10))
    # plt.subplot(4,1,1)
    # plt.contourf(X,Y,W,cmap='bwr');plt.colorbar()
    # plt.subplot(4,1,2)
    # plt.contourf(X,Y,aa_exact,cmap='bwr');plt.colorbar()
    # plt.subplot(4,1,3)
    # plt.contourf(X,Y,aa,cmap='bwr');plt.colorbar()
    # plt.subplot(4,1,4)
    # plt.contourf(X,Y,aa-aa_exact,cmap='bwr');plt.colorbar()
    #%% Not having X, Y in contour plots
    plt.figure(figsize=(4,10))
    plt.subplot(4,1,1)
    plt.contourf(X,Y,W,cmap='bwr');plt.colorbar()
    plt.title('INDEXING='+INDEXING)
    
    plt.subplot(4,1,2)
    plt.contourf(W,cmap='bwr');plt.colorbar()
    plt.title('INDEXING='+INDEXING)
