#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 23:05:58 2023

@author: rmojgani
"""

import scipy as sp

# Omega = sp.io.loadmat('results/Re20k_fkx4fky0_r0.1/DLEITH_Local_/NX128/dt0.0005_IC1/data/10.mat')['Omega']
Omega = sp.io.loadmat('results/Re20k_fkx4fky0_r0.1/DLEITH_/NX128/dt0.0005_IC1/data/10.mat')['Omega']

#%%
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from matplotlib import pyplot as plt
#%%
SGSModel_list = ['DLEITH_Local','DLeith']#, 'NoSGS', 'SMAG', 'DSMAG', 'LEITH', 'DLEITH', 'PiOmegaGM2', 'PiOmegaGM4', 'PiOmegaGM6']:
for SGSModel_string in SGSModel_list:

    Omega = sp.io.loadmat('../results/Re20k_fkx4fky0_r0.1/'+SGSModel_string+'_/NX128/dt0.0005_IC1/data/10.mat')['Omega']

    plt.contourf(Omega, cmap='bwr_r',levels=99)
