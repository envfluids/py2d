#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:50:40 2023

@author: rmojgani
"""
import h5py
import sys
import os
import scipy
from natsort import natsorted, ns

import numpy as np
from py2d.initialize import initialize_wavenumbers_2DFHIT
from py2d.convert import Omega2Psi_2DFHIT
from py2d.aposteriori_analysis import energy_2DFHIT, enstrophy_2DFHIT, eddyTurnoverTime_2DFHIT
from py2d.spectra import TKE_angled_average_2DFHIT, enstrophy_angled_average_2DFHIT
from py2d.apriori_analysis import *
from py2d.convert import *
from py2d.filter import *
from matplotlib import pyplot as plt
from PDE_KDE import myKDE, mybandwidth_scott
#%% Plot settings
import matplotlib.cbook as cbook
from matplotlib import cm
import matplotlib.pyplot as plt
DPI = 150

plt.rcParams.update({
    'font.size': 20,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
    'figure.dpi': DPI,
    'savefig.dpi': DPI
})
#%%
file_path = '/mnt/Mount/bridges2_phy/jakhar/DNS_data/Re20kNX1024nx4ny0r0p1/DNS/train1/'
mat_contents_RL = scipy.io.loadmat(file_path+'1000.mat')
path_RL = file_path
#%%
Omega = mat_contents_RL['Omega']

nx, ny = 1024, 1024
Lx, Ly = 2*np.pi,2*np.pi
Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly)
#%%
Psi = Omega2Psi_2DFHIT(Omega, invKsq, spectral=False)

U, V = Psi2UV_2DFHIT(Psi, Kx, Ky, spectral = False)

#%%
N_LES = 32
Delta = 2 * np.pi / N_LES
Uf = filter2D_2DFHIT(U, filterType='gaussian', coarseGrainType='spectral', Delta=Delta, Ngrid=N_LES, spectral=False)
Vf = filter2D_2DFHIT(V, filterType='gaussian', coarseGrainType='spectral', Delta=Delta, Ngrid=N_LES, spectral=False)

#%%
plt.figure(figsize=(10,12))
plt.subplot(3,2,1)
plt.pcolor(Omega)
plt.subplot(3,2,2)
plt.pcolor(Psi)

plt.subplot(3,2,3)
plt.pcolor(U)
plt.subplot(3,2,4)
plt.pcolor(V)

plt.subplot(3,2,5)
plt.pcolor(Uf)
plt.subplot(3,2,6)
plt.pcolor(Vf)
#%%
from py2d.SGSterms import Tau

Tau11, Tau12, Tau22 = Tau(Omega, filterType='spectral', coarseGrainType='spectral', Delta=Delta, N_LES=N_LES)
#%%
Tau11_hat = np.fft.fft2(Tau11)
Tau12_hat = np.fft.fft2(Tau12)
Tau22_hat = np.fft.fft2(Tau22)
#%%
nx, ny = 32, 32
Lx, Ly = 2*np.pi,2*np.pi
Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly)

PTau = energyTransfer_2DFHIT(Uf, Vf, Tau11, Tau12, Tau22, Kx, Ky)
#%%
BANDWIDTH = mybandwidth_scott(PTau)

Vecpoints, exp_log_kde, log_kde, kde = myKDE(PTau/np.std(PTau), BANDWIDTH=BANDWIDTH )

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.semilogy(Vecpoints, exp_log_kde, ':k', alpha=0.75, linewidth=2, label='')
# plt.ylim([1e-5,1e-1])
# plt.xlim([-5,5])
plt.xlabel(r'$T_E$')
plt.ylabel(r'$\mathcal{P}\left(T_E\right)$')
plt.tight_layout()
#%%
def Sigma_eddy_viscosity(eddy_viscosity, Omega_hat, Kx, Ky):
    '''
    https://github.com/envfluids/py2d/blob/c9df5333ea1b2f085c4d0f159b0e369b09cc221f/py2d/eddy_viscosity_models.py#L38
    Calculate the eddy viscosity term (Tau) in the momentum equation
    '''
    Omegax_hat = (1.j) * Kx * Omega_hat
    Omegay_hat = (1.j) * Ky * Omega_hat

    Sigma1 = -eddy_viscosity*np.fft.ifft2(Omegax_hat).real
    Sigma2 = -eddy_viscosity*np.fft.ifft2(Omegay_hat).real
    print('s1',Sigma1[0,0])
    return Sigma1, Sigma2

def enstrophyTransfer_2D_FHIT(Omega, Sigma1, Sigma2, Kx, Ky):
    """
    Enstrophy transfer of 2D_FHIT using SGS vorticity stress

    Inputs:
    Omega: Vorticity
    Sigma1, Sigma2: SGS vorticity stress

    Output:
    PZ: enstrophy transfer
    """

    Omegax = derivative_2DFHIT(Omega, [1,0], Kx=Kx, Ky=Ky, spectral=False)
    Omegay = derivative_2DFHIT(Omega, [0,1], Kx=Kx, Ky=Ky, spectral=False)
    print((Sigma1[0,0]))
    print((Omegax[0,0]))
    PZ = -Sigma1*Omegax - Sigma2*Omegay

    return PZ
#%%
from py2d.SGSterms import Tau
from py2d.SGSterms import Sigma


filterType = 'gaussian' # 'spectral', 'gaussian'
nx, ny = 1024, 1024
Lx, Ly = 2*np.pi,2*np.pi
Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly)
nx, ny = 32, 32
Kxf, Kyf, Kabsf, Ksqf, invKsqf = initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly)


NSAMPLE = 2000
PZ_M = []
PTau_M = []
cs_M = []
ve_M = []

icount = 0
filecount = 0
file_path = '/mnt/Mount/bridges2_phy/jakhar/DNS_data/Re20kNX1024nx4ny0r0p1/DNS/train'

for icount in range(1,5):
    file_path += file_path+str(icount)+'/'
    for file in natsorted(os.listdir(path_RL), alg=ns.PATH | ns.IGNORECASE):
        # Check if file ends with .mat
        if file.endswith('.mat') and filecount%1==0:

            file_path = os.path.join(path_RL, file)
            print(f'RL → {filecount}/{NSAMPLE}, Loaded: {file_path}')

            # Load .mat file
            mat_contents_RL = scipy.io.loadmat(file_path)
            Omega = mat_contents_RL['Omega']
            Psi = Omega2Psi_2DFHIT(Omega, invKsq, spectral=False)
            U, V = Psi2UV_2DFHIT(Psi, Kx, Ky, spectral = False)

            Tau11, Tau12, Tau22 = Tau(Omega, filterType=filterType, coarseGrainType='spectral', Delta=Delta, N_LES=N_LES)
            # Tau11_hat = np.fft.fft2(Tau11)
            # Tau12_hat = np.fft.fft2(Tau12)
            # Tau22_hat = np.fft.fft2(Tau22)
            Uf = filter2D_2DFHIT(U, filterType=filterType, coarseGrainType='spectral', Delta=Delta, Ngrid=N_LES, spectral=False)
            Vf = filter2D_2DFHIT(V, filterType=filterType, coarseGrainType='spectral', Delta=Delta, Ngrid=N_LES, spectral=False)
            PTau = energyTransfer_2DFHIT(Uf, Vf, Tau11, Tau12, Tau22, Kxf, Kyf)


            Omegaf = filter2D_2DFHIT(Omega, filterType=filterType, coarseGrainType='spectral', Delta=Delta, Ngrid=N_LES, spectral=False)

            Omega_hat = np.fft.ifft2(Omegaf)
            Sigma1, Sigma2 = Sigma(Omega, filterType=filterType, coarseGrainType='spectral', Delta=Delta, N_LES=N_LES)
            PZ = enstrophyTransfer_2D_FHIT(Omegaf, Sigma1, Sigma2, Kxf, Kyf)

            PTau_M = np.append(PTau_M, (PTau))
            PZ_M = np.append(PZ_M, (PZ))

            filecount += 1
            if filecount > NSAMPLE: break
#%%
BANDWIDTH = mybandwidth_scott(PTau_M)*1

# Vecpoints, exp_log_kde, log_kde, kde = myKDE(PTau_M/np.std(PTau_M), BANDWIDTH=BANDWIDTH )
# spe = np.stack((Vecpoints[:,], exp_log_kde[:,], log_kde[:,]),axis=1)
# gse = np.stack((Vecpoints[:,], exp_log_kde[:,], log_kde[:,]),axis=1)


plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
# plt.semilogy(Vecpoints, exp_log_kde, ':k', alpha=0.75, linewidth=2, label=filterType)
plt.semilogy(gse[:,0], gse[:,1], ':k', alpha=0.75, linewidth=2, label='gaussian')
plt.semilogy(spe[:,0], spe[:,1], '-b', alpha=0.75, linewidth=2, label='spectral')

plt.semilogy([0,0],[0,1],'r')
plt.ylim([1e-3,1e0])
plt.xlim([-5,5])
plt.xlabel(r'$T_E$')
plt.ylabel(r'$\mathcal{P}\left(T_E\right)$')
plt.tight_layout()
plt.legend(loc='upper left',fontsize=10)


BANDWIDTH = mybandwidth_scott(PZ_M)

# Vecpoints, exp_log_kde, log_kde, kde = myKDE(PZ_M/np.std(PZ_M),BANDWIDTH=BANDWIDTH)

# spz = np.stack((Vecpoints[:,], exp_log_kde[:,], log_kde[:,]),axis=1)
# gsz = np.stack((Vecpoints[:,], exp_log_kde[:,], log_kde[:,]),axis=1)

plt.subplot(2,2,2)

# plt.semilogy(Vecpoints, exp_log_kde, ':k', alpha=0.75, linewidth=2, label=filterType)
plt.semilogy(gse[:,0], gse[:,1], ':k', alpha=0.75, linewidth=2, label='gaussian')
plt.semilogy(spz[:,0], spz[:,1], '-b', alpha=0.75, linewidth=2, label='spectral')

plt.semilogy([0,0],[0,1],'r')

plt.ylim([1e-3,1e0])
plt.xlim([-5,5])
plt.xlabel(r'$T_Z$')
plt.ylabel(r'$\mathcal{P}\left(T_Z\right)$')
plt.tight_layout()
plt.legend(loc='upper left',fontsize=10)
