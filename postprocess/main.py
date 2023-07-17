#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 23:05:58 2023

@author: rmojgani
"""

from scipy.io import loadmat, savemat


from py2d.convert import *
from py2d.aposteriori_analysis import *

#%%
NX=128
Lx = 2*np.pi
dx = Lx/NX
#-----------------
x        = np.linspace(0, Lx-dx, num=NX)
kx       = (2*np.pi/Lx)*np.concatenate((
                            np.arange(0,NX/2+1,dtype=np.float64),
                            np.arange((-NX/2+1),0,dtype=np.float64)
                            ))
# [Y,X]    = np.meshgrid(x,x)
# [Ky,Kx]  = np.meshgrid(kx,kx)
# Ksq      = onp.array((Kx**2 + Ky**2))
# Ksq[0,0] = 1e16
# Kabs     = np.sqrt(Ksq)
# invKsq   = 1/Ksq

Kx, Ky, Ksq = initialize_wavenumbers_2DFHIT(NX, NX, Lx, Lx)
Kabs     = np.sqrt(Ksq)
invKsq   = 1/Ksq

# w1_hat = np.fft.ifft2(Omega)
# signal_hat = energy_es(w1_hat, invKsq, NX, Kabs)#, invKsq, NX )
# # signal_hat = enstrophy_es(w1_hat, Kabs, NX )
# Kplot, energy, kplot_str =  spectrum_angle_average_vec(signal_hat, Kabs, NX)#, kx, , invKsq)

# plt.loglog(Kplot,energy)
#%%
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from matplotlib import pyplot as plt


#%%
def spectrum_angle_average_vec(es, Kabs, NX):#, kx):#, , invKsq):
    '''
    Angle averaged energy spectrum
    '''
    arr_len = int(0.5*NX)#
    #arr_len = int(np.ceil(0.5*np.sqrt((NX*NX + NX*NX))))
    kplot = np.array(range(arr_len))
    eplot = np.zeros(arr_len)

    # spectrum for all wavenumbers
    for k in kplot[1:]:
        unmask = np.logical_and(Kabs>=(k-0.5), Kabs<(k+0.5))
        #eplot[k] = k*np.sum(es[unmask]);
        eplot = eplot.at[k].set(k*np.sum(es[unmask]))
    #eplot[0]=es[0,0]
    # eplot = eplot.at[1].set(k*np.sum(es[unmask]))

    kplot_str = '\sqrt{\kappa_x^2+\kappa_y^2}'
    return kplot, eplot, kplot_str
#%%
def energy_es(w1_hat, invKsq, NX , Kabs):

    psi_hat = -invKsq*w1_hat
    es = 0.5*np.abs((np.conj(psi_hat))*w1_hat)

    return es
#%%
def enstrophy_es(w1_hat, Kabs, NX ):
    # March 2023
    es = 0.5*np.abs((np.conj(w1_hat).T)*w1_hat)

    return es
#%%
plt.figure(figsize=(12,14))
SGSModel_list = ['DLEITH_Local','DLeith']#, 'NoSGS', 'SMAG', 'DSMAG', 'LEITH', 'DLEITH', 'PiOmegaGM2', 'PiOmegaGM4', 'PiOmegaGM6']:
for (SGSModel_string, icount) in zip(SGSModel_list,range(1,3,1)):

    Omega = loadmat('../results/Re20k_fkx4fky0_r0.1/'+SGSModel_string+'_/NX128/dt0.0005_IC1/data/10.mat')['Omega']

    w1_hat = np.fft.ifft2(Omega)
    # signal_hat = energy_es(w1_hat, invKsq, NX, Kabs)#, invKsq, NX )
    signal_hat = enstrophy_es(w1_hat, Kabs, NX )
    Kplot, energy, kplot_str =  spectrum_angle_average_vec(signal_hat, Kabs, NX)#, kx, , invKsq)

    Z_angled_average_spectra, kkx = enstrophy_angled_average_2DFHIT(Omega, spectral = False)


    plt.subplot(2,2,icount)
    plt.title(SGSModel_string)
    plt.contourf(Omega, cmap='bwr_r',levels=99)

    plt.subplot(2,2,3)
    plt.loglog(Kplot,energy,label=SGSModel_string)

    # plt.subplot(2,2,4)

    plt.loglog(kkx,Z_angled_average_spectra)
    plt.ylim([1e-6,1e0])
    plt.legend()


plt.show()
#%%
plt.figure(figsize=(12,14))

for (NX, icount, linewidth) in zip([128,512],range(1,3),[1,2]):

    print(NX)

    Omega = loadmat('../data/ICs/NX'+str(NX)+'/1.mat')['Omega']

    Lx = 2*np.pi
    dx = Lx/NX
    #-----------------
    x        = np.linspace(0, Lx-dx, num=NX)
    kx       = (2*np.pi/Lx)*np.concatenate((
                                np.arange(0,NX/2+1,dtype=np.float64),
                                np.arange((-NX/2+1),0,dtype=np.float64)
                                ))
    # [Y,X]    = np.meshgrid(x,x)
    # [Ky,Kx]  = np.meshgrid(kx,kx)
    # Ksq      = onp.array((Kx**2 + Ky**2))
    # Ksq[0,0] = 1e16
    # Kabs     = np.sqrt(Ksq)
    # invKsq   = 1/Ksq

    Kx, Ky, Ksq = initialize_wavenumbers_2DFHIT(NX, NX, Lx, Lx)
    Kabs     = np.sqrt(Ksq)
    invKsq   = 1/Ksq

    w1_hat = np.fft.ifft2(Omega)
    signal_hat = energy_es(w1_hat, invKsq, NX, Kabs)#, invKsq, NX )
    # signal_hat = enstrophy_es(w1_hat, Kabs, NX )
    Kplot, energy, kplot_str =  spectrum_angle_average_vec(signal_hat, Kabs, NX)#, kx, , invKsq)

    Omega_hat = w1_hat
    Omega = np.fft.ifft2(Omega_hat)
    Psi_hat = Omega2Psi_2DFHIT(Omega, Kx, Ky, Ksq)
    TKE_angled_average, kkx = TKE_angled_average_2DFHIT(Psi_hat, Omega_hat, spectral = True)
    Z_angled_average_spectra, kkx = enstrophy_angled_average_2DFHIT(Omega, spectral = False)

    plt.subplot(2,2,icount)
    plt.title('')
    plt.contourf(Omega, cmap='bwr_r',levels=99)

    plt.subplot(2,2,3)
    plt.loglog(Kplot,energy,label='envfluids/spectra',linewidth=linewidth)
    plt.legend()

    plt.subplot(2,2,4)
    # plt.loglog(kkx,Z_angled_average_spectra,label='py2d')
    plt.loglog(kkx,TKE_angled_average,label='py2d',linewidth=linewidth)

    plt.xlim([0,256])
    # plt.ylim([1e-9,1e1])
    plt.legend()


plt.show()