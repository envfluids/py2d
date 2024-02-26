#%% Import
import numpy as nnp
import numpy as np
from scipy.io import savemat
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from fastkde import fastKDE
import statsmodels.api as sm

import h5py
import sys
import os
import scipy
from py2d.initialize import initialize_wavenumbers_2DFHIT
from py2d.convert import Omega2Psi_2DFHIT
from py2d.aposteriori_analysis import energy_2DFHIT, enstrophy_2DFHIT, eddyTurnoverTime_2DFHIT
from py2d.spectra import TKE_angled_average_2DFHIT, enstrophy_angled_average_2DFHIT
import matplotlib.pyplot as plt

sys.path.append('/media/rmojgani/hdd/PostDoc/Projects/py2d_local/py2d')

from py2d.eddy_viscosity_models import *

try:
    from natsort import natsorted, ns
except:
    os.system("pip3 install natsort")
    from natsort import natsorted, ns
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
plt.set_cmap("bwr")

#%%
from mypathdictionary3 import *
# from mypathdictionary3_tests import *

# METHOD = 'DLEITH_tau_Local' #LEITH, DLEITH , DLEITH_sigma_Local, DLEITH_tau_Local,
METHOD = 'DSMAG_tau_Local' #SMAG, DSMAG , DSMAG_sigma_Local, DSMAG_tau_Local,
# METHOD = 'NoSGS'

# CASENO = 1
# NX = 128 #[32,64,128]
CASENO = 2
NX = 64 #[32,64,128]
# CASENO = 4
# NX = 256 #[128,192,256]

#%%
Delta = 2 * np.pi / NX

nx, ny = NX,NX
Lx,Ly = 2*np.pi,2*np.pi
Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly)
#%%
path_RL = mypathdictionaryclassic(CASENO, NX, METHOD)
NUM_DATA_RL=4
#%%
# directory = path_RL
#%%
NSAMPLE = 100#0#00#1#00
MYLOAD = 'not' #[RL, not]
energy_spectra_RL_list, enstrophy_spectra_RL_list = [], []
Omega_list = []

if 'LEITH' in METHOD :
    METHOD_RL = 'LEITH_RL'
elif 'SMAG' in METHOD :
    METHOD_RL = 'SMAG_RL'
METHOD_RL += ', data='+str(NUM_DATA_RL)


abs_grad_omega_M = []
S_M = []
veRL_M = []
cs_M = []

filecount = 0
for file in natsorted(os.listdir(path_RL), alg=ns.PATH | ns.IGNORECASE):
    print(file)
    # Check if file ends with .mat
    if file.endswith('.mat') and filecount%1==0:
        file_path = os.path.join(path_RL, file)
        print(f'RL → {filecount}/{NSAMPLE}, Loaded: {file_path}')

        # Load .mat file
        mat_contents_RL = scipy.io.loadmat(file_path)
        Omega = mat_contents_RL['Omega']
        Psi = Omega2Psi_2DFHIT(Omega, invKsq=invKsq)

        if MYLOAD == 'not':
            Omega_hat = np.fft.fft2(Omega)
            Psi_hat = -invKsq*Omega_hat
            Psi = np.fft.ifft2(Psi_hat).real

            w1_hat = Omega_hat
            psi_hat = Psi_hat
            #%% cl to nu
            # def dleithlocal_method(self):#, Psi_hat, Omega_hat, Kx, Ky, Ksq, Delta):

            # PiOmega_hat = 0.0
            # #
            if 'LEITH' in METHOD :
                characteristic_Omega = characteristic_omega_leith(Omega_hat, Kx, Ky)
                c_dynamic = coefficient_dleithlocal_PsiOmega(Psi_hat, Omega_hat, characteristic_Omega, Kx, Ky, Ksq, Delta)
                Cmodel = c_dynamic ** (1/3)
                veRL = eddy_viscosity_leith(Cmodel, Delta, characteristic_Omega)
            #%% cs to nu
            if 'SMAG' in METHOD :

                def eddy_viscosity_smag_local(Cs, Delta, characteristic_S):
                    '''
                    Smagorinsky Model (SMAG) - Local characteristic_S
                    '''
                    characteristic_S2 = characteristic_S ** 2
                    characteristic_S = np.sqrt(characteristic_S2)
                    eddy_viscosity = (Cs * Delta) ** 2 * characteristic_S
                    return eddy_viscosity

                characteristic_S = characteristic_strain_rate_smag(Psi_hat, Kx, Ky, Ksq)
                c_dynamic = coefficient_dsmaglocal_PsiOmega(Psi_hat, Omega_hat, characteristic_S, Kx, Ky, Ksq, Delta)
                Cmodel = np.sqrt(c_dynamic)

                veRL = eddy_viscosity_smag_local(Cmodel, Delta, characteristic_S)

            w1x_hat = -(1j*Kx)*w1_hat
            w1y_hat = (1j*Ky)*w1_hat
            w1x = np.real(np.fft.ifft2(w1x_hat))
            w1y = np.real(np.fft.ifft2(w1y_hat))
            abs_grad_omega = np.sqrt( w1x**2+w1y**2  )


            S1 = np.real(np.fft.ifft2(-Ky*Kx*psi_hat)) # make sure .*
            S2 = 0.5*np.real(np.fft.ifft2(-(Kx*Kx - Ky*Ky)*psi_hat))
            S  = 2.0*(S1*S1 + S2*S2)**0.5

            abs_grad_omega = nnp.array(abs_grad_omega)
            S = nnp.array(S)

            abs_grad_omega_M = nnp.append(abs_grad_omega_M, abs_grad_omega)
            S_M = nnp.append(S_M, S)
            veRL_M = nnp.append(veRL_M, veRL)
            cs_M = nnp.append(cs_M, Cmodel)

        filecount += 1
        if filecount > NSAMPLE: break
#%%
plt.figure(figsize=(6,5))
plt.pcolor(Omega,vmin=-25,vmax=25);plt.colorbar()
#%%
plt.figure(figsize=(11,5))
plt.subplot(1,2,1)
plt.pcolor(S)
plt.title(r'$\|S\|$')
plt.subplot(1,2,2)
plt.pcolor(abs_grad_omega);
plt.title(r'$\|\omega\|$')

for i in range(2):
    plt.subplot(1,2,i+1)

    plt.colorbar();plt.tight_layout();plt.axis('square')
#%% Plot 3D
#%% Plot 3D
from mpl_toolkits.mplot3d import Axes3D
LABELPAD = 10
camview = np.array([
                [45  , 45 ,45],
                [0   ,-90 , 0],
                [0   ,  0 , 0],
                [-90 ,  0 , 0],
                [0   ,  0 ,90],
                   ]).T
for icount in range(len(camview.T)):
    fig = plt.figure(figsize=(11, 5))

    ax = fig.add_subplot(1,2,1, projection='3d', proj_type = 'ortho')

    if icount>=1:
        ax.view_init(elev=camview[0,icount],
                      azim=camview[1,icount],
                      roll=camview[2,icount]
                      )


    ax.scatter(S_M, abs_grad_omega_M, cs_M, s= 2, c= cs_M, vmin=-0.12, vmax=0.12, alpha=0.15)

    ax.set_xlabel(r'$\|S\|$')
    ax.set_ylabel(r'$\|\omega\|$')
    ax.set_zlabel(r'$c$')
    ax.set_zlim(-0.3, 0.3)

    ax.xaxis.labelpad = LABELPAD
    ax.yaxis.labelpad = LABELPAD
    ax.zaxis.labelpad = LABELPAD

    plt.tight_layout()#;plt.axis('square')
    # plt.colorbar()

    ax = fig.add_subplot(1,2,2, projection='3d', proj_type = 'ortho')

    if icount>=1:
        ax.view_init(elev=camview[0,icount],
                      azim=camview[1,icount],
                      roll=camview[2,icount]
                      )

    ax.scatter(S_M, abs_grad_omega_M, veRL_M, s= 2, c= veRL_M, vmin=-0.001, vmax=0.001, alpha=0.15)
    ax.set_zlim(-0.02, 0.02)
    ax.set_xlabel(r'$\|S\|$')
    ax.set_ylabel(r'$\|\omega\|$')
    ax.set_zlabel(r'$\nu$')
    plt.tight_layout()#;plt.axis('square')

    plt.show()

    # filename_save = 'C'+str(CASENO)+'_N'+str(NX)+METHOD
    # plt.savefig(filename_save+"_view"+str(icount)+".png")
stop
#%%
veRL_cut = 0.0#np.mean(veRL_M)

veRLn_M =veRL_M*0+np.NaN
veRLn_M[veRL_M<veRL_cut] = veRL_M[veRL_M<veRL_cut]

veRLp_M =veRL_M*0+np.NaN
veRLp_M[veRL_M>veRL_cut] = veRL_M[veRL_M>veRL_cut]

veRLpn_M = veRL_M*0-1
veRLpn_M[veRL_M>veRL_cut] = 1
#%%
num_samples = int(2*5e3)
indices = np.random.choice(S_M.shape[0], num_samples, replace=False)

import pandas as pd
df = pd.DataFrame(np.array([
                            S_M[indices],
                            abs_grad_omega_M[indices],
                            veRL_M[indices],
                            veRLp_M[indices],
                            veRLn_M[indices],
                            veRLpn_M[indices]
                            ]).T,
                columns=["S", "$\omega$","$c$","$c>0$","$c<0$","sign"])
#%% 3D - Sampled
LABELPAD = 15
for icount in range(len(camview.T)):
    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(1,1,1, projection='3d', proj_type = 'ortho')

    if icount>=1:
        ax.view_init(elev=camview[0,icount],
                     azim=camview[1,icount],
                     roll=camview[2,icount]
                     )


    ax.scatter(S_M, abs_grad_omega_M, veRL_M, s= 2, c= veRL_M, vmin=-0.12, vmax=0.12, alpha=0.15)
    ax.set_xlabel(r'$\|S\|$')
    ax.set_ylabel(r'$\|\omega\|$')
    ax.set_zlabel(r'$\|c\|$')

    ax.xaxis.labelpad = LABELPAD
    ax.yaxis.labelpad = LABELPAD
    ax.zaxis.labelpad = LABELPAD

    plt.tight_layout()#;plt.axis('square')
    # plt.colorbar()

    filename_save = 'C'+str(CASENO)+'_N'+str(NX)
    plt.savefig(filename_save+"_view"+str(icount)+"_s.png")
    plt.show()

#%%
import seaborn as sns
sns.set(font_scale=1.5, rc={'text.usetex' : True}, style='ticks')
#%%
# g = sns.jointplot(data=df, x="S", y="nu", kind="kde")
# g.plot_joint(sns.kdeplot)

#%%
# print('Joint plot')
# g = sns.jointplot(data=df, x="S", y="omega", kind="kde")

# # g.plot_joint(plt.scatter, hue="nu", s=40, linewidth=1, marker="+")
# g.ax_joint.collections[0].set_alpha(0)
# # g.set_axis_labels("$SepalLength(Cm)$", "$SepalWidth(Cm)$")
# plt.show()
#%%
print('Joint grid')

h = sns.JointGrid(data=df, x="S", y="$\omega$", hue="sign",    palette = "coolwarm")
h.plot_joint(sns.scatterplot, s=25, alpha=.25)
# h.plot_marginals(sns.stripplot, hue="nu", dodge=True)

filename_save = 'C'+str(CASENO)+'_N'+str(NX)
plt.savefig(filename_save+"_cc.png")

#%%
#sns.jointplot(data=df, x="S", "$\omega$", kind="kde")
#%%
# g = sns.jointplot(data=df, x="S", y="omega")
# g.plot_joint(sns.kdeplot, color="nu")
# # plt.ylim([-25,25])
# # plt.xlim([-50,250])


# sns.kdeplot(
#     data=geyser, x="S", y="$\omega$",
#     fill=True, thresh=0, levels=100, cmap="mako",
# )
#%% Pair plot
# print('Pair of cc')

# sns.pairplot(
#     df,
#     kind="kde",
#     #plot_kws=dict(marker="+", linewidth=1),
#     #diag_kws=dict(fill=False),
#     hue = "sign"
# )
#%% Pair plot
print('Pair of cc (scatter)')
# sb.set_palette("coolwarm")
sns.color_palette("coolwarm")

sns.pairplot(
    df,
    kind='reg',
    hue = "sign",
    palette = "coolwarm",
    plot_kws={'scatter_kws': {"s":1, 'alpha': 0.25}},
    corner=True #to plot only the lower triangle:
)

filename_save = 'C'+str(CASENO)+'_N'+str(NX)
plt.savefig(filename_save+"_joint.png")
#%%
nn=100
xs = np.linspace(0, 300, nn)
ynu = np.linspace(0, 25, nn)
xv, yv = np.meshgrid(xs, ynu, indexing='ij')

# xv = np.pad(xv, 1, mode='wrap')[1:,1:]
# yv = np.pad(yv, 1, mode='wrap')[1:,1:]

zv = xv*0
zstd = xv*0

for icount in range(nn-1):
    for jcount in range(nn-1):

        x_l = xv[icount, jcount]
        x_r = xv[icount+1, jcount]

        y_b = yv[icount, jcount]
        y_t = yv[icount, jcount+1]

        # print(x_l, x_r,'|' , y_b, y_t)
        box = np.logical_and(
        np.logical_and(abs_grad_omega_M>=x_l ,  abs_grad_omega_M<x_r),
        np.logical_and(S_M>=y_b ,  S_M<y_t)
        )

        zv[icount,jcount] = np.mean(veRL_M[box])
        zstd[icount,jcount] = np.std(veRL_M[box])

#%%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xv, yv, zv,cmap='bwr')
ax.set_xlabel(r'$\|\omega\|$')
ax.set_ylabel(r'$\|S\|$')
ax.set_zlabel(r'$\overline{c}$')
#%%
vmin = min(min(zv[zv!=np.nan][1:]), -max(zv[zv!=np.nan]) )

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title(r'$\overline{c}$', fontsize=30)
plt.contourf(xv,yv,zv,cmap='bwr',vmin=vmin, vmax=-vmin, levels=21);
# plt.colorbar();plt.tight_layout();#plt.axis('square')
plt.subplot(1,2,2)
plt.title(r'$\sigma(c)$', fontsize=30)
plt.contourf(xv,yv,zstd,cmap='gray_r',levels=51);

for i in range(2):
    plt.subplot(1,2,i+1)
    plt.colorbar()

    plt.xlabel(r'$\|\omega\|$')
    plt.ylabel(r'$\|S\|$')
    plt.tight_layout();#plt.axis('square')

filename_save = 'C'+str(CASENO)+'_N'+str(NX)
plt.savefig(filename_save+".png")
