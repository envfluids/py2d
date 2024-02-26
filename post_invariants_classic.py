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
sys.path.append('/media/rmojgani/hdd/PostDoc/Projects/py2d_local/py2d')

from py2d.eddy_viscosity_models import *

try:
    from natsort import natsorted, ns
except:
    os.system("pip3 install natsort")
    from natsort import natsorted, ns
#%%
#from mypathdictionary import *
# from mypathdictionary2 import *
from mypathdictionary3 import *
# from mypathdictionary3_tests import *

#%%
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

# %% Case select
# METHOD = 'DLEITH_tau_Local' #LEITH, DLEITH , DLEITH_sigma_Local, DLEITH_tau_Local
METHOD = 'DSMAG_sigma_Local' #SMAG, DSMAG , DSMAG_sigma_Local, DSMAG_tau_Local, || DSMAG_sigma_Local_LocalS , DSMAG_tau_Local_LocalS/
# METHOD = 'NoSGS'

# case = str(sys.argv[1])
# percent_data = float(sys.argv[2])
# CASENO = 1
# NX = 64 #[32,64,128]
# CASENO = 2
# NX = 64 #[32,64,128]
CASENO = 4
NX = 256 #[128,192,256]

if CASENO == 1:
    NUM_DATA_Classic = 2_00
elif CASENO == 2:
    NUM_DATA_Classic = 5_00
elif CASENO == 4:
    NUM_DATA_Classic = 2_00
    NUM_DATA_RL =1000 #0
#%%
Delta = 2 * np.pi / NX
#%% Load DNS results
# directory = '/mnt/Mount/bridges2_phy/jakhar/DNS_data/'
# filename = 'Re20kNX1024nx4ny4r0p1b20_1024_0.01_aposteriori_data.mat'
# mat_contents = scipy.io.loadmat(filename)

if CASENO==1:
      patthadd = '/media/rmojgani/hdd/PostDoc/Projects/py2d_local/'
      mat_contents = scipy.io.loadmat('results/Re20000_fkx4fky4_r0.1_b0/Re20000_fkx4fky4_r0.1_b0.0_1024_1.0_aposteriori_data.mat')
      # NX = 1024
      # dataType = 'results/Re20000_fkx4fky4_r0.1_b0.10/NoSGS/NX1024/dt5e-05_IC1/data'
      # dataType = '/mnt/Mount/bridges2_phy/jakhar/py2d/results/Re20000_fkx4fky4_r0.1_b0.0/NoSGS/NX1024/dt5e-05_IC1/data'
elif CASENO==2:
    # mat_contents = scipy.io.loadmat('results/Re20000_fkx4fky4_r0.1_b20.0/Re20kNX1024nx4ny4r0p1b20_1024_1.0_aposteriori_data.mat')
    mat_contents = scipy.io.loadmat('results/Re20000_fkx4fky4_r0.1_b20/Re20kNX1024nx4ny4r0p1b20_aposteriori_data.mat')

elif CASENO==4:
    # mat_contents = scipy.io.loadmat('results/Re20000_fkx25fky25_r0.1_b0/Re20kNX1024nx25ny25r0p1_512_aposteriori_data.mat')
    mat_contents = scipy.io.loadmat('results/Re20000_fkx25fky25_r0.1_b0/Re20kNX1024nx25ny25r0p1_aposteriori_data.mat')

elif CASENO==10:
      patthadd = '/media/rmojgani/hdd/PostDoc/Projects/py2d_local/'
      mat_contents = scipy.io.loadmat('results/Re100kNX2048nx4ny0r0p1/Re100kNX2048nx4ny0r0p1_aposteriori_data.mat')


energy_spectra_DNS = mat_contents['energy_spectra'].reshape(-1,)
enstrophy_spectra_DNS = mat_contents['enstrophy_spectra'].reshape(-1,)
# wavenumbers_spectra_DNS = mat_contents['wavenumbers_spectra'].reshape(-1,)


pdf_DNS = np.array([mat_contents['Omega_bins_scipy'][0,:],mat_contents['Omega_pdf_scipy'][0,:]]).T
std_omega_DNS =  mat_contents['Omega_std'][0][0]

wavenumbers_spectra_DNS = np.arange(mat_contents['energy_spectra'].shape[1])
#%%
nx, ny = NX,NX
Lx,Ly = 2*np.pi,2*np.pi
Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly)

# select % of data for pdf calculation
percent_data = 1#0.75
# Compute the 2% of total elements in the matrix
num_samples = int(percent_data * NX * NX)
#%% Load RL results
import re
from py2d.spectra import *
energy_spectra_RL_list, enstrophy_spectra_RL_list = [], []
Omega_list = []

#path_RL, NUM_DATA_RL, BANDWIDTH_R = mypathdictionaryRL(CASENO, NX, METHOD)
path_RL = mypathdictionaryclassic(CASENO, NX, METHOD)

NUM_DATA_RL = 10#0

if 'LEITH' in METHOD :
    METHOD_RL = 'LEITH_RL'
elif 'SMAG' in METHOD :
    METHOD_RL = 'SMAG_RL'
METHOD_RL += ', data='+str(NUM_DATA_RL)

omega_M = []# np.zeros((NLES, NLES))
ve_M = []# np.zeros((NLES, NLES))

filecount = 0
for file in natsorted(os.listdir(path_RL), alg=ns.PATH | ns.IGNORECASE):
    # Check if file ends with .mat
    if (file.endswith('.mat') and filecount%1==0):#and int(re.findall(r'\d+', file)[1])>0:
        file_path = os.path.join(path_RL, file)
        print(f'RL → {filecount}/{NUM_DATA_RL}, Loaded: {file_path}')

        # Load .mat file
        mat_contents_RL = scipy.io.loadmat(file_path)
        #%% omega to cL
        Omega = mat_contents_RL['Omega']
        Omega_hat = np.fft.fft2(Omega)
        Psi_hat = -invKsq*Omega_hat
        Psi = np.fft.ifft2(Psi_hat).real
        #%% cl to nu
        # def dleithlocal_method(self):#, Psi_hat, Omega_hat, Kx, Ky, Ksq, Delta):

        # PiOmega_hat = 0.0
        # characteristic_Omega = characteristic_omega_leith(Omega_hat, Kx, Ky)
        # #
        # c_dynamic = coefficient_dleithlocal_PsiOmega(Psi_hat, Omega_hat, characteristic_Omega, Kx, Ky, Ksq, Delta)
        # Cl = c_dynamic ** (1/3)
        # eddy_viscosity = eddy_viscosity_leith(Cl, Delta, characteristic_Omega)
        #%%
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
        Cs = np.sqrt(c_dynamic)

        eddy_viscosity = eddy_viscosity_smag_local(Cs, Delta, characteristic_S)
        #%%
        veRL = eddy_viscosity

        #%%
        omega_M = np.append(omega_M, Omega)
        ve_M = np.append(ve_M, veRL)
        #%%

        energy_spectra_RL, wavenumbers_RL = TKE_angled_average_2DFHIT(Psi, Omega, spectral=False)
        energy_spectra_RL = -energy_spectra_RL

        enstrophy_spectra_RL, wavenumbers_RL  = enstrophy_angled_average_2DFHIT(Omega_hat, spectral=True)


        energy_spectra_RL_list.append(energy_spectra_RL)
        enstrophy_spectra_RL_list.append(enstrophy_spectra_RL)

        omega_flattened = Omega.flatten()
        indices = np.random.choice(omega_flattened.shape[0], num_samples, replace=False)
        sampled_data = omega_flattened[indices]
        Omega_list.append(sampled_data)
        Omega_list.append(-sampled_data)

        filecount += 1
        if filecount>NUM_DATA_RL: break

energy_spectra_RL_arr = nnp.array(energy_spectra_RL_list)
enstrophy_spectra_RL_arr = nnp.array(enstrophy_spectra_RL_list)

energy_spectra_RL_mean = energy_spectra_RL_arr.mean(axis=0)
enstrophy_spectra_RL_mean = enstrophy_spectra_RL_arr.mean(axis=0)

Omega_arr_RL = nnp.array(Omega_list).flatten()
#%%
del Omega_list
#%%
omega_M_2D = omega_M.reshape(NX, NX, -1, order='F')
ve_M_2D = ve_M.reshape(NX,NX, -1,order='F')
#%%
import os, sys
#os.chdir('../../py2d/')
sys.path.append('/media/rmojgani/hdd/PostDoc/Projects/py2d_local/py2d')

from py2d.convert import Omega2Psi_2DFHIT


from py2d.eddy_viscosity_models import Tau_eddy_viscosity

# ve = ve_M_2D[:,:,tcount]#mat_contents['slnWorDNS'][:,:,i]

# omega = omega_M_2D[:,:,tcount]
# w1_hat = np.fft.fft2(omega)
# psi_hat = -invKsq*w1_hat
# psi = np.fft.ifft2(psi_hat).real

# Tau11, Tau12, Tau22 = Tau_eddy_viscosity(ve, psi_hat, Kx, Ky)
#%% Distribution of # elif statetype=='invariantlocalandglobalgradgrad': commit 9ae57f1
#-----------------------------------------
def D_dir(u_hat, K_dir):
    Du_Ddir = 1j*K_dir*u_hat
    return Du_Ddir
#-----------------------------------------
def decompose_sym(A):
    S = 0.5*(A+A.T)
    R = 0.5*(A-A.T)
    return S, R
#-----------------------------------------
def invariant(A):
    S, R = decompose_sym(A)
    lambda1 = np.trace(S)
    lambda2 = np.trace(S@S)
    lambda3 = np.trace(R@R)
    return [lambda1, lambda2, lambda3]
#-----------------------------------------
def pickcenter(array, Nx, Ny, nAgents_v, nAgents_h=[]):
    if nAgents_h==[]:
        nAgents_h = nAgents_v

    ix = np.linspace(0,Nx,nAgents_h,endpoint=False).astype('int')
    iy = np.linspace(0,Ny,nAgents_v,endpoint=False).astype('int')

    array_agents = array[ix,:][:,iy]

    mystatelist = array_agents.reshape(-1,1).tolist()

    return mystatelist
#-----------------------------------------
# psi_M_2D = psi_M.reshape(-1,128,128)
# omega_M_2D = omega_M.reshape(-1,128,128)

DPI = 150
fig = plt.figure(figsize=(20,25),dpi=DPI)
xplotM = np.array([])
yplotM = np.array([])

icount = 0

mystatelist = []
mystatelistve = []
mystatelistom = []
mylisttau = []

# icount += 1
# for psi, omega in zip(psi_M_2D, omega_M_2D):
for tcount in range(NUM_DATA_RL):#omega_M_2D.shape[2]):

    ve = ve_M_2D[:,:,tcount]#mat_contents['slnWorDNS'][:,:,i]

    omega = omega_M_2D[:,:,tcount]
    w1_hat = np.fft.fft2(omega)
    psi_hat = -invKsq*w1_hat
    psi = np.fft.ifft2(psi_hat).real


    Tau11, Tau12, Tau22 = Tau_eddy_viscosity(ve, psi_hat, Kx, Ky)

    # plt.pcolor(psi,cmap='bwr')
    # plt.show()
    u1_hat = D_dir(psi_hat,Ky) # u_hat = (1j*Ky)*psi_hat
    v1_hat = -D_dir(psi_hat,Kx) # v_hat = -(1j*Kx)*psi_hat

    dudx_hat = D_dir(u1_hat,Kx)
    dudy_hat = D_dir(u1_hat,Ky)

    dvdx_hat = D_dir(v1_hat,Kx)
    dvdy_hat = D_dir(v1_hat,Ky)

    dudxx_hat = D_dir(dudx_hat,Kx)
    dudyy_hat = D_dir(dudy_hat,Ky)

    dvdxx_hat = D_dir(dvdx_hat,Kx)
    dvdyy_hat = D_dir(dvdy_hat,Ky)


    dudx = np.fft.ifft2(dudx_hat).real
    dudy = np.fft.ifft2(dudy_hat).real
    dvdx = np.fft.ifft2(dvdx_hat).real
    dvdy = np.fft.ifft2(dvdy_hat).real

    dudxx = np.fft.ifft2(dudxx_hat).real
    dudyy = np.fft.ifft2(dudyy_hat).real
    dvdxx = np.fft.ifft2(dvdxx_hat).real
    dvdyy = np.fft.ifft2(dvdyy_hat).real


    NY=NX
    nActiongrid = 16

    list1 =  pickcenter(dudx, NX, NY, nActiongrid)
    list2 =  pickcenter(dudy, NX, NY, nActiongrid)
    list3 =  pickcenter(dvdx, NX, NY, nActiongrid)
    list4 =  pickcenter(dvdy, NX, NY, nActiongrid)

    list5 =  pickcenter(dudxx, NX, NY, nActiongrid)
    list6 =  pickcenter(dudyy, NX, NY, nActiongrid)
    list7 =  pickcenter(dvdxx, NX, NY, nActiongrid)
    list8 =  pickcenter(dvdyy, NX, NY, nActiongrid)

    listomega =  pickcenter(omega, NX, NY, nActiongrid)
    listve =  pickcenter(ve, NX, NY, nActiongrid)


    for dudx,dudy,dvdx,dvdy,dudxx,dudxy,dvdyx,dvdyy, ome, vem in zip(list1, list2, list3, list4, list5, list6, list7, list8, listomega, listve):
        gradV = np.array([[dudx[0], dudy[0]],
                          [dvdx[0], dvdy[0]]])
        hessV = np.array([[dudxx[0], dudxy[0]],
                          [dvdyx[0], dvdyy[0]]])
        allinvariants = invariant(gradV)+invariant(hessV)
        mystatelist.append(allinvariants)

        mystatelistve.append(vem)
        mystatelistom.append(ome)


    list1 =  pickcenter(Tau11, NX, NY, nActiongrid)
    list2 =  pickcenter(Tau12, NX, NY, nActiongrid)
    list3 =  pickcenter(Tau22, NX, NY, nActiongrid)

    for tau11, tau12, tau22 in zip(list1, list2, list3):
        alltaus = tau11+tau12+tau22
        mylisttau.append(alltaus)


    list_str=['\lambda_1','\lambda_2','\lambda_3','\lambda_4','\lambda_5','\lambda_6', '\tau_{11}','\tau_{12}','\tau_{22}','\omega']
    # xplot = np.array(mystatelist+mylisttau)
    xplot = np.hstack((np.array(mystatelist),
                       np.array(mylisttau),
                       np.array(mystatelistom)
                   ))

    try:
        xplotM = np.vstack([xplotM,xplot])
    except:
        xplotM = xplot

    yplot = np.array(mystatelistve)
    try:
        yplotM = np.vstack([yplotM,yplot])
    except:
        yplotM = yplot

    # stop
    # mystatelist1 = []
    # mystatelist2 = []
    # mystatelistome = []
    # mystatelistve = []


    # for ii in range(0,6):
    #     for jj in range(0,6):

    #         mystatelist1.append(allinvariants[ii])
    #         mystatelist2.append(allinvariants[jj])
    #         mystatelistome.append(ome)
    #         mystatelistve.append(vem)
    #         stop

    #     # # plt.plot(np.array(mystatelist1[1]),np.array(mystatelist2[2]))
    #     # # print(mystatelist[0])

    #     # # xplot, yplot = np.array(mystatelistome), np.array(mystatelist2)

    #     #     if ii>0 and jj>0 and ii>jj:
    #     #         xplot, yplot = np.array(mystatelist1), np.array(mystatelist2)
    #     #         plt.subplot(6,6,icount+1)
    #     #         plt.plot(xplot, yplot,'.k',alpha=0.05, markersize=20)
    #     #         plt.xlabel(rf'$\lambda_{str(1+ii)}$', fontsize=20)
    #     #         plt.ylabel(rf'$\lambda_{str(1+jj)}$', fontsize=20)


    #     #         plt.tight_layout()

    #     #         # xplot, yplot = np.array(mystatelistome), np.array(mystatelist2)
    #     #         # plt.subplot(6,6,icount+1)
    #     #         # plt.plot(xplot, yplot,'.k',alpha=0.05, markersize=2)
    #     #         # plt.xlabel(rf'$\omega$', fontsize=20)
    #     #         # plt.ylabel(rf'$\lambda_{str(1+jj)}$', fontsize=20)

    #     #         # plt.xlim([0,400])
    #     #         # plt.ylim([-800,0])
    #     #         try:
    #     #             xplotM = np.vstack([xplotM,xplot])
    #     #         except:
    #     #             try:
    #     #                 xplotM = xplot
    #     #             except:
    #     #                 pass
    #     #         print('-----', ii, jj, xplotM.shape)

    #     #         icount += 1
    #     # plt.show()

# stop_tcount

# #%%
# invariant1 = 0*dudx
# invariant2 = 0*dudx
# invariant3 = 0*dudx
# invariant4 = 0*dudx
# invariant5 = 0*dudx
# invariant6 = 0*dudx
# for icount in range(NX):
#     for jcount in range(NX):

#         gradV = np.array([[dudx[icount,jcount], dudy[icount,jcount]],
#                           [dvdx[icount,jcount], dvdy[icount,jcount]]])
#         hessV = np.array([[dudxx[icount,jcount], dudyy[icount,jcount]],
#                           [dvdxx[icount,jcount], dvdyy[icount,jcount]]])

#         #allinvariants = invariant(gradV)+invariant(hessV)
#         invariant1[icount,jcount], invariant2[icount,jcount], invariant3[icount,jcount] =invariant(gradV)
#         invariant4[icount,jcount], invariant5[icount,jcount], invariant6[icount,jcount] =invariant(hessV)

# # %%
# # plt.figure(figsize=(12,12))

# fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
# plt.subplot(3,3,1);plt.title(rf'$\lambda_1$')
# plt.pcolor(invariant1);plt.colorbar()
# plt.subplot(3,3,2);plt.title(rf'$\lambda_2$')
# plt.pcolor(invariant2);plt.colorbar()
# plt.subplot(3,3,3);plt.title(rf'$\lambda_3$')
# plt.pcolor(invariant3);plt.colorbar()
# plt.subplot(3,3,4);plt.title(rf'$\lambda_4$')
# plt.pcolor(invariant4);plt.colorbar()
# plt.subplot(3,3,5);plt.title(rf'$\lambda_5$')
# plt.pcolor(invariant5);plt.colorbar()
# plt.subplot(3,3,6);plt.title(rf'$\lambda_6$')
# plt.pcolor(invariant6);plt.colorbar()
# plt.subplot(3,3,7);plt.title(rf'$\omega$')
# plt.pcolor(Omega);plt.colorbar()
# #%%
# plt.plot(invariant5.reshape(-1,1) , invariant6.reshape(-1,1), '.' )
# stop
#%%
sr_points = xplotM
sr_color = yplotM#np.array(mystatelistve)#np.array(mystatelistome)
##%%
from sklearn.preprocessing import normalize
sr_points_normalized = normalize(sr_points, norm='l2', axis=1, copy=True, return_norm=False)
# sr_points_normalized =sr_points
# #%%
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# scaler.fit(sr_points_normalized)
# sr_points_normalized = scaler.transform(sr_points_normalized)
# #%%
##%%
def mystand(a):
    from sklearn.preprocessing import StandardScaler
    a = a.reshape(-1,1)

    # Normalize
    a = normalize(a, norm='l2', axis=1, copy=True, return_norm=False)

    # Standarzie
    scaler = StandardScaler()
    scaler.fit(a)
    a = scaler.transform(a)
    return a
##%%
# sr_points_normalized =sr_points
# for icol in range(sr_points.shape[1]):
#     sr_points_normalized[:,icol] = mystand(sr_points[:,icol]).reshape(-1)
##%%
list_str=[r'\lambda_1',r'\lambda_2',r'\lambda_3',
          r'\lambda_4',r'\lambda_5',r'\lambda_6',
          r'\tau_{11}',r'\tau_{12}',r'\tau_{22}','\omega']

from openTSNE import TSNE
import sklearn

tsne = TSNE(
    n_components=2,
    perplexity=1,
    initialization="pca",
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)

for PERPLEXITY in [100]:#[1,5,10, 100,500,1e9]:#, 5000, 10000]:
    tsne.perplexity=PERPLEXITY
    embedding_train = tsne.fit(sr_points_normalized)

    ##%%
    for icount in range(0,10):

        # Scaler
        # scaler = sklearn.preprocessing.StandardScaler().fit(sr_points[:,icount].reshape(-1,1))
        # SR_POINT = scaler.transform(sr_points[:,icount].reshape(-1,1))

        # SR_POINT = sklearn.preprocessing.normalize(sr_points[:,icount].reshape(-1,1), norm='l2', axis=1, copy=True, return_norm=False)

        # SR_COLOR = sr_points[:,icount].reshape(-1,1)
        SR_COLOR = sr_points_normalized[:,icount].reshape(-1,1)
        # SR_COLOR = xplotM[:,icount]
        # SR_COLOR = mystand(xplotM[:,icount])

        # VMIN = min(SR_POINT.min() , -SR_POINT.max())
        # VMAX = -VMIN
        plt.scatter(embedding_train[:,0], embedding_train[:,1], c=SR_COLOR, s=0.1, cmap='bwr')#, vmin=VMIN, vmax=VMAX);

        clb = plt.colorbar()
        clb.ax.set_title(rf'${list_str[icount]}$')

        #plt.title(rf"TSNE, p={PERPLEXITY}")
        plt.axis('off')
        plt.annotate(rf'$C{CASENO}, N_x={NX}$', xy=(0.05, 0.05), xycoords='axes fraction',fontsize=10)
        plt.annotate(rf'T-SNE, $p={PERPLEXITY}$', xy=(0.05, 0.0), xycoords='axes fraction',fontsize=10)
        plt.annotate(rf'{METHOD}', xy=(0.05, 1.05), xycoords='axes fraction',fontsize=10)

        file_name = 'C'+str(CASENO)+'_Nx'+str(NX)+'_'+METHOD+'_p'+str(PERPLEXITY)+'l'+str(icount)
        plt.savefig(file_name+'.png',dpi=DPI)

        plt.show()
#%%
