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

try:
    from natsort import natsorted, ns
except:
    os.system("pip3 install natsort")
    from natsort import natsorted, ns
#%% Parameters
METHOD = 'DLEITH' #LEITH, DLEITH , DLEITH_sigma_Local, DLEITH_tau_Local,
# METHOD = 'DSMAG_tau_Local' #SMAG, DSMAG , DSMAG_sigma_Local, DSMAG_tau_Local,
# METHOD = 'NoSGS'

# case = str(sys.argv[1])
# percent_data = float(sys.argv[2])
CASENO = 1
NX = 32
# CASENO = 4
# NX = 256
# CASENO = 2
# NX = 64

NUM_DATA_Classic = 2_00
NUM_DATA_RL =1#_000
if CASENO == 2:
    NUM_DATA_Classic = 2_00
    # NUM_DATA_RL =3_00
    NUM_DATA_RL =2_500 #0

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
      KF = 4

elif CASENO==2:
    # mat_contents = scipy.io.loadmat('results/Re20000_fkx4fky4_r0.1_b20.0/Re20kNX1024nx4ny4r0p1b20_1024_1.0_aposteriori_data.mat')
    mat_contents = scipy.io.loadmat('results/Re20000_fkx4fky4_r0.1_b20.0/Re20kNX1024nx4ny4r0p1b20_aposteriori_data.mat')

    # patthadd = '/media/rmojgani/hdd/PostDoc/Projects/py2d_local/results/Re20000_fkx4fky4_r0.1_b20.0'
    # mat_contents = scipy.io.loadmat('results/Re20000_fkx4fky4_r0.1_b20.0/Re20kNX1024nx4ny4r0p1b20_1024_0.1_aposteriori_data.mat')
    KF = 4

elif CASENO==4:
    mat_contents = scipy.io.loadmat('results/Re20000_fkx25fky25_r0.1_b0/Re20kNX1024nx25ny25r0p1_aposteriori_data.mat')
    KF = 25

elif CASENO==10:
      patthadd = '/media/rmojgani/hdd/PostDoc/Projects/py2d_local/'
      mat_contents = scipy.io.loadmat('results/Re100kNX2048nx4ny0r0p1/Re100kNX2048nx4ny0r0p1_aposteriori_data.mat')

wavenumbers_spectra_DNS = np.arange(513)

energy_spectra_DNS = mat_contents['energy_spectra'].reshape(-1,)
enstrophy_spectra_DNS = mat_contents['enstrophy_spectra'].reshape(-1,)
# wavenumbers_spectra_DNS = mat_contents['wavenumbers_spectra'].reshape(-1,)

pdf_DNS = np.array([mat_contents['Omega_bins_scipy'][0,:],mat_contents['Omega_pdf_scipy'][0,:]]).T
std_omega_DNS =  mat_contents['Omega_std'][0][0]

#%% Load Classic results
# dataType = 'Re' + str(int(Re/1000)) + 'kNX' + str(NX) + 'nx' + str(fkx) + 'ny' + str(fky) + 'r0p1'
if CASENO==1:
    if 'LEITH' in METHOD:
        dataType = 'results/Re20000_fkx4fky4_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
        # dataType = '/mnt/Mount/envfluids/PostDoc/Projects/py2d/results/Re20000_fkx4fky4_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
    elif 'SMAG' in METHOD:
        dataType = 'results/Re20000_fkx4fky4_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
        # dataType = '/mnt/Mount/envfluids/PostDoc/Projects/py2d/results/Re20000_fkx4fky4_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'

elif CASENO==2:
    # dataType = '/mnt/Mount/envfluids/PostDoc/Projects/py2d/'
    # dataType = dataType+'results/Re20000_fkx4fky4_r0.1_b20/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
    # dataType = dataType+'results/Re20000_fkx4fky4_r0.1_b20.0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
    dataType = '/mnt/Mount/envfluids/PostDoc/Projects/py2d/results/Re20000_fkx4fky4_r0.1_b20/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------



if 'sigma_Local' in METHOD:
    METHOD_str = '∇.(ν_e ∇ω )'
elif 'tau_Local' in METHOD:
    METHOD_str = '∇×∇.(-2 ν_e S_{ij})'
else:
    METHOD_str = METHOD
METHOD_str += ', data='+str(NUM_DATA_Classic)


nx, ny = NX,NX
Lx,Ly = 2*np.pi,2*np.pi
Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly)

# select % of data for pdf calculation
percent_data = 1#0.75
# Compute the 2% of total elements in the matrix
num_samples = int(percent_data * NX * NX)

# Initialize lists to store data
energy_list, enstrophy_list, time_list = [], [], []
energy_spectra_list, enstrophy_spectra_list, Omega_list = [], [], []
eddy_turnover_time_list = []

DATA_DIR = dataType #+ '/DNS/' + ICname + '/'
print(DATA_DIR)

directory = DATA_DIR  # replace with your directory path

# Get a sorted list of all files
# files = sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0]))
# files = ['1.mat']#, '2.mat']

filecount = 0
# Iterate over sorted files
for file in natsorted(os.listdir(directory), alg=ns.PATH | ns.IGNORECASE):
    # Check if file ends with .mat
    if file.endswith('.mat') and filecount%1==0:
        file_path = os.path.join(directory, file)
        # Load .mat file
        mat = scipy.io.loadmat(file_path)
        print(f'{filecount}/{NUM_DATA_Classic}, Loaded: {file_path} \r')

        Omega = mat['Omega']
        Psi = Omega2Psi_2DFHIT(Omega, invKsq)

    filecount += 1

        # Now 'mat' is a dict with variable names as keys, and loaded matrices as values
        # You can process these matrices using 'mat' dict
    if filecount>NUM_DATA_Classic: break
    energy = energy_2DFHIT(Psi, Omega)
    enstrophy = enstrophy_2DFHIT(Omega)
    eddy_turnover_time = 1/np.sqrt(enstrophy)

    Psi_hat = np.fft.fft2(Psi)
    Omega_hat = np.fft.fft2(Omega)

    energy_spectra, wavenumbers = TKE_angled_average_2DFHIT(Psi_hat, Omega_hat, spectral=True)
    enstrophy_spectra, wavenumbers = enstrophy_angled_average_2DFHIT(Omega_hat, spectral=True)

    # Append to lists
    energy_list.append(energy)
    enstrophy_list.append(enstrophy)
    energy_spectra_list.append(energy_spectra)
    enstrophy_spectra_list.append(enstrophy_spectra)
    eddy_turnover_time_list.append(eddy_turnover_time)

    omega_flattened = Omega.flatten()
    indices = np.random.choice(omega_flattened.shape[0], num_samples, replace=False)
    sampled_data = omega_flattened[indices]
    Omega_list.append(sampled_data)

#%%
# Convert lists to numpy arrays
energy_arr = nnp.array(energy_list)
enstrophy_arr = nnp.array(enstrophy_list)
eddy_turnover_time_arr = nnp.array(eddy_turnover_time_list)
energy_spectra_arr = nnp.array(energy_spectra_list)
enstrophy_spectra_arr = nnp.array(enstrophy_spectra_list)
Omega_arr = nnp.array(Omega_list).flatten()

del Omega_list

# Calculate mean and standard deviation
Omega_mean, Omega_std = np.mean(Omega_arr), np.std(Omega_arr)

# Define bins within 10 standard deviations from the mean, but also limit them within the range of the data
bin_min = -40#(Omega_mean - 10*Omega_std)#, np.min(Omega_arr))
bin_max = 40#(Omega_mean + 10*Omega_std)#, np.max(Omega_arr))
if CASENO==2:
    bin_min = -50#(Omega_mean - 10*Omega_std)#, np.min(Omega_arr))
    bin_max = 50#(Omega_mean + 10*Omega_std)#, np.max(Omega_arr))
bins = np.linspace(bin_min, bin_max, 81)

print('bin min', bin_min)
print('bin max', bin_max)
print('Omega Shape', Omega_arr.shape)
print('Omega mean', Omega_mean)
print('Omega_std', Omega_std)
print('Total nans', np.sum(np.isnan(Omega_arr)))

# Take mean over the arrays
energy_spectra_mean = energy_spectra_arr.mean(axis=0)
enstrophy_spectra_mean = enstrophy_spectra_arr.mean(axis=0)

# Compute PDF
# Scipy
try:
    print('scipy')
    kde = gaussian_kde(Omega_arr, bw_method='scott')
    # # Define a range over which to evaluate the density
    bins_scipy = bins
    bw_scott = kde.factor
    # # Evaluate the density over the range
    pdf_scipy = kde.evaluate(bins_scipy)
except Exception as e:
    print(f"Scipy KDE failed with error: {str(e)}")
    pdf_scipy = np.empty(1)
    bins_scipy = np.empty(1)

#%% Load RL results
from py2d.spectra import *
# from mypathdictionary import *
# path_RL = mypathdictionaryRL(CASENO, NX, METHOD)
from mypathdictionary3 import *
path_RL, NUM_DATA_RL, BANDWIDTH_R = mypathdictionaryRL(CASENO, NX, METHOD)
#%%
energy_spectra_RL_list, enstrophy_spectra_RL_list = [], []
Omega_list = []

if 'LEITH' in METHOD :
    METHOD_RL = 'LEITH_RL'
elif 'SMAG' in METHOD :
    METHOD_RL = 'SMAG_RL'
METHOD_RL += ', data='+str(NUM_DATA_RL)

filecount = 0
for file in natsorted(os.listdir(path_RL), alg=ns.PATH | ns.IGNORECASE):
    # Check if file ends with .mat
    if file.endswith('.mat') and filecount%1==0:
        file_path = os.path.join(path_RL, file)
        print(f'RL → {filecount}/{NUM_DATA_RL}, Loaded: {file_path}')

        # Load .mat file
        mat_contents_RL = scipy.io.loadmat(file_path)

        Omega = np.fft.ifft2(mat_contents_RL['w_hat']).real
        Psi = np.fft.ifft2(mat_contents_RL['psi_hat']).real

        energy_spectra_RL, wavenumbers_RL = TKE_angled_average_2DFHIT(Psi, Omega, spectral=False)
        energy_spectra_RL = -energy_spectra_RL

        enstrophy_spectra_RL, wavenumbers_RL  = enstrophy_angled_average_2DFHIT(mat_contents_RL['w_hat'], spectral=True)


        energy_spectra_RL_list.append(energy_spectra_RL)
        enstrophy_spectra_RL_list.append(enstrophy_spectra_RL)

        omega_flattened = Omega.flatten()
        indices = np.random.choice(omega_flattened.shape[0], num_samples, replace=False)
        sampled_data = omega_flattened[indices]
        Omega_list.append(sampled_data)

        filecount += 1
        if filecount>NUM_DATA_RL: break

energy_spectra_RL_arr = nnp.array(energy_spectra_RL_list)
enstrophy_spectra_RL_arr = nnp.array(enstrophy_spectra_RL_list)

energy_spectra_RL_mean = energy_spectra_RL_arr.mean(axis=0)
enstrophy_spectra_RL_mean = enstrophy_spectra_RL_arr.mean(axis=0)

Omega_arr_RL = nnp.array(Omega_list).flatten()

del Omega_list

#%% Define error in specta
def error_spec(enstrophy_spectra_mean, enstrophy_spectra_DNS) :
    error_ens = np.abs(np.sum( enstrophy_spectra_mean - enstrophy_spectra_DNS[:int(NX/2+1)] )) / np.sum(enstrophy_spectra_DNS)

    return error_ens
#%% Plot - Spectra
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5),dpi=150)
plt.subplot(1,2,1)
plt.title('C'+str(CASENO)+': '+METHOD)
plt.loglog(wavenumbers_spectra_DNS, energy_spectra_DNS, linewidth=5, color="0.75")
plt.loglog(wavenumbers, energy_spectra_mean)
plt.loglog(wavenumbers_RL, energy_spectra_RL_mean,'k')


# plt.ylim([1e-6,1e0])
plt.ylim([1e-6,1e1])

error_tke = error_spec(energy_spectra_mean, energy_spectra_DNS)
error_tke_RL = error_spec(energy_spectra_RL_mean, energy_spectra_DNS)


plt.subplot(1,2,2)
plt.title('C'+str(CASENO)+': '+METHOD)

plt.loglog(wavenumbers_spectra_DNS, enstrophy_spectra_DNS, linewidth=5, color="0.75",label='DNS')
plt.loglog(wavenumbers, enstrophy_spectra_mean,label=METHOD_str)
plt.loglog(wavenumbers_RL, enstrophy_spectra_RL_mean,'k',label=METHOD_RL)

error_ens = error_spec(enstrophy_spectra_mean, enstrophy_spectra_DNS)
error_ens_RL = error_spec(enstrophy_spectra_RL_mean, enstrophy_spectra_DNS)

plt.legend()

# plt.ylim([1e-3,1e1])
plt.ylim([1e-3,1e2])
print(' Error spectra:                               error_tke', 'error_ens')
print(METHOD_RL   , NX, ':', error_tke_RL, error_ens_RL)
print(METHOD_str  , NX, ':', error_tke, error_ens)

#%% DNS DATA load .dat
# pdf_DNS2 = np.loadtxt('pdf_case01_FDNS.dat')
if CASENO==2:
    pdf_DNS2 = np.loadtxt('postprocess/pdf_case02_FDNS.dat')
    YMIN, YMAX = 1e-7, 1e0

# # std_omega_DNS = 1#6.0705
XMIN, XMAX = -5, 5
YMIN, YMAX = 1e-5, 1e-1


#%% KDE
from PDE_KDE import myKDE, mybandwidth_scott
# BANDWIDTH = mybandwidth_scott(Omega_arr)*1
# Vecpoints, exp_log_kde, log_kde, kde = myKDE(Omega_arr,BANDWIDTH=BANDWIDTH)

BANDWIDTH = mybandwidth_scott(Omega_arr_RL)
if CASENO==2:
    BANDWIDTH = mybandwidth_scott(Omega_arr_RL)*2
Vecpoints, exp_log_kde, log_kde, kde = myKDE(Omega_arr_RL,BANDWIDTH=BANDWIDTH)
#%% Plot - PDF
num_file = filecount
std_omega = 1#std_omega_DNS
plt.figure(figsize=(7,5),dpi=150)
plt.title(METHOD)
# plt.semilogy(bins_fastkde/std_omega,pdf_fastkde,label='fastked')
plt.semilogy(bins_scipy/std_omega,pdf_scipy,label='scipy')
# plt.semilogy(bins_skl/std_omega,pdf_skl,label='scikit-learn')
# plt.semilogy(bins_sm/std_omega,pdf_sm,label='sm')

plt.semilogy(Vecpoints/std_omega, exp_log_kde, ':k', alpha=0.75, linewidth=2, label=METHOD_RL)

# plt.ylim([1e-5,1e-1])
# plt.xlim([-5,+5])
# plt.legend()

try:
    plt.semilogy(pdf_DNS2[:,0], pdf_DNS2[:,1], 'r', linewidth=4.0, alpha=0.5, label='DNS_Yifei')
except:
    pass

plt.semilogy(pdf_DNS[:,0], pdf_DNS[:,1], 'b', linewidth=4.0, alpha=0.5, label='DNS')
# plt.semilogy(-pdf_DNS[:,0], pdf_DNS[:,1], 'b', linewidth=4.0, alpha=0.5, label='DNS')


plt.xlabel(r'$\omega / \sigma(\omega)$, $\sigma(\omega)$=' +
           str(np.round(std_omega, 2)))
plt.ylabel(r'$\mathcal{P}\left(\omega\right)$, w. '+str(num_file)+' samples')

plt.grid(which='major', linestyle='--',
         linewidth='1.0', color='black', alpha=0.25)
plt.grid(which='minor', linestyle='-',
         linewidth='0.5', color='red', alpha=0.25)

# minor_ticks = np.arange(-6, 7, 0.5)
# plt.xticks(minor_ticks, minor=True)
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))

#plt.xlim([XMIN, XMAX])
plt.xlim([-50, 50])
plt.ylim([YMIN/1000, YMAX])
plt.legend()
#%% Interpolate PDFs
from scipy import interpolate
x = pdf_DNS[:,0]
y = pdf_DNS[:,1]
f = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")

pdf_DNS_on_bins = f(bins)   # use interpolation function returned by `interp1d`

x = Vecpoints
y = exp_log_kde
f = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)

pdf_data_on_bins = f(bins)   # use interpolation function returned by `interp1d`

pdf_extreme_line = 3

in_band = np.abs(bins)<=(pdf_extreme_line*std_omega_DNS)
out_band = ~in_band
#%% Error pdfs
error_pdf_in = np.linalg.norm(np.abs(pdf_data_on_bins-pdf_DNS_on_bins)[in_band])
pdf_in = np.linalg.norm(np.abs(pdf_DNS_on_bins)[in_band])
error_in = error_pdf_in/pdf_in


error_pdf_out = np.linalg.norm(np.abs(pdf_data_on_bins-pdf_DNS_on_bins)[out_band])
pdf_out = np.linalg.norm(np.abs(pdf_DNS_on_bins)[out_band])
error_out = error_pdf_out/pdf_out

error_pdf_all = np.linalg.norm(np.abs(pdf_data_on_bins-pdf_DNS_on_bins))
pdf_all = np.linalg.norm(np.abs(pdf_DNS_on_bins))
error_all = error_pdf_all/pdf_all

print('error:   Bulk, Tail, All')
print(METHOD_RL,":", error_in, error_out, error_all)

##%% Error pdfs RL
error_pdf_in = np.linalg.norm(np.abs(pdf_scipy-pdf_DNS_on_bins)[in_band])
pdf_in = np.linalg.norm(np.abs(pdf_DNS_on_bins)[in_band])
error_in_RL = error_pdf_in/pdf_in


error_pdf_out = np.linalg.norm(np.abs(pdf_scipy-pdf_DNS_on_bins)[out_band])
pdf_out = np.linalg.norm(np.abs(pdf_DNS_on_bins)[out_band])
error_out_RL = error_pdf_out/pdf_out

error_pdf_all = np.linalg.norm(np.abs(pdf_scipy-pdf_DNS_on_bins))
pdf_all = np.linalg.norm(np.abs(pdf_DNS_on_bins))
error_all_RL = error_pdf_all/pdf_all

print(METHOD_str,":", error_in_RL, error_out_RL, error_all_RL)

#%% Plot PDFs
fig, axs= plt.subplots(1,3, sharex=True, figsize=(16,5), dpi=150)

yhor = np.ones_like(bins)*1e-12

axs[0].fill_between(bins, yhor, pdf_DNS_on_bins, where=(in_band), color='C0', alpha=0.3)

axs[0].semilogy(bins_scipy, pdf_scipy       , '+-r' , label=METHOD_str)
axs[0].semilogy(bins      , pdf_data_on_bins, '+-g', label=METHOD_RL)
axs[0].semilogy(bins      , pdf_DNS_on_bins , '.k' , label='DNS')
axs[0].legend(loc='upper left')
#axs[0][0].legend(bbox_to_anchor=(1.04, 1))

plt.subplot(1,3,1)
plt.xlim([bin_min, bin_max])
plt.ylim([YMIN/100, YMAX*10])

axs[1].fill_between(bins, yhor, np.abs(pdf_data_on_bins-pdf_DNS_on_bins), where=(in_band), color='C0', alpha=0.3)

axs[1].semilogy(bins,np.abs(pdf_data_on_bins-pdf_DNS_on_bins), '-')
axs[1].semilogy(bins[in_band],np.abs(pdf_data_on_bins-pdf_DNS_on_bins)[in_band], '.k')
axs[1].semilogy(bins[out_band],np.abs(pdf_data_on_bins-pdf_DNS_on_bins)[out_band], '.r')
plt.subplot(1,3,2)
plt.title('RL')
plt.xlim([bin_min, bin_max])
# plt.ylim([YMIN/1e2, YMAX/1e1])
plt.ylim([YMIN/1e2, YMAX/1e1])

#plt.text(-30, 1e-3, str(error_in_RL)+str(error_out_RL)+str(error_all_RL))

ii=2
axs[ii].fill_between(bins, yhor, np.abs(pdf_scipy-pdf_DNS_on_bins), where=(in_band), color='C1', alpha=0.3)

axs[ii].semilogy(bins,np.abs(pdf_scipy-pdf_DNS_on_bins), '-')
axs[ii].semilogy(bins[in_band],np.abs(pdf_scipy-pdf_DNS_on_bins)[in_band], '+k')
axs[ii].semilogy(bins[out_band],np.abs(pdf_scipy-pdf_DNS_on_bins)[out_band], '+r')

plt.subplot(1,3,3)
plt.title(METHOD)
plt.xlim([bin_min, bin_max])
plt.ylim([YMIN/1e2, YMAX/1e1])

plt.show()
#%% SAVE SPECTRA
np.savetxt('C'+str(CASENO)+'_'+'N'+str(NX)+'_'+METHOD+'_energy.dat', np.vstack([wavenumbers, energy_spectra_mean]).T, delimiter='\t')
np.savetxt('C'+str(CASENO)+'_'+'N'+str(NX)+'_'+METHOD_RL+'_energy.dat', np.vstack([wavenumbers_RL, energy_spectra_RL_mean]).T, delimiter='\t')

np.savetxt('C'+str(CASENO)+'_'+'N'+str(NX)+'_'+METHOD+'_enstrophy.dat', np.vstack([wavenumbers, enstrophy_spectra_mean]).T, delimiter='\t')
np.savetxt('C'+str(CASENO)+'_'+'N'+str(NX)+'_'+METHOD_RL+'_enstrophy.dat', np.vstack([wavenumbers_RL, enstrophy_spectra_RL_mean]).T, delimiter='\t')

#%% SAVE PDF
np.savetxt('C'+str(CASENO)+'_'+'N'+str(NX)+'_'+METHOD+'_pdf.dat', np.vstack([bins, pdf_DNS_on_bins, pdf_scipy ]).T, delimiter='\t')
np.savetxt('C'+str(CASENO)+'_'+'N'+str(NX)+'_'+METHOD_RL+'_pdf.dat', np.vstack([bins, pdf_DNS_on_bins, pdf_data_on_bins ]).T, delimiter='\t')
# #%%
# np.savetxt('C'+str(CASENO)+'_'+'N'+str(NX)+'_'+'DNS'+'_energy.dat', np.vstack([wavenumbers_spectra_DNS, energy_spectra_DNS]).T, delimiter='\t')
# np.savetxt('C'+str(CASENO)+'_'+'N'+str(NX)+'_'+'DNS'+'_enstrophy.dat', np.vstack([wavenumbers_spectra_DNS, enstrophy_spectra_DNS]).T, delimiter='\t')
# np.savetxt('C'+str(CASENO)+'_'+'N'+str(NX)+'_'+'DNS'+'_pdf.dat', np.vstack([bins, pdf_DNS_on_bins, pdf_DNS_on_bins ]).T, delimiter='\t')

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
#%% Definitions for interscale for DNS
import sys
sys.path.append('/media/rmojgani/hdd/PostDoc/ScratchBook/spectra/experiments/case2/')
from filters import filter_guassian, filter_cutoff, filter_cutoff_coarsen

myfilter = 'cutoff' # Gaussian , cutoff

NLES = NX
Delta_c = 2*np.pi/NLES
Delta = 2*Delta_c

if myfilter == 'Gaussian':
    filter = lambda a_hat: filter_guassian(a_hat, Delta, Ksq)
    # DELTA_STR=Fraction(str(Delta/np.pi)).as_integer_ratio()
    title_str = ''

    # YMINEflux, YMAXEflux = -0.05*(64/NLES), 0.05*(64/NLES)
    # YMINZflux, YMAXZflux = -0.25, 0.25
    YMINEflux, YMAXEflux = -1e-1*(32/NLES)**2, 1e-1*(32/NLES)**2
    YMINZflux, YMAXZflux = -2*(32/NLES)**2, 2*(32/NLES)**2

elif myfilter == 'cutoff':
    filter = lambda a_hat:  filter_cutoff(a_hat, NDNS, NLES)[0]
    YMINEflux, YMAXEflux = -5e-3*(32/NLES)**3, 5e-3*(32/NLES)**3
    YMINZflux, YMAXZflux = -5e-2*(64/NLES)**3, 5e-2*(64/NLES)**3

#% def convection_conserved
def convection_conserved(psi_hat, w1_hat, Kx, Ky):
    # Velocity
    u1_hat = -(1j*Ky)*psi_hat
    v1_hat = (1j*Kx)*psi_hat

    # Convservative form
    w1 = np.real(np.fft.ifft2(w1_hat))
    conu1 = 1j*Kx*np.fft.fft2((np.real(np.fft.ifft2(u1_hat))*w1))
    conv1 = 1j*Ky*np.fft.fft2((np.real(np.fft.ifft2(v1_hat))*w1))
    convec_hat = conu1 + conv1

    # Non-conservative form
    w1x_hat = 1j*Kx*w1_hat
    w1y_hat = 1j*Ky*w1_hat
    conu1 = np.fft.fft2(np.real(np.fft.ifft2(u1_hat))*np.real(np.fft.ifft2(w1x_hat)))
    conv1 = np.fft.fft2(np.real(np.fft.ifft2(v1_hat))*np.real(np.fft.ifft2(w1y_hat)))
    convecN_hat = conu1 + conv1

    convec_hat = 0.5*(convec_hat + convecN_hat)
    return convec_hat

#% def PI_from_convection_conserved
def PI_from_convection_conserved(psi_hat, w1_hat, Kx, Ky, filter):
    J1 = filter(convection_conserved(psi_hat, w1_hat, Kx, Ky))
    J2 = convection_conserved(filter(psi_hat), filter(w1_hat), Kx, Ky)
    PI_hat = J1 - J2
    PI = np.fft.ifft2(PI_hat).real
    return PI, PI_hat
#%%
from mypathdictionary3 import *
sys.path.append('/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/_model/py2d/')
# METHOD = 'DSMAG_tau_Local'
METHOD = 'DLEITH_tau_Local'

MYLOAD = 'RL'
NSAMPLE = 50#0#1#00

from py2d.spectra import *
if MYLOAD == 'RL':
    path_RL, _, _ = mypathdictionaryRL(CASENO, NX, METHOD)
else:
    path_RL = mypathdictionaryclassic(CASENO, NX, METHOD)

from convert import Tau2PiOmega_2DFHIT
from eddy_viscosity_models import Tau_eddy_viscosity

# NSAMPLE = 1
spec_tke_mean = 0
spec_ens_mean = 0

Eflux_hat_M = []
Zflux_hat_M = []

PI_M = []

icount = 0
filecount = 0
for file in natsorted(os.listdir(path_RL), alg=ns.PATH | ns.IGNORECASE):
    # Check if file ends with .mat
    if file.endswith('.mat') and filecount%1==0:
        file_path = os.path.join(path_RL, file)
        print(f'RL → {filecount}/{NSAMPLE}, Loaded: {file_path}')

        # Load .mat file
        mat_contents_RL = scipy.io.loadmat(file_path)

        if MYLOAD == 'RL':
            Omega = np.fft.ifft2(mat_contents_RL['w_hat']).real
            Psi = np.fft.ifft2(mat_contents_RL['psi_hat']).real
            psi_hat = mat_contents_RL['psi_hat']
        else:
            Omega = mat_contents_RL['Omega']
            Psi = Omega2Psi_2DFHIT(Omega, invKsq=invKsq)

        if MYLOAD == 'RL':
            ve = np.abs(mat_contents_RL['veRL'])
            print( 'min(ν_e)', np.min(ve) )
            Tau11, Tau12, Tau22 = Tau_eddy_viscosity(ve, psi_hat, Kx, Ky)


            Tau11_hat = np.fft.fft2(Tau11)
            Tau12_hat = np.fft.fft2(Tau12)
            Tau22_hat = np.fft.fft2(Tau22)

            PiOmega_hat = Tau2PiOmega_2DFHIT(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky, spectral=True)
            PI_hat = PiOmega_hat
            PI = np.fft.ifft2(PI_hat).real

            if  np.min(ve) < 0 :
                stop
        # PI, PI_hat = PI_from_convection_conserved(Psi_hat, Omega_hat, Kx, Ky, filter)


        # PI = np.fft.ifft2(PI_hat).real
        # Omega = w1Python

        # PI_hat = np.fft.fft2(PI).real
        # Omega_hat = np.fft.fft2(Omega).real

        spec_ens, k = enstrophyTransfer_spectra_2DFHIT(Kx, Ky, Omega=Omega, Sigma1=None, Sigma2=None, PiOmega=PI, method='PiOmega', spectral=False)
        # # spec, k = enstrophyTransfer_spectra_2DFHIT(Kx, Ky, Omega=Omega_hat, Sigma1=None, Sigma2=None, PiOmega=PI_hat, method='PiOmega', spectral=True)

        # needs :
        # Psi = Omega2Psi_2DFHIT(Omega, invKsq=invKsq)
        spec_tke, k = energyTransfer_spectra_2DFHIT(Kx, Ky, U=None, V=None, Tau11=None, Tau12=None, Tau22=None, Psi=Psi, PiOmega=PI, method='PiOmega', spectral=False)

        spec_tke_mean +=spec_tke
        spec_ens_mean +=spec_ens

        Eflux_hat_M = np.append(Eflux_hat_M, spec_tke)
        Zflux_hat_M = np.append(Zflux_hat_M, spec_ens)

        PI_M = np.append(PI_M, PI)


        filecount += 1
        if filecount > NSAMPLE: break
##%%
kplot_str='\kappa'
kmax = int(NLES/2)
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.title('Sample size:'+str(NSAMPLE))#+', filter:'+myfilter )
# plt.text(NLES/2, YMAXEflux*0.75, rf'$\kappa_c={kmax}$',fontsize=24)
plt.plot(k,spec_tke_mean/NSAMPLE)
# plt.ylim([YMINEflux, YMAXEflux])
plt.ylabel('$T_E=\mathbb{R}( \hat{\Pi} . \hat{\psi} )$')
# plt.ylim([-0.0006, 0.0006])


plt.subplot(1,2,2)
# plt.text(NLES/2, YMAXZflux*0.75, rf'$\kappa_c={kmax}$',fontsize=24)
plt.plot(k,spec_ens_mean/NSAMPLE)
# plt.ylim([YMINZflux, YMAXZflux])
plt.ylabel('$T_Z=\mathbb{R}( \hat{\Pi} . \hat{\omega} )$')
# plt.ylim([-0.05, 0.05])

for icount in [1,2]:
    plt.subplot(1,2,icount)
    plt.xlabel(rf'${kplot_str}$')
    # plt.plot([NLES/2,NLES/2],[-10,10],'--r', linewidth=2)


    plt.gca().set_xlim(left=1)
    plt.grid(which='major', linestyle='--',
              linewidth='1.0', color='black', alpha=0.25)
    plt.grid(which='minor', linestyle='-',
              linewidth='0.5', color='red', alpha=0.15)

    plt.gca().set_xscale('log')
    plt.xlim([1, 512])
    # plt.xlim([1, 32])
    plt.tight_layout()
##%%
# filename_save = METHOD_RL+'interscale_k'+str(KF)+'_NLES'+str(NLES)
# results = np.vstack((k, spec_ens_mean/NSAMPLE, spec_tke_mean/NSAMPLE)).T
#%%
matrix_action = mat_contents_RL['veRL']
xagent = np.linspace(1, NX, 4+1, endpoint= True)-1
yagent =np.linspace(1, NX, 4+1, endpoint= True)-1
Xagent, Yagent = np.meshgrid(xagent, yagent, indexing='ij')

plt.figure(figsize=(6,5))
plt.contourf(matrix_action, levels=101)#, vmin=-0.15, vmax=0.15);
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(Xagent, Yagent, c='k', marker='+',alpha=0.5); plt.colorbar();
#%%
from scipy.interpolate import RectBivariateSpline
##%%
L = 2*np.pi
nActiongrid = 4
xaction = np.linspace(0,L,nActiongrid, endpoint=True)
yaction = np.linspace(0,L,nActiongrid, endpoint=True)
upsamplesize = NX # 1 for testing, will be changed to grid size eventually

# def upsample(action, xaction, yaction, arr_action, upsamplesize):
#     '''
#     action: list of lenght  ...
#     forcing: np.array Nx x Nx
#     '''

#     upsample_action = RectBivariateSpline(xaction, yaction, arr_action, kx=2, ky=2)

#     # Initlize action
#     x2 = np.linspace(0,L, upsamplesize, endpoint=True)
#     y2 = np.linspace(0,L,  upsamplesize, endpoint=True)
#     forcing = upsample_action(x2, y2)
#     return forcing
def upsample(xaction, yaction, arr_action, upsamplesize):


    Nfine = upsamplesize
    Ncoarse = 4

    Ahat = np.zeros((Nfine,Nfine), dtype=np.complex128)
    Ahat[:Ncoarse,:Ncoarse] = np.fft.fft2(arr_action)
    forcing = np.fft.ifft2(Ahat).real*(NX/Ncoarse)*(NX/Ncoarse)

    return forcing


matrix_action = mat_contents_RL['veRL']
# forcing = upsample(action, Xaction, Yaction, arr_action, upsamplesize)
indices = np.linspace(0, NX-1, num=4).astype('int')
##%%
kdegree = 1
upsample_action = RectBivariateSpline(xaction, yaction, matrix_action[indices,][:,indices], kx=kdegree, ky=kdegree)

# Initlize action
x2 = np.linspace(0,L, upsamplesize, endpoint=True)
y2 = np.linspace(0,L,  upsamplesize, endpoint=True)
# forcing = upsample_action(x2, y2)

forcing = mat_contents_RL['veRL']#upsample( matrix_action[indices,][:,indices], xaction, yaction, 32)
##%%
import matplotlib as mpl

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
mesh = ax.contourf(forcing, levels=101,  vmin=-0.15, vmax=0.150);
mesh.set_clim(-0.15,0.15)
# plt.colorbar(mesh, boundaries=np.linspace(0, 2, 6))
# plt.clim(vmin=-0.15, vmax=0.15)
# cbar = plt.colorbar()
# ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=0.5)
# cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm,
                       # norm=mpl.colors.Normalize(vmin=-0.15, vmax=0.15))
# ax.set_clim(-0.15, 0.15)
# mesh.colorbar()
# cbar.set_ticks(np.linspace(-0.15, 0.15,3,endpoint=True))


plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(Xagent, Yagent, c='r', marker='o',alpha=1.0)
plt.scatter(Xagent, Yagent, c='w', marker='+',alpha=1.0)
plt.xlim([0,NX-1])
plt.ylim([0,NX-1])

##%%
plt.figure()

VMIN = np.min(forcing)
VMAX = np.max(forcing)

CS = plt.contourf(forcing, levels = 101, vmin = VMIN, vmax = VMAX, cmap=cm.bwr)
m = plt.cm.ScalarMappable(cmap=cm.bwr)
m.set_array(forcing)
m.set_clim(VMIN, VMAX)
plt.colorbar(m, boundaries=np.linspace(VMIN, VMAX, 51,endpoint=True))
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(Xagent, Yagent, c='k', marker='o',alpha=0.5)
plt.scatter(Xagent, Yagent, c='w', marker='+',alpha=1.0)
##%%
PPLOT = Omega
plt.figure()
VMIN = min( np.min(Omega),  -np.max(Omega))
VMAX = max( np.max(Omega),  np.min(-Omega))

CS = plt.contourf(Omega, levels = 101, vmin = VMIN, vmax = VMAX, cmap=cm.bwr)
m = plt.cm.ScalarMappable(cmap=cm.bwr_r)
m.set_array(forcing)
m.set_clim(VMIN, VMAX)
plt.colorbar(m, boundaries=np.linspace(VMIN, VMAX, 51,endpoint=True))
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(Xagent, Yagent, c='k', marker='o',alpha=0.5)
plt.scatter(Xagent, Yagent, c='w', marker='+',alpha=1.0)
#%%
def smag_cs(Kx, Ky, psi_hat, forcing):
    cs = forcing * ((2*np.pi/NX )**2)
    S1 = np.real(np.fft.ifft2(-Ky*Kx*psi_hat)) # make sure .*
    S2 = 0.5*np.real(np.fft.ifft2(-(Kx*Kx - Ky*Ky)*psi_hat))
    S  = 2.0*(S1*S1 + S2*S2)**0.5
#        cs = (0.17 * 2*np.pi/NX )**2  # for LX = 2 pi
    Smean = (np.mean(S**2.0))**0.5;
    ve = cs*Smean

    return ve, cs, S
##%%
ve, cs, S = smag_cs(Kx, Ky, psi_hat, forcing)
#%%
w1_hat = mat_contents_RL['w_hat']
Grad_Omega_hat_dirx = Kx*np.fft.fft2( ve * np.fft.ifft2(Kx*w1_hat) )
Grad_Omega_hat_diry = Ky*np.fft.fft2( ve * np.fft.ifft2(Ky*w1_hat) )
PiOmega_hat = Grad_Omega_hat_dirx + Grad_Omega_hat_diry

plt.figure(figsize=(26,4),dpi=250)

icount = 1
title_str = ['{S}','$c_s$','$\psi$','$\Pi$','']
for PPLOT in [S, forcing, np.fft.ifft2(psi_hat).real, np.fft.ifft2(PiOmega_hat).real]:
# for PPLOT in [np.fft.ifft2(PiOmega_hat).real]:
    plt.subplot(1,4,icount)
    plt.title(title_str[icount-1])

    VMIN = np.min(PPLOT)
    VMAX = np.max(PPLOT)

    CS = plt.contourf(PPLOT, levels = 101, vmin = VMIN, vmax = VMAX, cmap=cm.bwr)
    m = plt.cm.ScalarMappable(cmap=cm.bwr_r)
    m.set_array(PPLOT)
    m.set_clim(VMIN, VMAX)
    plt.colorbar(m, boundaries=np.linspace(VMIN, VMAX, 51,endpoint=True))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(Xagent, Yagent, c='k', marker='o',alpha=0.5)
    plt.scatter(Xagent, Yagent, c='w', marker='+',alpha=1.0)

    icount += 1
plt.show()
#%%
# PPLOT = PPLOT3-PPLOT1

# VMIN = -60#np.min(PPLOT)
# VMAX = 60#np.max(PPLOT)

# plt.figure()
# CS = plt.contourf(PPLOT, levels = 101)#, vmin = VMIN, vmax = VMAX, cmap=cm.bwr_r)

# m = plt.cm.ScalarMappable(cmap=cm.bwr_r)
# m.set_array(PPLOT)
# m.set_clim(VMIN, VMAX)
# plt.colorbar(m, boundaries=np.linspace(VMIN, VMAX, 51,endpoint=True))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.scatter(Xagent, Yagent, c='k', marker='o',alpha=0.5)
# plt.scatter(Xagent, Yagent, c='w', marker='+',alpha=1.0)
# plt.show()
#%%
degree = 2
xcoarse = np.linspace(0,L,NLES, endpoint=True)
ycoarse = np.linspace(0,L,NLES, endpoint=True)
xfine = np.linspace(0,L,1024, endpoint=True)
yfine = np.linspace(0,L,1024, endpoint=True)

fig,ax = plt.subplots()

LEVELS=101
ax.clear()
w1Python = Omega
upsample_action = RectBivariateSpline(xcoarse, ycoarse, w1Python, kx=degree, ky=degree)
w1Python = upsample_action(xfine, yfine)
plt.contourf(w1Python,vmin=-26,vmax=26,levels=LEVELS, cmap='bwr');plt.axis('square')
#ax.set_title('%03d'%(i))
ax.grid(False)
ax.axis('off')
#%%
Omega = scipy.io.loadmat('10001.mat')['Omega']

w1Python_down = Omega[::32,::32]
upsample_action = RectBivariateSpline(xcoarse, ycoarse, w1Python_down, kx=degree, ky=degree)
w1Python_down_up = upsample_action(xfine, yfine)


from py2d.filter import *
w1Python = np.fft.fft2(Omega)
w1Python_hat = coarse_spectral_filter_square_2DFHIT(w1Python, 32)
w1Python_f = np.fft.ifft2(w1Python_hat).real

upsample_action = RectBivariateSpline(xcoarse, ycoarse, w1Python_f, kx=degree, ky=degree)
w1Python_f_up = upsample_action(xfine, yfine)
#%%
NCoarse = 32
Delta = 2*L/NCoarse
w1Python_f_gauss = filter2D_2DFHIT(Omega, filterType='gaussian', coarseGrainType='spectral', Delta=Delta, Ngrid=NCoarse)
# w1Python_f_gauss = np.fft.ifft2()

upsample_action = RectBivariateSpline(xcoarse, ycoarse, w1Python_f_gauss, kx=degree, ky=degree)
w1Python_f_gauss_up = upsample_action(xfine, yfine)

#%% Plot 3x3
plt.figure(figsize=(17,22), dpi=150)
plt.subplot(4,3,1);plt.title('DNS')
plt.contourf(Omega,vmin=-26,vmax=26,levels=LEVELS, cmap='bwr');plt.axis('square')


plt.subplot(4,3,4);plt.title('Downsample')
plt.pcolor(w1Python_down, cmap='bwr');plt.axis('square')

plt.subplot(4,3,5);plt.title('Downsample: Contourf, default')
plt.contourf(w1Python_down,vmin=-26,vmax=26,levels=LEVELS, cmap='bwr');plt.axis('square')

plt.subplot(4,3,6);plt.title('Downsample: Contourf, Bi-quadratic')
plt.contourf(w1Python_down_up,vmin=-26,vmax=26,levels=LEVELS, cmap='bwr');plt.axis('square')


plt.subplot(4,3,7);plt.title('Spectral Filter')
plt.pcolor(w1Python_f, cmap='bwr');plt.axis('square')

plt.subplot(4,3,8);plt.title('S Filter: Contourf, default')
plt.contourf(w1Python_f,vmin=-26,vmax=26,levels=LEVELS, cmap='bwr');plt.axis('square')

plt.subplot(4,3,9);plt.title('S Filter: ontourf, Bi-quadratic')
plt.contourf(w1Python_f_up,vmin=-26,vmax=26,levels=LEVELS, cmap='bwr');plt.axis('square')

plt.subplot(4,3,10);plt.title('Gaussian Filter')
plt.pcolor(w1Python_f_gauss, cmap='bwr');plt.axis('square')

plt.subplot(4,3,11);plt.title('S Filter: Contourf, default')
plt.contourf(w1Python_f_gauss,vmin=-26,vmax=26,levels=LEVELS, cmap='bwr');plt.axis('square')

plt.subplot(4,3,12);plt.title('S Filter: ontourf, Bi-quadratic')
plt.contourf(w1Python_f_gauss_up,vmin=-26,vmax=26,levels=LEVELS, cmap='bwr');plt.axis('square')
#%% Movie - Preload
METHOD = 'DLEITH_tau_Local' #LEITH, DLEITH , DLEITH_sigma_Local, DLEITH_tau_Local,
# METHOD = 'DSMAG_tau_Local' #SMAG, DSMAG , DSMAG_sigma_Local, DSMAG_tau_Local,
for METHOD in ['DLEITH_tau_Local' ]:#['LEITH', 'DLEITH' , 'DLEITH_sigma_Local', 'DLEITH_tau_Local']:
    path_classic = mypathdictionaryclassic(CASENO, NX, METHOD)
    file_path = os.path.join(path_classic, file)

    mat_contents_classic = scipy.io.loadmat(file_path)
    Omega =  mat_contents_classic['Omega']

    plt.figure(figsize=(17,22), dpi=150)

    plt.subplot(4,3,12);plt.title(METHOD)
    plt.contourf(Omega,vmin=-26,vmax=26,levels=LEVELS, cmap='bwr');plt.axis('square')
    plt.colorbar()

#%% Movie
import matplotlib.animation as animation
from IPython import display
from mpl_toolkits.axes_grid1 import make_axes_locatable
DPI = 300
max_val = 25
LEVELS = np.linspace(-max_val, max_val, 100)
vmin=-max_val
vmax=max_val
#for METHOD in ['DSMAG_tau_Local']:
# for METHOD in ['LEITH', 'DLEITH' , 'DLEITH_sigma_Local', 'DLEITH_tau_Local',
#                'SMAG', 'DSMAG' , 'DSMAG_sigma_Local', 'DSMAG_tau_Local']:
for METHOD in [ 'DSMAG' ,'DSMAG_sigma_Local', 'DSMAG_tau_Local']:
    print('Animation generation, method', METHOD)

    filename_save = METHOD
    path_classic = mypathdictionaryclassic(CASENO, NX, METHOD)

    fig = plt.figure(dpi=DPI)
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    #cax = div.append_axes('right', '5%', '5%')

    im = ax.contourf(Omega, vmin=-max_val, vmax=max_val, levels=LEVELS, cmap='bwr');
    plt.axis('square')

    cb = plt.colorbar(im, ax=ax, ticks=np.linspace(-max_val, max_val, 11))
    # cb.ax.set_yticklabels(['{:.2f}'.format(i) for i in np.linspace(-max_val, max_val, 11)])
    #cb = fig.colorbar(im, cax=cax)

    def animate(i,LEVELS=100):
        cax.cla()
        print(i)
        file= str(i+1)+'.mat'
        file_path = os.path.join(path_classic, file)

        mat_contents_classic = scipy.io.loadmat(file_path)
        Omega =  mat_contents_classic['Omega']
        #upsample_action = RectBivariateSpline(xcoarse, ycoarse, w1Python, kx=2, ky=2)
        #w1Python = upsample_action(xfine, yfine)
        im = ax.contourf(Omega,vmin=-max_val, vmax=max_val,levels=LEVELS, cmap='bwr');plt.axis('square')
        #fig.colorbar(im, cax=cax)
        #cb = fig.colorbar(im, cax=cax)

        #ax.set_title('%03d'%(i))
        ax.set_title(METHOD)
        ax.grid(False)
        ax.axis('off')

    MAX_FRAMES= 200#0
    interval = 0.005#in seconds
    ani = animation.FuncAnimation(fig,animate,save_count=MAX_FRAMES,blit=False)
    FFwriter = animation.FFMpegWriter()
    ani.save(filename_save+'.mp4', writer = FFwriter, dpi=DPI)
    plt.show()
#%% PDF of Π
BANDWIDTH = mybandwidth_scott(PI_M)

Vecpoints, exp_log_kde, log_kde, kde = myKDE(PI_M,BANDWIDTH=BANDWIDTH)
plt.figure(figsize=(5,5))
plt.semilogy(Vecpoints/np.std(PI_M), exp_log_kde, ':k', alpha=0.75, linewidth=2, label=METHOD_RL)
plt.ylim([1e-5,1e-1])
plt.xlim([-5,5])
plt.xlabel(r'$\Pi/\sigma(\Pi)$')
plt.ylabel(r'$\mathcal{P}\left(\Pi\right)$')
#%%
from py2d.apriori_analysis import *
from py2d.convert import *
# from py2d.eddy_viscosity_models import Sigma_eddy_viscosity
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
# input Psi_hat, ve, Omega

U_hat, V_hat = Psi2UV_2DFHIT_spectral(Psi_hat, Kx, Ky)
U = np.fft.ifft2(U_hat)
V = np.fft.ifft2(V_hat)

Tau11, Tau12, Tau22 = Tau_eddy_viscosity(ve, Psi_hat, Kx, Ky)

Tau11_hat = np.fft.fft2(Tau11)
Tau12_hat = np.fft.fft2(Tau12)
Tau22_hat = np.fft.fft2(Tau22)

PTau = energyTransfer_2DFHIT(U, V, Tau11, Tau12, Tau22, Kx, Ky)


Omega_hat = np.fft.ifft2(Omega)
Sigma1, Sigma2 = Sigma_eddy_viscosity(ve, Omega_hat, Kx, Ky)


PZ = enstrophyTransfer_2D_FHIT(Omega, Sigma1, Sigma2, Kx, Ky)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)

plt.pcolor(PTau,cmap='gray_r')
plt.colorbar();plt.tight_layout();plt.axis('square')
plt.title(r'$P_{\tau}$')

plt.subplot(1,2,2)
plt.pcolor(PZ)

plt.colorbar();plt.tight_layout();plt.axis('square')
plt.title(r'$P_{Z}$')

#%% Distribution of P_tau, P_Z
BANDWIDTH = mybandwidth_scott(PTau)

Vecpoints, exp_log_kde, log_kde, kde = myKDE(PTau/np.std(PTau), BANDWIDTH=BANDWIDTH )

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.semilogy(Vecpoints, exp_log_kde, ':k', alpha=0.75, linewidth=2, label=METHOD_RL)
# plt.ylim([1e-5,1e-1])
# plt.xlim([-5,5])
plt.xlabel(r'$T_E$')
plt.ylabel(r'$\mathcal{P}\left(T_E\right)$')
plt.tight_layout()

BANDWIDTH = mybandwidth_scott(PZ)

Vecpoints, exp_log_kde, log_kde, kde = myKDE(PZ/np.std(PZ),BANDWIDTH=BANDWIDTH)
plt.subplot(1,2,2)

plt.semilogy(Vecpoints, exp_log_kde, ':k', alpha=0.75, linewidth=2, label=METHOD_RL)
# plt.ylim([1e-5,1e-1])
# plt.xlim([-5,5])
plt.xlabel(r'$T_Z$')
plt.ylabel(r'$\mathcal{P}\left(T_Z\right)$')
plt.tight_layout()
#%%

# METHOD = 'DSMAG_tau_Local'
METHOD = 'DLEITH_tau_Local'

MYLOAD = 'RL'
NSAMPLE = 50#0#1#00

from py2d.spectra import *
if MYLOAD == 'RL':
    path_RL, _, _ = mypathdictionaryRL(CASENO, NX, METHOD)
else:
    path_RL = mypathdictionaryclassic(CASENO, NX, METHOD)

plt.figure(figsize=(10,5))

NSAMPLE = 200#0
PZ_M = []
PTau_M = []
cs_M = []
ve_M = []

icount = 0
filecount = 0
for file in natsorted(os.listdir(path_RL), alg=ns.PATH | ns.IGNORECASE):
    print(file)
    # Check if file ends with .mat
    if file.endswith('.mat') and filecount%1==0:

        file_path = os.path.join(path_RL, file)
        print(f'RL → {filecount}/{NSAMPLE}, Loaded: {file_path}')

        # Load .mat file
        mat_contents_RL = scipy.io.loadmat(file_path)

        if MYLOAD == 'RL':
            Omega = np.fft.ifft2(mat_contents_RL['w_hat']).real
            Psi = np.fft.ifft2(mat_contents_RL['psi_hat']).real
            psi_hat = mat_contents_RL['psi_hat']
        else:
            Omega = mat_contents_RL['Omega']
            Psi = Omega2Psi_2DFHIT(Omega, invKsq=invKsq)

        if MYLOAD == 'RL':
            ve = (mat_contents_RL['veRL']) # abs removed!
    #         # cs = np.abs(mat_contents_RL['veRL']) # abs removed!
    #         # cs = (mat_contents_RL['veRL']) # abs removed!

    #         S1 = np.real(np.fft.ifft2(-Ky*Kx*psi_hat)) # make sure .*
    #         S2 = 0.5*np.real(np.fft.ifft2(-(Kx*Kx - Ky*Ky)*psi_hat))
    #         S  = 2.0*(S1*S1 + S2*S2)**0.5
    # #        cs = (0.17 * 2*np.pi/NX )**2  # for LX = 2 pi
    #         S = (np.mean(S**2.0))**0.5;
    #         ve = cs*S
    #         print( 'min(ν_e)', np.min(ve) )


        U_hat, V_hat = Psi2UV_2DFHIT_spectral(Psi_hat, Kx, Ky)
        U = np.fft.ifft2(U_hat)
        V = np.fft.ifft2(V_hat)

        Tau11, Tau12, Tau22 = Tau_eddy_viscosity(ve, Psi_hat, Kx, Ky)

        Tau11_hat = np.fft.fft2(Tau11)
        Tau12_hat = np.fft.fft2(Tau12)
        Tau22_hat = np.fft.fft2(Tau22)

        PTau = energyTransfer_2DFHIT(U, V, Tau11, Tau12, Tau22, Kx, Ky)


        Omega_hat = np.fft.ifft2(Omega)
        Sigma1, Sigma2 = Sigma_eddy_viscosity(ve, Omega_hat, Kx, Ky)


        PZ = enstrophyTransfer_2D_FHIT(Omega, Sigma1, Sigma2, Kx, Ky)


        cs_M = np.append(cs_M, cs)
        ve_M = np.append(ve_M, ve)

        # PTau_M = np.append(PTau_M, np.mean(PTau))
        # PZ_M = np.append(PZ_M, np.mean(PZ))
        PTau_M = np.append(PTau_M, (PTau))
        PZ_M = np.append(PZ_M, (PZ))

        filecount += 1
        if filecount > NSAMPLE: break

#%%
BANDWIDTH = mybandwidth_scott(PTau_M)

Vecpoints, exp_log_kde, log_kde, kde = myKDE(PTau_M/np.std(PTau_M), BANDWIDTH=BANDWIDTH )

plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.semilogy(Vecpoints, exp_log_kde, ':k', alpha=0.75, linewidth=4, label=METHOD_RL)
# plt.semilogy(bb[:,0], bb[:,1], '--r', alpha=0.75, linewidth=2, label='RL')

plt.semilogy(aa[:,0], aa[:,1], '--b', alpha=0.75, linewidth=2, label='FDNS')

plt.semilogy([0,0],[0,1],'r')
plt.ylim([1e-3,1e0])
plt.xlim([-5,5])
plt.xlabel(r'$T_E$')
plt.ylabel(r'$\mathcal{P}\left(T_E\right)$')
plt.tight_layout()
plt.legend(loc='upper left')
#%%
BANDWIDTH = mybandwidth_scott(PZ_M)

Vecpoints, exp_log_kde, log_kde, kde = myKDE(PZ_M/np.std(PZ_M),BANDWIDTH=BANDWIDTH)
plt.subplot(2,2,2)

plt.semilogy(Vecpoints, exp_log_kde, ':k', alpha=0.75, linewidth=2, label=METHOD_RL)

plt.ylim([1e-3,1e0])
plt.xlim([-5,5])
plt.xlabel(r'$T_Z$')
plt.ylabel(r'$\mathcal{P}\left(T_Z\right)$')
plt.tight_layout()


BANDWIDTH = mybandwidth_scott(cs_M)

Vecpoints, exp_log_kde, log_kde, kde = myKDE(cs_M/np.std(cs_M),BANDWIDTH=BANDWIDTH)
plt.subplot(2,2,3)

plt.semilogy(Vecpoints, exp_log_kde, ':k', alpha=0.75, linewidth=2, label=METHOD_RL)

plt.ylim([1e-3,1e0])
plt.xlim([-5,5])
plt.xlabel(r'$c_s$')
plt.ylabel(r'$\mathcal{P}\left(c_s\right)$')
plt.tight_layout()


BANDWIDTH = mybandwidth_scott(ve_M)

Vecpoints, exp_log_kde, log_kde, kde = myKDE(ve_M/np.std(ve_M),BANDWIDTH=BANDWIDTH)
plt.subplot(2,2,4)

plt.semilogy(Vecpoints, exp_log_kde, ':k', alpha=0.75, linewidth=2, label=METHOD_RL)

plt.ylim([1e-3,1e0])
plt.xlim([-5,5])
plt.xlabel(r'$\nu_e$')
plt.ylabel(r'$\mathcal{P}\left(\nu_e\right)$')
plt.tight_layout()
