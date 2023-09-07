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
#%% Case select
METHOD = 'DLEITH' #LEITH, DLEITH , DLEITH_sigma_Local, DLEITH_tau_Local,
# METHOD = 'DSMAG' #SMAG, DSMAG , DSMAG_sigma_Local, DSMAG_tau_Local,
# METHOD = 'NoSGS'

# case = str(sys.argv[1])
# percent_data = float(sys.argv[2])
CASENO = 2
NX = 64
# CASENO = 2
# NX = 128

NUM_DATA_Classic = 2_00
NUM_DATA_RL =1_000
if CASENO == 2:
    NUM_DATA_Classic = 2_00
    # NUM_DATA_RL =3_00
    NUM_DATA_RL =1000 #0
    # NUM_DATA_RL =5000 #0
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
    mat_contents = scipy.io.loadmat('results/Re20000_fkx4fky4_r0.1_b20.0/Re20kNX1024nx4ny4r0p1b20_aposteriori_data.mat')

    # patthadd = '/media/rmojgani/hdd/PostDoc/Projects/py2d_local/results/Re20000_fkx4fky4_r0.1_b20.0'
    # mat_contents = scipy.io.loadmat('results/Re20000_fkx4fky4_r0.1_b20.0/Re20kNX1024nx4ny4r0p1b20_1024_0.1_aposteriori_data.mat')
elif CASENO==10:
      patthadd = '/media/rmojgani/hdd/PostDoc/Projects/py2d_local/'
      mat_contents = scipy.io.loadmat('results/Re100kNX2048nx4ny0r0p1/Re100kNX2048nx4ny0r0p1_aposteriori_data.mat')


energy_spectra_DNS = mat_contents['energy_spectra'].reshape(-1,)
enstrophy_spectra_DNS = mat_contents['enstrophy_spectra'].reshape(-1,)
# wavenumbers_spectra_DNS = mat_contents['wavenumbers_spectra'].reshape(-1,)

pdf_DNS = np.array([mat_contents['Omega_bins_scipy'][0,:],mat_contents['Omega_pdf_scipy'][0,:]]).T
std_omega_DNS =  mat_contents['Omega_std'][0][0]

wavenumbers_spectra_DNS = np.arange(mat_contents['energy_spectra'].shape[1])


#%% Load Classic results


# dataType = 'Re' + str(int(Re/1000)) + 'kNX' + str(NX) + 'nx' + str(fkx) + 'ny' + str(fky) + 'r0p1'
if CASENO==1 or CASENO==10:
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
from mypathdictionary import mypathdictionary
energy_spectra_RL_list, enstrophy_spectra_RL_list = [], []
Omega_list = []

path_RL = mypathdictionary(CASENO, NX, METHOD)

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
plt.ylim([1e-8,1e1])

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
plt.show()
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

