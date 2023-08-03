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

# Re=20e3
# fkx=25
# fky=25
# alpha=0.1
# NX=1024

# case = str(sys.argv[1])
# percent_data = float(sys.argv[2])

case = 'K1'
percent_data = 0.1

if case == 'K1':
    Re = 20000
    fkx = 4
    fky = 0
    NX = 1024
elif case == 'K2':
    Re = 20000
    fkx = 25
    fky = 25
    NX = 1024
elif case == 'K3':
    Re = 100000
    fkx = 4
    fky = 0
    NX = 2048
elif case == 'K4':
    Re = 100000
    fkx = 25
    fky = 25
    NX = 2048

NX = 64
# dataType = 'Re' + str(int(Re/1000)) + 'kNX' + str(NX) + 'nx' + str(fkx) + 'ny' + str(fky) + 'r0p1'
# dataType = 'results/Re20000_fkx4fky4_r0.1_b0/DSMAG/NX64/dt0.0005_IC1/data'
# dataType = 'results/Re20000_fkx4fky4_r0.1_b0/DSMAG_sigma_Local/NX64/dt0.0005_IC1/data'
dataType = 'results/Re20000_fkx4fky4_r0.1_b0/DSMAG_tau_Local/NX64/dt0.0005_IC1/data'

# DATA_DIR = dataType + '/train1/'

# 10% of total number of files np.rint(0.1*last_file_number_data)
# 100 th file

nx, ny = NX,NX
Lx,Ly = 2*np.pi,2*np.pi
Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly)

if Re == 20e3:
    fileNo = 6
elif Re == 100e3:
    fileNo = 11

# select % of data for pdf calculation
# percent_data = 0.02
# Compute the 2% of total elements in the matrix
num_samples = int(percent_data * NX * NX)

# Initialize lists to store data
energy_list, enstrophy_list, time_list = [], [], []
energy_spectra_list, enstrophy_spectra_list, Omega_list = [], [], []
eddy_turnover_time_list = []

if case == 'K1':
    ICArr = ['train1', 'train2', 'train3', 'train4', 'test1']
elif case == 'K2':
    ICArr = ['train1']
elif case == 'K3':
    ICArr = ['train1', 'train2', 'test1', 'test2', 'test3']
elif case == 'K4':
    ICArr = ['train1', 'train2', 'test1', 'test2', 'test3']

for ICname in ICArr:
    # if ICname == 'train1' or ICname == 'train2':
    #     fileNo = 16
    # elif ICname == 'test1' or ICname == 'test2' or ICname == 'test3':
    #     fileNo = 9

    DATA_DIR = dataType #+ '/DNS/' + ICname + '/'
    print(DATA_DIR)

    directory = DATA_DIR  # replace with your directory path

    # Get a sorted list of all files
    files = sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0]))
    files = ['1.mat', '2.mat']

    filecount = 0
    # Iterate over sorted files
    for file in os.listdir(directory):
        # Check if file ends with .mat
        if file.endswith('.mat'):
            file_path = os.path.join(directory, file)
            # Load .mat file
            mat = scipy.io.loadmat(file_path)
            print(f'Loaded file {file_path}')

            Omega = mat['Omega']
            Psi = Omega2Psi_2DFHIT(Omega, invKsq)

            filecount += 1

            # Now 'mat' is a dict with variable names as keys, and loaded matrices as values
            # You can process these matrices using 'mat' dict
            if filecount>100: break
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
bin_min = max(Omega_mean - 10*Omega_std, np.min(Omega_arr))
bin_max = min(Omega_mean + 10*Omega_std, np.max(Omega_arr))
bins = np.linspace(bin_min, bin_max, 1000)

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

# FastKDE
try:
    pdf_fastkde, bins_fastkde = fastKDE.pdf(Omega_arr)
except Exception as e:
    print(f"FastKDE failed with error: {str(e)}")
    pdf_fastkde = np.empty(1)
    bins_fastkde = np.empty(1)

# Scipy
try:
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

# StatsModels
try:
    kde_sm = sm.nonparametric.KDEUnivariate(Omega_arr)
    kde_sm.fit(bw='scott')
    bins_sm = bins
    pdf_sm = np.interp(bins_sm, kde_sm.support, kde_sm.density)
except Exception as e:
    print(f"StatsModels failed error: {str(e)}")
    pdf_sm = np.empty(1)
    bins_sm = np.empty(1)

try:
    # Scikit-learn
    # Scikit-learn doesn't have 'scott' use scipy to calculate the bandwidth
    kde_skl = KernelDensity(kernel='gaussian',bandwidth=bw_scott)
    kde_skl.fit(Omega_arr[:, np.newaxis])
    log_pdf_skl = kde_skl.score_samples(bins[:, np.newaxis])
    pdf_skl = np.exp(log_pdf_skl)
    bins_skl = bins
    
except Exception as e:
    print(f"Scikit-learn failed with error: {str(e)}")

output_filename = dataType + '_' + str(percent_data) + '_aposteriori_dataRM.mat'
savemat(output_filename, {
    'wavenumbers': wavenumbers,
    'energy_spectra': energy_spectra_mean,
    'enstrophy_spectra': enstrophy_spectra_mean,
    'energy': energy_arr,
    'enstrophy': enstrophy_arr,
    'eddyTurnoverTime': eddy_turnover_time_arr,
    'Omega_mean': Omega_mean,
    'Omega_std': Omega_std,
    'Omega_pdf_fastkde': pdf_fastkde,
    'Omega_bins_fastkde': bins_fastkde,
    'Omega_pdf_scipy': pdf_scipy,
    'Omega_bins_scipy': bins_scipy,
    'Omega_pdf_skl': pdf_skl,
    'Omega_bins_skl': bins_skl,
    'Omega_pdf_sm': pdf_sm,
    'Omega_bins_sm': bins_sm,
})
#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5),dpi=150)
plt.subplot(2,1,1)
plt.title(dataType)
plt.loglog(energy_spectra_mean)
plt.ylim([1e-6,1e0])
plt.subplot(2,1,1)
plt.title(dataType)
plt.loglog(enstrophy_spectra_mean)
plt.ylim([1e-6,1e0])

plt.figure(figsize=(5,5),dpi=150)
plt.semilogy(Omega_std,)

    'Omega_std': ,
    'Omega_pdf_fastkde': pdf_fastkde,
    'Omega_bins_fastkde': bins_fastkde,
    'Omega_pdf_scipy': pdf_scipy,
    'Omega_bins_scipy': bins_scipy,
    'Omega_pdf_skl': pdf_skl,
    'Omega_bins_skl': bins_skl,
    'Omega_pdf_sm': pdf_sm,
    'Omega_bins_sm': bins_sm,