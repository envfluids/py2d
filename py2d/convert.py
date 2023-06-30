# ----------------------------------------------------------------------
# Created : Karan Jakhar May 2023
# ----------------------------------------------------------------------

import numpy as nnp
import jax.numpy as np
from jax import jit
from functools import partial
import glob
import os

def eddyTurnoverTime_2DFHIT(Omega, definition='Enstrophy'):
    """
    Compute eddy turnover time for 2D_FHIT using Omega.
    
    Args:
    A (ndarray): 2D array of Omega U.
    definition (str): Optional string to define eddy turnover time. Default is 'Enstrophy'.
                      Possible values: 'Enstrophy', 'Omega', 'Velocity'
                      
    Returns:
    float: Eddy turnover time.
    """

    eddyTurnoverTime = 1/np.sqrt(np.mean(Omega**2))
    
    return eddyTurnoverTime

def get_last_file(file_path):
    # Get all .mat files in the specified directory
    mat_files = glob.glob(os.path.join(file_path, '*.mat'))
    
    # Extract the integer values from the filenames
    file_numbers = [int(os.path.splitext(os.path.basename(file))[0]) for file in mat_files]
    
    # Find the highest integer value
    if file_numbers:
        last_file_number = max(file_numbers)
        return last_file_number
    else:
        return None

def initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly):
    '''
    Initialize the wavenumbers for 2D Forced Homogeneous Isotropic Turbulence (2D-FHIT).
    
    Parameters:
    -----------
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    Lx : float
        Length of the domain in the x-direction.
    Ly : float
        Length of the domain in the y-direction.

    Returns:
    --------
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.
    '''
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)
    (Kx, Ky) = np.meshgrid(kx, ky, indexing='ij')
    Ksq = Kx ** 2 + Ky ** 2
    return Kx, Ky, Ksq

def Omega2Psi_2DFHIT(Omega, Kx, Ky, Ksq):
    """
    Calculate the stream function from vorticity.

    This function calculates the stream function (Psi) from vorticity (Omega) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    both physical and spectral space calculations.

    Parameters:
    -----------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.
    spectral : bool, optional
        If True, assumes input vorticity is in spectral space and returns stream function in
        spectral space. If False (default), assumes input vorticity is in physical space and
        returns stream function in physical space.

    Returns:
    --------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.

    """
    Omega_hat = Omega
    lap_Psi_hat = -Omega_hat
    Psi_hat = lap_Psi_hat / -Ksq
    Psi_hat = Psi_hat.at[(0, 0)].set(0)
    return Psi_hat


def Psi2Omega_2DFHIT(Psi, Kx, Ky, Ksq):
    """
    Calculate the vorticity from the stream function.

    This function calculates the vorticity (Omega) from the stream function (Psi) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    both physical and spectral space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.
    spectral : bool, optional
        If True, assumes input stream function is in spectral space and returns vorticity in
        spectral space. If False (default), assumes input stream function is in physical space and
        returns vorticity in physical space.

    Returns:
    --------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical or spectral space, depending on the 'spectral' flag.

    """
    Psi_hat = Psi
    lap_Psi_hat = -Ksq * Psi_hat
    Omega_hat = -lap_Psi_hat
    return Omega_hat


def Psi2UV_2DFHIT(Psi, Kx, Ky, Ksq):
    """
    Calculate the velocity components U and V from the stream function.

    This function calculates the velocity components U and V from the stream function (Psi)
    using the relationship between them. The function can handle both physical and spectral
    space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    spectral : bool, optional
        If True, assumes input stream function is in spectral space and returns velocity components
        in spectral space. If False (default), assumes input stream function is in physical space and
        returns velocity components in physical space.

    Returns:
    --------
    U, V : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in physical or spectral space, depending on the 'spectral' flag.

    """
    Psi_hat = Psi
    U_hat = (1.j) * Ky * Psi_hat
    V_hat = -(1.j) * Kx * Psi_hat
    return U_hat, V_hat


def Tau2PiOmega_2DFHIT(Tau11, Tau12, Tau22, Kx, Ky, Ksq):
    """
    Calculate PiOmega, the curl of the divergence of Tau, where Tau is a 2D symmetric tensor.

    Parameters:
    -----------
    Tau11 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, depending on the 'spectral' flag.
    Tau12 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, depending on the 'spectral' flag.
    Tau22 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    spectral : bool, optional
        If True, assumes input Tau elements are in spectral space and returns PiOmega in spectral space.
        If False (default), assumes input Tau elements are in physical space and returns PiOmega in physical space.

    Returns:
    --------
    PiOmega : numpy.ndarray
        PiOmega (2D array) in physical or spectral space, depending on the 'spectral' flag.

    """
    Tau11_hat = Tau11
    Tau12_hat = Tau12
    Tau22_hat = Tau22
    PiOmega_hat = Kx * Ky * (Tau11_hat - Tau22_hat) - (Kx * Kx - Ky * Ky) * Tau12_hat
    return PiOmega_hat

@jit
def strain_rate_2DFHIT(Psi, Kx, Ky, Ksq):
    """
    Code Validated
    Author: Karan Jakhar
    Data Created: May 6th 2023
    Last Modified: May 6th 2023
    
    Calculate the Strain rate components S11, S12, and S22 from the stream function.

    This function calculates the velocity components U and V from the stream function (Psi)
    using the relationship between them. The function can handle both physical and spectral
    space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.

    Returns:
    --------
    S11, S12, S22 : tuple of numpy.ndarray
        Strain rate components S11, S12, and S22 (2D arrays) in physical or spectral space, depending on the 'spectral' flag.

    """
    Psi_hat = Psi
    Ux_hat = -(Kx * Ky) * Psi_hat
    Vx_hat = Kx * Kx * Psi_hat
    Uy_hat = -(Ky * Ky) * Psi_hat
    S11_hat = Ux_hat
    S12_hat = 0.5 * (Uy_hat + Vx_hat)
    S22_hat = -Ux_hat
    return S11_hat, S12_hat, S22_hat

def derivative_2D_FHIT(T_hat, order, Kx, Ky):
    """
    Calculate spatial derivatives for 2D_FHIT in spectral space.
    Boundary conditions are periodic in x and y spatial dimensions
    Length of domain 2*pi

    Input:
    T_hat: Input flow field in spectral space: Square Matrix NxN
    order [orderX, orderY]: Array of order of derivatives in x and y spatial dimensions: [Interger (>=0), Integer (>=0)] 
    Kx, Ky: Kx and Ky values calculated beforehand.

    Output:
    Tderivative_hat: derivative of the flow field T in spectral space: Square Matrix NxN
    """

    orderX = order[0]
    orderY = order[1]

    # Calculating derivatives in spectral space
    Tderivative_hat = ((1j*Kx)**orderX) * ((1j*Ky)**orderY) * T_hat

    return Tderivative_hat

#  CNN functions


# def prepare_data_cnn(Psi1_hat, Kx, Ky, Ksq):
#     (U_hat, V_hat) = Psi2UV_2DFHIT(Psi1_hat, Kx, Ky, Ksq)
#     U = np.real(np.fft.ifft2(U_hat))
#     V = np.real(np.fft.ifft2(V_hat))
#     input_data = np.stack((U, V), axis=0)
#     return input_data


# def postproccess_data_cnn(Tau11CNN, Tau12CNN, Tau22CNN, Kx, Ky, Ksq):
#     Tau11CNN_hat = np.fft.fft2(Tau11CNN)
#     Tau12CNN_hat = np.fft.fft2(Tau12CNN)
#     Tau22CNN_hat = np.fft.fft2(Tau22CNN)
#     PiOmega_hat = Tau2PiOmega_2DFHIT(Tau11CNN_hat, Tau12CNN_hat, Tau22CNN_hat, Kx, Ky, Ksq)
#     print(type(PiOmega_hat))
#     return PiOmega_hat

def prepare_data_cnn(Psi1_hat, Kx, Ky, Ksq):
    U_hat, V_hat = Psi2UV_2DFHIT(Psi1_hat, Kx, Ky, Ksq)
    # I am using a Transpose here which I should not. After fixing CNN, this should be removed
    U = np.real(np.fft.ifft2(U_hat)) # This should be removed after I fixed CNN code loading data
    V = np.real(np.fft.ifft2(V_hat))
    input_data = np.stack((U, V), axis=0)
    return input_data

def postproccess_data_cnn(Tau11CNN, Tau12CNN, Tau22CNN, Kx, Ky, Ksq):
    Tau11CNN_hat = np.fft.fft2(Tau11CNN)
    Tau12CNN_hat = np.fft.fft2(Tau12CNN)
    Tau22CNN_hat = np.fft.fft2(Tau22CNN)
    PiOmega_hat = Tau2PiOmega_2DFHIT(Tau11CNN_hat, Tau12CNN_hat, Tau22CNN_hat, Kx, Ky, Ksq)
    return PiOmega_hat

def postproccess_data_cnn_mcwiliams_ani(Tau1, Tau2, Kx, Ky, Ksq):
    Tau1_hat = np.fft.fft2(Tau1)
    Tau2_hat = np.fft.fft2(Tau2)
    PiOmega_hat = Tau2PiOmega_2DFHIT(Tau1_hat, Tau2_hat, -1*Tau1_hat, Kx, Ky, Ksq)
    return PiOmega_hat

def postproccess_data_cnn_PiOmega(PiOmega, Kx, Ky, Ksq):
    PiOmega_hat = np.fft.fft2(PiOmega)
    return PiOmega_hat
        
def preprocess_data_cnn_PsiVor_PiOmega(Psi1_hat, Kx, Ky, Ksq):
    Omega_hat = Psi2Omega_2DFHIT(Psi1_hat, Kx, Ky, Ksq)
    Omega = np.real(np.fft.ifft2(Omega_hat)) # This should be removed after I fixed CNN code loading data
    Psi = np.real(np.fft.ifft2(Psi1_hat))
    input_data = np.stack((Psi, Omega), axis=0)
    return input_data

def normalize_data(data):
    # Calculate mean and standard deviation
    # Input data : (2, NX, NY)
    mean = np.mean(data, axis=(1,2), keepdims=True)
    std = np.std(data, axis=(1,2), keepdims=True)
    
    # Normalize data
    # normalized_data = (data - mean) / (std + np.finfo(np.float32).eps)
    normalized_data = (data - mean) / (std)
    
    return normalized_data

def denormalize_data(data, mean, std):
    # Denormalize data
    denormalized_data = data * std + mean
    
    return denormalized_data