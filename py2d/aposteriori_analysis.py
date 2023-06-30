# ----------------------------------------------------------------------
# Created : Karan Jakhar May 2023
# ----------------------------------------------------------------------

import numpy as nnp
import jax.numpy as np
from jax import jit 

def eddyTurnoverTime_2DFHIT(Omega):
    """
    Compute eddy turnover time for 2D_FHIT using Omega.
    
    Args:
    A (ndarray): 2D array of Omega U.
    definition (str): Optional string to define eddy turnover time. Default is 'Enstrophy'.
                      Possible values: 'Enstrophy', 'Omega', 'Velocity'
                      
    Returns:
    float: Eddy turnover time.
    """
    eddyTurnoverTime = 1 / np.sqrt(np.mean(Omega ** 2))
    return eddyTurnoverTime


def energy_2DFHIT(Psi, Omega, spectral = False):
    '''Calculates energy as the mean of 0.5 * Psi * Omega.

    Args:
        Psi (np.ndarray): The stream function matrix.
        Omega (np.ndarray): The vorticity matrix.
        spectral (bool): Whether to perform inverse Fast Fourier Transform on Psi and Omega. Default is False.

    Returns:
        float: The calculated energy.
    '''
    if spectral:
        Psi = np.fft.ifft2(Psi).real
        Omega = np.fft.ifft2(Omega).real
        
    energy = np.mean(0.5 * Psi * Omega)
    return energy


def enstrophy_2DFHIT(Omega, spectral = False):
    '''Calculates enstrophy as the mean of 0.5 * Omega * Omega.

    Args:
        Omega (np.ndarray): The vorticity matrix.
        spectral (bool): Whether to perform inverse Fast Fourier Transform on Omega. Default is False.

    Returns:
        float: The calculated enstrophy.
    '''
    if spectral:
        Omega = np.fft.ifft2(Omega).real
        
    enstrophy = np.mean(0.5 * Omega * Omega)
    return enstrophy


def spectrum_angled_average_2DFHIT(A, spectral = False):
    '''
    Compute the radially/angle-averaged spectrum of a 2D square matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The input 2D square matrix. If `spectral` is False, `A` is in the physical domain; otherwise, it is in the spectral domain.
    spectral : bool, optional
        Whether `A` is in the spectral domain. Default is False.

    Returns
    -------
    spectrum : numpy.ndarray
        The radially/angle-averaged spectrum of `A`.
    wavenumbers : numpy.ndarray
        The corresponding wavenumbers.

    Raises
    ------
    ValueError
        If `A` is not a 2D square matrix or `spectral` is not a boolean.
    '''
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('Input is not a 2D square matrix. Please input a 2D square matrix')
    if not isinstance(spectral, bool):
        raise ValueError('Invalid input for spectral. It should be a boolean value')
    if not np.issubdtype(A.dtype, np.number):
        raise ValueError('Input contains non-numeric values')
        
    nx = A.shape[0]
    L = 2 * np.pi
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=L/nx)
    ky = 2 * np.pi * np.fft.fftfreq(nx, d=L/nx)
    (wavenumber_x, wavenumber_y) = np.meshgrid(kx, ky, indexing='ij')
    absolute_wavenumber = np.sqrt(wavenumber_x ** 2 + wavenumber_y ** 2)
    absolute_wavenumber = np.fft.fftshift(absolute_wavenumber)
    
    if not spectral:
        spectral_A = np.fft.fft2(A)
    else:
        spectral_A = A
    spectral_A = np.abs(spectral_A) / nx ** 2
    spectral_A = np.fft.fftshift(spectral_A)
    bin_edges = np.arange(-0.5, nx / 2 + 0.5)
    binnumber = np.digitize(absolute_wavenumber.ravel(), bins=bin_edges)
    spectrum = np.bincount(binnumber, weights=spectral_A.ravel())[1:]
    wavenumbers = np.arange(0, nx // 2 + 1)
    return spectrum, wavenumbers


def TKE_angled_average_2DFHIT(Psi_hat, Omega_hat, spectral = False):
    '''
    Calculate the energy spectrum and its angled average.

    Parameters
    ----------
    Psi_hat : numpy.ndarray
        The input 2D square matrix of stream function. If `spectral` is False, `Psi_hat` is in the physical domain; otherwise, it is in the spectral domain.
    Omega_hat : numpy.ndarray
        The input 2D square matrix of vorticity. If `spectral` is False, `Omega_hat` is in the physical domain; otherwise, it is in the spectral domain.
    spectral : bool, optional
        Whether `Psi_hat` and `Omega_hat` are in the spectral domain. Default is False.

    Returns
    -------
    TKE_angled_average : numpy.ndarray
        The radially/angle-averaged energy spectrum of `E_hat`.
    kkx : numpy.ndarray
        The corresponding wavenumbers.
    '''
    NX = Psi_hat.shape[0]
    
    if not spectral:
        Psi_hat = np.fft.fft2(Psi_hat)
        Omega_hat = np.fft.fft2(Omega_hat)
        
    E_hat = 0.5 * np.abs(np.conj(Psi_hat) * Omega_hat) / NX ** 2
    TKE_angled_average, kkx = spectrum_angled_average_2DFHIT(E_hat, spectral=True)
    return TKE_angled_average, kkx

def enstrophy_angled_average_2DFHIT(Omega_hat, spectral = False):
    '''
    Calculate the enstrophy spectrum and its angled average.

    Parameters
    ----------
    Omega_hat : numpy.ndarray
        The input 2D square matrix of vorticity. If `spectral` is False, `Omega_hat` is in the physical domain; otherwise, it is in the spectral domain.
    spectral : bool, optional
        Whether `Omega_hat` is in the spectral domain. Default is False.

    Returns
    -------
    Z_angled_average_spectra : numpy.ndarray
        The radially/angle-averaged enstrophy spectrum of `Z_hat`.
    kkx : numpy.ndarray
        The corresponding wavenumbers.
    '''
    NX = Omega_hat.shape[0]
    if not spectral:
        Omega_hat = np.fft.fft2(Omega_hat)
        
    Z_hat = 0.5 * np.abs(np.conj(Omega_hat) * Omega_hat) / NX ** 2
    Z_angled_average_spectra, kkx = spectrum_angled_average_2DFHIT(Z_hat, spectral=True)
    return Z_angled_average_spectra, kkx

