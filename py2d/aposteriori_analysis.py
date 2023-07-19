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


