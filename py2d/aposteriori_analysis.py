# ----------------------------------------------------------------------
# Created : Karan Jakhar May 2023
# ----------------------------------------------------------------------

import numpy as np

from py2d.convert import strain_rate_2DFHIT
from py2d.derivative import derivative_2DFHIT

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

def energyDissipationRate(Psi, Re, Kx, Ky, spectral=False):
    """
    Compute energy dissipation rate for 2D_FHIT using Omega.
    
    Args:
    A (ndarray): 2D array of Omega U.
    Re (float): Reynolds number.
    
    Returns:
    float: Energy dissipation rate.

    Note:
    The energy dissipation rate is computed in Eq.16 of [1].
    [1] Buaria, D., & Sreenivasan, K. R. (2023). 
        Forecasting small-scale dynamics of fluid turbulence using deep neural networks. 
        Proceedings of the National Academy of Sciences, 120(30), e2305765120.
        https://www.pnas.org/doi/abs/10.1073/pnas.2305765120
    """
    
    # Compute strain rate tensor
    if spectral:
        S11_hat, S12_hat, S22_hat = strain_rate_2DFHIT(Psi, Kx, Ky, spectral=True)
        S11, S12, S22 = np.fft.ifft2(S11_hat).real, np.fft.ifft2(S12_hat).real, np.fft.ifft2(S22_hat).real
    else:
        S11, S12, S22 = strain_rate_2DFHIT(Psi, Kx, Ky, spectral=False)

    # Compute energy dissipation rate
    energyDissipationRate = (2/Re) * np.mean(S11 ** 2 + 2 * S12 ** 2 + S22 ** 2)

    return energyDissipationRate


def enstrophyDissipationRate(Omega, Re, Kx, Ky, spectral=False):
    """
    Compute enstrophy dissipation rate for 2D_FHIT using Omega.
    
    Args:
    A (ndarray): 2D array of Omega U.
    Re (float): Reynolds number.
    
    Returns:
    float: Enstrophy dissipation rate.

    Note:
    The energy dissipation rate is computed in Eq.11 of [1].
    [1] Alexakis, A., & Doering, C. R. (2006). 
    Energy and enstrophy dissipation in steady state 2d turbulence. 
    Physics letters A, 359(6), 652-657.
    https://doi.org/10.1016/j.physleta.2006.07.048
    """
    
    # Compute vorticity gradient
    if spectral:
        Omega_hat = np.fft.ifft2(Omega).real
        Omegax_hat = derivative_2DFHIT(Omega_hat, [1,0], Kx=Kx, Ky=Ky, spectral=True)
        Omegay_hat = derivative_2DFHIT(Omega_hat, [0,1], Kx=Kx, Ky=Ky, spectral=True)
        Omegax, Omegay = np.fft.ifft2(Omegax_hat).real, np.fft.ifft2(Omegay_hat).real

    else:
        Omegax = derivative_2DFHIT(Omega, [1,0], Kx=Kx, Ky=Ky, spectral=False)
        Omegay = derivative_2DFHIT(Omega, [0,1], Kx=Kx, Ky=Ky, spectral=False)

    # Compute enstrophy dissipation rate
    enstrophyDissipationRate = (1/Re) * np.mean(Omegax ** 2 + Omegay ** 2)

    return enstrophyDissipationRate