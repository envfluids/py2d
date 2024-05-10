from py2d.derivative import derivative
import numpy as np
from py2d.dealias import multiply_dealias

def energy(Psi, Omega, spectral = False, dealias = False):
    '''Calculates energy as the mean of 0.5 * Psi * Omega.

    Args:
        Psi (np.ndarray): The stream function matrix.
        Omega (np.ndarray): The vorticity matrix.
        spectral (bool): Whether to perform inverse Fast Fourier Transform on Psi and Omega. Default is False.

    Returns:
        float: The calculated energy.
    '''
    N = Psi.shape[0]

    if spectral:
        Psi = np.fft.irfft2(Psi, s=[N, N])
        Omega = np.fft.irfft2(Omega, s=[N, N])

    PsiOmega = multiply_dealias(Psi, Omega, dealias=dealias)
        
    energy = np.mean(0.5 * PsiOmega)
    return energy


def enstrophy(Omega, spectral = False, dealias=False):
    '''Calculates enstrophy as the mean of 0.5 * Omega * Omega.

    Args:
        Omega (np.ndarray): The vorticity matrix.
        spectral (bool): Whether to perform inverse Fast Fourier Transform on Omega. Default is False.

    Returns:
        float: The calculated enstrophy.
    '''
    N = Omega.shape[0]

    if spectral:
        Omega = np.fft.irfft2(Omega, s=[N, N])

    OmegaOmega = multiply_dealias(Omega, Omega, dealias=dealias)
        
    enstrophy = np.mean(0.5 * OmegaOmega)
    return enstrophy


def energyTransfer(U, V, Tau11, Tau12, Tau22, Kx, Ky, dealias = False):
    """
    Energy transfer of 2D_FHIT using SGS stress
    Input is single snapshot (N x N matrix)

    Inputs:
    U,V: Velocities
    Tau11, Tau12, Tau22: SGS stress

    Output: 
    PTau: energy transfer

    Note: Dealias[a,b,c] != Dealias[Dealias[a,b],c] 
    Dealiasing energy transfer directly may not render correct result and you may need to explore how Tau is caluclated.
    Considering Tau is usually composed fo multiplying 2 fields.
    """

    Ux = derivative(U, [1,0], Kx=Kx, Ky=Ky, spectral=False)
    Uy = derivative(U, [0,1], Kx=Kx, Ky=Ky, spectral=False)
    Vx = derivative(V, [1,0], Kx=Kx, Ky=Ky, spectral=False)

    Tau11Ux = multiply_dealias(Tau11, Ux, dealias=dealias)
    Tau22Ux = multiply_dealias(Tau22, Ux, dealias=dealias)
    Tau12Uy = multiply_dealias(Tau12, Uy, dealias=dealias)
    Tau12Vx = multiply_dealias(Tau12, Vx, dealias=dealias)

    PTau = -Tau11Ux + Tau22Ux - Tau12Uy - Tau12Vx

    return PTau

def enstrophyTransfer(Omega, Sigma1, Sigma2, Kx, Ky, dealias = False):
    """
    Enstrophy transfer of 2D_FHIT using SGS vorticity stress

    Inputs:
    Omega: Vorticity
    Sigma1, Sigma2: SGS vorticity stress

    Output: 
    PZ: enstrophy transfer

    Note: Dealias[a,b,c] != Dealias[Dealias[a,b],c] 
    Dealiasing enstrophy transfer directly may not render correct result and you may need to explore how Sigma is caluclated.
    Considering Sigma is usually composed fo multiplying 2 fields.
    """

    Omegax = derivative(Omega, [1,0], Kx=Kx, Ky=Ky, spectral=False)
    Omegay = derivative(Omega, [0,1], Kx=Kx, Ky=Ky, spectral=False)

    Sigma1Omegax = multiply_dealias(Sigma1, Omegax, dealias=dealias)
    Sigma2Omegay = multiply_dealias(Sigma2, Omegay, dealias=dealias)

    PZ = -Sigma1Omegax - Sigma2Omegay

    return PZ

def corr2(a_var,b_var):
    # Correlation coefficient of N x N x T array

    a = a_var - np.mean(a_var)
    b = b_var - np.mean(b_var)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
    
    return r

