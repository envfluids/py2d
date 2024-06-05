import numpy as np
from py2d.derivative import derivative
from py2d.convert import Tau2AnisotropicTau, strain_rate
from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.dealias import multiply_dealias
from py2d.util import eig_vec_2D

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

def angle_Tau_strainRate(Tau11, Tau12, Tau22, Psi, anisotropic=False, dealias=False):
    """
    Calculate the angle between the stress tensor and the strain rate tensor.

    This function takes the components of the stress tensor and the strain rate tensor and computes the angle between them.

    Parameters:
    Tau11 (np.ndarray): A 2D array representing the Tau11 component of the stress tensor.
    Tau12 (np.ndarray): A 2D array representing the Tau12 component of the stress tensor.
    Tau22 (np.ndarray): A 2D array representing the Tau22 component of the stress tensor.
    Psi (np.ndarray): A 2D array representing the Psi component of the strain rate tensor.
    anisotropic (bool): A boolean flag to indicate whether the material is anisotropic.

    Returns:
    np.ndarray: A 2D array containing the angle between the stress tensor and the strain rate tensor at each point.
    """

    if anisotropic:
        Tauu11r = Tau11
        Tau12r = Tau12
        Tau22r = Tau22
    else:
        Tauu11r, Tau12r, Tau22r = Tau2AnisotropicTau(Tau11, Tau12, Tau22)

    # Calculate the components of the strain rate tensor
    nx, ny = Tau11.shape
    Lx, Ly = 2*np.pi, 2*np.pi
    Kx, Ky, _, _, _ = initialize_wavenumbers_rfft2(nx, ny, Lx, Ly)

    S11, S12, S22 = strain_rate(Psi, Kx, Ky)

    # Calculate the angle between the stress tensor and the strain rate tensor
    Tau_S = multiply_dealias(Tauu11r, S11, dealias=dealias) + 2*multiply_dealias(
        Tau12r, S12, dealias=dealias) + multiply_dealias(Tau22r, S22, dealias=dealias)
    Tau_Tau = multiply_dealias(Tauu11r, Tauu11r, dealias=dealias) + 2*multiply_dealias(
        Tau12r, Tau12r, dealias=dealias) + multiply_dealias(Tau22r, Tau22r, dealias=dealias)
    S_S = multiply_dealias(S11, S11, dealias=dealias) + 2*multiply_dealias(
        S12, S12, dealias=dealias) + multiply_dealias(S22, S22, dealias=dealias)
    
    angle = np.arccos(Tau_S/(np.sqrt(Tau_Tau)*np.sqrt(S_S)))

    # Calculate the angle between eigenvectors
    S_eigVec1, S_eigVec2, _, _ = eig_vec_2D(S11, S12, S12, S22)
    Tau_eigVec1, Tau_eigVec2, _, _ = eig_vec_2D(Tauu11r, Tau12r, Tau12r, Tau22r)

    # Dot product between eigenvectors
    Tau_S_eigVec1 = np.sum(Tau_eigVec1*S_eigVec1, axis=1)
    Tau_S_eigVec2 = np.sum(Tau_eigVec2*S_eigVec2, axis=1)

    Tau_Tau_eigVec1 = np.sum(Tau_eigVec1*Tau_eigVec1, axis=1)
    Tau_Tau_eigVec2 = np.sum(Tau_eigVec2*Tau_eigVec2, axis=1)

    S_S_eigVec1 = np.sum(S_eigVec1*S_eigVec1, axis=1)
    S_S_eigVec2 = np.sum(S_eigVec2*S_eigVec2, axis=1)

    angle_eigVec1 = np.arccos(Tau_S_eigVec1/(np.sqrt(Tau_Tau_eigVec1)*np.sqrt(S_S_eigVec1)))
    angle_eigVec2 = np.arccos(Tau_S_eigVec2/(np.sqrt(Tau_Tau_eigVec2)*np.sqrt(S_S_eigVec2)))

    return angle, angle_eigVec1, angle_eigVec2

