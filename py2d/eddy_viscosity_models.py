# ----------------------------------------------------------------------
# Created : Karan Jakhar May 2023
# ----------------------------------------------------------------------

# Eddy Viscosity SGS Models for 2D Turbulence solver


import numpy as nnp
import jax.numpy as np
from jax import jit

from py2d.convert import strain_rate_2DFHIT_spectral

strain_rate_2DFHIT_spectral = jit(strain_rate_2DFHIT_spectral)

@jit
def Tau_eddy_viscosity(eddy_viscosity, Psi_hat, Kx, Ky):
    '''
    Calculate the eddy viscosity term (Tau) in the momentum equation 
    '''
    S11_hat, S12_hat, S22_hat = strain_rate_2DFHIT_spectral(Psi_hat, Kx, Ky)
    S11 = np.real(np.fft.ifft2(S11_hat))
    S12 = np.real(np.fft.ifft2(S12_hat))
    S22 = np.real(np.fft.ifft2(S22_hat))

    Tau11 = -2*eddy_viscosity_smag*S11
    Tau12 = -2*eddy_viscosity_smag*(S12)
    Tau22 = -2*eddy_viscosity_smag*S22

    return Tau11, Tau12, Tau22

@jit
def eddy_viscosity_smag(Cs, Delta, characteristic_S):
    '''
    Smagorinsky Model (SMAG)
    '''
    characteristic_S2 = characteristic_S ** 2
    characteristic_S_mean = np.sqrt(np.mean(characteristic_S2))
    eddy_viscosity = (Cs * Delta) ** 2 * characteristic_S_mean
    return eddy_viscosity

@jit
def characteristic_strain_rate_smag(Psi_hat, Kx, Ky, Ksq):
    '''
    Characteristic strain rate
    Required for Smagorinsky Model
    '''
    (S11_hat, S12_hat, _) = strain_rate_2DFHIT_spectral(Psi_hat, Kx, Ky)
    S11 = np.real(np.fft.ifft2(S11_hat))
    S12 = np.real(np.fft.ifft2(S12_hat))
    characteristic_S = 2 * np.sqrt(S11 ** 2 + S12 ** 2)
    return characteristic_S

@jit
def eddy_viscosity_leith(Cl, Delta, characteristic_Omega):
    '''
    Leith Model (LEITH)
    '''
    characteristic_Omega_mean = np.mean(characteristic_Omega)
    ls = Cl * Delta
    eddy_viscosity = ls ** 3 * characteristic_Omega_mean
    return eddy_viscosity

@jit
def characteristic_omega_leith(Omega_hat, Kx, Ky):
    '''
    Characteristic Gradient of Omega
    Required for Leith Model
    '''
    Omegax_hat = (1.j) * Kx * Omega_hat
    Omegay_hat = (1.j) * Ky * Omega_hat
    Omegax = np.real(np.fft.ifft2(Omegax_hat))
    Omegay = np.real(np.fft.ifft2(Omegay_hat))
    characteristic_Omega = np.sqrt(Omegax ** 2 + Omegay ** 2)
    return characteristic_Omega

@jit
def coefficient_dsmag_PsiOmega(Psi_hat, Omega_hat, characteristic_S, Kx, Ky, Ksq, Delta):
    '''
    cs = Cs**2
    Dynamic Coefficient for Dynamic Smagorinsky Model (DSMAG)
    '''
    (Psif_hat, Omegaf_hat, Omega_lap, Omegaf_lap, Delta_test, nx_test) = initialize_filtered_variables_PsiOmega(
        Psi_hat, Omega_hat, Ksq, Delta)
    L = residual_jacobian_PsiOmega(Psi_hat, Omega_hat, Psif_hat, Omegaf_hat, Kx, Ky, nx_test)
    M = residual_dsmag_PsiOmega(Omega_lap, Omegaf_lap, characteristic_S, Delta, Delta_test, nx_test)
    cs = coefficient_dynamic_PsiOmega(L, M)
    return cs

@jit
def coefficient_dleith_PsiOmega(Psi_hat, Omega_hat, characteristic_Omega, Kx, Ky, Ksq, Delta):
    '''
    cl = Cl**3
    Dynamic Coefficient for Dynamic Leith Model (DLEITH)
    '''
    (Psif_hat, Omegaf_hat, Omega_lap, Omegaf_lap, Delta_test, nx_test) = initialize_filtered_variables_PsiOmega(
        Psi_hat, Omega_hat, Ksq, Delta)
    L = residual_jacobian_PsiOmega(Psi_hat, Omega_hat, Psif_hat, Omegaf_hat, Kx, Ky, nx_test)
    M = residual_dleith_PsiOmega(Omega_lap, Omegaf_lap, characteristic_Omega, Delta, Delta_test, nx_test)
    cl = coefficient_dynamic_PsiOmega(L, M)
    return cl

@jit
def coefficient_dleithlocal_PsiOmega(Psi_hat, Omega_hat, characteristic_Omega, Kx, Ky, Ksq, Delta):
    '''
    cl = Cl**3
    Dynamic Coefficient for Dynamic Leith Model (DLEITH)
    '''
    (Psif_hat, Omegaf_hat, Omega_lap, Omegaf_lap, Delta_test, nx_test) = initialize_filtered_variables_PsiOmega(
        Psi_hat, Omega_hat, Ksq, Delta)
    L = residual_jacobian_PsiOmega(Psi_hat, Omega_hat, Psif_hat, Omegaf_hat, Kx, Ky, nx_test)
    M = residual_dleith_PsiOmega(Omega_lap, Omegaf_lap, characteristic_Omega, Delta, Delta_test, nx_test)
    cl = coefficient_dynamiclocal_PsiOmega(L, M)
    return cl

@jit
def coefficient_dynamic_PsiOmega(L, M):
    '''
    Dynamic Coefficient
    Required for DSMAG and DLEITH model
    '''
    LM = L * M
    MM = M * M
    LM_pos = 0.5 * (LM + np.abs(LM))

    c_dynamic = np.mean(LM_pos) / np.mean(MM)
    return c_dynamic


@jit
def coefficient_dynamiclocal_PsiOmega(L, M):
    '''
    Dynamic Coefficient
    Required for DSMAG and DLEITH model
    Attemps for: LOCAL
    '''
    LM = L * M
    MM = M * M
    LM_pos = 0.5 * (LM + np.abs(LM))

    #c_dynamic = np.mean(LM_pos) / np.mean(MM)
    c_dynamic = (LM_pos) / np.mean(MM)
    return c_dynamic

@jit
def initialize_filtered_variables_PsiOmega(Psi_hat, Omega_hat, Ksq, Delta):
    '''
    Calculate filtered flow variables for DSMAG and DLEITH model - Using Psi and Omega
    '''
    nx = Psi_hat.shape[0]
    nx_test = nx // 2
    Delta_test = 2 * Delta
    Psif_hat = spectral_filter_square_same_size_2DFHIT(Psi_hat, nx_test)
    Omegaf_hat = spectral_filter_square_same_size_2DFHIT(Omega_hat, nx_test)
    Omega_lap = np.real(np.fft.ifft2(-Ksq * Omega_hat))
    Omegaf_lap = np.real(np.fft.ifft2(-Ksq * Omegaf_hat))
    return Psif_hat, Omegaf_hat, Omega_lap, Omegaf_lap, Delta_test, nx_test

@jit
def residual_jacobian_PsiOmega(Psi_hat, Omega_hat, Psif_hat, Omegaf_hat, Kx, Ky, nx_test):
    '''
    Residual of Jacobian
    Difference between Filtered Jacobian and Jacobian of Filtered flow variables
    '''
    J1 = jacobian_Spectral2Physical(Omega_hat, Psi_hat, Kx, Ky)
    J1f_hat = spectral_filter_square_same_size_2DFHIT(np.fft.fft2(J1), nx_test)
    J1f = np.real(np.fft.ifft2(J1f_hat))
    J2f = jacobian_Spectral2Physical(Omegaf_hat, Psif_hat, Kx, Ky)
    L = J1f - J2f
    return L

@jit
def residual_dsmag_PsiOmega(Omega_lap, Omegaf_lap, characteristic_S, Delta, Delta_test, nx_test):
    '''
    Residual of SMAG
    Difference between Filtered SMAG and SMAG of Filtered flow variables
    Required for DSMAG
    '''
    M1_hat = spectral_filter_square_same_size_2DFHIT(np.fft.fft2(characteristic_S * Omega_lap), nx_test)
    M1 = Delta ** 2 * np.real(np.fft.ifft2(M1_hat))
    characteristic_Sf_hat = spectral_filter_square_same_size_2DFHIT(np.fft.fft2(characteristic_S), nx_test)
    characteristic_Sf = np.real(np.fft.ifft2(characteristic_Sf_hat))
    M2 = Delta_test ** 2 * characteristic_Sf * Omegaf_lap
    M = M1 - M2
    return M

@jit
def residual_dleith_PsiOmega(Omega_lap, Omegaf_lap, characteristic_Omega, Delta, Delta_test, nx_test):
    '''
    Residual of LEITH
    Difference between Filtered LEITH and LEITH of Filtered flow variables
    Required for DLEITH
    '''
    M1_hat = spectral_filter_square_same_size_2DFHIT(np.fft.fft2(characteristic_Omega * Omega_lap), nx_test)
    M1 = Delta ** 3 * np.real(np.fft.ifft2(M1_hat))
    characteristic_Omegaf_hat = spectral_filter_square_same_size_2DFHIT(np.fft.fft2(characteristic_Omega), nx_test)
    characteristic_Omegaf = np.real(np.fft.ifft2(characteristic_Omegaf_hat))
    M2 = Delta_test ** 3 * characteristic_Omegaf * Omegaf_lap
    M = M1 - M2
    return M

@jit
def jacobian_Spectral2Physical(a_hat, b_hat, Kx, Ky):
    """
    Calculate the Jacobian (in physical space) of two scalar fields  (spectral space) a and b for 2DFHIT.

    Parameters:
    -----------
    a_hat : numpy.ndarray
        First scalar field (2D array) in spectral space, depending on the 'spectral' flag.
    b_hat : numpy.ndarray
        Second scalar field (2D array) in spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.

    Returns:
    --------
    J : numpy.ndarray
        Jacobian of scalar fields a and b (2D array) in physical space.
    """
    ax_hat = (1.j)*Kx * a_hat
    ay_hat = (1.j)*Ky * a_hat
    bx_hat = (1.j)*Kx * b_hat
    by_hat = (1.j)*Ky * b_hat
    ax = np.real(np.fft.ifft2(ax_hat))
    ay = np.real(np.fft.ifft2(ay_hat))
    bx = np.real(np.fft.ifft2(bx_hat))
    by = np.real(np.fft.ifft2(by_hat))
    J = ax * by - ay * bx
    return J

@jit
def spectral_filter_square_same_size_2DFHIT(q_hat, N_LES):
    '''
    A sharp spectral filter for 2D flow variables. The function takes a 2D square matrix and a cutoff
    frequency, performs a FFT, applies a filter in the frequency domain, and then performs an inverse FFT
    to return the filtered data.

    Parameters:
    q (numpy.ndarray): The input 2D square matrix.
    N_LES (int): The cutoff frequency.

    Returns:
    numpy.ndarray: The filtered data. The data is in the frequency domain.
    '''
    kc = N_LES / 2
    (nx, ny) = q_hat.shape
    Lx = 2 * np.pi
    Ly = 2 * np.pi

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Lx/nx)
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    k = np.sqrt(Kx ** 2 + Ky ** 2)

    q_filtered_hat = np.where(k < kc, q_hat, 0)
    return q_filtered_hat

# @jit
# def coefficient_dsmag(
#     kappa = 2;
#     nx_test = nx/2



#     ax_hat = Kx * a_hat
#     ay_hat = Ky * a_hat
#     bx_hat = Kx * b_hat
#     by_hat = Ky * b_hat
#     ax = np.real(np.fft.ifft2(ax_hat))
#     ay = np.real(np.fft.ifft2(ay_hat))
#     bx = np.real(np.fft.ifft2(bx_hat))
#     by = np.real(np.fft.ifft2(by_hat))
#     J = ax * by - ay * bx

#     J1_hat = (-1Ky)

#     J1 = jacobian_Spectral2Physical(Omega_hat, Psi_hat, Kx, Ky)
#     J1f_hat = spectral_filter_square_same_size_2DFHIT(np.fft.fft2(J1), nx_test)
#     J1f = np.real(np.fft.ifft2(J1f_hat))
#     J2f = jacobian_Spectral2Physical(Omegaf_hat, Psif_hat, Kx, Ky)
#     L = J1f - J2f
# )