# ----------------------------------------------------------------------
# Created : Karan Jakhar May 2023
# ----------------------------------------------------------------------

# Eddy Viscosity SGS Models for 2D Turbulence solver


import numpy as np
import jax.numpy as jnp
from jax import jit

from py2d.convert import strain_rate_spectral
from py2d.filter import spectral_filter_square_same_size_jit

strain_rate_spectral_jit = jit(strain_rate_spectral)

def Tau_eddy_viscosity(eddy_viscosity, Psi_hat, Kx, Ky):
    '''
    Calculate the eddy viscosity term (Tau) in the momentum equation
    '''
    N = Psi_hat.shape[0]

    S11_hat, S12_hat, S22_hat = strain_rate_spectral(Psi_hat, Kx, Ky)
    S11 = np.fft.irfft2(S11_hat, s=[N, N])
    S12 = np.fft.irfft2(S12_hat, s=[N, N])
    S22 = np.fft.irfft2(S22_hat, s=[N, N])

    Tau11 = -2*eddy_viscosity*S11
    Tau12 = -2*eddy_viscosity*S12
    Tau22 = -2*eddy_viscosity*S22

    return Tau11, Tau12, Tau22

def Sigma_eddy_viscosity(eddy_viscosity, Omega_hat, Kx, Ky):
    '''
    Calculate the eddy viscosity term (Tau) in the momentum equation
    '''
    N = Omega_hat.shape[0]

    Omegax_hat = (1.j) * Kx * Omega_hat
    Omegay_hat = (1.j) * Ky * Omega_hat

    Omegax = np.fft.irfft2(Omegax_hat, s=[N, N])
    Omegay = np.fft.irfft2(Omegay_hat, s=[N, N])

    Sigma1 = -eddy_viscosity*Omegax
    Sigma2 = -eddy_viscosity*Omegay

    return Sigma1, Sigma2

@jit
def eddy_viscosity_smag(Cs, Delta, characteristic_S):
    '''
    Smagorinsky Model (SMAG)
    '''
    characteristic_S2 = characteristic_S ** 2
    characteristic_S_mean = jnp.sqrt(jnp.mean(characteristic_S2))
    eddy_viscosity = (Cs * Delta) ** 2 * characteristic_S_mean
    return eddy_viscosity

@jit
def eddy_viscosity_smag_local(Cs, Delta, characteristic_S):
    '''
    Smagorinsky Model (SMAG) - Local characteristic_S
    '''
    characteristic_S2 = characteristic_S ** 2
    characteristic_S = jnp.sqrt(characteristic_S2)
    eddy_viscosity = (Cs * Delta) ** 2 * characteristic_S
    return eddy_viscosity

@jit
def characteristic_strain_rate_smag(Psi_hat, Kx, Ky, Ksq):
    '''
    Characteristic strain rate
    Required for Smagorinsky Model
    '''
    N = Psi_hat.shape[0]

    (S11_hat, S12_hat, _) = strain_rate_spectral_jit(Psi_hat, Kx, Ky)
    S11 = jnp.fft.irfft2(S11_hat, s=[N, N])
    S12 = jnp.fft.irfft2(S12_hat, s=[N, N])
    characteristic_S = 2 * jnp.sqrt(S11 ** 2 + S12 ** 2)
    return characteristic_S

@jit
def eddy_viscosity_leith(Cl, Delta, characteristic_Omega):
    '''
    Leith Model (LEITH)
    '''
    characteristic_Omega_mean = jnp.mean(characteristic_Omega)
    ls = Cl * Delta
    eddy_viscosity = ls ** 3 * characteristic_Omega_mean
    return eddy_viscosity

@jit
def characteristic_omega_leith(Omega_hat, Kx, Ky):
    '''
    Characteristic Gradient of Omega
    Required for Leith Model
    '''
    N = Omega_hat.shape[0]

    Omegax_hat = (1.j) * Kx * Omega_hat
    Omegay_hat = (1.j) * Ky * Omega_hat
    Omegax = jnp.fft.irfft2(Omegax_hat, s=[N, N])
    Omegay = jnp.fft.irfft2(Omegay_hat, s=[N, N])
    characteristic_Omega = jnp.sqrt(Omegax ** 2 + Omegay ** 2)
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
def coefficient_dsmaglocal_PsiOmega(Psi_hat, Omega_hat, characteristic_S, Kx, Ky, Ksq, Delta):
    '''
    cs = Cs**2
    Dynamic Coefficient for Dynamic Smagorinsky Model (DSMAG) with local Cs
    '''
    (Psif_hat, Omegaf_hat, Omega_lap, Omegaf_lap, Delta_test, nx_test) = initialize_filtered_variables_PsiOmega(
        Psi_hat, Omega_hat, Ksq, Delta)
    L = residual_jacobian_PsiOmega(Psi_hat, Omega_hat, Psif_hat, Omegaf_hat, Kx, Ky, nx_test)
    M = residual_dsmag_PsiOmega(Omega_lap, Omegaf_lap, characteristic_S, Delta, Delta_test, nx_test)
    cs = coefficient_dynamiclocal_PsiOmega(L, M)
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
    LM_pos = 0.5 * (LM + jnp.abs(LM))

    c_dynamic = jnp.mean(LM_pos) / jnp.mean(MM)
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
    LM_pos = 0.5 * (LM + jnp.abs(LM))

    #c_dynamic = np.mean(LM_pos) / np.mean(MM)
    c_dynamic = (LM_pos) / jnp.mean(MM)
    return c_dynamic

@jit
def initialize_filtered_variables_PsiOmega(Psi_hat, Omega_hat, Ksq, Delta):
    '''
    Calculate filtered flow variables for DSMAG and DLEITH model - Using Psi and Omega
    '''
    nx = Psi_hat.shape[0]
    nx_test = nx // 2
    Delta_test = 2 * Delta
    Psif_hat = spectral_filter_square_same_size_jit(Psi_hat, kc=nx_test//2)
    Omegaf_hat = spectral_filter_square_same_size_jit(Omega_hat, kc=nx_test//2)
    Omega_lap = jnp.fft.irfft2(-Ksq * Omega_hat, s=[nx,nx])
    Omegaf_lap = jnp.fft.irfft2(-Ksq * Omegaf_hat, s=[nx,nx])
    return Psif_hat, Omegaf_hat, Omega_lap, Omegaf_lap, Delta_test, nx_test

@jit
def residual_jacobian_PsiOmega(Psi_hat, Omega_hat, Psif_hat, Omegaf_hat, Kx, Ky, nx_test):
    '''
    Residual of Jacobian
    Difference between Filtered Jacobian and Jacobian of Filtered flow variables
    '''
    N = Psi_hat.shape[0]

    J1 = jacobian_Spectral2Physical(Omega_hat, Psi_hat, Kx, Ky)
    J1f_hat = spectral_filter_square_same_size_jit(jnp.fft.rfft2(J1), kc=nx_test//2)
    J1f = jnp.fft.irfft2(J1f_hat, s=[N,N])
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
    N = Omega_lap.shape[0]

    M1_hat = spectral_filter_square_same_size_jit(jnp.fft.rfft2(characteristic_S * Omega_lap), kc=nx_test//2)
    M1 = Delta ** 2 * jnp.fft.irfft2(M1_hat, s=[N,N])
    characteristic_Sf_hat = spectral_filter_square_same_size_jit(jnp.fft.rfft2(characteristic_S), kc=nx_test//2)
    characteristic_Sf = jnp.fft.irfft2(characteristic_Sf_hat, s=[N,N])
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
    N = Omega_lap.shape[0]

    M1_hat = spectral_filter_square_same_size_jit(jnp.fft.rfft2(characteristic_Omega * Omega_lap), kc=nx_test//2)
    M1 = Delta ** 3 * jnp.fft.irfft2(M1_hat, s=[N,N])
    characteristic_Omegaf_hat = spectral_filter_square_same_size_jit(jnp.fft.rfft2(characteristic_Omega), kc=nx_test//2)
    characteristic_Omegaf = jnp.fft.irfft2(characteristic_Omegaf_hat, s=[N,N])
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
    N = a_hat.shape[0]

    ax_hat = (1.j)*Kx * a_hat
    ay_hat = (1.j)*Ky * a_hat
    bx_hat = (1.j)*Kx * b_hat
    by_hat = (1.j)*Ky * b_hat
    ax = jnp.fft.irfft2(ax_hat, s=[N,N])
    ay = jnp.fft.irfft2(ay_hat, s=[N,N])
    bx = jnp.fft.irfft2(bx_hat, s=[N,N])
    by = jnp.fft.irfft2(by_hat, s=[N,N])
    J = ax * by - ay * bx
    return J