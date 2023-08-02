import numpy as nnp
import jax.numpy as np
from jax import jit

from py2d.derivative import derivative_2DFHIT_spectral

@jit
def real_ifft2(val):
    return np.real(np.fft.ifft2(val))

# PiOmegaGM2
def PiOmegaGM2_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian' or filterType=='box':
        # GM2 for gaussian and box is same
        PiOmegaGM2 = PiOmegaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM2_hat = np.fft.fft2(PiOmegaGM2)
        return PiOmegaGM2_hat
    else:
        return PiOmegaGM2

@jit
# Needed for jit
def PiOmegaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    A = Delta**2 / 12

    Ux = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [1, 0], Kx, Ky))

    Omegaxx = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [2, 0], Kx, Ky))
    Omegayy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [0, 2], Kx, Ky))
    Omegaxy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [1, 1], Kx, Ky))

    PiOmegaGM2 = A * (Omegaxy*(Uy + Vx) + Ux*(Omegaxx - Omegayy))

    return PiOmegaGM2

# PiOmegaGM4
def PiOmegaGM4_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        PiOmegaGM4 = PiOmegaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM4_hat = np.fft.fft2(PiOmegaGM4)
        return PiOmegaGM4_hat
    else:
        return PiOmegaGM4

@jit
def PiOmegaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B2 = Delta**4 / 288

    PiOmegaGM2 = PiOmegaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Uxx = real_ifft2(derivative_2DFHIT_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [2, 0], Kx, Ky))

    Omegaxxx = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [3, 0], Kx, Ky))
    Omegaxxy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [2, 1], Kx, Ky))
    Omegaxyy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [1, 2], Kx, Ky))
    Omegayyy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [0, 3], Kx, Ky))

    PiOmegaGM4 = PiOmegaGM2 + B2 * (Omegaxxy * (2*Uxy + Vxx) + 
                                     Uxx * (Omegaxxx - 2*Omegaxyy) - Uxy*Omegayyy + Uyy*Omegaxyy)
    
    return PiOmegaGM4

# PiOmegaGM6
def PiOmegaGM6_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        PiOmegaGM6 = PiOmegaGM6_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM6_hat = np.fft.fft2(PiOmegaGM6)
        return PiOmegaGM6_hat
    else:
        return PiOmegaGM6

@jit
def PiOmegaGM6_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    PiOmegaGM4 = PiOmegaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    C1 = Delta**6 / 3456
    C2 = Delta**6 / 10368

    Uxxx = real_ifft2(derivative_2DFHIT_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [3, 0], Kx, Ky))

    Omegaxxxx = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [4, 0], Kx, Ky))
    Omegaxxxy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [3, 1], Kx, Ky))
    Omegaxxyy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [2, 2], Kx, Ky))
    Omegaxyyy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [1, 3], Kx, Ky))
    Omegayyyy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [0, 4], Kx, Ky))

    PiOmegaGM6 = PiOmegaGM4 + C2 * (Omegaxxxy*(3*Uxxy + Vxxx) + Omegaxyyy*(Uyyy - 3*Uxxy) + 
                                            3*Omegaxxyy*Uxyy + Uxxx*(Omegaxxxx - 3*Omegaxxyy) - Omegayyyy*Uxyy)
    return PiOmegaGM6


