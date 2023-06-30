import numpy as nnp
import jax.numpy as np
from jax import jit

from py2d.derivative_2DFHIT import derivative_2DFHIT

@jit
def real_ifft2(val):
    return np.real(np.fft.ifft2(val))

@jit
def GM2(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    A = Delta**2 / 12

    Ux = real_ifft2(derivative_2DFHIT(U_hat, [1, 0], Kx, Ky))
    Uy = real_ifft2(derivative_2DFHIT(U_hat, [0, 1], Kx, Ky))
    Vx = real_ifft2(derivative_2DFHIT(V_hat, [1, 0], Kx, Ky))

    Omegaxx = real_ifft2(derivative_2DFHIT(Omega_hat, [2, 0], Kx, Ky))
    Omegayy = real_ifft2(derivative_2DFHIT(Omega_hat, [0, 2], Kx, Ky))
    Omegaxy = real_ifft2(derivative_2DFHIT(Omega_hat, [1, 1], Kx, Ky))

    PiOmegaGM2 = A * (Omegaxy*(Uy + Vx) + Ux*(Omegaxx - Omegayy))

    return PiOmegaGM2

@jit
def GM4(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B2 = Delta**4 / 288

    PiOmegaGM2 = GM2(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Uxx = real_ifft2(derivative_2DFHIT(U_hat, [2, 0], Kx, Ky))
    Uxy = real_ifft2(derivative_2DFHIT(U_hat, [1, 1], Kx, Ky))
    Uyy = real_ifft2(derivative_2DFHIT(U_hat, [0, 2], Kx, Ky))
    Vxx = real_ifft2(derivative_2DFHIT(V_hat, [2, 0], Kx, Ky))

    Omegaxxx = real_ifft2(derivative_2DFHIT(Omega_hat, [3, 0], Kx, Ky))
    Omegaxxy = real_ifft2(derivative_2DFHIT(Omega_hat, [2, 1], Kx, Ky))
    Omegaxyy = real_ifft2(derivative_2DFHIT(Omega_hat, [1, 2], Kx, Ky))
    Omegayyy = real_ifft2(derivative_2DFHIT(Omega_hat, [0, 3], Kx, Ky))

    PiOmegaGM4 = PiOmegaGM2 + B2 * (Omegaxxy * (2*Uxy + Vxx) + 
                                     Uxx * (Omegaxxx - 2*Omegaxyy) - Uxy*Omegayyy + Uyy*Omegaxyy)
    
    return PiOmegaGM4

@jit
def GM6(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    PiOmegaGM4 = GM4(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    C1 = Delta**6 / 3456
    C2 = Delta**6 / 10368

    Uxxx = real_ifft2(derivative_2DFHIT(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_ifft2(derivative_2DFHIT(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_ifft2(derivative_2DFHIT(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_ifft2(derivative_2DFHIT(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_ifft2(derivative_2DFHIT(V_hat, [3, 0], Kx, Ky))

    Omegaxxxx = real_ifft2(derivative_2DFHIT(Omega_hat, [4, 0], Kx, Ky))
    Omegaxxxy = real_ifft2(derivative_2DFHIT(Omega_hat, [3, 1], Kx, Ky))
    Omegaxxyy = real_ifft2(derivative_2DFHIT(Omega_hat, [2, 2], Kx, Ky))
    Omegaxyyy = real_ifft2(derivative_2DFHIT(Omega_hat, [1, 3], Kx, Ky))
    Omegayyyy = real_ifft2(derivative_2DFHIT(Omega_hat, [0, 4], Kx, Ky))

    PiOmegaGM6 = PiOmegaGM4 + C2 * (Omegaxxxy*(3*Uxxy + Vxxx) + Omegaxyyy*(Uyyy - 3*Uxxy) + 
                                            3*Omegaxxyy*Uxyy + Uxxx*(Omegaxxxx - 3*Omegaxxyy) - Omegayyyy*Uxyy)
    return PiOmegaGM6
