import numpy as np
import jax.numpy as jnp
from jax import jit

from py2d.derivative import derivative_spectral
from py2d.dealias import multiply_dealias_spectral, multiply_dealias_spectral_jit

@jit
def real_irfft2_jit(val):
    return jnp.real(jnp.fft.irfft2(val, s=(val.shape[0], val.shape[0])))

def real_irfft2(val):
    return np.real(np.fft.irfft2(val, s=(val.shape[0], val.shape[0])))

derivative_spectral_jit = jit(derivative_spectral)

# PiOmegaGM2
def PiOmegaGM2(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = jnp.fft.rfft2(Omega)
        U_hat = jnp.fft.rfft2(U)
        V_hat = jnp.fft.rfft2(V)

    if filterType=='gaussian' or filterType=='box':
        # GM2 for gaussian and box is same
        # Two function for dealias and alias are made to avoid if else in the main function and make it jax/jit compatible
        if dealias:
            PiOmegaGM2_hat = PiOmegaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
            PiOmegaGM2 = real_irfft2_jit(PiOmegaGM2_hat)
        else:
            PiOmegaGM2 = PiOmegaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM2_hat = jnp.fft.rfft2(PiOmegaGM2)
        return PiOmegaGM2_hat
    else:
        return PiOmegaGM2

@jit
# Two function for dealias and alias are made to avoid if else in the main function and make it jax/jit compatible
def PiOmegaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # Not dealiased

    A = Delta**2 / 12

    Ux = real_irfft2_jit(derivative_spectral_jit(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2_jit(derivative_spectral_jit(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2_jit(derivative_spectral_jit(V_hat, [1, 0], Kx, Ky))

    Omegaxx = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [2, 0], Kx, Ky))
    Omegayy = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [0, 2], Kx, Ky))
    Omegaxy = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [1, 1], Kx, Ky))

    PiOmegaGM2 = A * (Omegaxy*(Uy + Vx) + Ux*(Omegaxx - Omegayy))

    return PiOmegaGM2

@jit
def PiOmegaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # dealiased

    A = Delta**2 / 12

    Ux_hat = derivative_spectral_jit(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral_jit(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral_jit(V_hat, [1, 0], Kx, Ky)

    Omegaxx_hat = derivative_spectral_jit(Omega_hat, [2, 0], Kx, Ky)
    Omegayy_hat = derivative_spectral_jit(Omega_hat, [0, 2], Kx, Ky)
    Omegaxy_hat = derivative_spectral_jit(Omega_hat, [1, 1], Kx, Ky)

    UyOmegaxy_hat = multiply_dealias_spectral_jit(Uy_hat, Omegaxy_hat)
    VxOmegaxy_hat = multiply_dealias_spectral_jit(Vx_hat, Omegaxy_hat)
    UxOmegaxx_hat = multiply_dealias_spectral_jit(Ux_hat, Omegaxx_hat)
    UxOmegayy_hat = multiply_dealias_spectral_jit(Ux_hat, Omegayy_hat)

    PiOmegaGM2_hat = A * (UyOmegaxy_hat + VxOmegaxy_hat + UxOmegaxx_hat - UxOmegayy_hat)
    
    return PiOmegaGM2_hat

# PiOmegaGM4
def PiOmegaGM4(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = jnp.fft.rfft2(Omega)
        U_hat = jnp.fft.rfft2(U)
        V_hat = jnp.fft.rfft2(V)

    if dealias:
        if filterType == 'gaussian':
            PiOmegaGM4_hat = PiOmegaGM4_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        elif filterType == 'box':
            PiOmegaGM4_hat = PiOmegaGM4_box_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        PiOmegaGM4 = real_irfft2_jit(PiOmegaGM4_hat)
    else:
        if filterType == 'gaussian':
            PiOmegaGM4 = PiOmegaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        elif filterType == 'box':
            PiOmegaGM4 = PiOmegaGM4_box(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM4_hat = jnp.fft.rfft2(PiOmegaGM4)
        return PiOmegaGM4_hat
    else:
        return PiOmegaGM4

@jit
def PiOmegaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased

    B2 = Delta**4 / 288

    PiOmegaGM2 = PiOmegaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Uxx = real_irfft2_jit(derivative_spectral_jit(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2_jit(derivative_spectral_jit(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2_jit(derivative_spectral_jit(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2_jit(derivative_spectral_jit(V_hat, [2, 0], Kx, Ky))

    Omegaxxx = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [3, 0], Kx, Ky))
    Omegaxxy = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [2, 1], Kx, Ky))
    Omegaxyy = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [1, 2], Kx, Ky))
    Omegayyy = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [0, 3], Kx, Ky))

    PiOmegaGM4 = PiOmegaGM2 + B2 * (Omegaxxy * (2*Uxy + Vxx) + 
                                     Uxx * (Omegaxxx - 2*Omegaxyy) - Uxy*Omegayyy + Uyy*Omegaxyy)
    
    return PiOmegaGM4

@jit
def PiOmegaGM4_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased

    B2 = Delta**4 / 288

    PiOmegaGM2_hat = PiOmegaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Uxx_hat = derivative_spectral_jit(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral_jit(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral_jit(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral_jit(V_hat, [2, 0], Kx, Ky)

    Omegaxxx_hat = derivative_spectral_jit(Omega_hat, [3, 0], Kx, Ky)
    Omegaxxy_hat = derivative_spectral_jit(Omega_hat, [2, 1], Kx, Ky)
    Omegaxyy_hat = derivative_spectral_jit(Omega_hat, [1, 2], Kx, Ky)
    Omegayyy_hat = derivative_spectral_jit(Omega_hat, [0, 3], Kx, Ky)

    UxyOmegaxxy_hat = multiply_dealias_spectral_jit(Uxy_hat, Omegaxxy_hat)
    VxxOmegaxxy_hat = multiply_dealias_spectral_jit(Vxx_hat, Omegaxxy_hat)
    UxxOmegaxxx_hat = multiply_dealias_spectral_jit(Uxx_hat, Omegaxxx_hat)
    UxxOmegaxyy_hat = multiply_dealias_spectral_jit(Uxx_hat, Omegaxyy_hat)
    UxyOmegayyy_hat = multiply_dealias_spectral_jit(Uxy_hat, Omegayyy_hat)
    UyyOmegaxyy_hat = multiply_dealias_spectral_jit(Uyy_hat, Omegaxyy_hat)

    PiOmegaGM4_hat = PiOmegaGM2_hat + B2 * (2*UxyOmegaxxy_hat + VxxOmegaxxy_hat + 
                                            UxxOmegaxxx_hat - 2*UxxOmegaxyy_hat - UxyOmegayyy_hat + UyyOmegaxyy_hat)
    
    return PiOmegaGM4_hat

@jit
def PiOmegaGM4_box(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    B3 = Delta**4 / 720

    PiOmegaGM2 = PiOmegaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))

    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))

    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    # Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))

    # Omegax = real_irfft2(derivative_spectral(Omega_hat, [1, 0], Kx, Ky))
    # Omegay = real_irfft2(derivative_spectral(Omega_hat, [0, 1], Kx, Ky))

    Omegaxx = real_irfft2(derivative_spectral(Omega_hat, [2, 0], Kx, Ky))
    Omegaxy = real_irfft2(derivative_spectral(Omega_hat, [1, 1], Kx, Ky))
    Omegayy = real_irfft2(derivative_spectral(Omega_hat, [0, 2], Kx, Ky))

    Omegaxxx = real_irfft2(derivative_spectral(Omega_hat, [3, 0], Kx, Ky))
    Omegaxxy = real_irfft2(derivative_spectral(Omega_hat, [2, 1], Kx, Ky))
    Omegaxyy = real_irfft2(derivative_spectral(Omega_hat, [1, 2], Kx, Ky))
    Omegayyy = real_irfft2(derivative_spectral(Omega_hat, [0, 3], Kx, Ky))

    Omegaxxxx = real_irfft2(derivative_spectral(Omega_hat, [4, 0], Kx, Ky))
    Omegaxxxy = real_irfft2(derivative_spectral(Omega_hat, [3, 1], Kx, Ky))
    # Omegaxxyy = real_irfft2(derivative_spectral(Omega_hat, [2, 2], Kx, Ky))
    Omegaxyyy = real_irfft2(derivative_spectral(Omega_hat, [1, 3], Kx, Ky))
    Omegayyyy = real_irfft2(derivative_spectral(Omega_hat, [0, 4], Kx, Ky))

    PiOmegaGM4 = PiOmegaGM2 - B3*( Omegaxy*(Uyyy + Vxxx) - Omegaxxy*(5*Uxy+Vxx) - Omegaxyy*(Uyy-5*Uxx)
                                  + Uy*Omegaxyyy + Uxy*Omegayyy - Uxyy*Omegayy + Vx*Omegaxxxy  
                                  + Ux*(Omegaxxxx-Omegayyyy) + Uxxx*Omegaxx - Uxx*Omegaxxx)
    
    return PiOmegaGM4

@jit
def PiOmegaGM4_box_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    B3 = Delta**4 / 720

    PiOmegaGM2_hat = PiOmegaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)

    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)

    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    # Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)

    # Omegax_hat = derivative_spectral(Omega_hat, [1, 0], Kx, Ky)
    # Omegay_hat = derivative_spectral(Omega_hat, [0, 1], Kx, Ky)

    Omegaxx_hat = derivative_spectral(Omega_hat, [2, 0], Kx, Ky)
    Omegaxy_hat = derivative_spectral(Omega_hat, [1, 1], Kx, Ky)
    Omegayy_hat = derivative_spectral(Omega_hat, [0, 2], Kx, Ky)

    Omegaxxx_hat = derivative_spectral(Omega_hat, [3, 0], Kx, Ky)
    Omegaxxy_hat = derivative_spectral(Omega_hat, [2, 1], Kx, Ky)
    Omegaxyy_hat = derivative_spectral(Omega_hat, [1, 2], Kx, Ky)
    Omegayyy_hat = derivative_spectral(Omega_hat, [0, 3], Kx, Ky)

    Omegaxxxx_hat = derivative_spectral(Omega_hat, [4, 0], Kx, Ky)
    Omegaxxxy_hat = derivative_spectral(Omega_hat, [3, 1], Kx, Ky)
    # Omegaxxyy_hat = derivative_spectral(Omega_hat, [2, 2], Kx, Ky)
    Omegaxyyy_hat = derivative_spectral(Omega_hat, [1, 3], Kx, Ky)
    Omegayyyy_hat = derivative_spectral(Omega_hat, [0, 4], Kx, Ky)

    UyyyOmegaxy_hat = multiply_dealias_spectral_jit(Uyyy_hat, Omegaxy_hat)
    VxxxOmegaxy_hat = multiply_dealias_spectral_jit(Vxxx_hat, Omegaxy_hat)
    UxyOmegaxxy_hat = multiply_dealias_spectral_jit(Uxy_hat, Omegaxxy_hat)
    VxxOmegaxxy_hat = multiply_dealias_spectral_jit(Vxx_hat, Omegaxxy_hat)
    UyyOmegaxyy_hat = multiply_dealias_spectral_jit(Uyy_hat, Omegaxyy_hat)
    UxxOmegaxyy_hat = multiply_dealias_spectral_jit(Uxx_hat, Omegaxyy_hat)
    UyOmegaxyyy_hat = multiply_dealias_spectral_jit(Uy_hat, Omegaxyyy_hat)
    UxyOmegayyy_hat = multiply_dealias_spectral_jit(Uxy_hat, Omegayyy_hat)
    UxyyOmegayy_hat = multiply_dealias_spectral_jit(Uxyy_hat, Omegayy_hat)
    VxOmegaxxxy_hat = multiply_dealias_spectral_jit(Vx_hat, Omegaxxxy_hat)
    UxOmegaxxxx_hat = multiply_dealias_spectral_jit(Ux_hat, Omegaxxxx_hat)
    UxOmegayyyy_hat = multiply_dealias_spectral_jit(Ux_hat, Omegayyyy_hat)
    UxxxOmegaxx_hat = multiply_dealias_spectral_jit(Uxxx_hat, Omegaxx_hat)
    UxxOmegaxxx_hat = multiply_dealias_spectral_jit(Uxx_hat, Omegaxxx_hat)

    PiOmegaGM4_hat = PiOmegaGM2_hat - B3*( UyyyOmegaxy_hat + VxxxOmegaxy_hat - 5*UxyOmegaxxy_hat - VxxOmegaxxy_hat - UyyOmegaxyy_hat + 5*UxxOmegaxyy_hat
                                        + UyOmegaxyyy_hat + UxyOmegayyy_hat - UxyyOmegayy_hat + VxOmegaxxxy_hat 
                                        + UxOmegaxxxx_hat - UxOmegayyyy_hat + UxxxOmegaxx_hat - UxxOmegaxxx_hat)
    
    return PiOmegaGM4_hat

# PiOmegaGM6
def PiOmegaGM6(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = jnp.fft.rfft2(Omega)
        U_hat = jnp.fft.rfft2(U)
        V_hat = jnp.fft.rfft2(V)

    if dealias:
        if filterType == 'gaussian':
            PiOmegaGM6_hat = PiOmegaGM6_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        elif filterType == 'box':
            PiOmegaGM6_hat = PiOmegaGM6_box_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        PiOmegaGM6 = real_irfft2_jit(PiOmegaGM6_hat)
    else:
        if filterType == 'gaussian':
            PiOmegaGM6 = PiOmegaGM6_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        elif filterType == 'box':
            PiOmegaGM6 = PiOmegaGM6_box(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM6_hat = jnp.fft.rfft2(PiOmegaGM6)
        return PiOmegaGM6_hat
    else:
        return PiOmegaGM6

@jit
def PiOmegaGM6_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased

    PiOmegaGM4 = PiOmegaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    C2 = Delta**6 / 10368

    Uxxx = real_irfft2_jit(derivative_spectral_jit(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2_jit(derivative_spectral_jit(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2_jit(derivative_spectral_jit(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2_jit(derivative_spectral_jit(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2_jit(derivative_spectral_jit(V_hat, [3, 0], Kx, Ky))

    Omegaxxxx = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [4, 0], Kx, Ky))
    Omegaxxxy = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [3, 1], Kx, Ky))
    Omegaxxyy = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [2, 2], Kx, Ky))
    Omegaxyyy = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [1, 3], Kx, Ky))
    Omegayyyy = real_irfft2_jit(derivative_spectral_jit(Omega_hat, [0, 4], Kx, Ky))

    PiOmegaGM6 = PiOmegaGM4 + C2 * (Omegaxxxy*(3*Uxxy + Vxxx) + Omegaxyyy*(Uyyy - 3*Uxxy) + 
                                            3*Omegaxxyy*Uxyy + Uxxx*(Omegaxxxx - 3*Omegaxxyy) - Omegayyyy*Uxyy)
    return PiOmegaGM6

@jit
def PiOmegaGM6_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased

    PiOmegaGM4_hat = PiOmegaGM4_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    C2 = Delta**6 / 10368

    Uxxx_hat = derivative_spectral_jit(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral_jit(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral_jit(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral_jit(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral_jit(V_hat, [3, 0], Kx, Ky)

    Omegaxxxx_hat = derivative_spectral_jit(Omega_hat, [4, 0], Kx, Ky)
    Omegaxxxy_hat = derivative_spectral_jit(Omega_hat, [3, 1], Kx, Ky)
    Omegaxxyy_hat = derivative_spectral_jit(Omega_hat, [2, 2], Kx, Ky)
    Omegaxyyy_hat = derivative_spectral_jit(Omega_hat, [1, 3], Kx, Ky)
    Omegayyyy_hat = derivative_spectral_jit(Omega_hat, [0, 4], Kx, Ky)

    UxxyOmegaxxxy_hat = multiply_dealias_spectral_jit(Uxxy_hat, Omegaxxxy_hat)
    VxxxOmegaxxxy_hat = multiply_dealias_spectral_jit(Vxxx_hat, Omegaxxxy_hat)
    UyyyOmegaxyyy_hat = multiply_dealias_spectral_jit(Uyyy_hat, Omegaxyyy_hat)
    UxxyOmegaxyyy_hat = multiply_dealias_spectral_jit(Uxxy_hat, Omegaxyyy_hat)
    UxyyOmegaxxyy_hat = multiply_dealias_spectral_jit(Uxyy_hat, Omegaxxyy_hat)
    UxxxOmegaxxxx_hat = multiply_dealias_spectral_jit(Uxxx_hat, Omegaxxxx_hat)
    UxxxOmegaxxyy_hat = multiply_dealias_spectral_jit(Uxxx_hat, Omegaxxyy_hat)
    UxyyOmegayyyy_hat = multiply_dealias_spectral_jit(Uxyy_hat, Omegayyyy_hat)

    PiOmegaGM6_hat = PiOmegaGM4_hat + C2 * (3*UxxyOmegaxxxy_hat + VxxxOmegaxxxy_hat + UyyyOmegaxyyy_hat - 3*UxxyOmegaxyyy_hat + 
                                        3*UxyyOmegaxxyy_hat + UxxxOmegaxxxx_hat - 3*UxxxOmegaxxyy_hat - UxyyOmegayyyy_hat)

    return PiOmegaGM6_hat

@jit
def PiOmegaGM6_box(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    C5 = Delta**6 / 60480

    PiOmegaGM4 = PiOmegaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))

    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))

    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))

    Uxxxx = real_irfft2(derivative_spectral(U_hat, [4, 0], Kx, Ky))
    Uxxxy = real_irfft2(derivative_spectral(U_hat, [3, 1], Kx, Ky))
    Uxxyy = real_irfft2(derivative_spectral(U_hat, [2, 2], Kx, Ky))
    Uxyyy = real_irfft2(derivative_spectral(U_hat, [1, 3], Kx, Ky))
    Uyyyy = real_irfft2(derivative_spectral(U_hat, [0, 4], Kx, Ky))
    Vxxxx = real_irfft2(derivative_spectral(V_hat, [4, 0], Kx, Ky))

    Uxxxxx = real_irfft2(derivative_spectral(U_hat, [5, 0], Kx, Ky))
    # Uxxxxy = real_irfft2(derivative_spectral(U_hat, [4, 1], Kx, Ky))
    # Uxxxyy = real_irfft2(derivative_spectral(U_hat, [3, 2], Kx, Ky))
    # Uxxyyy = real_irfft2(derivative_spectral(U_hat, [2, 3], Kx, Ky))
    Uxyyyy = real_irfft2(derivative_spectral(U_hat, [1, 4], Kx, Ky))
    Uyyyyy = real_irfft2(derivative_spectral(U_hat, [0, 5], Kx, Ky))
    Vxxxxx = real_irfft2(derivative_spectral(V_hat, [5, 0], Kx, Ky))

    # Omegax = real_irfft2(derivative_spectral(Omega_hat, [1, 0], Kx, Ky))
    # Omegay = real_irfft2(derivative_spectral(Omega_hat, [0, 1], Kx, Ky))

    Omegaxx = real_irfft2(derivative_spectral(Omega_hat, [2, 0], Kx, Ky))
    Omegaxy = real_irfft2(derivative_spectral(Omega_hat, [1, 1], Kx, Ky))
    Omegayy = real_irfft2(derivative_spectral(Omega_hat, [0, 2], Kx, Ky))

    Omegaxxx = real_irfft2(derivative_spectral(Omega_hat, [3, 0], Kx, Ky))
    Omegaxxy = real_irfft2(derivative_spectral(Omega_hat, [2, 1], Kx, Ky))
    Omegaxyy = real_irfft2(derivative_spectral(Omega_hat, [1, 2], Kx, Ky))
    Omegayyy = real_irfft2(derivative_spectral(Omega_hat, [0, 3], Kx, Ky))

    Omegaxxxx = real_irfft2(derivative_spectral(Omega_hat, [4, 0], Kx, Ky))
    Omegaxxxy = real_irfft2(derivative_spectral(Omega_hat, [3, 1], Kx, Ky))
    Omegaxxyy = real_irfft2(derivative_spectral(Omega_hat, [2, 2], Kx, Ky))
    Omegaxyyy = real_irfft2(derivative_spectral(Omega_hat, [1, 3], Kx, Ky))
    Omegayyyy = real_irfft2(derivative_spectral(Omega_hat, [0, 4], Kx, Ky))

    Omegaxxxxx = real_irfft2(derivative_spectral(Omega_hat, [5, 0], Kx, Ky))
    Omegaxxxxy = real_irfft2(derivative_spectral(Omega_hat, [4, 1], Kx, Ky))
    Omegaxxxyy = real_irfft2(derivative_spectral(Omega_hat, [3, 2], Kx, Ky))
    Omegaxxyyy = real_irfft2(derivative_spectral(Omega_hat, [2, 3], Kx, Ky))
    Omegaxyyyy = real_irfft2(derivative_spectral(Omega_hat, [1, 4], Kx, Ky))
    Omegayyyyy = real_irfft2(derivative_spectral(Omega_hat, [0, 5], Kx, Ky))

    Omegaxxxxxx = real_irfft2(derivative_spectral(Omega_hat, [6, 0], Kx, Ky))
    Omegaxxxxxy = real_irfft2(derivative_spectral(Omega_hat, [5, 1], Kx, Ky))
    # Omegaxxxxyy = real_irfft2(derivative_spectral(Omega_hat, [4, 2], Kx, Ky))
    # Omegaxxxyyy = real_irfft2(derivative_spectral(Omega_hat, [3, 3], Kx, Ky))
    # Omegaxxyyyy = real_irfft2(derivative_spectral(Omega_hat, [2, 4], Kx, Ky))
    Omegaxyyyyy = real_irfft2(derivative_spectral(Omega_hat, [1, 5], Kx, Ky))
    Omegayyyyyy = real_irfft2(derivative_spectral(Omega_hat, [0, 6], Kx, Ky))

    PiOmegaGM6 = PiOmegaGM4 + C5 * (2*Omegaxy*(Uyyyyy + Vxxxxx) 
                                    + 2*(Uy*Omegaxyyyyy + Vx*Omegaxxxxxy + Uxxxxx*Omegaxx)
                                    + Omegaxxxy*(7*Uxxy + 2*Vxxx) - Omegaxxy*(7*(Uxxxy + Uxyyy) +2*Vxxxx)
                                    + Omegaxyyy*(2*Uyyy - 7*Uxxy) + Omegaxyy*(7*Uxxyy -2*Uyyyy) 
                                    + Uxxxx*(7*Omegaxyy - 2*Omegaxxx) - 7*Uxy*(Omegaxxxxy - Omegaxxyyy)
                                    + 7*Omegaxxyy*(Uxyy - Uxxx) + Uxx*(7*(Omegaxxxyy + Omegaxyyyy) -2*Omegaxxxxx)
                                    + 2*(Uxy*Omegayyyyy + Uxyyy*Omegayyy - Uxyyyy*Omegayy - Uyy*Omegaxyyyy)
                                    + 2*(-Uxyy*Omegayyyy - Vxx*Omegaxxxxy + Ux*(Omegaxxxxxx - Omegayyyyyy) +Uxxx*Omegaxxxx))

    return PiOmegaGM6

@jit
def PiOmegaGM6_box_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    C5 = Delta**6 / 60480

    PiOmegaGM4_hat = PiOmegaGM4_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)

    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)

    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)

    Uxxxx_hat = derivative_spectral(U_hat, [4, 0], Kx, Ky)
    Uxxxy_hat = derivative_spectral(U_hat, [3, 1], Kx, Ky)
    Uxxyy_hat = derivative_spectral(U_hat, [2, 2], Kx, Ky)
    Uxyyy_hat = derivative_spectral(U_hat, [1, 3], Kx, Ky)
    Uyyyy_hat = derivative_spectral(U_hat, [0, 4], Kx, Ky)
    Vxxxx_hat = derivative_spectral(V_hat, [4, 0], Kx, Ky)

    Uxxxxx_hat = derivative_spectral(U_hat, [5, 0], Kx, Ky)
    # Uxxxxy_hat = derivative_spectral(U_hat, [4, 1], Kx, Ky)
    # Uxxxyy_hat = derivative_spectral(U_hat, [3, 2], Kx, Ky)
    # Uxxyyy_hat = derivative_spectral(U_hat, [2, 3], Kx, Ky)
    Uxyyyy_hat = derivative_spectral(U_hat, [1, 4], Kx, Ky)
    Uyyyyy_hat = derivative_spectral(U_hat, [0, 5], Kx, Ky)
    Vxxxxx_hat = derivative_spectral(V_hat, [5, 0], Kx, Ky)

    # Omegax_hat = derivative_spectral(Omega_hat, [1, 0], Kx, Ky)
    # Omegay_hat = derivative_spectral(Omega_hat, [0, 1], Kx, Ky)

    Omegaxx_hat = derivative_spectral(Omega_hat, [2, 0], Kx, Ky)
    Omegaxy_hat = derivative_spectral(Omega_hat, [1, 1], Kx, Ky)
    Omegayy_hat = derivative_spectral(Omega_hat, [0, 2], Kx, Ky)

    Omegaxxx_hat = derivative_spectral(Omega_hat, [3, 0], Kx, Ky)
    Omegaxxy_hat = derivative_spectral(Omega_hat, [2, 1], Kx, Ky)
    Omegaxyy_hat = derivative_spectral(Omega_hat, [1, 2], Kx, Ky)
    Omegayyy_hat = derivative_spectral(Omega_hat, [0, 3], Kx, Ky)

    Omegaxxxx_hat = derivative_spectral(Omega_hat, [4, 0], Kx, Ky)
    Omegaxxxy_hat = derivative_spectral(Omega_hat, [3, 1], Kx, Ky)
    Omegaxxyy_hat = derivative_spectral(Omega_hat, [2, 2], Kx, Ky)
    Omegaxyyy_hat = derivative_spectral(Omega_hat, [1, 3], Kx, Ky)
    Omegayyyy_hat = derivative_spectral(Omega_hat, [0, 4], Kx, Ky)

    Omegaxxxxx_hat = derivative_spectral(Omega_hat, [5, 0], Kx, Ky)
    Omegaxxxxy_hat = derivative_spectral(Omega_hat, [4, 1], Kx, Ky)
    Omegaxxxyy_hat = derivative_spectral(Omega_hat, [3, 2], Kx, Ky)
    Omegaxxyyy_hat = derivative_spectral(Omega_hat, [2, 3], Kx, Ky)
    Omegaxyyyy_hat = derivative_spectral(Omega_hat, [1, 4], Kx, Ky)
    Omegayyyyy_hat = derivative_spectral(Omega_hat, [0, 5], Kx, Ky)

    Omegaxxxxxx_hat = derivative_spectral(Omega_hat, [6, 0], Kx, Ky)
    Omegaxxxxxy_hat = derivative_spectral(Omega_hat, [5, 1], Kx, Ky)
    # Omegaxxxxyy_hat = derivative_spectral(Omega_hat, [4, 2], Kx, Ky)
    # Omegaxxxyyy_hat = derivative_spectral(Omega_hat, [3, 3], Kx, Ky)
    # Omegaxxyyyy_hat = derivative_spectral(Omega_hat, [2, 4], Kx, Ky)
    Omegaxyyyyy_hat = derivative_spectral(Omega_hat, [1, 5], Kx, Ky)
    Omegayyyyyy_hat = derivative_spectral(Omega_hat, [0, 6], Kx, Ky)

    UyyyyyOmegaxy_hat = multiply_dealias_spectral_jit(Uyyyyy_hat, Omegaxy_hat)
    VxxxxxOmegaxy_hat = multiply_dealias_spectral_jit(Vxxxxx_hat, Omegaxy_hat)
    UyOmegaxyyyyy_hat = multiply_dealias_spectral_jit(Uy_hat, Omegaxyyyyy_hat)
    VxOmegaxxxxxy_hat = multiply_dealias_spectral_jit(Vx_hat, Omegaxxxxxy_hat)
    UxxxxxOmegaxx_hat = multiply_dealias_spectral_jit(Uxxxxx_hat, Omegaxx_hat)
    UxxyOmegaxxxy_hat = multiply_dealias_spectral_jit(Uxxy_hat, Omegaxxxy_hat)
    VxxxOmegaxxxy_hat = multiply_dealias_spectral_jit(Vxxx_hat, Omegaxxxy_hat)
    UxxxyOmegaxxy_hat = multiply_dealias_spectral_jit(Uxxxy_hat, Omegaxxy_hat)
    UxyyyOmegaxxy_hat = multiply_dealias_spectral_jit(Uxyyy_hat, Omegaxxy_hat)
    VxxxxOmegaxxy_hat = multiply_dealias_spectral_jit(Vxxxx_hat, Omegaxxy_hat)
    UyyyOmegaxyyy_hat = multiply_dealias_spectral_jit(Uyyy_hat, Omegaxyyy_hat)
    UxxyOmegaxyyy_hat = multiply_dealias_spectral_jit(Uxxy_hat, Omegaxyyy_hat)
    UxxyyOmegaxyy_hat = multiply_dealias_spectral_jit(Uxxyy_hat, Omegaxyy_hat)
    UyyyyOmegaxyy_hat = multiply_dealias_spectral_jit(Uyyyy_hat, Omegaxyy_hat)
    UxxxxOmegaxyy_hat = multiply_dealias_spectral_jit(Uxxxx_hat, Omegaxyy_hat)
    UxxxxOmegaxxx_hat = multiply_dealias_spectral_jit(Uxxxx_hat, Omegaxxx_hat)
    UxyOmegaxxxxy_hat = multiply_dealias_spectral_jit(Uxy_hat, Omegaxxxxy_hat)
    UxyOmegaxxyyy_hat = multiply_dealias_spectral_jit(Uxy_hat, Omegaxxyyy_hat)
    UxyyOmegaxxyy = multiply_dealias_spectral_jit(Uxyy_hat, Omegaxxyy_hat)
    UxxxOmegaxxyy = multiply_dealias_spectral_jit(Uxxx_hat, Omegaxxyy_hat)
    UxxOmegaxxxyy = multiply_dealias_spectral_jit(Uxx_hat, Omegaxxxyy_hat)
    UxxOmegaxyyyy = multiply_dealias_spectral_jit(Uxx_hat, Omegaxyyyy_hat)
    UxxOmegaxxxxx_hat = multiply_dealias_spectral_jit(Uxx_hat, Omegaxxxxx_hat)
    UxyOmegayyyyy_hat = multiply_dealias_spectral_jit(Uxy_hat, Omegayyyyy_hat)
    UxyyyOmegayyy_hat = multiply_dealias_spectral_jit(Uxyyy_hat, Omegayyy_hat)
    UxyyyyOmegayy_hat = multiply_dealias_spectral_jit(Uxyyyy_hat, Omegayy_hat)
    UyyOmegaxyyyy_hat = multiply_dealias_spectral_jit(Uyy_hat, Omegaxyyyy_hat)
    UxyyOmegayyyy_hat = multiply_dealias_spectral_jit(Uxyy_hat, Omegayyyy_hat)
    VxxOmegaxxxxy_hat = multiply_dealias_spectral_jit(Vxx_hat, Omegaxxxxy_hat)
    UxOmegaxxxxxx_hat = multiply_dealias_spectral_jit(Ux_hat, Omegaxxxxxx_hat)
    UxOmegayyyyyy_hat = multiply_dealias_spectral_jit(Ux_hat, Omegayyyyyy_hat)
    UxxxOmegaxxxx_hat = multiply_dealias_spectral_jit(Uxxx_hat, Omegaxxxx_hat)

    PIOmegaGM6_hat = PiOmegaGM4_hat + C5 * (2*(UyyyyyOmegaxy_hat + VxxxxxOmegaxy_hat) 
                                        + 2*(UyOmegaxyyyyy_hat + VxOmegaxxxxxy_hat + UxxxxxOmegaxx_hat)
                                        + 7*UxxyOmegaxxxy_hat + 2*VxxxOmegaxxxy_hat - 7*(UxxxyOmegaxxy_hat + UxyyyOmegaxxy_hat) - 2*VxxxxOmegaxxy_hat
                                        + 2*UyyyOmegaxyyy_hat - 7*UxxyOmegaxyyy_hat + 7*UxxyyOmegaxyy_hat - 2*UyyyyOmegaxyy_hat
                                        + 7*UxxxxOmegaxyy_hat - 2*UxxxxOmegaxxx_hat - 7*(UxyOmegaxxxxy_hat - UxyOmegaxxyyy_hat)
                                        + 7*(UxyyOmegaxxyy - UxxxOmegaxxyy) + 7*(UxxOmegaxxxyy + UxxOmegaxyyyy) - 2*UxxOmegaxxxxx_hat
                                        + 2*(UxyOmegayyyyy_hat + UxyyyOmegayyy_hat - UxyyyyOmegayy_hat - UyyOmegaxyyyy_hat)
                                        + 2*(-UxyyOmegayyyy_hat - VxxOmegaxxxxy_hat + UxOmegaxxxxxx_hat - UxOmegayyyyyy_hat + UxxxOmegaxxxx_hat))
    
    return PIOmegaGM6_hat


def PiOmegaGM8(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.rfft2(Omega)
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        PiOmegaGM8 = PiOmegaGM8_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM8_hat = np.fft.rfft2(PiOmegaGM8)
        return PiOmegaGM8_hat
    else:
        return PiOmegaGM8


# PiOmegaGM10
def PiOmegaGM10(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.rfft2(Omega)
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        PiOmegaGM10 = PiOmegaGM10_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM10_hat = np.fft.rfft2(PiOmegaGM10)
        return PiOmegaGM10_hat
    else:
        return PiOmegaGM10


##############################################################################################################

# TauGM2
def TauGM2(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian' or filterType=='box':
    # GM2 for gaussian and box is same
        if dealias:
            Tau11GM2_hat, Tau12GM2_hat, Tau22GM2_hat = TauGM2_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)
            Tau11GM2 = real_irfft2(Tau11GM2_hat)
            Tau12GM2 = real_irfft2(Tau12GM2_hat)
            Tau22GM2 = real_irfft2(Tau22GM2_hat)
        else:
            Tau11GM2, Tau12GM2, Tau22GM2 = TauGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11GM2_hat = np.fft.rfft2(Tau11GM2)
        Tau12GM2_hat = np.fft.rfft2(Tau12GM2)
        Tau22GM2_hat = np.fft.rfft2(Tau22GM2)
        return Tau11GM2_hat, Tau12GM2_hat, Tau22GM2_hat
    else:
        return Tau11GM2, Tau12GM2, Tau22GM2
    

def TauGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta):
# Not dealiased
    A = Delta**2 / 12

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux

    Tau11GM2 = A * (Ux**2 + Uy**2)
    Tau12GM2 = A * (Ux*Vx + Uy*Vy)
    Tau22GM2 = A * (Vx**2 + Vy**2)

    return Tau11GM2, Tau12GM2, Tau22GM2


def TauGM2_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta):
# dealiased
    A = Delta**2 / 12

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)
    Vy_hat = -Ux_hat

    UxUx_hat = multiply_dealias_spectral(Ux_hat, Ux_hat)
    UyUy_hat = multiply_dealias_spectral(Uy_hat, Uy_hat)
    UxVx_hat = multiply_dealias_spectral(Ux_hat, Vx_hat)
    UyVy_hat = multiply_dealias_spectral(Uy_hat, Vy_hat)
    VxVx_hat = multiply_dealias_spectral(Vx_hat, Vx_hat)
    VyVy_hat = multiply_dealias_spectral(Vy_hat, Vy_hat)

    Tau11GM2_hat = A * (UxUx_hat + UyUy_hat)
    Tau12GM2_hat = A * (UxVx_hat + UyVy_hat)
    Tau22GM2_hat = A * (VxVx_hat + VyVy_hat)

    return Tau11GM2_hat, Tau12GM2_hat, Tau22GM2_hat

# TauGM4
def TauGM4(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if dealias:
        if filterType=='gaussian':
            Tau11GM4_hat, Tau12GM4_hat, Tau22GM4_hat = TauGM4_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)
        elif filterType=='box':
            Tau11GM4_hat, Tau12GM4_hat, Tau22GM4_hat = TauGM4_box_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)
        Tau11GM4 = real_irfft2(Tau11GM4_hat)
        Tau12GM4 = real_irfft2(Tau12GM4_hat)
        Tau22GM4 = real_irfft2(Tau22GM4_hat)
    else:
        if filterType=='gaussian':
            Tau11GM4, Tau12GM4, Tau22GM4 = TauGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta)
        elif filterType=='box':
            Tau11GM4, Tau12GM4, Tau22GM4 = TauGM4_box(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11GM4_hat = np.fft.rfft2(Tau11GM4)
        Tau12GM4_hat = np.fft.rfft2(Tau12GM4)
        Tau22GM4_hat = np.fft.rfft2(Tau22GM4)
        return Tau11GM4_hat, Tau12GM4_hat, Tau22GM4_hat
    else:
        return Tau11GM4, Tau12GM4, Tau22GM4
    

def TauGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B2 = Delta**4 / 288

    Tau11GM2, Tau12GM2, Tau22GM2 = TauGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy

    Tau11GM4 = Tau11GM2 + B1*(Uxy**2)   + B2*(Uxx**2 + Uyy**2)
    Tau12GM4 = Tau12GM2 + B1*(Uxy*Vxy)  + B2*(Uxx*Vxx + Uyy*Vyy)
    Tau22GM4 = Tau22GM2 + B1*(Vxy**2)   + B2*(Vxx**2 + Vyy**2)

    return Tau11GM4, Tau12GM4, Tau22GM4


def TauGM4_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B2 = Delta**4 / 288

    Tau11GM2_hat, Tau12GM2_hat, Tau22GM2_hat = TauGM2_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)

    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)
    Vxy_hat = -Uxx_hat
    Vyy_hat = -Uxy_hat

    UxxUxx_hat = multiply_dealias_spectral(Uxx_hat, Uxx_hat)
    UyyUyy_hat = multiply_dealias_spectral(Uyy_hat, Uyy_hat)
    UxyUxy_hat = multiply_dealias_spectral(Uxy_hat, Uxy_hat)
    UxxVxx_hat = multiply_dealias_spectral(Uxx_hat, Vxx_hat)
    UxyVxy_hat = multiply_dealias_spectral(Uxy_hat, Vxy_hat)
    UyyVyy_hat = multiply_dealias_spectral(Uyy_hat, Vyy_hat)
    VxxVxx_hat = multiply_dealias_spectral(Vxx_hat, Vxx_hat)
    VyyVyy_hat = multiply_dealias_spectral(Vyy_hat, Vyy_hat)
    VxyVxy_hat = multiply_dealias_spectral(Vxy_hat, Vxy_hat)

    Tau11GM4_hat = Tau11GM2_hat + B1*(UxyUxy_hat) + B2*(UxxUxx_hat + UyyUyy_hat)
    Tau12GM4_hat = Tau12GM2_hat + B1*(UxyVxy_hat) + B2*(UxxVxx_hat + UyyVyy_hat)
    Tau22GM4_hat = Tau22GM2_hat + B1*(VxyVxy_hat) + B2*(VxxVxx_hat + VyyVyy_hat)

    return Tau11GM4_hat, Tau12GM4_hat, Tau22GM4_hat

def TauGM4_box(U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B3 = Delta**4 / 720

    Tau11GM2, Tau12GM2, Tau22GM2 = TauGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta) # GM2 box and gaussian are equal

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux

    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy

    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    # Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    # Vxxy = -Uxxx
    # Vxyy = -Uxxy
    Vyyy = -Uxyy

    Tau11GM4 = Tau11GM2 + B1*(Uxy**2) + B3*(Uxx**2 + Uyy**2 - 2*Ux*Uxxx - 2*Uy*Uyyy)
    Tau12GM4 = Tau12GM2 + B1*(Uxy*Vxy) + B3*(Uxx*Vxx + Uyy*Vyy - Ux*Vxxx -Uxxx*Vx - Uy*Vyyy - Uyyy*Vy)
    Tau22GM4 = Tau22GM2 + B1*(Vxy**2) + B3*(Vxx**2 + Vyy**2 - 2*Vx*Vxxx - 2*Vy*Vyyy)

    return Tau11GM4, Tau12GM4, Tau22GM4

def TauGM4_box_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B3 = Delta**4 / 720

    Tau11GM2_hat, Tau12GM2_hat, Tau22GM2_hat = TauGM2_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta) # GM2 box and gaussian are equal

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)
    Vy_hat = -Ux_hat

    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)
    Vxy_hat = -Uxx_hat
    Vyy_hat = -Uxy_hat

    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    # Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    # Vxxy_hat = -Uxxx_hat
    # Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat

    UxyUxy_hat = multiply_dealias_spectral(Uxy_hat, Uxy_hat)
    UxxUxx_hat = multiply_dealias_spectral(Uxx_hat, Uxx_hat)
    UyyUyy_hat = multiply_dealias_spectral(Uyy_hat, Uyy_hat)
    UxUxxx_hat = multiply_dealias_spectral(Ux_hat, Uxxx_hat)
    UyUyyy_hat = multiply_dealias_spectral(Uy_hat, Uyyy_hat)

    VxyVxy_hat = multiply_dealias_spectral(Vxy_hat, Vxy_hat)
    VxxVxx_hat = multiply_dealias_spectral(Vxx_hat, Vxx_hat)
    VyyVyy_hat = multiply_dealias_spectral(Vyy_hat, Vyy_hat)
    VxVxxx_hat = multiply_dealias_spectral(Vx_hat, Vxxx_hat)
    VyVyyy_hat = multiply_dealias_spectral(Vy_hat, Vyyy_hat)

    UxyVxy_hat = multiply_dealias_spectral(Uxy_hat, Vxy_hat)
    UxxVxx_hat = multiply_dealias_spectral(Uxx_hat, Vxx_hat)
    UyyVyy_hat = multiply_dealias_spectral(Uyy_hat, Vyy_hat)
    UxVxxx_hat = multiply_dealias_spectral(Ux_hat, Vxxx_hat)
    UyVyyy_hat = multiply_dealias_spectral(Uy_hat, Vyyy_hat)
    UxxxVx_hat = multiply_dealias_spectral(Uxxx_hat, Vx_hat)
    UyyyVy_hat = multiply_dealias_spectral(Uyyy_hat, Vy_hat)

    Tau11GM4_hat = Tau11GM2_hat + B1*(UxyUxy_hat) + B3*(UxxUxx_hat + UyyUyy_hat - 2*UxUxxx_hat - 2*UyUyyy_hat)
    Tau12GM4_hat = Tau12GM2_hat + B1*(UxyVxy_hat) + B3*(UxxVxx_hat + UyyVyy_hat - UxVxxx_hat - UxxxVx_hat - UyVyyy_hat - UyyyVy_hat)
    Tau22GM4_hat = Tau22GM2_hat + B1*(VxyVxy_hat) + B3*(VxxVxx_hat + VyyVyy_hat - 2*VxVxxx_hat - 2*VyVyyy_hat)

    return Tau11GM4_hat, Tau12GM4_hat, Tau22GM4_hat

# TauGM6
def TauGM6(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if dealias:
        if filterType=='gaussian':
            Tau11GM6_hat, Tau12GM6_hat, Tau22GM6_hat = TauGM6_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)
        elif filterType=='box':
            Tau11GM6_hat, Tau12GM6_hat, Tau22GM6_hat = TauGM6_box_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)
        Tau11GM6 = real_irfft2(Tau11GM6_hat)
        Tau12GM6 = real_irfft2(Tau12GM6_hat)
        Tau22GM6 = real_irfft2(Tau22GM6_hat)
    else:
        if filterType=='gaussian':
            Tau11GM6, Tau12GM6, Tau22GM6 = TauGM6_gaussian(U_hat, V_hat, Kx, Ky, Delta)
        elif filterType=='box':
            Tau11GM6, Tau12GM6, Tau22GM6 = TauGM6_box(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11GM6_hat = np.fft.rfft2(Tau11GM6)
        Tau12GM6_hat = np.fft.rfft2(Tau12GM6)
        Tau22GM6_hat = np.fft.rfft2(Tau22GM6)
        return Tau11GM6_hat, Tau12GM6_hat, Tau22GM6_hat
    else:
        return Tau11GM6, Tau12GM6, Tau22GM6
    

def TauGM6_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    C1 = Delta**6 / 3456
    C2 = Delta**6 / 10368

    Tau11GM4, Tau12GM4, Tau22GM4 = TauGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy

    Tau11GM6 = Tau11GM4 + C1*(Uxxy*Uxxy + Uxyy*Uxyy) + C2*(Uxxx*Uxxx + Uyyy*Uyyy)
    Tau12GM6 = Tau12GM4 + C1*(Uxxy*Vxxy + Uxyy*Vxyy) + C2*(Uxxx*Vxxx + Uyyy*Vyyy)
    Tau22GM6 = Tau22GM4 + C1*(Vxxy*Vxxy + Vxyy*Vxyy) + C2*(Vxxx*Vxxx + Vyyy*Vyyy)

    return Tau11GM6, Tau12GM6, Tau22GM6


def TauGM6_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta):

    C1 = Delta**6 / 3456
    C2 = Delta**6 / 10368

    Tau11GM4_hat, Tau12GM4_hat, Tau22GM4_hat = TauGM4_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)

    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    Vxxy_hat = -Uxxx_hat
    Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat

    UxxyUxxy_hat = multiply_dealias_spectral(Uxxy_hat, Uxxy_hat)
    UxyyUxyy_hat = multiply_dealias_spectral(Uxyy_hat, Uxyy_hat)
    UxxxUxxx_hat = multiply_dealias_spectral(Uxxx_hat, Uxxx_hat)
    UyyyUyyy_hat = multiply_dealias_spectral(Uyyy_hat, Uyyy_hat)
    UxxyVxxy_hat = multiply_dealias_spectral(Uxxy_hat, Vxxy_hat)
    UxyyVxyy_hat = multiply_dealias_spectral(Uxyy_hat, Vxyy_hat)
    UxxxVxxx_hat = multiply_dealias_spectral(Uxxx_hat, Vxxx_hat)
    UyyyVyyy_hat = multiply_dealias_spectral(Uyyy_hat, Vyyy_hat)
    VxxyVxxy_hat = multiply_dealias_spectral(Vxxy_hat, Vxxy_hat)
    VxyyVxyy_hat = multiply_dealias_spectral(Vxyy_hat, Vxyy_hat)
    VxxxVxxx_hat = multiply_dealias_spectral(Vxxx_hat, Vxxx_hat)
    VyyyVyyy_hat = multiply_dealias_spectral(Vyyy_hat, Vyyy_hat)

    Tau11GM6_hat = Tau11GM4_hat + C1*(UxxyUxxy_hat + UxyyUxyy_hat) + C2*(UxxxUxxx_hat + UyyyUyyy_hat)
    Tau12GM6_hat = Tau12GM4_hat + C1*(UxxyVxxy_hat + UxyyVxyy_hat) + C2*(UxxxVxxx_hat + UyyyVyyy_hat)
    Tau22GM6_hat = Tau22GM4_hat + C1*(VxxyVxxy_hat + VxyyVxyy_hat) + C2*(VxxxVxxx_hat + VyyyVyyy_hat)

    return Tau11GM6_hat, Tau12GM6_hat, Tau22GM6_hat

def TauGM6_box(U_hat, V_hat, Kx, Ky, Delta):

    C3 = Delta**6 / 8640
    C4 = Delta**6 / 30240

    Tau11GM4, Tau12GM4, Tau22GM4 = TauGM4_box(U_hat, V_hat, Kx, Ky, Delta)

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux

    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy

    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy

    Uxxxx = real_irfft2(derivative_spectral(U_hat, [4, 0], Kx, Ky))
    Uxxxy = real_irfft2(derivative_spectral(U_hat, [3, 1], Kx, Ky))
    Uxxyy = real_irfft2(derivative_spectral(U_hat, [2, 2], Kx, Ky))
    Uxyyy = real_irfft2(derivative_spectral(U_hat, [1, 3], Kx, Ky))
    Uyyyy = real_irfft2(derivative_spectral(U_hat, [0, 4], Kx, Ky))
    Vxxxx = real_irfft2(derivative_spectral(V_hat, [4, 0], Kx, Ky))
    Vxxxy = -Uxxxx
    # Vxxyy = -Uxxxy
    Vxyyy = -Uxxyy
    Vyyyy = -Uxyyy

    Uxxxxx = real_irfft2(derivative_spectral(U_hat, [5, 0], Kx, Ky))
    # Uxxxxy = real_irfft2(derivative_spectral(U_hat, [4, 1], Kx, Ky))
    # Uxxxyy = real_irfft2(derivative_spectral(U_hat, [3, 2], Kx, Ky))
    # Uxxyyy = real_irfft2(derivative_spectral(U_hat, [2, 3], Kx, Ky))
    Uxyyyy = real_irfft2(derivative_spectral(U_hat, [1, 4], Kx, Ky))
    Uyyyyy = real_irfft2(derivative_spectral(U_hat, [0, 5], Kx, Ky))
    Vxxxxx = real_irfft2(derivative_spectral(V_hat, [5, 0], Kx, Ky))
    # Vxxxxy = -Uxxxxx
    # Vxxxyy = -Uxxxxy
    # Vxxyyy = -Uxxxyy
    # Vxyyyy = -Uxxyyy
    Vyyyyy = -Uxyyyy

    Tau11GM6 = Tau11GM4 + C3*(Uxxy**2 + Uxyy**2 - 2*Uxxxy*Uxy - 2*Uxyyy*Uxy) + C4*(
        Uxxx**2 + 2*Ux*Uxxxxx -2*Uxx*Uxxxx + Uyyy**2 + 2*Uy*Uyyyyy - 2*Uyy*Uyyyy)
    Tau12GM6 = Tau12GM4 + C3*(Uxxy*Vxxy + Uxyy*Vxyy - Uxxxy*Vxy - Uxy*Vxxxy - Uxyyy*Vxy - Uxy*Vxyyy) + C4*(
        Uxxx*Vxxx + Ux*Vxxxxx +Uxxxxx*Vx - Uxx*Vxxxx - Uxxxx*Vxx + Uyyy*Vyyy + Uy*Vyyyyy +Uyyyyy*Vy - Uyy*Vyyyy - Uyyyy*Vyy)
    Tau22GM6 = Tau22GM4 + C3*(Vxxy**2 + Vxyy**2 - 2*Vxxxy*Vxy - 2*Vxyyy*Vxy) + C4*(
        Vxxx**2 + 2*Vx*Vxxxxx -2*Vxx*Vxxxx + Vyyy**2 + 2*Vy*Vyyyyy - 2*Vyy*Vyyyy)
    
    return Tau11GM6, Tau12GM6, Tau22GM6

def TauGM6_box_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta):

    C3 = Delta**6 / 8640
    C4 = Delta**6 / 30240

    Tau11GM4_hat, Tau12GM4_hat, Tau22GM4_hat = TauGM4_box_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)
    Vy_hat = -Ux_hat

    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)
    Vxy_hat = -Uxx_hat
    Vyy_hat = -Uxy_hat

    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    Vxxy_hat = -Uxxx_hat
    Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat

    Uxxxx_hat = derivative_spectral(U_hat, [4, 0], Kx, Ky)
    Uxxxy_hat = derivative_spectral(U_hat, [3, 1], Kx, Ky)
    Uxxyy_hat = derivative_spectral(U_hat, [2, 2], Kx, Ky)
    Uxyyy_hat = derivative_spectral(U_hat, [1, 3], Kx, Ky)
    Uyyyy_hat = derivative_spectral(U_hat, [0, 4], Kx, Ky)
    Vxxxx_hat = derivative_spectral(V_hat, [4, 0], Kx, Ky)
    Vxxxy_hat = -Uxxxx_hat
    # Vxxyy_hat = - Uxxxy_hat
    Vxyyy_hat = -Uxxyy_hat
    Vyyyy_hat = -Uxyyy_hat

    Uxxxxx_hat = derivative_spectral(U_hat, [5, 0], Kx, Ky)
    # Uxxxxy_hat = derivative_spectral(U_hat, [4, 1], Kx, Ky)
    # Uxxxyy_hat = derivative_spectral(U_hat, [3, 2], Kx, Ky)
    # Uxxyyy_hat = derivative_spectral(U_hat, [2, 3], Kx, Ky)
    Uxyyyy_hat = derivative_spectral(U_hat, [1, 4], Kx, Ky)
    Uyyyyy_hat = derivative_spectral(U_hat, [0, 5], Kx, Ky)
    Vxxxxx_hat = derivative_spectral(V_hat, [5, 0], Kx, Ky)
    # Vxxxxy_hat = -Uxxxxx_hat
    # Vxxxyy_hat = -Uxxxxy_hat
    # Vxxyyy_hat = -Uxxxyy_hat
    # Vxyyyy_hat = -Uxxyyy_hat
    Vyyyyy_hat = -Uxyyyy_hat

    UxxyUxxy_hat = multiply_dealias_spectral(Uxxy_hat, Uxxy_hat)
    UxyyUxyy_hat = multiply_dealias_spectral(Uxyy_hat, Uxyy_hat)
    UxyUxxxy_hat = multiply_dealias_spectral(Uxy_hat, Uxxxy_hat)
    UxyUxyyy_hat = multiply_dealias_spectral(Uxy_hat, Uxyyy_hat)
    UxxxUxxx_hat = multiply_dealias_spectral(Uxxx_hat, Uxxx_hat)
    UxUxxxxx_hat = multiply_dealias_spectral(Ux_hat, Uxxxxx_hat)
    UxxUxxxx_hat = multiply_dealias_spectral(Uxx_hat, Uxxxx_hat)
    UyyyUyyy_hat = multiply_dealias_spectral(Uyyy_hat, Uyyy_hat)
    UyUyyyyy_hat = multiply_dealias_spectral(Uy_hat, Uyyyyy_hat)
    UyyUyyyy_hat = multiply_dealias_spectral(Uyy_hat, Uyyyy_hat)

    VxxyVxxy_hat = multiply_dealias_spectral(Vxxy_hat, Vxxy_hat)
    VxyyVxyy_hat = multiply_dealias_spectral(Vxyy_hat, Vxyy_hat)
    VxyVxxxy_hat = multiply_dealias_spectral(Vxy_hat, Vxxxy_hat)
    VxyVxyyy_hat = multiply_dealias_spectral(Vxy_hat, Vxyyy_hat)
    VxxxVxxx_hat = multiply_dealias_spectral(Vxxx_hat, Vxxx_hat)
    VxVxxxxx_hat = multiply_dealias_spectral(Vx_hat, Vxxxxx_hat)
    VxxVxxxx_hat = multiply_dealias_spectral(Vxx_hat, Vxxxx_hat)
    VyyyVyyy_hat = multiply_dealias_spectral(Vyyy_hat, Vyyy_hat)
    VyVyyyyy_hat = multiply_dealias_spectral(Vy_hat, Vyyyyy_hat)    
    VyyVyyyy_hat = multiply_dealias_spectral(Vyy_hat, Vyyyy_hat)

    UxxyVxxy_hat = multiply_dealias_spectral(Uxxy_hat, Vxxy_hat)
    UxyyVxyy_hat = multiply_dealias_spectral(Uxyy_hat, Vxyy_hat)
    UxxxyVxy_hat = multiply_dealias_spectral(Uxxxy_hat, Vxy_hat)
    UxyVxxxy_hat = multiply_dealias_spectral(Uxy_hat, Vxxxy_hat)
    UxyyyVxy_hat = multiply_dealias_spectral(Uxyyy_hat, Vxy_hat)
    UxyVxyyy_hat = multiply_dealias_spectral(Uxy_hat, Vxyyy_hat)
    UxxxVxxx_hat = multiply_dealias_spectral(Uxxx_hat, Vxxx_hat)
    UxVxxxxx_hat = multiply_dealias_spectral(Ux_hat, Vxxxxx_hat)
    UxxxxxVx_hat = multiply_dealias_spectral(Uxxxxx_hat, Vx_hat)
    UxxVxxxx_hat = multiply_dealias_spectral(Uxx_hat, Vxxxx_hat)
    UxxxxVxx_hat = multiply_dealias_spectral(Uxxxx_hat, Vxx_hat)
    UyyyVyyy_hat = multiply_dealias_spectral(Uyyy_hat, Vyyy_hat)
    UyVyyyyy_hat = multiply_dealias_spectral(Uy_hat, Vyyyyy_hat)
    UyyyyyVy_hat = multiply_dealias_spectral(Uyyyyy_hat, Vy_hat)
    UyyVyyyy_hat = multiply_dealias_spectral(Uyy_hat, Vyyyy_hat)
    UyyyyVyy_hat = multiply_dealias_spectral(Uyyyy_hat, Vyy_hat)

    Tau11GM6_hat = Tau11GM4_hat + C3*(UxxyUxxy_hat + UxyyUxyy_hat - 2*UxyUxxxy_hat - 2*UxyUxyyy_hat) + C4*(
        UxxxUxxx_hat + 2*UxUxxxxx_hat - 2*UxxUxxxx_hat + UyyyUyyy_hat + 2*UyUyyyyy_hat - 2*UyyUyyyy_hat)
    Tau12GM6_hat = Tau12GM4_hat + C3*(UxxyVxxy_hat + UxyyVxyy_hat - UxxxyVxy_hat - UxyVxxxy_hat - UxyyyVxy_hat - UxyVxyyy_hat) + C4*(
        UxxxVxxx_hat + UxVxxxxx_hat + UxxxxxVx_hat - UxxVxxxx_hat - UxxxxVxx_hat + UyyyVyyy_hat + UyVyyyyy_hat + UyyyyyVy_hat - UyyVyyyy_hat - UyyyyVyy_hat)
    Tau22GM6_hat = Tau22GM4_hat + C3*(VxxyVxxy_hat + VxyyVxyy_hat - 2*VxyVxxxy_hat - 2*VxyVxyyy_hat) + C4*(
        VxxxVxxx_hat + 2*VxVxxxxx_hat - 2*VxxVxxxx_hat + VyyyVyyy_hat + 2*VyVyyyyy_hat - 2*VyyVyyyy_hat)

    return Tau11GM6_hat, Tau12GM6_hat, Tau22GM6_hat


# TauGM8
def TauGM8(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        Tau11GM8, Tau12GM8, Tau22GM8 = TauGM8_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11GM8_hat = np.fft.rfft2(Tau11GM8)
        Tau12GM8_hat = np.fft.rfft2(Tau12GM8)
        Tau22GM8_hat = np.fft.rfft2(Tau22GM8)
        return Tau11GM8_hat, Tau12GM8_hat, Tau22GM8_hat
    else:
        return Tau11GM8, Tau12GM8, Tau22GM8

 # TauGM10
def TauGM10(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        Tau11GM10, Tau12GM10, Tau22GM10 = TauGM10_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11GM10_hat = np.fft.rfft2(Tau11GM10)
        Tau12GM10_hat = np.fft.rfft2(Tau12GM10)
        Tau22GM10_hat = np.fft.rfft2(Tau22GM10)
        return Tau11GM10_hat, Tau12GM10_hat, Tau22GM10_hat
    else:
        return Tau11GM10, Tau12GM10, Tau22GM10


##############################################################################################################

# SigmaGM2
def SigmaGM2(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.rfft2(Omega)
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian' or filterType=='box':
        # GM2 for gaussian and box is same
        if dealias:
            Sigma1GM2_hat, Sigma2GM2_hat = SigmaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
            Sigma1GM2 = real_irfft2(Sigma1GM2_hat)
            Sigma2GM2 = real_irfft2(Sigma2GM2_hat)
        else:
            Sigma1GM2, Sigma2GM2 = SigmaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Sigma1GM2_hat = np.fft.rfft2(Sigma1GM2)
        Sigma2GM2_hat = np.fft.rfft2(Sigma2GM2)
        return Sigma1GM2_hat, Sigma2GM2_hat
    else:
        return Sigma1GM2, Sigma2GM2
    

def SigmaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased
    A1 = Delta**2 / 12

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux

    Omegax = real_irfft2(derivative_spectral(Omega_hat, [1, 0], Kx, Ky))
    Omegay = real_irfft2(derivative_spectral(Omega_hat, [0, 1], Kx, Ky))

    Sigma1GM2 = A1 * (Ux*Omegax + Uy*Omegay)
    Sigma2GM2 = A1 * (Vx*Omegax + Vy*Omegay)

    return Sigma1GM2, Sigma2GM2


def SigmaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # dealiased
    A1 = Delta**2 / 12

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)
    Vy_hat = -Ux_hat

    Omegax_hat = derivative_spectral(Omega_hat, [1, 0], Kx, Ky)
    Omegay_hat = derivative_spectral(Omega_hat, [0, 1], Kx, Ky)

    UxOmegax_hat = multiply_dealias_spectral(Ux_hat, Omegax_hat)
    UyOmegay_hat = multiply_dealias_spectral(Uy_hat, Omegay_hat)
    VxOmegax_hat = multiply_dealias_spectral(Vx_hat, Omegax_hat)
    VyOmegay_hat = multiply_dealias_spectral(Vy_hat, Omegay_hat)

    Sigma1GM2_hat = A1 * (UxOmegax_hat + UyOmegay_hat)
    Sigma2GM2_hat = A1 * (VxOmegax_hat + VyOmegay_hat)

    return Sigma1GM2_hat, Sigma2GM2_hat

# SigmaGM4
def SigmaGM4(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.rfft2(Omega)
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if dealias:
        if filterType=='gaussian':
            Sigma1GM4_hat, Sigma2GM4_hat = SigmaGM4_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        elif filterType=='box':
            Sigma1GM4_hat, Sigma2GM4_hat = SigmaGM4_box_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        Sigma1GM4 = real_irfft2(Sigma1GM4_hat)
        Sigma2GM4 = real_irfft2(Sigma2GM4_hat)
    else:
        if filterType=='gaussian':
            Sigma1GM4, Sigma2GM4 = SigmaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        elif filterType=='box':
            Sigma1GM4, Sigma2GM4 = SigmaGM4_box(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Sigma1GM4_hat = np.fft.rfft2(Sigma1GM4)
        Sigma2GM4_hat = np.fft.rfft2(Sigma2GM4)
        return Sigma1GM4_hat, Sigma2GM4_hat
    else:
        return Sigma1GM4, Sigma2GM4
    

def SigmaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased
    B1 = Delta**4 / 144
    B2 = Delta**4 / 288

    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy

    Omegaxx = real_irfft2(derivative_spectral(Omega_hat, [2, 0], Kx, Ky))
    Omegaxy = real_irfft2(derivative_spectral(Omega_hat, [1, 1], Kx, Ky))
    Omegayy = real_irfft2(derivative_spectral(Omega_hat, [0, 2], Kx, Ky))

    Sigma1GM2, Sigma2GM2 = SigmaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Sigma1GM4 = Sigma1GM2 + B1*(Uxy*Omegaxy) + B2*(Uxx*Omegaxx + Uyy*Omegayy)
    Sigma2GM4 = Sigma2GM2 + B1*(Vxy*Omegaxy) + B2*(Vxx*Omegaxx + Vyy*Omegayy)

    return Sigma1GM4, Sigma2GM4


def SigmaGM4_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased
    B1 = Delta**4 / 144
    B2 = Delta**4 / 288

    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)
    Vxy_hat = -Uxx_hat
    Vyy_hat = -Uxy_hat

    Omegaxx_hat = derivative_spectral(Omega_hat, [2, 0], Kx, Ky)
    Omegaxy_hat = derivative_spectral(Omega_hat, [1, 1], Kx, Ky)
    Omegayy_hat = derivative_spectral(Omega_hat, [0, 2], Kx, Ky)

    Sigma1GM2_hat, Sigma2GM2_hat = SigmaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    UxyOmegaxy_hat = multiply_dealias_spectral(Uxy_hat, Omegaxy_hat)
    UxxOmegaxx_hat = multiply_dealias_spectral(Uxx_hat, Omegaxx_hat)
    UyyOmegayy_hat = multiply_dealias_spectral(Uyy_hat, Omegayy_hat)
    VxyOmegaxy_hat = multiply_dealias_spectral(Vxy_hat, Omegaxy_hat)
    VxxOmegaxx_hat = multiply_dealias_spectral(Vxx_hat, Omegaxx_hat)
    VyyOmegayy_hat = multiply_dealias_spectral(Vyy_hat, Omegayy_hat)

    Sigma1GM4_hat = Sigma1GM2_hat + B1*(UxyOmegaxy_hat) + B2*(UxxOmegaxx_hat + UyyOmegayy_hat)
    Sigma2GM4_hat = Sigma2GM2_hat + B1*(VxyOmegaxy_hat) + B2*(VxxOmegaxx_hat + VyyOmegayy_hat)

    return Sigma1GM4_hat, Sigma2GM4_hat

def SigmaGM4_box(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B3 = Delta**4 / 720

    Sigma1GM2, Sigma2GM2 = SigmaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta) # same for box and gaussian

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux

    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy

    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    # Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    # Vxxy = -Uxxx
    # Vxyy = -Uxxy
    Vyyy = -Uxyy

    Omegax = real_irfft2(derivative_spectral(Omega_hat, [1, 0], Kx, Ky))
    Omegay = real_irfft2(derivative_spectral(Omega_hat, [0, 1], Kx, Ky))

    Omegaxx = real_irfft2(derivative_spectral(Omega_hat, [2, 0], Kx, Ky))
    Omegaxy = real_irfft2(derivative_spectral(Omega_hat, [1, 1], Kx, Ky))
    Omegayy = real_irfft2(derivative_spectral(Omega_hat, [0, 2], Kx, Ky))

    Omegaxxx = real_irfft2(derivative_spectral(Omega_hat, [3, 0], Kx, Ky))
    # Omegaxxy = real_irfft2(derivative_spectral(Omega_hat, [2, 1], Kx, Ky))
    # Omegaxyy = real_irfft2(derivative_spectral(Omega_hat, [1, 2], Kx, Ky))
    Omegayyy = real_irfft2(derivative_spectral(Omega_hat, [0, 3], Kx, Ky))

    Sigma1GM4 = Sigma1GM2 + B1*(Uxy*Omegaxy) + B3*(Uxx*Omegaxx + Uyy*Omegayy - Ux*Omegaxxx - Uy*Omegayyy + Uxxx*Omegax + Uyyy*Omegay)
    Sigma2GM4 = Sigma2GM2 + B1*(Vxy*Omegaxy) + B3*(Vxx*Omegaxx + Vyy*Omegayy - Vx*Omegaxxx - Vy*Omegayyy + Vxxx*Omegax + Vyyy*Omegay)

    return Sigma1GM4, Sigma2GM4

def SigmaGM4_box_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B3 = Delta**4 / 720

    Sigma1GM2_hat, Sigma2GM2_hat = SigmaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta) # same for box and gaussian

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)
    Vy_hat = -Ux_hat

    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)
    Vxy_hat = -Uxx_hat
    Vyy_hat = -Uxy_hat

    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    # Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    # Vxxy_hat = -Uxxx_hat
    # Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat

    Omegax_hat = derivative_spectral(Omega_hat, [1, 0], Kx, Ky)
    Omegay_hat = derivative_spectral(Omega_hat, [0, 1], Kx, Ky)

    Omegaxx_hat = derivative_spectral(Omega_hat, [2, 0], Kx, Ky)
    Omegaxy_hat = derivative_spectral(Omega_hat, [1, 1], Kx, Ky)
    Omegayy_hat = derivative_spectral(Omega_hat, [0, 2], Kx, Ky)

    Omegaxxx_hat = derivative_spectral(Omega_hat, [3, 0], Kx, Ky)
    # Omegaxxy_hat = derivative_spectral(Omega_hat, [2, 1], Kx, Ky)
    # Omegaxyy_hat = derivative_spectral(Omega_hat, [1, 2], Kx, Ky)
    Omegayyy_hat = derivative_spectral(Omega_hat, [0, 3], Kx, Ky)

    UxyOmegaxy_hat = multiply_dealias_spectral(Uxy_hat, Omegaxy_hat)
    UxxOmegaxx_hat = multiply_dealias_spectral(Uxx_hat, Omegaxx_hat)
    UyyOmegayy_hat = multiply_dealias_spectral(Uyy_hat, Omegayy_hat)
    UxOmegaxxx_hat = multiply_dealias_spectral(Ux_hat, Omegaxxx_hat)
    UyOmegayyy_hat = multiply_dealias_spectral(Uy_hat, Omegayyy_hat)
    UxxxOmegax_hat = multiply_dealias_spectral(Uxxx_hat, Omegax_hat)
    UyyyOmegay_hat = multiply_dealias_spectral(Uyyy_hat, Omegay_hat)

    VxyOmegaxy_hat = multiply_dealias_spectral(Vxy_hat, Omegaxy_hat)
    VxxOmegaxx_hat = multiply_dealias_spectral(Vxx_hat, Omegaxx_hat)
    VyyOmegayy_hat = multiply_dealias_spectral(Vyy_hat, Omegayy_hat)
    VxOmegaxxx_hat = multiply_dealias_spectral(Vx_hat, Omegaxxx_hat)
    VyOmegayyy_hat = multiply_dealias_spectral(Vy_hat, Omegayyy_hat)
    VxxxOmegax_hat = multiply_dealias_spectral(Vxxx_hat, Omegax_hat)
    VyyyOmegay_hat = multiply_dealias_spectral(Vyyy_hat, Omegay_hat)

    Sigma1GM4_hat = Sigma1GM2_hat + B1*(UxyOmegaxy_hat) + B3*(UxxOmegaxx_hat + UyyOmegayy_hat - UxOmegaxxx_hat - UyOmegayyy_hat + UxxxOmegax_hat + UyyyOmegay_hat)
    Sigma2GM4_hat = Sigma2GM2_hat + B1*(VxyOmegaxy_hat) + B3*(VxxOmegaxx_hat + VyyOmegayy_hat - VxOmegaxxx_hat - VyOmegayyy_hat + VxxxOmegax_hat + VyyyOmegay_hat)

    return Sigma1GM4_hat, Sigma2GM4_hat

# SigmaGM6
def SigmaGM6(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.rfft2(Omega)
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if dealias:
        if filterType=='gaussian':
            Sigma1GM6_hat, Sigma2GM6_hat = SigmaGM6_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        elif filterType=='box':
            Sigma1GM6_hat, Sigma2GM6_hat = SigmaGM6_box_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        Sigma1GM6 = real_irfft2(Sigma1GM6_hat)
        Sigma2GM6 = real_irfft2(Sigma2GM6_hat)
    else:
        if filterType == 'gaussian':
            Sigma1GM6, Sigma2GM6 = SigmaGM6_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
        elif filterType == 'box':
            Sigma1GM6, Sigma2GM6 = SigmaGM6_box(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Sigma1GM6_hat = np.fft.rfft2(Sigma1GM6)
        Sigma2GM6_hat = np.fft.rfft2(Sigma2GM6)
        return Sigma1GM6_hat, Sigma2GM6_hat
    else:
        return Sigma1GM6, Sigma2GM6
    
def SigmaGM6_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealias
    C1 = Delta**6 / 3456
    C2 = Delta**6 / 10368

    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy

    Omegaxxx = real_irfft2(derivative_spectral(Omega_hat, [3, 0], Kx, Ky))
    Omegaxxy = real_irfft2(derivative_spectral(Omega_hat, [2, 1], Kx, Ky))
    Omegaxyy = real_irfft2(derivative_spectral(Omega_hat, [1, 2], Kx, Ky))
    Omegayyy = real_irfft2(derivative_spectral(Omega_hat, [0, 3], Kx, Ky))

    Sigma1GM4, Sigma2GM4 = SigmaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Sigma1GM6 = Sigma1GM4 + C1*(Uxxy*Omegaxxy + Uxyy*Omegaxyy) + C2*(Uxxx*Omegaxxx + Uyyy*Omegayyy)
    Sigma2GM6 = Sigma2GM4 + C1*(Vxxy*Omegaxxy + Vxyy*Omegaxyy) + C2*(Vxxx*Omegaxxx + Vyyy*Omegayyy)

    return Sigma1GM6, Sigma2GM6

def SigmaGM6_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # dealias
    C1 = Delta**6 / 3456
    C2 = Delta**6 / 10368

    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    Vxxy_hat = -Uxxx_hat
    Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat

    Omegaxxx_hat = derivative_spectral(Omega_hat, [3, 0], Kx, Ky)
    Omegaxxy_hat = derivative_spectral(Omega_hat, [2, 1], Kx, Ky)
    Omegaxyy_hat = derivative_spectral(Omega_hat, [1, 2], Kx, Ky)
    Omegayyy_hat = derivative_spectral(Omega_hat, [0, 3], Kx, Ky)

    UxxyOmegaxxy_hat = multiply_dealias_spectral(Uxxy_hat, Omegaxxy_hat)
    UxyyOmegaxyy_hat = multiply_dealias_spectral(Uxyy_hat, Omegaxyy_hat)
    UxxxOmegaxxx_hat = multiply_dealias_spectral(Uxxx_hat, Omegaxxx_hat)
    UyyyOmegayyy_hat = multiply_dealias_spectral(Uyyy_hat, Omegayyy_hat)
    VxxyOmegaxxy_hat = multiply_dealias_spectral(Vxxy_hat, Omegaxxy_hat)
    VxyyOmegaxyy_hat = multiply_dealias_spectral(Vxyy_hat, Omegaxyy_hat)
    VxxxOmegaxxx_hat = multiply_dealias_spectral(Vxxx_hat, Omegaxxx_hat)
    VyyyOmegayyy_hat = multiply_dealias_spectral(Vyyy_hat, Omegayyy_hat)

    Sigma1GM4_hat, Sigma2GM4_hat = SigmaGM4_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Sigma1GM6_hat = Sigma1GM4_hat + C1*(UxxyOmegaxxy_hat + UxyyOmegaxyy_hat) + C2*(UxxxOmegaxxx_hat + UyyyOmegayyy_hat)
    Sigma2GM6_hat = Sigma2GM4_hat + C1*(VxxyOmegaxxy_hat + VxyyOmegaxyy_hat) + C2*(VxxxOmegaxxx_hat + VyyyOmegayyy_hat)

    return Sigma1GM6_hat, Sigma2GM6_hat

def SigmaGM6_box(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    C3 = Delta**6 / 8640
    C4 = Delta**6 / 30240

    Sigma1GM4, Sigma2GM4 = SigmaGM4_box(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux

    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy

    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy

    Uxxxx = real_irfft2(derivative_spectral(U_hat, [4, 0], Kx, Ky))
    Uxxxy = real_irfft2(derivative_spectral(U_hat, [3, 1], Kx, Ky))
    Uxxyy = real_irfft2(derivative_spectral(U_hat, [2, 2], Kx, Ky))
    Uxyyy = real_irfft2(derivative_spectral(U_hat, [1, 3], Kx, Ky))
    Uyyyy = real_irfft2(derivative_spectral(U_hat, [0, 4], Kx, Ky))
    Vxxxx = real_irfft2(derivative_spectral(V_hat, [4, 0], Kx, Ky))
    Vxxxy = -Uxxxx
    # Vxxyy = -Uxxxy
    Vxyyy = -Uxxyy
    Vyyyy = -Uxyyy

    Uxxxxx = real_irfft2(derivative_spectral(U_hat, [5, 0], Kx, Ky))
    # Uxxxxy = real_irfft2(derivative_spectral(U_hat, [4, 1], Kx, Ky))
    # Uxxxyy = real_irfft2(derivative_spectral(U_hat, [3, 2], Kx, Ky))
    # Uxxyyy = real_irfft2(derivative_spectral(U_hat, [2, 3], Kx, Ky))
    Uxyyyy = real_irfft2(derivative_spectral(U_hat, [1, 4], Kx, Ky))
    Uyyyyy = real_irfft2(derivative_spectral(U_hat, [0, 5], Kx, Ky))
    Vxxxxx = real_irfft2(derivative_spectral(V_hat, [5, 0], Kx, Ky))
    # Vxxxxy = -Uxxxxx
    # Vxxxyy = -Uxxxxy
    # Vxxyyy = -Uxxxyy
    # Vxyyyy = -Uxxyyy
    Vyyyyy = -Uxyyyy

    Omegax = real_irfft2(derivative_spectral(Omega_hat, [1, 0], Kx, Ky))
    Omegay = real_irfft2(derivative_spectral(Omega_hat, [0, 1], Kx, Ky))

    Omegaxx = real_irfft2(derivative_spectral(Omega_hat, [2, 0], Kx, Ky))
    Omegaxy = real_irfft2(derivative_spectral(Omega_hat, [1, 1], Kx, Ky))
    Omegayy = real_irfft2(derivative_spectral(Omega_hat, [0, 2], Kx, Ky))

    Omegaxxx = real_irfft2(derivative_spectral(Omega_hat, [3, 0], Kx, Ky))
    Omegaxxy = real_irfft2(derivative_spectral(Omega_hat, [2, 1], Kx, Ky))
    Omegaxyy = real_irfft2(derivative_spectral(Omega_hat, [1, 2], Kx, Ky))
    Omegayyy = real_irfft2(derivative_spectral(Omega_hat, [0, 3], Kx, Ky))

    Omegaxxxx = real_irfft2(derivative_spectral(Omega_hat, [4, 0], Kx, Ky))
    Omegaxxxy = real_irfft2(derivative_spectral(Omega_hat, [3, 1], Kx, Ky))
    # Omegaxxyy = real_irfft2(derivative_spectral(Omega_hat, [2, 2], Kx, Ky))
    Omegaxyyy = real_irfft2(derivative_spectral(Omega_hat, [1, 3], Kx, Ky))
    Omegayyyy = real_irfft2(derivative_spectral(Omega_hat, [0, 4], Kx, Ky))

    Omegaxxxxx = real_irfft2(derivative_spectral(Omega_hat, [5, 0], Kx, Ky))
    # Omegaxxxxy = real_irfft2(derivative_spectral(Omega_hat, [4, 1], Kx, Ky))
    # Omegaxxxyy = real_irfft2(derivative_spectral(Omega_hat, [3, 2], Kx, Ky))
    # Omegaxxyyy = real_irfft2(derivative_spectral(Omega_hat, [2, 3], Kx, Ky))
    # Omegaxyyyy = real_irfft2(derivative_spectral(Omega_hat, [1, 4], Kx, Ky))
    Omegayyyyy = real_irfft2(derivative_spectral(Omega_hat, [0, 5], Kx, Ky))

    Sigma1GM6 = Sigma1GM4 + C3*(Uxxy*Omegaxxy + Uxyy*Omegaxyy - 
                                Uxxxy*Omegaxy - Uxyyy*Omegaxy - Uxy*Omegaxxxy - Uxy*Omegaxyyy) + C4*(
                                    Uxxx*Omegaxxx + Ux*Omegaxxxxx + Uxxxxx*Omegax - Uxx*Omegaxxxx - Uxxxx*Omegaxx - 
                                    Uyyy*Omegayyy + Uy*Omegayyyyy + Uyyyyy*Omegay - Uyy*Omegayyyy - Uyyyy*Omegayy)
    Sigma2GM6 = Sigma2GM4 + C3*(Vxxy*Omegaxxy + Vxyy*Omegaxyy -
                                Vxxxy*Omegaxy - Vxyyy*Omegaxy - Vxy*Omegaxxxy - Vxy*Omegaxyyy) + C4*(
                                    Vxxx*Omegaxxx + Vx*Omegaxxxxx + Vxxxxx*Omegax - Vxx*Omegaxxxx - Vxxxx*Omegaxx - 
                                    Vyyy*Omegayyy + Vy*Omegayyyyy + Vyyyyy*Omegay - Vyy*Omegayyyy - Vyyyy*Omegayy)
    
    return Sigma1GM6, Sigma2GM6

def SigmaGM6_box_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    C3 = Delta**6 / 8640
    C4 = Delta**6 / 30240

    Sigma1GM4_hat, Sigma2GM4_hat = SigmaGM4_box_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)
    Vy_hat = -Ux_hat

    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)
    Vxy_hat = -Uxx_hat
    Vyy_hat = -Uxy_hat

    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    Vxxy_hat = -Uxxx_hat
    Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat

    Uxxxx_hat = derivative_spectral(U_hat, [4, 0], Kx, Ky)
    Uxxxy_hat = derivative_spectral(U_hat, [3, 1], Kx, Ky)
    Uxxyy_hat = derivative_spectral(U_hat, [2, 2], Kx, Ky)
    Uxyyy_hat = derivative_spectral(U_hat, [1, 3], Kx, Ky)
    Uyyyy_hat = derivative_spectral(U_hat, [0, 4], Kx, Ky)
    Vxxxx_hat = derivative_spectral(V_hat, [4, 0], Kx, Ky)
    Vxxxy_hat = -Uxxxx_hat
    # Vxxyy_hat = - Uxxxy_hat
    Vxyyy_hat = -Uxxyy_hat
    Vyyyy_hat = -Uxyyy_hat

    Uxxxxx_hat = derivative_spectral(U_hat, [5, 0], Kx, Ky)
    # Uxxxxy_hat = derivative_spectral(U_hat, [4, 1], Kx, Ky)
    # Uxxxyy_hat = derivative_spectral(U_hat, [3, 2], Kx, Ky)
    # Uxxyyy_hat = derivative_spectral(U_hat, [2, 3], Kx, Ky)
    Uxyyyy_hat = derivative_spectral(U_hat, [1, 4], Kx, Ky)
    Uyyyyy_hat = derivative_spectral(U_hat, [0, 5], Kx, Ky)
    Vxxxxx_hat = derivative_spectral(V_hat, [5, 0], Kx, Ky)
    # Vxxxxy_hat = -Uxxxxx_hat
    # Vxxxyy_hat = -Uxxxxy_hat
    # Vxxyyy_hat = -Uxxxyy_hat
    # Vxyyyy_hat = -Uxxyyy_hat
    Vyyyyy_hat = -Uxyyyy_hat

    Omegax_hat = derivative_spectral(Omega_hat, [1, 0], Kx, Ky)
    Omegay_hat = derivative_spectral(Omega_hat, [0, 1], Kx, Ky)

    Omegaxx_hat = derivative_spectral(Omega_hat, [2, 0], Kx, Ky)
    Omegaxy_hat = derivative_spectral(Omega_hat, [1, 1], Kx, Ky)
    Omegayy_hat = derivative_spectral(Omega_hat, [0, 2], Kx, Ky)

    Omegaxxx_hat = derivative_spectral(Omega_hat, [3, 0], Kx, Ky)
    Omegaxxy_hat = derivative_spectral(Omega_hat, [2, 1], Kx, Ky)
    Omegaxyy_hat = derivative_spectral(Omega_hat, [1, 2], Kx, Ky)
    Omegayyy_hat = derivative_spectral(Omega_hat, [0, 3], Kx, Ky)

    Omegaxxxx_hat = derivative_spectral(Omega_hat, [4, 0], Kx, Ky)
    Omegaxxxy_hat = derivative_spectral(Omega_hat, [3, 1], Kx, Ky)
    # Omegaxxyy_hat = derivative_spectral(Omega_hat, [2, 2], Kx, Ky)
    Omegaxyyy_hat = derivative_spectral(Omega_hat, [1, 3], Kx, Ky)
    Omegayyyy_hat = derivative_spectral(Omega_hat, [0, 4], Kx, Ky)

    Omegaxxxxx_hat = derivative_spectral(Omega_hat, [5, 0], Kx, Ky)
    # Omegaxxxxy_hat = derivative_spectral(Omega_hat, [4, 1], Kx, Ky)
    # Omegaxxxyy_hat = derivative_spectral(Omega_hat, [3, 2], Kx, Ky)
    # Omegaxxyyy_hat = derivative_spectral(Omega_hat, [2, 3], Kx, Ky)
    # Omegaxyyyy_hat = derivative_spectral(Omega_hat, [1, 4], Kx, Ky)
    Omegayyyyy_hat = derivative_spectral(Omega_hat, [0, 5], Kx, Ky)

    UxxyOmegaxxy_hat = multiply_dealias_spectral(Uxxy_hat, Omegaxxy_hat)
    UxyyOmegaxyy_hat = multiply_dealias_spectral(Uxyy_hat, Omegaxyy_hat)
    UxxxyOmegaxy_hat = multiply_dealias_spectral(Uxxxy_hat, Omegaxy_hat)
    UxyyyOmegaxy_hat = multiply_dealias_spectral(Uxyyy_hat, Omegaxy_hat)
    UxyOmegaxxxy_hat = multiply_dealias_spectral(Uxy_hat, Omegaxxxy_hat)
    UxyOmegaxyyy_hat = multiply_dealias_spectral(Uxy_hat, Omegaxyyy_hat)
    UxxxOmegaxxx_hat = multiply_dealias_spectral(Uxxx_hat, Omegaxxx_hat)
    UxOmegaxxxxx_hat = multiply_dealias_spectral(Ux_hat, Omegaxxxxx_hat)
    UxxxxxOmegax_hat = multiply_dealias_spectral(Uxxxxx_hat, Omegax_hat)
    UxxOmegaxxxx_hat = multiply_dealias_spectral(Uxx_hat, Omegaxxxx_hat)
    UxxxxOmegaxx_hat = multiply_dealias_spectral(Uxxxx_hat, Omegaxx_hat)
    UyyyOmegayyy_hat = multiply_dealias_spectral(Uyyy_hat, Omegayyy_hat)
    UyOmegayyyyy_hat = multiply_dealias_spectral(Uy_hat, Omegayyyyy_hat)
    UyyyyyOmegay_hat = multiply_dealias_spectral(Uyyyyy_hat, Omegay_hat)
    UyyOmegayyyy_hat = multiply_dealias_spectral(Uyy_hat, Omegayyyy_hat)
    UyyyyOmegayy_hat = multiply_dealias_spectral(Uyyyy_hat, Omegayy_hat)

    VxxyOmegaxxy_hat = multiply_dealias_spectral(Vxxy_hat, Omegaxxy_hat)
    VxyyOmegaxyy_hat = multiply_dealias_spectral(Vxyy_hat, Omegaxyy_hat)
    VxxxyOmegaxy_hat = multiply_dealias_spectral(Vxxxy_hat, Omegaxy_hat)
    VxyyyOmegaxy_hat = multiply_dealias_spectral(Vxyyy_hat, Omegaxy_hat)
    VxyOmegaxxxy_hat = multiply_dealias_spectral(Vxy_hat, Omegaxxxy_hat)
    VxyOmegaxyyy_hat = multiply_dealias_spectral(Vxy_hat, Omegaxyyy_hat)
    VxxxOmegaxxx_hat = multiply_dealias_spectral(Vxxx_hat, Omegaxxx_hat)
    VxOmegaxxxxx_hat = multiply_dealias_spectral(Vx_hat, Omegaxxxxx_hat)
    VxxxxxOmegax_hat = multiply_dealias_spectral(Vxxxxx_hat, Omegax_hat)
    VxxOmegaxxxx_hat = multiply_dealias_spectral(Vxx_hat, Omegaxxxx_hat)
    VxxxxOmegaxx_hat = multiply_dealias_spectral(Vxxxx_hat, Omegaxx_hat)
    VyyyOmegayyy_hat = multiply_dealias_spectral(Vyyy_hat, Omegayyy_hat)
    VyOmegayyyyy_hat = multiply_dealias_spectral(Vy_hat, Omegayyyyy_hat)
    VyyyyyOmegay_hat = multiply_dealias_spectral(Vyyyyy_hat, Omegay_hat)
    VyyOmegayyyy_hat = multiply_dealias_spectral(Vyy_hat, Omegayyyy_hat)
    VyyyyOmegayy_hat = multiply_dealias_spectral(Vyyyy_hat, Omegayy_hat)

    Sigma1GM6_hat = Sigma1GM4_hat + C3*(UxxyOmegaxxy_hat + UxyyOmegaxyy_hat - 
                                        UxxxyOmegaxy_hat - UxyyyOmegaxy_hat - UxyOmegaxxxy_hat - UxyOmegaxyyy_hat) + C4*(
                                            UxxxOmegaxxx_hat + UxOmegaxxxxx_hat + UxxxxxOmegax_hat - UxxOmegaxxxx_hat - UxxxxOmegaxx_hat - 
                                            UyyyOmegayyy_hat + UyOmegayyyyy_hat + UyyyyyOmegay_hat - UyyOmegayyyy_hat - UyyyyOmegayy_hat)
    Sigma2GM6_hat = Sigma2GM4_hat + C3*(VxxyOmegaxxy_hat + VxyyOmegaxyy_hat -
                                        VxxxyOmegaxy_hat - VxyyyOmegaxy_hat - VxyOmegaxxxy_hat - VxyOmegaxyyy_hat) + C4*(
                                            VxxxOmegaxxx_hat + VxOmegaxxxxx_hat + VxxxxxOmegax_hat - VxxOmegaxxxx_hat - VxxxxOmegaxx_hat - 
                                            VyyyOmegayyy_hat + VyOmegayyyyy_hat + VyyyyyOmegay_hat - VyyOmegayyyy_hat - VyyyyOmegayy_hat)
    
    return Sigma1GM6_hat, Sigma2GM6_hat

# SigmaGM8
def SigmaGM8(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.rfft2(Omega)
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        Sigma1GM8, Sigma2GM8 = SigmaGM8_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Sigma1GM8_hat = np.fft.rfft2(Sigma1GM8)
        Sigma2GM8_hat = np.fft.rfft2(Sigma2GM8)
        return Sigma1GM8_hat, Sigma2GM8_hat
    else:
        return Sigma1GM8, Sigma2GM8

# SigmaGM10
def SigmaGM10(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.rfft2(Omega)
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        Sigma1GM10, Sigma2GM10 = SigmaGM10_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Sigma1GM10_hat = np.fft.rfft2(Sigma1GM10)
        Sigma2GM10_hat = np.fft.rfft2(Sigma2GM10)
        return Sigma1GM10_hat, Sigma2GM10_hat
    else:
        return Sigma1GM10, Sigma2GM10

##############################################################################################################
# The gradient models (Taylor series expansion) for Leonard, Cross, and Reynolds - This decomoposition of SGS term was proposed by Germano(1986)
# TauLeonard
    
# TauLeonardGM2 
def TauLeonardGM2(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    # TauLeonardGM2 is equal to TauGM2
    Tau11Leonard, Tau12Leonard, Ta22Leonard = TauGM2(U, V, Kx, Ky, Delta, filterType=filterType, spectral=spectral, dealias=dealias)

    return Tau11Leonard, Tau12Leonard, Ta22Leonard

# TauLeonardGM4
def TauLeonardGM4(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        if dealias:
            Tau11LeonardGM4_hat, Tau12LeonardGM4_hat, Tau22LeonardGM4_hat = TauLeonardGM4_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)
            Tau11LeonardGM4 = real_irfft2(Tau11LeonardGM4_hat)
            Tau12LeonardGM4 = real_irfft2(Tau12LeonardGM4_hat)
            Tau22LeonardGM4 = real_irfft2(Tau22LeonardGM4_hat)
        else:
            Tau11LeonardGM4, Tau12LeonardGM4, Tau22LeonardGM4 = TauLeonardGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11LeonardGM4_hat = np.fft.rfft2(Tau11LeonardGM4)
        Tau12LeonardGM4_hat = np.fft.rfft2(Tau12LeonardGM4)
        Tau22LeonardGM4_hat = np.fft.rfft2(Tau22LeonardGM4)
        return Tau11LeonardGM4_hat, Tau12LeonardGM4_hat, Tau22LeonardGM4_hat
    else:
        return Tau11LeonardGM4, Tau12LeonardGM4, Tau22LeonardGM4


def TauLeonardGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B2 = Delta**4 / 288

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux
    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy
    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy

    TauLeonard11GM2, TauLeonard12GM2, TauLeonard22GM2 = TauGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta)   
    
    TauLeonard11GM4 = TauLeonard11GM2 + B1*(Uxy*Uxy + Ux*Uxxx + Uy*Uyyy + Ux*Uxyy + Uy*Uxxy) + B2*(Uxx*Uxx + Uyy*Uyy)
    TauLeonard12GM4 = TauLeonard12GM2 + B1*(Uxy*Vxy)  + B2*(Uxx*Vxx + Uyy*Vyy + Uxxx*Vx + Uyyy*Vy + Ux*Vxxx + Uy*Vyyy
                                              + Uxyy*Vx + Uxxy*Vy + Ux*Vxyy + Uy*Vxxy)
    TauLeonard22GM4 = TauLeonard22GM2 + B1*(Vxy*Vxy + Vx*Vxxx + Vy*Vyyy + Vx*Vxyy + Vy*Vxxy) + B2*(Vxx*Vxx + Vyy*Vyy)

    return TauLeonard11GM4, TauLeonard12GM4, TauLeonard22GM4
    

def TauLeonardGM4_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B2 = Delta**4 / 288


    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)
    Vy_hat = -Ux_hat
    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)
    Vxy_hat = -Uxx_hat
    Vyy_hat = -Uxy_hat
    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    Vxxy_hat = -Uxxx_hat
    Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat

    UxyUxy_hat = multiply_dealias_spectral(Uxy_hat, Uxy_hat)
    UxxUxx_hat = multiply_dealias_spectral(Uxx_hat, Uxx_hat)
    UyyUyy_hat = multiply_dealias_spectral(Uyy_hat, Uyy_hat)
    UxUxxx_hat = multiply_dealias_spectral(Ux_hat, Uxxx_hat)
    UyUyyy_hat = multiply_dealias_spectral(Uy_hat, Uyyy_hat)
    UxUxyy_hat = multiply_dealias_spectral(Ux_hat, Uxyy_hat)
    UyUxxy_hat = multiply_dealias_spectral(Uy_hat, Uxxy_hat)

    VxyVxy_hat = multiply_dealias_spectral(Vxy_hat, Vxy_hat)
    VxxVxx_hat = multiply_dealias_spectral(Vxx_hat, Vxx_hat)
    VyyVyy_hat = multiply_dealias_spectral(Vyy_hat, Vyy_hat)
    VxVxxx_hat = multiply_dealias_spectral(Vx_hat, Vxxx_hat)
    VyVyyy_hat = multiply_dealias_spectral(Vy_hat, Vyyy_hat)
    VxVxyy_hat = multiply_dealias_spectral(Vx_hat, Vxyy_hat)
    VyVxxy_hat = multiply_dealias_spectral(Vy_hat, Vxxy_hat)

    UxyVxy_hat = multiply_dealias_spectral(Uxy_hat, Vxy_hat)
    UxxVxx_hat = multiply_dealias_spectral(Uxx_hat, Vxx_hat)
    UyyVyy_hat = multiply_dealias_spectral(Uyy_hat, Vyy_hat)
    UxVxxx_hat = multiply_dealias_spectral(Ux_hat, Vxxx_hat)
    UxxxVx_hat = multiply_dealias_spectral(Uxxx_hat, Vx_hat)
    UyVyyy_hat = multiply_dealias_spectral(Uy_hat, Vyyy_hat)
    UyyyVy_hat = multiply_dealias_spectral(Uyyy_hat, Vy_hat)
    UxVxyy_hat = multiply_dealias_spectral(Ux_hat, Vxxy_hat)
    UyVxxy_hat = multiply_dealias_spectral(Uy_hat, Vxxy_hat)
    UxyyVx_hat = multiply_dealias_spectral(Uxyy_hat, Vx_hat)
    UxxyVy_hat = multiply_dealias_spectral(Uxxy_hat, Vy_hat)

    Tau11LeonardGM2_hat, Tau12LeonardGM2_hat, Tau22LeonardGM2_hat = TauGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    Tau11LeonardGM4_hat = Tau11LeonardGM2_hat + B1*(UxyUxy_hat + UxUxxx_hat + UyUyyy_hat + UxUxyy_hat + UyUxxy_hat) + B2*(UxxUxx_hat + UyyUyy_hat)
    Tau12LeonardGM4_hat = Tau12LeonardGM2_hat + B1*(UxyVxy_hat)  + B2*(UxxVxx_hat + UyyVyy_hat + UxVxxx_hat + UyVyyy_hat + UxxxVx_hat + UyyyVy_hat 
                                                                       + UxVxyy_hat + UyVxxy_hat + UxyyVx_hat + UxxyVy_hat)
    Tau22LeonardGM4_hat = Tau22LeonardGM2_hat + B1*(VxyVxy_hat + VxVxxx_hat + VyVyyy_hat + VxVxyy_hat + VyVxxy_hat) + B2*(VxxVxx_hat + VyyVyy_hat)
    
    return Tau11LeonardGM4_hat, Tau12LeonardGM4_hat, Tau22LeonardGM4_hat

# TauLeonardGM6
def TauLeonardGM6(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        if dealias:
            Tau11LeonardGM6_hat, Tau12LeonardGM6_hat, Tau22LeonardGM6_hat = TauLeonardGM6_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)
            Tau11LeonardGM6 = real_irfft2(Tau11LeonardGM6_hat)
            Tau12LeonardGM6 = real_irfft2(Tau12LeonardGM6_hat)
            Tau22LeonardGM6 = real_irfft2(Tau22LeonardGM6_hat)
        else:
            Tau11LeonardGM6, Tau12LeonardGM6, Tau22LeonardGM6 = TauLeonardGM6_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11LeonardGM6_hat = np.fft.rfft2(Tau11LeonardGM6)
        Tau12LeonardGM6_hat = np.fft.rfft2(Tau12LeonardGM6)
        Tau22LeonardGM6_hat = np.fft.rfft2(Tau22LeonardGM6)
        return Tau11LeonardGM6_hat, Tau12LeonardGM6_hat, Tau22LeonardGM6_hat
    else:
        return Tau11LeonardGM6, Tau12LeonardGM6, Tau22LeonardGM6
        

def TauLeonardGM6_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    C1 = Delta**6 / 1728
    C2 = Delta**6 / 2304
    C3 = Delta**6 / 3456
    C4 = Delta**6 / 6912
    C5 = Delta**6 / 10368
    C6 = Delta**6 / 13824
    C7 = Delta**6 / 20736

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux
    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy
    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy
    Uxxxx = real_irfft2(derivative_spectral(U_hat, [4, 0], Kx, Ky))
    Uxxxy = real_irfft2(derivative_spectral(U_hat, [3, 1], Kx, Ky))
    Uxxyy = real_irfft2(derivative_spectral(U_hat, [2, 2], Kx, Ky))
    Uxyyy = real_irfft2(derivative_spectral(U_hat, [1, 3], Kx, Ky))
    Uyyyy = real_irfft2(derivative_spectral(U_hat, [0, 4], Kx, Ky))
    Vxxxx = real_irfft2(derivative_spectral(V_hat, [4, 0], Kx, Ky))
    Vxxxy = -Uxxxx
    Vxxyy = -Uxxxy
    Vxyyy = -Uxxyy
    Vyyyy = -Uxyyy
    Uxxxxx = real_irfft2(derivative_spectral(U_hat, [5, 0], Kx, Ky))
    Uxxxxy = real_irfft2(derivative_spectral(U_hat, [4, 1], Kx, Ky))
    Uxxxyy = real_irfft2(derivative_spectral(U_hat, [3, 2], Kx, Ky))
    Uxxyyy = real_irfft2(derivative_spectral(U_hat, [2, 3], Kx, Ky))
    Uxyyyy = real_irfft2(derivative_spectral(U_hat, [1, 4], Kx, Ky))
    Uyyyyy = real_irfft2(derivative_spectral(U_hat, [0, 5], Kx, Ky))
    Vxxxxx = real_irfft2(derivative_spectral(V_hat, [5, 0], Kx, Ky))
    Vxxxxy = -Uxxxxx
    Vxxxyy = -Uxxxxx
    Vxxyyy = -Uxxxyy
    Vxyyyy = -Uxxyyy
    Vyyyyy = -Uxyyyy

    Tau11LeonardGM4, Tau12LeonardGM4, Tau22LeonardGM4 = TauLeonardGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    Tau11LeonardGM6 = Tau11LeonardGM4 + C1*(Uxy*Uxyyy + Uxy*Uxxxy) + C2*(Uxyy*Uxyy + Uxxy*Uxxy) +C3*(
        Uxx*Uxxxx + Uyy*Uyyyy + Uxxx*Uxyy + Uyyy*Uxxy + Uxx*Uxxyy + Uyy*Uxxyy + Ux*Uxxxyy + Uy*Uxxyyy) +C4*(
            Ux*Uxxxxx + Uy*Uyyyyy + Ux*Uxyyyy + Uy*Uxxxxy) + C7*(Uxxx*Uxxx + Uyyy*Uyyy)
    
    Tau12LeonardGM6 =  Tau12LeonardGM4 + C3*(Uxyyy*Vxy + Uxxxy*Vxy + Uxy*Vxyyy + Uxy*Vxxxy) + C4*(
        Uxxx*Vxyy + Uyyy*Vxxy + Uxyy*Vxxx + Uxxy*Vxxx  
        + Uxxxx*Vxx + Uxx*Vxxxx + Uyyyy*Vyy + Uyy*Vyyyy  + Uxxyy*Vxx + Uxxyy*Vyy + Uxx*Vxxyy + Uyy*Vxxyy 
        + Uxxyyy*Vy + Uxxxyy*Vx + Uy*Vxxyyy + Ux*Vxxxyy) + C6*(
            Uxxxxx*Vx + Uyyyyy*Vy + Uy*Vyyyyy + Ux*Vxxxxx + Uxyyyy*Vx + Uxxxxy*Vy + Ux*Vxyyyy + Uy*Vxxxxy) + 5*C7*(
                Uxxx*Vxxx + Uyyy*Vyyy)
    
    Tau22LeonardGM6 = Tau22LeonardGM4 + C1*(Vxy*Vxyyy + Vxy*Vxxxy) + C2*(Vxyy*Vxyy + Vxxy*Vxxy) +C3*(
        Vxx*Vxxxx + Vyy*Vyyyy + Vxxx*Vxyy + Vyyy*Vxxy + Vxx*Vxxyy + Vyy*Vxxyy + Vx*Vxxxyy + Vy*Vxxyyy) +C4*(
            Vx*Vxxxxx + Vy*Vyyyyy + Vx*Vxyyyy + Vy*Vxxxxy) + C7*(Vxxx*Vxxx + Vyyy*Vyyy)
    
    return Tau11LeonardGM6, Tau12LeonardGM6, Tau22LeonardGM6
    

def TauLeonardGM6_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta):

    C1 = Delta**6 / 1728
    C2 = Delta**6 / 2304
    C3 = Delta**6 / 3456
    C4 = Delta**6 / 6912
    C5 = Delta**6 / 10368
    C6 = Delta**6 / 13824
    C7 = Delta**6 / 20736

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)
    Vy_hat = -Ux_hat
    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)
    Vxy_hat = -Uxx_hat
    Vyy_hat = -Uxy_hat
    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    Vxxy_hat = -Uxxx_hat
    Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat
    Uxxxx_hat = derivative_spectral(U_hat, [4, 0], Kx, Ky)
    Uxxxy_hat = derivative_spectral(U_hat, [3, 1], Kx, Ky)
    Uxxyy_hat = derivative_spectral(U_hat, [2, 2], Kx, Ky)
    Uxyyy_hat = derivative_spectral(U_hat, [1, 3], Kx, Ky)
    Uyyyy_hat = derivative_spectral(U_hat, [0, 4], Kx, Ky)
    Vxxxx_hat = derivative_spectral(V_hat, [4, 0], Kx, Ky)
    Vxxxy_hat = -Uxxxx_hat
    Vxxyy_hat = -Uxxxy_hat
    Vxyyy_hat = -Uxxyy_hat
    Vyyyy_hat = -Uxyyy_hat
    Uxxxxx_hat = derivative_spectral(U_hat, [5, 0], Kx, Ky)
    Uxxxxy_hat = derivative_spectral(U_hat, [4, 1], Kx, Ky)
    Uxxxyy_hat = derivative_spectral(U_hat, [3, 2], Kx, Ky)
    Uxxyyy_hat = derivative_spectral(U_hat, [2, 3], Kx, Ky)
    Uxyyyy_hat = derivative_spectral(U_hat, [1, 4], Kx, Ky)
    Uyyyyy_hat = derivative_spectral(U_hat, [0, 5], Kx, Ky)
    Vxxxxx_hat = derivative_spectral(V_hat, [5, 0], Kx, Ky)
    Vxxxxy_hat = -Uxxxxx_hat
    Vxxxyy_hat = -Uxxxxx_hat
    Vxxyyy_hat = -Uxxxyy_hat
    Vxyyyy_hat = -Uxxyyy_hat
    Vyyyyy_hat = -Uxyyyy_hat

    UxyUxyyy_hat = multiply_dealias_spectral(Uxy_hat, Uxyy_hat)
    UxyUxxxy_hat = multiply_dealias_spectral(Uxy_hat, Uxx_hat)
    UxyyUxyy_hat = multiply_dealias_spectral(Uxyy_hat, Uxyy_hat)
    UxxyUxxy_hat = multiply_dealias_spectral(Uxxy_hat, Uxxy_hat)
    UxxUxxxx_hat = multiply_dealias_spectral(Uxx_hat, Uxxx_hat)
    UyyUyyyy_hat = multiply_dealias_spectral(Uyy_hat, Uyyy_hat)
    UxxxUxyy_hat = multiply_dealias_spectral(Uxxx_hat, Uxyy_hat)
    UyyyUxxy_hat = multiply_dealias_spectral(Uyyy_hat, Uxxy_hat)
    UxxUxxyy_hat = multiply_dealias_spectral(Uxx_hat, Uxxyy_hat)
    UyyUxxyy_hat = multiply_dealias_spectral(Uyy_hat, Uxxyy_hat)
    UxUxxxyy_hat = multiply_dealias_spectral(Ux_hat, Uxxxyy_hat)
    UyUxxyyy_hat = multiply_dealias_spectral(Uy_hat, Uxxyyy_hat)
    UxUxxxxx_hat = multiply_dealias_spectral(Ux_hat, Uxxxxx_hat)
    UyUyyyyy_hat = multiply_dealias_spectral(Uy_hat, Uyyyyy_hat)
    UxUxyyyy_hat = multiply_dealias_spectral(Ux_hat, Uxyyyy_hat)
    UyUxxxxy_hat = multiply_dealias_spectral(Uy_hat, Uxxxxy_hat)
    UxxxUxxx_hat = multiply_dealias_spectral(Uxxx_hat, Uxxx_hat)
    UyyyUyyy_hat = multiply_dealias_spectral(Uyyy_hat, Uyyy_hat)

    VxyVxyyy_hat = multiply_dealias_spectral(Vxy_hat, Vxyy_hat)
    VxyVxxxy_hat = multiply_dealias_spectral(Vxy_hat, Vxx_hat)
    VxyyVxyy_hat = multiply_dealias_spectral(Vxyy_hat, Vxyy_hat)
    VxxyVxxy_hat = multiply_dealias_spectral(Vxxy_hat, Vxxy_hat)
    VxxVxxxx_hat = multiply_dealias_spectral(Vxx_hat, Vxxx_hat)
    VyyVyyyy_hat = multiply_dealias_spectral(Vyy_hat, Vyyy_hat)
    VxxxVxyy_hat = multiply_dealias_spectral(Vxxx_hat, Vxyy_hat)
    VyyyVxxy_hat = multiply_dealias_spectral(Vyyy_hat, Vxxy_hat)
    VxxVxxyy_hat = multiply_dealias_spectral(Vxx_hat, Vxxyy_hat)
    VyyVxxyy_hat = multiply_dealias_spectral(Vyy_hat, Vxxyy_hat)
    VxVxxxyy_hat = multiply_dealias_spectral(Vx_hat, Vxxxyy_hat)
    VyVxxyyy_hat = multiply_dealias_spectral(Vy_hat, Vxxyyy_hat)
    VxVxxxxx_hat = multiply_dealias_spectral(Vx_hat, Vxxxxx_hat)
    VyVyyyyy_hat = multiply_dealias_spectral(Vy_hat, Vyyyyy_hat)
    VxVxyyyy_hat = multiply_dealias_spectral(Vx_hat, Vxyyyy_hat)
    VyVxxxxy_hat = multiply_dealias_spectral(Vy_hat, Vxxxxy_hat)
    VxxxVxxx_hat = multiply_dealias_spectral(Vxxx_hat, Vxxx_hat)
    VyyyVyyy_hat = multiply_dealias_spectral(Vyyy_hat, Vyyy_hat)

    UxyyyVxy_hat = multiply_dealias_spectral(Uxyyy_hat, Vxy_hat)
    UxxxyVxy_hat = multiply_dealias_spectral(Uxxxy_hat, Vxy_hat)
    UxyVxyyy_hat = multiply_dealias_spectral(Uxy_hat, Vxyyy_hat)
    UxyVxxxy_hat = multiply_dealias_spectral(Uxy_hat, Vxxxy_hat)
    UxxxVxyy_hat = multiply_dealias_spectral(Uxxx_hat, Vxyy_hat)
    UyyyVxxy_hat = multiply_dealias_spectral(Uyyy_hat, Vxxy_hat)
    UxyyVxxx_hat = multiply_dealias_spectral(Uxyy_hat, Vxxx_hat)
    UxxyVxxx_hat = multiply_dealias_spectral(Uxxy_hat, Vxxx_hat)
    UxxxxVxx_hat = multiply_dealias_spectral(Uxxxx_hat, Vxx_hat)
    UxxVxxxx_hat = multiply_dealias_spectral(Uxx_hat, Vxxxx_hat)
    UyyyyVyy_hat = multiply_dealias_spectral(Uyyyy_hat, Vyy_hat)
    UyyVyyyy_hat = multiply_dealias_spectral(Uyy_hat, Vyyyy_hat)
    UxxyyVxx_hat = multiply_dealias_spectral(Uxxyy_hat, Vxx_hat)
    UxxyyVyy_hat = multiply_dealias_spectral(Uxxyy_hat, Vyy_hat)
    UxxVxxyy_hat = multiply_dealias_spectral(Uxx_hat, Vxxyy_hat)
    UyyVxxyy_hat = multiply_dealias_spectral(Uyy_hat, Vxxyy_hat)
    UxxyyyVx_hat = multiply_dealias_spectral(Uxxyyy_hat, Vx_hat)
    UxxxyyVy_hat = multiply_dealias_spectral(Uxxxyy_hat, Vy_hat)
    UyVxxyyy_hat = multiply_dealias_spectral(Uy_hat, Vxxyyy_hat)
    UxVxxxyy_hat = multiply_dealias_spectral(Ux_hat, Vxxxyy_hat)
    UxxxxxVx_hat = multiply_dealias_spectral(Uxxxxx_hat, Vx_hat)
    UyyyyyVy_hat = multiply_dealias_spectral(Uyyyyy_hat, Vy_hat)
    UyVyyyyy_hat = multiply_dealias_spectral(Uy_hat, Vyyyyy_hat)
    UxVxxxxx_hat = multiply_dealias_spectral(Ux_hat, Vxxxxx_hat)
    UxyyyyVx_hat = multiply_dealias_spectral(Uxyyyy_hat, Vx_hat)
    UxxxxyVy_hat = multiply_dealias_spectral(Uxxxxy_hat, Vy_hat)
    UxVxyyyy_hat = multiply_dealias_spectral(Ux_hat, Vxyyyy_hat)
    UyVxxxxy_hat = multiply_dealias_spectral(Uy_hat, Vxxxxy_hat)
    UxxxVxxx_hat = multiply_dealias_spectral(Uxxx_hat, Vxxx_hat)
    UyyyVyyy_hat = multiply_dealias_spectral(Uyyy_hat, Vyyy_hat)

    Tau11LeonardGM4_hat, Tau12LeonardGM4_hat, Tau22LeonardGM4_hat = TauLeonardGM4_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)

    Tau11LeonardGM6_hat = Tau11LeonardGM4_hat + C1*(UxyUxyyy_hat + UxyUxxxy_hat) + C2*(UxyyUxyy_hat + UxxyUxxy_hat) +C3*(
        UxxUxxxx_hat + UyyUyyyy_hat + UxxxUxyy_hat + UyyyUxxy_hat + UxxUxxyy_hat + UyyUxxyy_hat + UxUxxxyy_hat + UyUxxyyy_hat) +C4*(
            UxUxxxxx_hat + UyUyyyyy_hat + UxUxyyyy_hat + UyUxxxxy_hat) + C7*(UxxxUxxx_hat + UyyyUyyy_hat)
    
    Tau12LeonardGM6_hat =  Tau12LeonardGM4_hat + C3*(UxyyyVxy_hat + UxxxyVxy_hat + UxyVxyyy_hat + UxyVxxxy_hat) + C4*(
        UxxxVxyy_hat + UyyyVxxy_hat + UxyyVxxx_hat + UxxyVxxx_hat  
        + UxxxxVxx_hat + UxxVxxxx_hat + UyyyyVyy_hat + UyyVyyyy_hat  + UxxyyVxx_hat + UxxyyVyy_hat + UxxVxxyy_hat + UyyVxxyy_hat 
        + UxxyyyVx_hat + UxxxyyVy_hat + UyVxxyyy_hat + UxVxxxyy_hat) + C6*(
            UxxxxxVx_hat + UyyyyyVy_hat + UyVyyyyy_hat + UxVxxxxx_hat + UxyyyyVx_hat + UxxxxyVy_hat + UxVxyyyy_hat + UyVxxxxy_hat) + 5*C7*(
                UxxxVxxx_hat + UyyyVyyy_hat)
    
    Tau22LeonardGM6_hat = Tau22LeonardGM4_hat + C1*(VxyVxyyy_hat + VxyVxxxy_hat) + C2*(VxyyVxyy_hat + VxxyVxxy_hat) +C3*(
        VxxVxxxx_hat + VyyVyyyy_hat + VxxxVxyy_hat + VyyyVxxy_hat + VxxVxxyy_hat + VyyVxxyy_hat + VxVxxxyy_hat + VyVxxyyy_hat) +C4*(
            VxVxxxxx_hat + VyVyyyyy_hat + VxVxyyyy_hat + VyVxxxxy_hat) + C7*(VxxxVxxx_hat + VyyyVyyy_hat)
    
    return Tau11LeonardGM6_hat, Tau12LeonardGM6_hat, Tau22LeonardGM6_hat

# TauCrossGM2
def TauCrossGM2(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):
    Tau11CrossGM2, Tau12CrossGM2, Tau22CrossGM2 = 0,0,0
    return Tau11CrossGM2, Tau12CrossGM2, Tau22CrossGM2

# TauCrossGM4
def TauCrossGM4(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):
    
    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        if dealias:
            Tau11CrossGM4_hat, Tau12CrossGM4_hat, Tau22CrossGM4_hat = TauCrossGM4_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)
            Tau11CrossGM4 = real_irfft2(Tau11CrossGM4_hat)
            Tau12CrossGM4 = real_irfft2(Tau12CrossGM4_hat)
            Tau22CrossGM4 = real_irfft2(Tau22CrossGM4_hat)
        else:
            Tau11CrossGM4, Tau12CrossGM4, Tau22CrossGM4 = TauCrossGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11CrossGM4_hat = np.fft.rfft2(Tau11CrossGM4)
        Tau12CrossGM4_hat = np.fft.rfft2(Tau12CrossGM4)
        Tau22CrossGM4_hat = np.fft.rfft2(Tau22CrossGM4)
        return Tau11CrossGM4_hat, Tau12CrossGM4_hat, Tau22CrossGM4_hat
    else:
        return Tau11CrossGM4, Tau12CrossGM4, Tau22CrossGM4
    

def TauCrossGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B2 = Delta**4 / 144

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux
    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy

    Tau11CrossGM4 = -B1*(Ux*Uxxx + Uy*Uyyy + Ux*Uxyy + Uy*Uxxy)
    Tau12CrossGM4 = -B2*(Uxxx*Vx + Uyyy*Vy + Ux*Vxxx + Uy*Vyyy + Uxyy*Vx + Uxxy*Vy + Ux*Vxyy + Uy*Vxxy)
    Tau22CrossGM4 = -B1*(Vx*Vxxx + Vy*Vyyy + Vx*Vxyy + Vy*Vxxy)

    return Tau11CrossGM4, Tau12CrossGM4, Tau22CrossGM4


def TauCrossGM4_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta):
    
    B1 = Delta**4 / 144
    B2 = Delta**4 / 144

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)
    Vy_hat = -Ux_hat
    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    Vxxy_hat = -Uxxx_hat
    Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat

    UxUxxx_hat = multiply_dealias_spectral(Ux_hat, Uxxx_hat)
    UyUyyy_hat = multiply_dealias_spectral(Uy_hat, Uyyy_hat)
    UxUxyy_hat = multiply_dealias_spectral(Ux_hat, Uxyy_hat)
    UyUxxy_hat = multiply_dealias_spectral(Uy_hat, Uxxy_hat)

    VxVxxx_hat = multiply_dealias_spectral(Vx_hat, Vxxx_hat)
    VyVyyy_hat = multiply_dealias_spectral(Vy_hat, Vyyy_hat)
    VxVxyy_hat = multiply_dealias_spectral(Vx_hat, Vxyy_hat)
    VyVxxy_hat = multiply_dealias_spectral(Vy_hat, Vxxy_hat)

    UxVxxx_hat = multiply_dealias_spectral(Ux_hat, Vxxx_hat)
    UxxxVx_hat = multiply_dealias_spectral(Uxxx_hat, Vx_hat)
    UyVyyy_hat = multiply_dealias_spectral(Uy_hat, Vyyy_hat)
    UyyyVy_hat = multiply_dealias_spectral(Uyyy_hat, Vy_hat)
    UxVxyy_hat = multiply_dealias_spectral(Ux_hat, Vxxy_hat)
    UyVxxy_hat = multiply_dealias_spectral(Uy_hat, Vxxy_hat)
    UxyyVx_hat = multiply_dealias_spectral(Uxyy_hat, Vx_hat)
    UxxyVy_hat = multiply_dealias_spectral(Uxxy_hat, Vy_hat)

    Tau11CrossGM4_hat = -B1*(UxUxxx_hat + UyUyyy_hat + UxUxyy_hat + UyUxxy_hat)
    Tau12CrossGM4_hat = -B2*(UxxxVx_hat + UyyyVy_hat + UxVxxx_hat + UyVyyy_hat + UxyyVx_hat + UxxyVy_hat + UxVxyy_hat + UyVxxy_hat)
    Tau22CrossGM4_hat = -B1*(VxVxxx_hat + VyVyyy_hat + VxVxyy_hat + VyVxxy_hat)

    return Tau11CrossGM4_hat, Tau12CrossGM4_hat, Tau22CrossGM4_hat

# TauCrossGM6
def TauCrossGM6(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):
    
    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        if dealias:
            Tau11CrossGM6_hat, Tau12CrossGM6_hat, Tau22CrossGM6_hat = TauCrossGM6_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)
            Tau11CrossGM6 = real_irfft2(Tau11CrossGM6_hat)
            Tau12CrossGM6 = real_irfft2(Tau12CrossGM6_hat)
            Tau22CrossGM6 = real_irfft2(Tau22CrossGM6_hat)
        else:
            Tau11CrossGM6, Tau12CrossGM6, Tau22CrossGM6 = TauCrossGM6_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11CrossGM6_hat = np.fft.rfft2(Tau11CrossGM6)
        Tau12CrossGM6_hat = np.fft.rfft2(Tau12CrossGM6)
        Tau22CrossGM6_hat = np.fft.rfft2(Tau22CrossGM6)
        return Tau11CrossGM6_hat, Tau12CrossGM6_hat, Tau22CrossGM6_hat
    else:
        return Tau11CrossGM6, Tau12CrossGM6, Tau22CrossGM6
    

def TauCrossGM6_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    C1 = Delta**6 / 1728
    C2 = Delta**6 / 2304
    C3 = Delta**6 / 3456
    C4 = Delta**6 / 6912
    C5 = Delta**6 / 10368
    C6 = Delta**6 / 13824
    C7 = Delta**6 / 20736

    Ux = real_irfft2(derivative_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_irfft2(derivative_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_irfft2(derivative_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux
    Uxx = real_irfft2(derivative_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_irfft2(derivative_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_irfft2(derivative_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_irfft2(derivative_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy
    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy
    Uxxxx = real_irfft2(derivative_spectral(U_hat, [4, 0], Kx, Ky))
    Uxxxy = real_irfft2(derivative_spectral(U_hat, [3, 1], Kx, Ky))
    Uxxyy = real_irfft2(derivative_spectral(U_hat, [2, 2], Kx, Ky))
    Uxyyy = real_irfft2(derivative_spectral(U_hat, [1, 3], Kx, Ky))
    Uyyyy = real_irfft2(derivative_spectral(U_hat, [0, 4], Kx, Ky))
    Vxxxx = real_irfft2(derivative_spectral(V_hat, [4, 0], Kx, Ky))
    Vxxxy = -Uxxxx
    Vxxyy = -Uxxxy
    Vxyyy = -Uxxyy
    Vyyyy = -Uxyyy
    Uxxxxx = real_irfft2(derivative_spectral(U_hat, [5, 0], Kx, Ky))
    Uxxxxy = real_irfft2(derivative_spectral(U_hat, [4, 1], Kx, Ky))
    Uxxxyy = real_irfft2(derivative_spectral(U_hat, [3, 2], Kx, Ky))
    Uxxyyy = real_irfft2(derivative_spectral(U_hat, [2, 3], Kx, Ky))
    Uxyyyy = real_irfft2(derivative_spectral(U_hat, [1, 4], Kx, Ky))
    Uyyyyy = real_irfft2(derivative_spectral(U_hat, [0, 5], Kx, Ky))
    Vxxxxx = real_irfft2(derivative_spectral(V_hat, [5, 0], Kx, Ky))
    Vxxxxy = -Uxxxxx
    Vxxxyy = -Uxxxxx
    Vxxyyy = -Uxxxyy
    Vxyyyy = -Uxxyyy
    Vyyyyy = -Uxyyyy

    Tau11CrossGM4, Tau12CrossGM4, Tau22CrossGM4 = TauCrossGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    Tau11CrossGM6 = Tau11CrossGM4 - C1*(Uxxx*Uxyy + Uxy*Uxyyy + Uyyy*Uxxy + Uxy*Uxxxy) - C3*(
        Uxxx*Uxxx + Uyyy*Uyyy + Uxyy*Uxyy + Uxxy*Uxxy + Uxx*Uxxxx + Uyy*Uyyyy + Uxx*Uxxyy + Uyy*Uxxyy + Uy*Uxxyyy + Ux*Uxxxyy)- C4(
            Ux*Uxxxxx + Uy*Uyyyyy + Ux*Uxyyyy + Uy*Uxxxxy)
    
    Tau12CrossGM6 = Tau12CrossGM4 - C3*(Uxxx*Vxxx + Uyyy*Vyyy + Uxyy*Vxyy + Uxxy*Vxxy + Uxxx*Vxyy + Uyyy*Vxxy + Uxyy*Vxxx + Uxxy*Vyyy + Uxy*Vxyyy + Uxy*Vxxxy + Uxyyy*Vxy + Uxxxy*Vxy) - C4*(
        Uxx*Vxxxx + Uxxxx*Vxx + Uyy*Vyyyy + Uyyyy*Vyy + Uxxyy*Vxx + Uxxyy*Vyy + Uxx*Vxxyy + Uyy*Vxxyy + Uxxyyy*Vy + Uxxxyy*Vx + Uy*Vxxyyy + Ux*Vxxxyy) - C6*(
            Uxxxxx*Vx + Uyyyyy*Vy + Ux*Vxxxxx + Uy*Vyyyyy + Uxyyyy*Vx + Uxxxxy*Vy + Ux*Vxyyyy + Uy*Vxxxxy) 
    
    Tau22CrossGM6 = Tau22CrossGM4 - C1*(Vxxx*Vxyy + Vxy*Vxyyy + Vyyy*Vxxy + Vxy*Vxxxy) - C3*(
        Vxxx*Vxxx + Vyyy*Vyyy + Vxyy*Vxyy + Vxxy*Vxxy + Vxx*Vxxxx + Vyy*Vyyyy + Vxx*Vxxyy + Vyy*Vxxyy + Vy*Vxxyyy + Vx*Vxxxyy)- C4(
            Vx*Vxxxxx + Vy*Vyyyyy + Vx*Vxyyyy + Vy*Vxxxxy)
    
    return Tau11CrossGM6, Tau12CrossGM6, Tau22CrossGM6


def TauCrossGM6_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta):

    C1 = Delta**6 / 1728
    C2 = Delta**6 / 2304
    C3 = Delta**6 / 3456
    C4 = Delta**6 / 6912
    C5 = Delta**6 / 10368
    C6 = Delta**6 / 13824
    C7 = Delta**6 / 20736

    Ux_hat = derivative_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_spectral(V_hat, [1, 0], Kx, Ky)
    Vy_hat = -Ux_hat
    Uxx_hat = derivative_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_spectral(V_hat, [2, 0], Kx, Ky)
    Vxy_hat = -Uxx_hat
    Vyy_hat = -Uxy_hat
    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    Vxxy_hat = -Uxxx_hat
    Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat
    Uxxxx_hat = derivative_spectral(U_hat, [4, 0], Kx, Ky)
    Uxxxy_hat = derivative_spectral(U_hat, [3, 1], Kx, Ky)
    Uxxyy_hat = derivative_spectral(U_hat, [2, 2], Kx, Ky)
    Uxyyy_hat = derivative_spectral(U_hat, [1, 3], Kx, Ky)
    Uyyyy_hat = derivative_spectral(U_hat, [0, 4], Kx, Ky)
    Vxxxx_hat = derivative_spectral(V_hat, [4, 0], Kx, Ky)
    Vxxxy_hat = -Uxxxx_hat
    Vxxyy_hat = -Uxxxy_hat
    Vxyyy_hat = -Uxxyy_hat
    Vyyyy_hat = -Uxyyy_hat
    Uxxxxx_hat = derivative_spectral(U_hat, [5, 0], Kx, Ky)
    Uxxxxy_hat = derivative_spectral(U_hat, [4, 1], Kx, Ky)
    Uxxxyy_hat = derivative_spectral(U_hat, [3, 2], Kx, Ky)
    Uxxyyy_hat = derivative_spectral(U_hat, [2, 3], Kx, Ky)
    Uxyyyy_hat = derivative_spectral(U_hat, [1, 4], Kx, Ky)
    Uyyyyy_hat = derivative_spectral(U_hat, [0, 5], Kx, Ky)
    Vxxxxx_hat = derivative_spectral(V_hat, [5, 0], Kx, Ky)
    Vxxxxy_hat = -Uxxxxx_hat
    Vxxxyy_hat = -Uxxxxx_hat
    Vxxyyy_hat = -Uxxxyy_hat
    Vxyyyy_hat = -Uxxyyy_hat
    Vyyyyy_hat = -Uxyyyy_hat

    UxyUxyyy_hat = multiply_dealias_spectral(Uxy_hat, Uxyy_hat)
    UxyUxxxy_hat = multiply_dealias_spectral(Uxy_hat, Uxx_hat)
    UxyyUxyy_hat = multiply_dealias_spectral(Uxyy_hat, Uxyy_hat)
    UxxyUxxy_hat = multiply_dealias_spectral(Uxxy_hat, Uxxy_hat)
    UxxUxxxx_hat = multiply_dealias_spectral(Uxx_hat, Uxxx_hat)
    UyyUyyyy_hat = multiply_dealias_spectral(Uyy_hat, Uyyy_hat)
    UxxxUxyy_hat = multiply_dealias_spectral(Uxxx_hat, Uxyy_hat)
    UyyyUxxy_hat = multiply_dealias_spectral(Uyyy_hat, Uxxy_hat)
    UxxUxxyy_hat = multiply_dealias_spectral(Uxx_hat, Uxxyy_hat)
    UyyUxxyy_hat = multiply_dealias_spectral(Uyy_hat, Uxxyy_hat)
    UxUxxxyy_hat = multiply_dealias_spectral(Ux_hat, Uxxxyy_hat)
    UyUxxyyy_hat = multiply_dealias_spectral(Uy_hat, Uxxyyy_hat)
    UxUxxxxx_hat = multiply_dealias_spectral(Ux_hat, Uxxxxx_hat)
    UyUyyyyy_hat = multiply_dealias_spectral(Uy_hat, Uyyyyy_hat)
    UxUxyyyy_hat = multiply_dealias_spectral(Ux_hat, Uxyyyy_hat)
    UyUxxxxy_hat = multiply_dealias_spectral(Uy_hat, Uxxxxy_hat)
    UxxxUxxx_hat = multiply_dealias_spectral(Uxxx_hat, Uxxx_hat)
    UyyyUyyy_hat = multiply_dealias_spectral(Uyyy_hat, Uyyy_hat)

    VxyVxyyy_hat = multiply_dealias_spectral(Vxy_hat, Vxyy_hat)
    VxyVxxxy_hat = multiply_dealias_spectral(Vxy_hat, Vxx_hat)
    VxyyVxyy_hat = multiply_dealias_spectral(Vxyy_hat, Vxyy_hat)
    VxxyVxxy_hat = multiply_dealias_spectral(Vxxy_hat, Vxxy_hat)
    VxxVxxxx_hat = multiply_dealias_spectral(Vxx_hat, Vxxx_hat)
    VyyVyyyy_hat = multiply_dealias_spectral(Vyy_hat, Vyyy_hat)
    VxxxVxyy_hat = multiply_dealias_spectral(Vxxx_hat, Vxyy_hat)
    VyyyVxxy_hat = multiply_dealias_spectral(Vyyy_hat, Vxxy_hat)
    VxxVxxyy_hat = multiply_dealias_spectral(Vxx_hat, Vxxyy_hat)
    VyyVxxyy_hat = multiply_dealias_spectral(Vyy_hat, Vxxyy_hat)
    VxVxxxyy_hat = multiply_dealias_spectral(Vx_hat, Vxxxyy_hat)
    VyVxxyyy_hat = multiply_dealias_spectral(Vy_hat, Vxxyyy_hat)
    VxVxxxxx_hat = multiply_dealias_spectral(Vx_hat, Vxxxxx_hat)
    VyVyyyyy_hat = multiply_dealias_spectral(Vy_hat, Vyyyyy_hat)
    VxVxyyyy_hat = multiply_dealias_spectral(Vx_hat, Vxyyyy_hat)
    VyVxxxxy_hat = multiply_dealias_spectral(Vy_hat, Vxxxxy_hat)
    VxxxVxxx_hat = multiply_dealias_spectral(Vxxx_hat, Vxxx_hat)
    VyyyVyyy_hat = multiply_dealias_spectral(Vyyy_hat, Vyyy_hat)

    UxyyyVxy_hat = multiply_dealias_spectral(Uxyyy_hat, Vxy_hat)
    UxxxyVxy_hat = multiply_dealias_spectral(Uxxxy_hat, Vxy_hat)
    UxyVxyyy_hat = multiply_dealias_spectral(Uxy_hat, Vxyyy_hat)
    UxyVxxxy_hat = multiply_dealias_spectral(Uxy_hat, Vxxxy_hat)
    UxxxVxyy_hat = multiply_dealias_spectral(Uxxx_hat, Vxyy_hat)
    UyyyVxxy_hat = multiply_dealias_spectral(Uyyy_hat, Vxxy_hat)
    UxyyVxxx_hat = multiply_dealias_spectral(Uxyy_hat, Vxxx_hat)
    UxxxxVxx_hat = multiply_dealias_spectral(Uxxxx_hat, Vxx_hat)
    UxxVxxxx_hat = multiply_dealias_spectral(Uxx_hat, Vxxxx_hat)
    UyyyyVyy_hat = multiply_dealias_spectral(Uyyyy_hat, Vyy_hat)
    UyyVyyyy_hat = multiply_dealias_spectral(Uyy_hat, Vyyyy_hat)
    UxxyyVxx_hat = multiply_dealias_spectral(Uxxyy_hat, Vxx_hat)
    UxxyyVyy_hat = multiply_dealias_spectral(Uxxyy_hat, Vyy_hat)
    UxxVxxyy_hat = multiply_dealias_spectral(Uxx_hat, Vxxyy_hat)
    UyyVxxyy_hat = multiply_dealias_spectral(Uyy_hat, Vxxyy_hat)
    UxxyyyVx_hat = multiply_dealias_spectral(Uxxyyy_hat, Vx_hat)
    UxxxyyVy_hat = multiply_dealias_spectral(Uxxxyy_hat, Vy_hat)
    UyVxxyyy_hat = multiply_dealias_spectral(Uy_hat, Vxxyyy_hat)
    UxVxxxyy_hat = multiply_dealias_spectral(Ux_hat, Vxxxyy_hat)
    UxxxxxVx_hat = multiply_dealias_spectral(Uxxxxx_hat, Vx_hat)
    UyyyyyVy_hat = multiply_dealias_spectral(Uyyyyy_hat, Vy_hat)
    UyVyyyyy_hat = multiply_dealias_spectral(Uy_hat, Vyyyyy_hat)
    UxVxxxxx_hat = multiply_dealias_spectral(Ux_hat, Vxxxxx_hat)
    UxyyyyVx_hat = multiply_dealias_spectral(Uxyyyy_hat, Vx_hat)
    UxxxxyVy_hat = multiply_dealias_spectral(Uxxxxy_hat, Vy_hat)
    UxVxyyyy_hat = multiply_dealias_spectral(Ux_hat, Vxyyyy_hat)
    UyVxxxxy_hat = multiply_dealias_spectral(Uy_hat, Vxxxxy_hat)
    UxxxVxxx_hat = multiply_dealias_spectral(Uxxx_hat, Vxxx_hat)
    UyyyVyyy_hat = multiply_dealias_spectral(Uyyy_hat, Vyyy_hat)
    UxyyVxyy_hat = multiply_dealias_spectral(Uxyy_hat, Vxyy_hat)
    UxxyVxxy_hat = multiply_dealias_spectral(Uxxy_hat, Vxxy_hat)
    UxxyVyyy_hat = multiply_dealias_spectral(Uxxy_hat, Vyyy_hat)

    Tau11CrossGM4_hat, Tau12CrossGM4_hat, Tau22CrossGM4_hat = TauCrossGM4_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)

    Tau11CrossGM6_hat = Tau11CrossGM4_hat - C1*(UxxxUxyy_hat + UxyUxyyy_hat + UyyyUxxy_hat + UxyUxxxy_hat) - C3*(
        UxxxUxxx_hat + UyyyUyyy_hat + UxyyUxyy_hat + UxxyUxxy_hat + UxxUxxxx_hat + UyyUyyyy_hat + UxxUxxyy_hat + UyyUxxyy_hat + UyUxxyyy_hat + UxUxxxyy_hat)- C4(
            UxUxxxxx_hat + UyUyyyyy_hat + UxUxyyyy_hat + UyUxxxxy_hat) 
    
    Tau12CrossGM6_hat = Tau12CrossGM4_hat - C3*(UxxxVxxx_hat + UyyyVyyy_hat + UxyyVxyy_hat + UxxyVxxy_hat + UxxxVxyy_hat + UyyyVxxy_hat + UxyyVxxx_hat + UxxyVyyy_hat + UxyVxyyy_hat + UxyVxxxy_hat + UxyyyVxy_hat + UxxxyVxy_hat) - C4*(
        UxxVxxxx_hat + UxxxxVxx_hat + UyyVyyyy_hat + UyyyyVyy_hat + UxxyyVxx_hat + UxxyyVyy_hat + UxxVxxyy_hat + UyyVxxyy_hat + UxxyyyVx_hat + UxxxyyVy_hat + UyVxxyyy_hat + UxVxxxyy_hat) - C6*(
            UxxxxxVx_hat + UyyyyyVy_hat + UyVyyyyy_hat + UxVxxxxx_hat + UxyyyyVx_hat + UxxxxyVy_hat + UxVxyyyy_hat + UyVxxxxy_hat)
    
    Tau22CrossGM6_hat = Tau22CrossGM4_hat - C1*(VxxxVxyy_hat + VxyVxyyy_hat + VyyyVxxy_hat + VxyVxxxy_hat) - C3*(
        VxxxVxxx_hat + VyyyVyyy_hat + VxyyVxyy_hat + VxxyVxxy_hat + VxxVxxxx_hat + VyyVyyyy_hat + VxxVxxyy_hat + VyyVxxyy_hat + VyVxxyyy_hat + VxVxxxyy_hat)- C4(
            VxVxxxxx_hat + VyVyyyyy_hat + VxVxyyyy_hat + VyVxxxxy_hat)

    return Tau11CrossGM6_hat, Tau12CrossGM6_hat, Tau22CrossGM6_hat

# TauReynoldsGM2
def TauReynoldsGM2(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):
    Tau11ReynoldsGM2, Tau12ReynoldsGM2, Tau22ReynoldsGM2 = 0,0,0
    return Tau11ReynoldsGM2, Tau12ReynoldsGM2, Tau22ReynoldsGM2

# TauReynoldsGM4
def TauReynoldsGM4(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):
    Tau11ReynoldsGM4, Tau12ReynoldsGM4, Tau22ReynoldsGM4 = 0,0,0
    return Tau11ReynoldsGM4, Tau12ReynoldsGM4, Tau22ReynoldsGM4

# TauReynoldsGM6
def TauReynoldsGM6(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.rfft2(U)
        V_hat = np.fft.rfft2(V)

    if filterType=='gaussian':
        if dealias:
            Tau11ReynoldsGM6_hat, Tau12ReynoldsGM6_hat, Tau22ReynoldsGM6_hat = TauReynoldsGM6_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta)
            Tau11ReynoldsGM6 = real_irfft2(Tau11ReynoldsGM6_hat)
            Tau12ReynoldsGM6 = real_irfft2(Tau12ReynoldsGM6_hat)
            Tau22ReynoldsGM6 = real_irfft2(Tau22ReynoldsGM6_hat)
        else:
            Tau11ReynoldsGM6, Tau12ReynoldsGM6, Tau22ReynoldsGM6 = TauReynoldsGM6_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11ReynoldsGM6_hat = np.fft.rfft2(Tau11ReynoldsGM6)
        Tau12ReynoldsGM6_hat = np.fft.rfft2(Tau12ReynoldsGM6)
        Tau22ReynoldsGM6_hat = np.fft.rfft2(Tau22ReynoldsGM6)
        return Tau11ReynoldsGM6_hat, Tau12ReynoldsGM6_hat, Tau22ReynoldsGM6_hat
    else:
        return Tau11ReynoldsGM6, Tau12ReynoldsGM6, Tau22ReynoldsGM6
    

def TauReynoldsGM6_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    C3 = Delta**6 / 3456
    C4 = Delta**6 / 6912

    Uxxx = real_irfft2(derivative_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_irfft2(derivative_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_irfft2(derivative_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_irfft2(derivative_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_irfft2(derivative_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy

    Tau11ReynoldsGM6 = C3*(Uxxx*Uxyy + Uyyy*Uxxy) + C4*(Uxxx*Uxxx + Uyyy*Uyyy + Uxyy*Uxyy + Uxxy*Uxxy)
    Tau12ReynoldsGM6 = C4*(Uxxx*Vxxx + Uyyy*Vyyy + Uxyy*Vxxx + Uxxy*Vyyy + Uxxx*Vxyy + Uxyy*Vxyy + Uyyy*Vxxy + Uxxy*Vxxy)
    Tau22ReynoldsGM6 = C3*(Vxxx*Vxyy + Vyyy*Vxxy) + C4*(Vxxx*Vxxx + Vyyy*Vyyy + Vxyy*Vxyy + Vxxy*Vxxy)

    return Tau11ReynoldsGM6, Tau12ReynoldsGM6, Tau22ReynoldsGM6


def TauReynoldsGM6_gaussian_dealias_spectral(U_hat, V_hat, Kx, Ky, Delta):

    C3 = Delta**6 / 3456
    C4 = Delta**6 / 6912

    Uxxx_hat = derivative_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_spectral(V_hat, [3, 0], Kx, Ky)
    Vxxy_hat = -Uxxx_hat
    Vxyy_hat = -Uxxy_hat
    Vyyy_hat = -Uxyy_hat

    UxyyUxyy_hat = multiply_dealias_spectral(Uxyy_hat, Uxyy_hat)
    UxxyUxxy_hat = multiply_dealias_spectral(Uxxy_hat, Uxxy_hat)
    UxxxUxyy_hat = multiply_dealias_spectral(Uxxx_hat, Uxyy_hat)
    UyyyUxxy_hat = multiply_dealias_spectral(Uyyy_hat, Uxxy_hat)
    UxxxUxxx_hat = multiply_dealias_spectral(Uxxx_hat, Uxxx_hat)
    UyyyUyyy_hat = multiply_dealias_spectral(Uyyy_hat, Uyyy_hat)

    VxyyVxyy_hat = multiply_dealias_spectral(Vxyy_hat, Vxyy_hat)
    VxxyVxxy_hat = multiply_dealias_spectral(Vxxy_hat, Vxxy_hat)
    VxxxVxyy_hat = multiply_dealias_spectral(Vxxx_hat, Vxyy_hat)
    VyyyVxxy_hat = multiply_dealias_spectral(Vyyy_hat, Vxxy_hat)
    VxxxVxxx_hat = multiply_dealias_spectral(Vxxx_hat, Vxxx_hat)
    VyyyVyyy_hat = multiply_dealias_spectral(Vyyy_hat, Vyyy_hat)

    UxxxVxyy_hat = multiply_dealias_spectral(Uxxx_hat, Vxyy_hat)
    UyyyVxxy_hat = multiply_dealias_spectral(Uyyy_hat, Vxxy_hat)
    UxyyVxxx_hat = multiply_dealias_spectral(Uxyy_hat, Vxxx_hat)
    UxxxVxxx_hat = multiply_dealias_spectral(Uxxx_hat, Vxxx_hat)
    UyyyVyyy_hat = multiply_dealias_spectral(Uyyy_hat, Vyyy_hat)
    UxyyVxyy_hat = multiply_dealias_spectral(Uxyy_hat, Vxyy_hat)
    UxxyVxxy_hat = multiply_dealias_spectral(Uxxy_hat, Vxxy_hat)
    UxxyVyyy_hat = multiply_dealias_spectral(Uxxy_hat, Vyyy_hat)

    Tau11ReynoldsGM6_hat = C3*(UxxxUxyy_hat + UyyyUxxy_hat) + C4*(UxxxUxxx_hat + UyyyUyyy_hat + UxyyUxyy_hat + UxxyUxxy_hat)
    Tau12ReynoldsGM6_hat = C4*(UxxxVxxx_hat + UyyyVyyy_hat + UxyyVxxx_hat + UxxyVyyy_hat + UxxxVxyy_hat + UxyyVxyy_hat + UyyyVxxy_hat + UxxyVxxy_hat)
    Tau22ReynoldsGM6_hat = C3*(VxxxVxyy_hat + VyyyVxxy_hat) + C4*(VxxxVxxx_hat + VyyyVyyy_hat + VxyyVxyy_hat + VxxyVxxy_hat)

    return Tau11ReynoldsGM6_hat, Tau12ReynoldsGM6_hat, Tau22ReynoldsGM6_hat