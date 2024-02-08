import numpy as nnp
import jax.numpy as np
from jax import jit

from py2d.derivative import derivative_2DFHIT_spectral
from py2d.dealias import multiply_dealias_spectral

@jit
def real_ifft2(val):
    return np.real(np.fft.ifft2(val))

# Take fft2 if input data flag is not spectral
@jit
def fft2_if_not_spectral(data, spectral):
    return data if spectral else np.fft.fft2(data)

# Take fft2 if input data flag is not spectral
@jit 
def fft2_if_spectral(data, spectral):
    return  np.fft.fft2(data) if not spectral else data

def process_PiOmegaGM(Omega, U, V, Kx, Ky, Delta, filterType, spectral, function):
    Omega_hat, U_hat, V_hat = (fft2_if_not_spectral(Omega, spectral), fft2_if_not_spectral(U, spectral), fft2_if_not_spectral(V, spectral))
    result = function(Omega_hat, U_hat, V_hat, Kx, Ky, Delta) if filterType == 'gaussian' else None
    return fft2_if_spectral(result, spectral)

def process_TauGM(U, V, Kx, Ky, Delta, filterType, spectral, function):
    U_hat, V_hat = fft2_if_not_spectral(U, spectral), fft2_if_not_spectral(V, spectral)
    Tau11, Tau12, Tau22 = function(U_hat, V_hat, Kx, Ky, Delta)
    return fft2_if_spectral(Tau11, spectral), fft2_if_spectral(Tau12, spectral), fft2_if_spectral(Tau22, spectral)

# PiOmegaGM2
def PiOmegaGM2_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian' or filterType=='box':
        # GM2 for gaussian and box is same
        # Two function for dealias and alias are made to avoid if else in the main function and make it jax/jit compatible
        if dealias:
            PiOmegaGM2_hat = PiOmegaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
            PiOmegaGM2 = real_ifft2(PiOmegaGM2_hat)
        else:
            PiOmegaGM2 = PiOmegaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM2_hat = np.fft.fft2(PiOmegaGM2)
        return PiOmegaGM2_hat
    else:
        return PiOmegaGM2

@jit
# Two function for dealias and alias are made to avoid if else in the main function and make it jax/jit compatible
def PiOmegaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # Not dealiased

    A = Delta**2 / 12

    Ux = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [1, 0], Kx, Ky))

    Omegaxx = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [2, 0], Kx, Ky))
    Omegayy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [0, 2], Kx, Ky))
    Omegaxy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [1, 1], Kx, Ky))

    PiOmegaGM2 = A * (Omegaxy*(Uy + Vx) + Ux*(Omegaxx - Omegayy))

    return PiOmegaGM2

@jit
def PiOmegaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # dealiased

    A = Delta**2 / 12

    Ux_hat = derivative_2DFHIT_spectral(U_hat, [1, 0], Kx, Ky)
    Uy_hat = derivative_2DFHIT_spectral(U_hat, [0, 1], Kx, Ky)
    Vx_hat = derivative_2DFHIT_spectral(V_hat, [1, 0], Kx, Ky)

    Omegaxx_hat = derivative_2DFHIT_spectral(Omega_hat, [2, 0], Kx, Ky)
    Omegayy_hat = derivative_2DFHIT_spectral(Omega_hat, [0, 2], Kx, Ky)
    Omegaxy_hat = derivative_2DFHIT_spectral(Omega_hat, [1, 1], Kx, Ky)

    UyOmegaxy_hat = multiply_dealias_spectral(Uy_hat, Omegaxy_hat)
    VxOmegaxy_hat = multiply_dealias_spectral(Vx_hat, Omegaxy_hat)
    UxOmegaxx_hat = multiply_dealias_spectral(Ux_hat, Omegaxx_hat)
    UxOmegayy_hat = multiply_dealias_spectral(Ux_hat, Omegayy_hat)

    PiOmegaGM2_hat = A * (UyOmegaxy_hat + VxOmegaxy_hat + UxOmegaxx_hat - UxOmegayy_hat)
    
    return PiOmegaGM2_hat

# PiOmegaGM4
def PiOmegaGM4_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        if dealias:
            PiOmegaGM4_hat = PiOmegaGM4_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
            PiOmegaGM4 = real_ifft2(PiOmegaGM4_hat)
        else:
            PiOmegaGM4 = PiOmegaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM4_hat = np.fft.fft2(PiOmegaGM4)
        return PiOmegaGM4_hat
    else:
        return PiOmegaGM4

@jit
def PiOmegaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased

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

@jit
def PiOmegaGM4_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased

    B2 = Delta**4 / 288

    PiOmegaGM2_hat = PiOmegaGM2_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Uxx_hat = derivative_2DFHIT_spectral(U_hat, [2, 0], Kx, Ky)
    Uxy_hat = derivative_2DFHIT_spectral(U_hat, [1, 1], Kx, Ky)
    Uyy_hat = derivative_2DFHIT_spectral(U_hat, [0, 2], Kx, Ky)
    Vxx_hat = derivative_2DFHIT_spectral(V_hat, [2, 0], Kx, Ky)

    Omegaxxx_hat = derivative_2DFHIT_spectral(Omega_hat, [3, 0], Kx, Ky)
    Omegaxxy_hat = derivative_2DFHIT_spectral(Omega_hat, [2, 1], Kx, Ky)
    Omegaxyy_hat = derivative_2DFHIT_spectral(Omega_hat, [1, 2], Kx, Ky)
    Omegayyy_hat = derivative_2DFHIT_spectral(Omega_hat, [0, 3], Kx, Ky)

    UxyOmegaxxy_hat = multiply_dealias_spectral(Uxy_hat, Omegaxxy_hat)
    VxxOmegaxxy_hat = multiply_dealias_spectral(Vxx_hat, Omegaxxy_hat)
    UxxOmegaxxx_hat = multiply_dealias_spectral(Uxx_hat, Omegaxxx_hat)
    UxxOmegaxyy_hat = multiply_dealias_spectral(Uxx_hat, Omegaxyy_hat)
    UxyOmegayyy_hat = multiply_dealias_spectral(Uxy_hat, Omegayyy_hat)
    UyyOmegaxyy_hat = multiply_dealias_spectral(Uyy_hat, Omegaxyy_hat)

    PiOmegaGM4_hat = PiOmegaGM2_hat + B2 * (2*UxyOmegaxxy_hat + VxxOmegaxxy_hat + 
                                            UxxOmegaxxx_hat - 2*UxxOmegaxyy_hat - UxyOmegayyy_hat + UyyOmegaxyy_hat)
    
    return PiOmegaGM4_hat

# PiOmegaGM6
def PiOmegaGM6_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False, dealias=True):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        if dealias:
            PiOmegaGM6_hat = PiOmegaGM6_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)
            PiOmegaGM6 = real_ifft2(PiOmegaGM6_hat)
        else:
            PiOmegaGM6 = PiOmegaGM6_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM6_hat = np.fft.fft2(PiOmegaGM6)
        return PiOmegaGM6_hat
    else:
        return PiOmegaGM6

@jit
def PiOmegaGM6_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased

    PiOmegaGM4 = PiOmegaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

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

@jit
def PiOmegaGM6_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
    # not dealiased

    PiOmegaGM4_hat = PiOmegaGM4_gaussian_dealias_spectral(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    C2 = Delta**6 / 10368

    Uxxx_hat = derivative_2DFHIT_spectral(U_hat, [3, 0], Kx, Ky)
    Uxxy_hat = derivative_2DFHIT_spectral(U_hat, [2, 1], Kx, Ky)
    Uxyy_hat = derivative_2DFHIT_spectral(U_hat, [1, 2], Kx, Ky)
    Uyyy_hat = derivative_2DFHIT_spectral(U_hat, [0, 3], Kx, Ky)
    Vxxx_hat = derivative_2DFHIT_spectral(V_hat, [3, 0], Kx, Ky)

    Omegaxxxx_hat = derivative_2DFHIT_spectral(Omega_hat, [4, 0], Kx, Ky)
    Omegaxxxy_hat = derivative_2DFHIT_spectral(Omega_hat, [3, 1], Kx, Ky)
    Omegaxxyy_hat = derivative_2DFHIT_spectral(Omega_hat, [2, 2], Kx, Ky)
    Omegaxyyy_hat = derivative_2DFHIT_spectral(Omega_hat, [1, 3], Kx, Ky)
    Omegayyyy_hat = derivative_2DFHIT_spectral(Omega_hat, [0, 4], Kx, Ky)

    UxxyOmegaxxxy_hat = multiply_dealias_spectral(Uxxy_hat, Omegaxxxy_hat)
    VxxxOmegaxxxy_hat = multiply_dealias_spectral(Vxxx_hat, Omegaxxxy_hat)
    UyyyOmegaxyyy_hat = multiply_dealias_spectral(Uyyy_hat, Omegaxyyy_hat)
    UxxyOmegaxyyy_hat = multiply_dealias_spectral(Uxxy_hat, Omegaxyyy_hat)
    UxyyOmegaxxyy_hat = multiply_dealias_spectral(Uxyy_hat, Omegaxxyy_hat)
    UxxxOmegaxxxx_hat = multiply_dealias_spectral(Uxxx_hat, Omegaxxxx_hat)
    UxxxOmegaxxyy_hat = multiply_dealias_spectral(Uxxx_hat, Omegaxxyy_hat)
    UxyyOmegayyyy_hat = multiply_dealias_spectral(Uxyy_hat, Omegayyyy_hat)

    PiOmegaGM6_hat = PiOmegaGM4_hat + C2 * (3*UxxyOmegaxxxy_hat + VxxxOmegaxxxy_hat + UyyyOmegaxyyy_hat - 3*UxxyOmegaxyyy_hat + 
                                        3*UxyyOmegaxxyy_hat + UxxxOmegaxxxx_hat - 3*UxxxOmegaxxyy_hat - UxyyOmegayyyy_hat)

    return PiOmegaGM6_hat


# PiOmegaGM8
def PiOmegaGM8_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        PiOmegaGM8 = PiOmegaGM8_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM8_hat = np.fft.fft2(PiOmegaGM8)
        return PiOmegaGM8_hat
    else:
        return PiOmegaGM8


# PiOmegaGM10
def PiOmegaGM10_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        PiOmegaGM10 = PiOmegaGM10_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        PiOmegaGM10_hat = np.fft.fft2(PiOmegaGM10)
        return PiOmegaGM10_hat
    else:
        return PiOmegaGM10


##############################################################################################################

# TauGM2
def TauGM2_2DFHIT(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)


    if filterType=='gaussian' or filterType=='box':
        # GM2 for gaussian and box is same
        Tau11GM2, Tau12GM2, Tau22GM2 = TauGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11GM2_hat = np.fft.fft2(Tau11GM2)
        Tau12GM2_hat = np.fft.fft2(Tau12GM2)
        Tau22GM2_hat = np.fft.fft2(Tau22GM2)
        return Tau11GM2_hat, Tau12GM2_hat, Tau22GM2_hat
    else:
        return Tau11GM2, Tau12GM2, Tau22GM2
    
@jit
def TauGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    A = Delta**2 / 12

    Ux = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux

    Tau11GM2 = A * (Ux**2 + Uy**2)
    Tau12GM2 = A * (Ux*Vx + Uy*Vy)
    Tau22GM2 = A * (Vx**2 + Vy**2)

    return Tau11GM2, Tau12GM2, Tau22GM2


# TauGM4
def TauGM4_2DFHIT(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        Tau11GM4, Tau12GM4, Tau22GM4 = TauGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11GM4_hat = np.fft.fft2(Tau11GM4)
        Tau12GM4_hat = np.fft.fft2(Tau12GM4)
        Tau22GM4_hat = np.fft.fft2(Tau22GM4)
        return Tau11GM4_hat, Tau12GM4_hat, Tau22GM4_hat
    else:
        return Tau11GM4, Tau12GM4, Tau22GM4
    
@jit
def TauGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B2 = Delta**4 / 288

    Tau11GM2, Tau12GM2, Tau22GM2 = TauGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    Uxx = real_ifft2(derivative_2DFHIT_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy

    Tau11GM4 = Tau11GM2 + B1*(Uxy**2)   + B2*(Uxx**2 + Uyy**2)
    Tau12GM4 = Tau12GM2 + B1*(Uxy*Vxy)  + B2*(Uxx*Vxx + Uyy*Vyy)
    Tau22GM4 = Tau22GM2 + B1*(Vxy**2)   + B2*(Vxx**2 + Vyy**2)

    return Tau11GM4, Tau12GM4, Tau22GM4

# TauGM6
def TauGM6_2DFHIT(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        Tau11GM6, Tau12GM6, Tau22GM6 = TauGM6_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11GM6_hat = np.fft.fft2(Tau11GM6)
        Tau12GM6_hat = np.fft.fft2(Tau12GM6)
        Tau22GM6_hat = np.fft.fft2(Tau22GM6)
        return Tau11GM6_hat, Tau12GM6_hat, Tau22GM6_hat
    else:
        return Tau11GM6, Tau12GM6, Tau22GM6
    
@jit
def TauGM6_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    C1 = Delta**6 / 3456
    C2 = Delta**6 / 10368

    Tau11GM4, Tau12GM4, Tau22GM4 = TauGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    Uxxx = real_ifft2(derivative_2DFHIT_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy

    Tau11GM6 = Tau11GM4 + C1*(Uxxy*Uxxy + Uxyy*Uxyy) + C2*(Uxxx*Uxxx + Uyyy*Uyyy)
    Tau12GM6 = Tau12GM4 + C1*(Uxxy*Vxxy + Uxyy*Vxyy) + C2*(Uxxx*Vxxx + Uyyy*Vyyy)
    Tau22GM6 = Tau22GM4 + C1*(Vxxy*Vxxy + Vxyy*Vxyy) + C2*(Vxxx*Vxxx + Vyyy*Vyyy)

    return Tau11GM6, Tau12GM6, Tau22GM6

# TauGM8
def TauGM8_2DFHIT(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        Tau11GM8, Tau12GM8, Tau22GM8 = TauGM8_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11GM8_hat = np.fft.fft2(Tau11GM8)
        Tau12GM8_hat = np.fft.fft2(Tau12GM8)
        Tau22GM8_hat = np.fft.fft2(Tau22GM8)
        return Tau11GM8_hat, Tau12GM8_hat, Tau22GM8_hat
    else:
        return Tau11GM8, Tau12GM8, Tau22GM8

 # TauGM10
def TauGM10_2DFHIT(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        U_hat, V_hat = U, V
    else:
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        Tau11GM10, Tau12GM10, Tau22GM10 = TauGM10_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Tau11GM10_hat = np.fft.fft2(Tau11GM10)
        Tau12GM10_hat = np.fft.fft2(Tau12GM10)
        Tau22GM10_hat = np.fft.fft2(Tau22GM10)
        return Tau11GM10_hat, Tau12GM10_hat, Tau22GM10_hat
    else:
        return Tau11GM10, Tau12GM10, Tau22GM10


##############################################################################################################

# SigmaGM2
def SigmaGM2_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian' or filterType=='box':
        # GM2 for gaussian and box is same
        Sigma1GM2, Sigma2GM2 = SigmaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Sigma1GM2_hat = np.fft.fft2(Sigma1GM2)
        Sigma2GM2_hat = np.fft.fft2(Sigma2GM2)
        return Sigma1GM2_hat, Sigma2GM2_hat
    else:
        return Sigma1GM2, Sigma2GM2
    
@jit
def SigmaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    A1 = Delta**2 / 12

    Ux = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux

    Omegax = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [1, 0], Kx, Ky))
    Omegay = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [0, 1], Kx, Ky))

    Sigma1GM2 = A1 * (Ux*Omegax + Uy*Omegay)
    Sigma2GM2 = A1 * (Vx*Omegax + Vy*Omegay)

    return Sigma1GM2, Sigma2GM2

# SigmaGM4
def SigmaGM4_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        Sigma1GM4, Sigma2GM4 = SigmaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Sigma1GM4_hat = np.fft.fft2(Sigma1GM4)
        Sigma2GM4_hat = np.fft.fft2(Sigma2GM4)
        return Sigma1GM4_hat, Sigma2GM4_hat
    else:
        return Sigma1GM4, Sigma2GM4
    
@jit
def SigmaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B2 = Delta**4 / 288

    Uxx = real_ifft2(derivative_2DFHIT_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy

    Omegaxx = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [2, 0], Kx, Ky))
    Omegaxy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [1, 1], Kx, Ky))
    Omegayy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [0, 2], Kx, Ky))

    Sigma1GM2, Sigma2GM2 = SigmaGM2_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Sigma1GM4 = Sigma1GM2 + B1*(Uxy*Omegaxy) + B2*(Uxx*Omegaxx + Uyy*Omegayy)
    Sigma2GM4 = Sigma2GM2 + B1*(Vxy*Omegaxy) + B2*(Vxx*Omegaxx + Vyy*Omegayy)

    return Sigma1GM4, Sigma2GM4

# SigmaGM6
def SigmaGM6_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        Sigma1GM6, Sigma2GM6 = SigmaGM6_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Sigma1GM6_hat = np.fft.fft2(Sigma1GM6)
        Sigma2GM6_hat = np.fft.fft2(Sigma2GM6)
        return Sigma1GM6_hat, Sigma2GM6_hat
    else:
        return Sigma1GM6, Sigma2GM6
    
def SigmaGM6_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta):

    C1 = Delta**6 / 3456
    C2 = Delta**6 / 10368

    Uxxx = real_ifft2(derivative_2DFHIT_spectral(U_hat, [3, 0], Kx, Ky))
    Uxxy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [2, 1], Kx, Ky))
    Uxyy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 2], Kx, Ky))
    Uyyy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 3], Kx, Ky))
    Vxxx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [3, 0], Kx, Ky))
    Vxxy = -Uxxx
    Vxyy = -Uxxy
    Vyyy = -Uxyy

    Omegaxxx = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [3, 0], Kx, Ky))
    Omegaxxy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [2, 1], Kx, Ky))
    Omegaxyy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [1, 2], Kx, Ky))
    Omegayyy = real_ifft2(derivative_2DFHIT_spectral(Omega_hat, [0, 3], Kx, Ky))

    Sigma1GM4, Sigma2GM4 = SigmaGM4_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    Sigma1GM6 = Sigma1GM4 + C1*(Uxxy*Omegaxxy + Uxyy*Omegaxyy) + C2*(Uxxx*Omegaxxx + Uyyy*Omegayyy)
    Sigma2GM6 = Sigma2GM4 + C1*(Vxxy*Omegaxxy + Vxyy*Omegaxyy) + C2*(Vxxx*Omegaxxx + Vyyy*Omegayyy)

    return Sigma1GM6, Sigma2GM6

# SigmaGM8
def SigmaGM8_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        Sigma1GM8, Sigma2GM8 = SigmaGM8_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Sigma1GM8_hat = np.fft.fft2(Sigma1GM8)
        Sigma2GM8_hat = np.fft.fft2(Sigma2GM8)
        return Sigma1GM8_hat, Sigma2GM8_hat
    else:
        return Sigma1GM8, Sigma2GM8

# SigmaGM10
def SigmaGM10_2DFHIT(Omega, U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):

    if spectral:
        Omega_hat, U_hat, V_hat = Omega, U, V
    else:
        Omega_hat = np.fft.fft2(Omega)
        U_hat = np.fft.fft2(U)
        V_hat = np.fft.fft2(V)

    if filterType=='gaussian':
        Sigma1GM10, Sigma2GM10 = SigmaGM10_gaussian(Omega_hat, U_hat, V_hat, Kx, Ky, Delta)

    if spectral:
        Sigma1GM10_hat = np.fft.fft2(Sigma1GM10)
        Sigma2GM10_hat = np.fft.fft2(Sigma2GM10)
        return Sigma1GM10_hat, Sigma2GM10_hat
    else:
        return Sigma1GM10, Sigma2GM10

##############################################################################################################

# TauLeonard
    
# TauGM2
def TauLeonardGM2_2DFHIT(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):
    process_TauGM(U, V, Kx, Ky, Delta, filterType, spectral, TauLeonardGM2_gaussian)
    # Map filterType to the corresponding function
    filter_functions = {
        'gaussian': TauLeonardGM2_gaussian,
        # Add more filter types and their corresponding functions here
        'box': TauLeonardGM2_box,  # Assuming example_filter_function is defined
    }
    process_function = filter_functions.get(filterType)
    if process_function:
        process_TauGM(U, V, Kx, Ky, Delta, filterType, spectral, process_function)
    else:
        raise ValueError(f"Unsupported filter type: {filterType}")
    
@jit
def TauLeonardGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    A = Delta**2 / 12

    Ux = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 0], Kx, Ky))
    Uy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 1], Kx, Ky))
    Vx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [1, 0], Kx, Ky))
    Vy = -Ux

    Tau11GM2 = A * (Ux**2 + Uy**2)
    Tau12GM2 = A * (Ux*Vx + Uy*Vy)
    Tau22GM2 = A * (Vx**2 + Vy**2)

    return Tau11GM2, Tau12GM2, Tau22GM2

@jit
def TauLeonardGM2_box(U_hat, V_hat, Kx, Ky, Delta):

    return 

# TauGM4

def TauLeonardGM4_2DFHIT(U, V, Kx, Ky, Delta, filterType='gaussian', spectral=False):
    process_TauGM(U, V, Kx, Ky, Delta, filterType, spectral, TauLeonardGM4_gaussian)
    # Map filterType to the corresponding function
    filter_functions = {
        'gaussian': TauLeonardGM4_gaussian,
        # Add more filter types and their corresponding functions here
        'box': TauLeonardGM4_box,  # Assuming example_filter_function is defined
    }
    process_function = filter_functions.get(filterType)
    if process_function:
        process_TauGM(U, V, Kx, Ky, Delta, filterType, spectral, process_function)
    else:
        raise ValueError(f"Unsupported filter type: {filterType}")
    
@jit
def TauLeonardGM4_gaussian(U_hat, V_hat, Kx, Ky, Delta):

    B1 = Delta**4 / 144
    B2 = Delta**4 / 288

    Tau11GM2, Tau12GM2, Tau22GM2 = TauLeonardGM2_gaussian(U_hat, V_hat, Kx, Ky, Delta)

    Uxx = real_ifft2(derivative_2DFHIT_spectral(U_hat, [2, 0], Kx, Ky))
    Uxy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [1, 1], Kx, Ky))
    Uyy = real_ifft2(derivative_2DFHIT_spectral(U_hat, [0, 2], Kx, Ky))
    Vxx = real_ifft2(derivative_2DFHIT_spectral(V_hat, [2, 0], Kx, Ky))
    Vxy = -Uxx
    Vyy = -Uxy

    Tau11GM4 = Tau11GM2 + B1*(Uxy**2)   + B2*(Uxx**2 + Uyy**2)
    Tau12GM4 = Tau12GM2 + B1*(Uxy*Vxy)  + B2*(Uxx*Vxx + Uyy*Vyy)
    Tau22GM4 = Tau22GM2 + B1*(Vxy**2)   + B2*(Vxx**2 + Vyy**2)

    return Tau11GM4, Tau12GM4, Tau22GM4
