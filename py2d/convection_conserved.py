import numpy as np
import numpy as nnp
import jax.numpy as np
from jax import grad, jit, vmap, pmap

@jit
def convection_conserved(Omega1_hat, U1_hat, V1_hat, Kx, Ky):
    
    # Convservative form
    # U1_hat = (1.j) * Ky * Psi1_hat
    # V1_hat = -(1.j) * Kx * Psi1_hat
    U1 = np.real(np.fft.ifft2(U1_hat))
    V1 = np.real(np.fft.ifft2(V1_hat))
    Omega1 = np.real(np.fft.ifft2(Omega1_hat))
    conu1 = (1.j) * Kx * np.fft.fft2(U1 * Omega1)
    conv1 = (1.j) * Ky * np.fft.fft2(V1 * Omega1)
    convec_hat = conu1 + conv1
    
    # Non-conservative form
    Omega1x = np.real(np.fft.ifft2((1.j) * Kx * Omega1_hat))
    Omega1y = np.real(np.fft.ifft2((1.j) * Ky * Omega1_hat))
    conu1 = np.fft.fft2(U1 * Omega1x)
    conv1 = np.fft.fft2(V1 * Omega1y)
    convecN_hat = conu1 + conv1
    
    convec_hat = 0.5 * (convec_hat + convecN_hat)
    
    return convec_hat