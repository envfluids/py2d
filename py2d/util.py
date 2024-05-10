import numpy as np
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator

# Function to convert fft2 outputs to rfft2 outputs
def fft2_to_rfft2(a_hat_fft):
    if a_hat_fft.shape[0] % 2 == 0:
        # Shape of matrix is even
        return a_hat_fft[:,:a_hat_fft.shape[1]//2+1]
    else:
        # Shape of matrix is odd
        return a_hat_fft[:,:(a_hat_fft.shape[1]-1)//2+1]

# Function to convert rfft2 outputs to fft2 outputs
def rfft2_to_fft2(a_hat_rfft2, backend=np): 

    a_hat_fft2 = backend.zeros([a_hat_rfft2.shape[0], a_hat_rfft2.shape[0]], dtype=backend.complex128)

    if a_hat_rfft2.shape[0] % 2 == 0:
        # Shape of matrix is even
        kn = a_hat_rfft2.shape[0]//2  # Index of Nyquist wavenumber
        a_hat_fft2[:,:kn+1] = a_hat_rfft2
        a_hat_fft2[0,kn+1:] = backend.flip(backend.conj(a_hat_rfft2[0,1:kn])) # Making the first wavenumber conjugate symmetric
        a_hat_fft2[1:,kn+1:] = backend.flip(backend.flip(backend.conj(a_hat_rfft2[1:,1:kn]),axis=0),axis=1) # Making the rest of the matrix conjugate symmetric

    else:
        # Shape of matrix is odd
        kn = (a_hat_rfft2.shape[0]-1)//2
        a_hat_fft2[:,:kn+1] = a_hat_rfft2
        a_hat_fft2[0,kn+1:] = backend.flip(backend.conj(a_hat_rfft2[0,1:kn+1])) # Making the first wavenumber conjugate symmetric
        a_hat_fft2[1:,kn+1:] = backend.flip(backend.flip(backend.conj(a_hat_rfft2[1:,1:kn+1]),axis=0),axis=1) # Making the rest of the matrix conjugate symmetric

    return a_hat_fft2

def regrid(data, out_x, out_y):
    # Regrid data to a new grid size
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)

    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))

    return interpolating_function((xv, yv))

