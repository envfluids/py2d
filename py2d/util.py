import numpy as np
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator

def corr2(a_var,b_var):
    # Correlation coefficient of N x N x T array

    a = a_var - np.mean(a_var)
    b = b_var - np.mean(b_var)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
    
    return r

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

def eig_vec_2D(A11, A12, A21, A22):
    """
    Calculate eigenvectors and eigenvalues of a 2D tensor field.

    This function takes three 2D arrays representing the components of a 2D tensor field and computes
    the eigenvalues and eigenvectors for each 2x2 tensor at every point in the field.

    Parameters:
    A11 (np.ndarray): A 2D array representing the A11 component of the tensor field.
    A12 (np.ndarray): A 2D array representing the A12 component of the tensor field.
    A22 (np.ndarray): A 2D array representing the A22 component of the tensor field.

    Returns:
    tuple: A tuple containing the following elements:
        - eigVec1 (np.ndarray): A 2xN array where N is the number of elements in A11. Each column represents the first eigenvector of the corresponding tensor.
        - eigVec2 (np.ndarray): A 2xN array where N is the number of elements in A11. Each column represents the second eigenvector of the corresponding tensor.
        - eigVal1 (np.ndarray): A 1D array of length N, containing the first eigenvalue of each tensor.
        - eigVal2 (np.ndarray): A 1D array of length N, containing the second eigenvalue of each tensor.
    """
    # Initialize eigenvalues and eigenvectors
    eigVal1 = np.zeros(A11.size)
    eigVal2 = np.zeros(A11.size)
    eigVec1 = np.zeros((A11.size,2))
    eigVec2 = np.zeros((A11.size,2))

    # Loop through each element
    for countGrid in range(A11.size):
        A = np.array([[A11.flat[countGrid], A12.flat[countGrid]], 
                      [A12.flat[countGrid], A22.flat[countGrid]]])
        eigVals, eigVecs = np.linalg.eig(A)
        # Sort the eigenvalues and eigenvectors
        idx = eigVals.argsort()[::-1]
        eigVals = eigVals[idx]
        eigVecs = eigVecs[:,idx]

        eigVec1[countGrid,:] = eigVecs[:, 0]
        eigVec2[countGrid,:] = eigVecs[:, 1]
        eigVal1[countGrid] = eigVals[0]
        eigVal2[countGrid] = eigVals[1]

    return eigVec1, eigVec2, eigVal1, eigVal2


