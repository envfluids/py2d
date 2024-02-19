import numpy as np
import jax.numpy as jnp
from jax import jit

from py2d.filter import coarse_spectral_filter_square_2DFHIT, coarse_spectral_filter_square_2DFHIT_jit

def multiply_dealias(u, v, dealias=True):
    """
    Multiply two fields in physical space, with an option to dealias the product.
    Reference: https://math.jhu.edu/~feilu/notes/DealiasingFFT.pdf
    Orszag 2/3 Rule

    Parameters:
    - u, v: 2D numpy arrays representing the fields to be multiplied.
    - dealias: Boolean indicating if dealiasing is required. Default is True.

    Returns:
    - Product of the two fields, optionally dealiased, in physical space.
    """
    if dealias:
        # Convert to spectral space for dealiasing
        u_hat_alias = np.fft.fft2(u)
        v_hat_alias = np.fft.fft2(v)
        # dealising
        uv_alias_hat = multiply_dealias_spectral(u_hat_alias, v_hat_alias)
        uv_alias = np.fft.ifft2(uv_alias_hat).real
    else:
        # Direct multiplication in physical space if no dealiasing needed
        uv_alias = u * v

    return uv_alias

def multiply_dealias_spectral(u_hat, v_hat):
    """
    Multiply two fields and dealias the result, Input is in spectral space and returned in physical space.

    Parameters:
    - u_hat, v_hat: 2D numpy arrays representing the spectral coefficients of the fields.

    Returns:
    - Product of the two fields, dealiased and converted to physical space.
    """

    Ncoarse = u_hat.shape[0]

    # Dealias each field
    u_dealias_hat = padding_for_dealias(u_hat, spectral=True)
    v_dealias_hat = padding_for_dealias(v_hat, spectral=True)

    u_dealias = np.fft.ifft2(u_dealias_hat).real
    v_dealias = np.fft.ifft2(v_dealias_hat).real

    u_dealias_v_dealias_hat = np.fft.fft2(u_dealias * v_dealias)

    # Multiply on the dealise grid and coarse grain to alias grid
    uv_dealias_hat = coarse_spectral_filter_square_2DFHIT(u_dealias_v_dealias_hat, Ncoarse)
    
    return uv_dealias_hat

def padding_for_dealias(u, spectral=False, K=3/2):
    """
    Padding zeros to the high wavenumber in the
    spectral space for a 2D square grid. This is required for handling
    nonlinear terms in simulations of 2D Homogeneous Isotropic Turbulence.
    Reference: https://math.jhu.edu/~feilu/notes/DealiasingFFT.pdf

    Parameters:
    - u: 2D numpy array representing the field in physical or spectral space.
    - spectral: Boolean indicating if 'u' is already in spectral space. Default is False.
    - K: Scaling factor to increase the grid size for dealiasing. Default is 3/2.

    Returns:
    - u_dealias: 2D numpy array in spectral space (if spectral=True) or
                 in physical space (if spectral=False) after padding.

    Note:
    - The u_hat_dealias needs to be conjugate symmetric for the inverse FFT to be real.
    """
    # Determine the original and dealiased grid sizes
    N_alias = u.shape[0]
    N_dealias = int(K * N_alias)

    if spectral:
        u_hat_alias = u
    else:
        # Compute the spectral coefficients of the input field
        u_hat_alias = np.fft.fft2(u)

    # Scale the spectral data to account for the increased grid size
    u_hat_alias_scaled = K**2 * u_hat_alias

    # ********** Numpy Code ************

    # Initialize a 2D array of zeros with the dealiased shape
    u_hat_dealias = np.zeros((N_dealias, N_dealias), dtype=complex)

    # Compute indices for padding
    indvpad = np.r_[0:int(N_alias/2)+1, N_dealias-int(N_alias/2)+1:N_dealias]
    
    # Apply scaling and pad the spectral data with zeros
    u_hat_dealias[np.ix_(indvpad, indvpad)] = u_hat_alias_scaled

    # u_hat_dealias[int(N_alias/2)+1,:] = 0
    # u_hat_dealias[:,int(N_alias/2)+1] = 0

    if spectral:
        return u_hat_dealias
    else:
        # Return in the appropriate space
        return np.fft.ifft2(u_hat_dealias).real

@jit
def multiply_dealias_physical_jit(u, v):
    """
    Multiply two fields and dealias the result, Input is in spectral space and returned in physical space.

    Parameters:
    - u_hat, v_hat: 2D numpy arrays representing the spectral coefficients of the fields.

    Returns:
    - Product of the two fields, dealiased and converted to physical space.
    """

    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)

    uv_dealias_hat = multiply_dealias_spectral_jit(u_hat, v_hat)

    uv_dealias = jnp.fft.ifft2(uv_dealias_hat).real
    
    return uv_dealias

@jit
def multiply_dealias_spectral_jit(u_hat, v_hat):
    """
    Multiply two fields and dealias the result, Input is in spectral space and returned in physical space.

    Parameters:
    - u_hat, v_hat: 2D numpy arrays representing the spectral coefficients of the fields.

    Returns:
    - Product of the two fields, dealiased and converted to physical space.
    """

    Ncoarse = u_hat.shape[0]

    # Dealias each field
    u_dealias_hat = padding_for_dealias(u_hat, spectral=True)
    v_dealias_hat = padding_for_dealias(v_hat, spectral=True)

    u_dealias = jnp.fft.ifft2(u_dealias_hat).real
    v_dealias = jnp.fft.ifft2(v_dealias_hat).real

    u_dealias_v_dealias_hat = jnp.fft.fft2(u_dealias * v_dealias)

    # Multiply on the dealise grid and coarse grain to alias grid
    uv_dealias_hat = coarse_spectral_filter_square_2DFHIT_jit(u_dealias_v_dealias_hat, Ncoarse)
    
    return uv_dealias_hat

@jit
def padding_for_dealias_spectral_jit(u_hat_alias, K=3/2):
    """
    Padding zeros to the high wavenumber in the
    spectral space for a 2D square grid. This is required for handling
    nonlinear terms in simulations of 2D Homogeneous Isotropic Turbulence.
    Reference: https://math.jhu.edu/~feilu/notes/DealiasingFFT.pdf

    Parameters:
    - u: 2D numpy array representing the field in spectral space.
    - spectral: Boolean indicating if 'u' is already in spectral space. Default is False.
    - K: Scaling factor to increase the grid size for dealiasing. Default is 3/2.

    Returns:
    - u_dealias_hat: 2D numpy array in spectral space after padding.

    Note:
    - The u_hat_dealias needs to be conjugate symmetric for the inverse FFT to be real.
    """
    # Determine the original and dealiased grid sizes
    N_alias = u_hat_dealias.shape[0]
    N_dealias = int(K * N_alias)

    # Scale the spectral data to account for the increased grid size
    u_hat_alias_scaled = K**2 * u_hat_alias

    # ********** Jax Code ************

    u_hat_dealias = jnp.zeros((N_dealias, N_dealias), dtype=complex)

  # Extract the corners of the scaled array
    utopleft = u_hat_alias_scaled[0:int(N_alias/2)+1, 0:int(N_alias/2)+1]
    utopright = u_hat_alias_scaled[0:int(N_alias/2)+1, N_alias-int(N_alias/2)+1:N_alias]
    ubottomleft = u_hat_alias_scaled[N_alias-int(N_alias/2)+1:N_alias, 0:int(N_alias/2)+1]
    ubottomright = u_hat_alias_scaled[N_alias-int(N_alias/2)+1:N_alias, N_alias-int(N_alias/2)+1:N_alias]

    # Since JAX arrays are immutable, use the .at[].set() method for updates
    u_hat_dealias = u_hat_dealias.at[0:int(N_alias/2)+1, 0:int(N_alias/2)+1].set(utopleft)
    u_hat_dealias = u_hat_dealias.at[0:int(N_alias/2)+1, N_dealias-int(N_alias/2)+1:N_dealias].set(utopright)
    u_hat_dealias = u_hat_dealias.at[N_dealias-int(N_alias/2)+1:N_dealias, 0:int(N_alias/2)+1].set(ubottomleft)
    u_hat_dealias = u_hat_dealias.at[N_dealias-int(N_alias/2)+1:N_dealias, N_dealias-int(N_alias/2)+1:N_dealias].set(ubottomright)
    
    # Return in the appropriate space
    return u_hat_dealias