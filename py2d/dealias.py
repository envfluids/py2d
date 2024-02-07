import numpy as np
import jax.numpy as jnp

from py2d.filter import filter2D_2DFHIT

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
        u_hat_alias = jnp.fft.fft2(u)
        v_hat_alias = jnp.fft.fft2(v)
        # dealising
        uv_alias = multiply_dealias_spectral2physical(u_hat_alias, v_hat_alias)
    else:
        # Direct multiplication in physical space if no dealiasing needed
        uv_alias = u * v

    return uv_alias

def multiply_dealias_spectral2physical(u_hat, v_hat):
    """
    Multiply two fields and dealias the result, Input is in spectral space and returned in physical space.

    Parameters:
    - u_hat, v_hat: 2D numpy arrays representing the spectral coefficients of the fields.

    Returns:
    - Product of the two fields, dealiased and converted to physical space.
    """
    # Dealias each field
    u_dealias_hat = padding_for_dealias(u_hat, spectral=True)
    v_dealias_hat = padding_for_dealias(v_hat, spectral=True)

    u_dealias = jnp.fft.ifft2(u_dealias_hat).real
    v_dealias = jnp.fft.ifft2(v_dealias_hat).real

    # Multiply on the dealise grid and coarse grain to alias grid
    uv_alias = filter2D_2DFHIT(u_dealias * v_dealias, filterType=None, coarseGrainType='spectral', Ngrid=u_hat.shape)
    
    return uv_alias

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
                 in physical space (if spectral=False) after dealiasing.

    Note:
    - The u_hat_dealias needs to be conjugate symmetric for the inverse FFT to be real.
    """
    # Determine the original and dealiased grid sizes
    N_alias = u.shape[0]
    N_dealias = int(K * N_alias)

    # Convert to spectral space if not already
    if spectral:
        u_hat_alias = u
    else:
        u_hat_alias = jnp.fft.fft2(u)
    
    # Initialize a 2D array of zeros with the dealiased shape
    u_hat_dealias = np.zeros((N_dealias, N_dealias), dtype=complex)
    
    # Compute indices for padding
    indvpad = np.r_[0:int(N_alias/2)+1, N_dealias-int(N_alias/2)+1:N_dealias]
    
    # Apply scaling and pad the spectral data with zeros
    u_hat_dealias[np.ix_(indvpad, indvpad)] = K**2 * u_hat_alias

    # u_hat_dealias[int(N_alias/2)+1,:] = 0
    # u_hat_dealias[:,int(N_alias/2)+1] = 0
    
    # Return in the appropriate space
    if spectral:
        return u_hat_dealias
    else:
        # Convert back to physical space if needed
        u_dealias = jnp.fft.ifft2(u_hat_dealias).real
        return u_dealias