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

    # Initialize a 2D array of zeros with the dealiased shape
    u_hat_dealias = np.zeros((N_dealias, N_dealias), dtype=complex)

    # Compute indices for padding
    indvpad = np.r_[0:int(N_alias/2)+1, N_dealias-int(N_alias/2)+1:N_dealias]
    
    # Apply scaling and pad the spectral data with zeros
    u_hat_dealias[np.ix_(indvpad, indvpad)] = u_hat_alias_scaled

    # Making the padded array conjugate symmetric
    u_hat_dealias_pad = conjugate_symmetrize_padding(u_hat_dealias.copy())

    if spectral:
        return u_hat_dealias_pad
    else:
        # Return in the appropriate space
        return np.fft.ifft2(u_hat_dealias_pad).real
    
def conjugate_symmetrize_padding(a_hat_fft_pad):
    
    N_dealias = int(a_hat_fft_pad.shape[0])

    a_hat_fft_pad_sym = a_hat_fft_pad.copy()

    # # First Making first row and column conjugate symmetric (0th wavenumber data)
    # a_hat_fft_pad_sym[0,N_dealias//2+1] = np.conj(a_hat_fft_pad_sym[0,N_dealias//2-1])
    # a_hat_fft_pad_sym[N_dealias//2+1,0] = np.conj(a_hat_fft_pad_sym[N_dealias//2-1,0])

    # # Conjugate symmetry of 'N_dealias//2-1 'th' row and column to 'N_dealias//2+1' 'th' row and column
    # # Uneven padding of zeros creates this error
    # a_hat_fft_pad_sym[-(N_dealias//2-1),1:] = np.conj(np.flip(a_hat_fft_pad_sym[(N_dealias//2-1),1:]))
    # a_hat_fft_pad_sym[1:,-(N_dealias//2-1)] = np.conj(np.flip(a_hat_fft_pad_sym[1:,(N_dealias//2-1)]))

    ############ Alternatively equate the data to zero at wavenumber (N//2-1) to (N//2+1) to make it conjugate symmetric
    a_hat_fft_pad_sym[int(N_dealias/2)-1,:] = 0
    a_hat_fft_pad_sym[:,int(N_dealias/2)-1] = 0
    a_hat_fft_pad_sym[int(N_dealias/2)+1,:] = 0
    a_hat_fft_pad_sym[:,int(N_dealias/2)+1] = 0

    return a_hat_fft_pad_sym

@jit
def multiply_dealias_physical_jit(a, b):
    """
    Multiply two fields and dealias the result, Input is in spectral space and returned in physical space.

    Parameters:
    - u_hat, v_hat: 2D numpy arrays representing the spectral coefficients of the fields.

    Returns:
    - Product of the two fields, dealiased and converted to physical space.
    """

    a_hat = jnp.fft.fft2(a)
    b_hat = jnp.fft.fft2(b)

    ab_dealias_hat = multiply_dealias_spectral_jit(a_hat, b_hat)

    ab_dealias = jnp.fft.ifft2(ab_dealias_hat).real
    
    return ab_dealias

@jit
def multiply_dealias_spectral_jit(a_hat, b_hat):
    """
    Multiply two fields and dealias the result, Input is in spectral space and returned in physical space.

    Parameters:
    - u_hat, v_hat: 2D numpy arrays representing the spectral coefficients of the fields.

    Returns:
    - Product of the two fields, dealiased and converted to physical space.
    """

    Ncoarse = a_hat.shape[0]

    # Dealias each field
    a_dealias_hat = padding_for_dealias_spectral_jit(a_hat)
    b_dealias_hat = padding_for_dealias_spectral_jit(b_hat)

    a_dealias = jnp.fft.ifft2(a_dealias_hat).real
    b_dealias = jnp.fft.ifft2(b_dealias_hat).real

    a_dealias_b_dealias_hat = jnp.fft.fft2(a_dealias * b_dealias)

    # Multiply on the dealise grid and coarse grain to alias grid
    ab_dealias_hat = coarse_spectral_filter_square_2DFHIT_jit(a_dealias_b_dealias_hat, Ncoarse)
    
    return ab_dealias_hat

@jit
def padding_for_dealias_spectral_jit(a_hat_alias, K=3/2):
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
    N_alias = a_hat_alias.shape[0]
    N_dealias = int(K * N_alias)

    # Scale the spectral data to account for the increased grid size
    a_hat_alias_scaled = K**2 * a_hat_alias

    # ********** Jax Code ************

    a_hat_dealias = jnp.zeros((N_dealias, N_dealias), dtype=complex)

  # Extract the corners of the scaled array
    utopleft = a_hat_alias_scaled[0:int(N_alias/2)+1, 0:int(N_alias/2)+1]
    utopright = a_hat_alias_scaled[0:int(N_alias/2)+1, N_alias-int(N_alias/2)+1:N_alias]
    ubottomleft = a_hat_alias_scaled[N_alias-int(N_alias/2)+1:N_alias, 0:int(N_alias/2)+1]
    ubottomright = a_hat_alias_scaled[N_alias-int(N_alias/2)+1:N_alias, N_alias-int(N_alias/2)+1:N_alias]

    # Since JAX arrays are immutable, use the .at[].set() method for updates
    a_hat_dealias = a_hat_dealias.at[0:int(N_alias/2)+1, 0:int(N_alias/2)+1].set(utopleft)
    a_hat_dealias = a_hat_dealias.at[0:int(N_alias/2)+1, N_dealias-int(N_alias/2)+1:N_dealias].set(utopright)
    a_hat_dealias = a_hat_dealias.at[N_dealias-int(N_alias/2)+1:N_dealias, 0:int(N_alias/2)+1].set(ubottomleft)
    a_hat_dealias = a_hat_dealias.at[N_dealias-int(N_alias/2)+1:N_dealias, N_dealias-int(N_alias/2)+1:N_dealias].set(ubottomright)

    # Making first row,col conjugate symmetric as well as the Nyquist +/- 1 row,col
    a_hat_dealias_pad = conjugate_symmetrize_padding_jit(a_hat_dealias.copy())

    # Return in the appropriate space
    return a_hat_dealias_pad

@jit
def conjugate_symmetrize_padding_jit(a_hat_fft_pad):

    N_dealias = int(a_hat_fft_pad.shape[0])
    a_hat_fft_pad_sym = a_hat_fft_pad.copy()

    # # First Row and Column (using at.set)
    # a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[0, N_dealias//2+1].set(jnp.conj(a_hat_fft_pad_sym[0, N_dealias//2-1]))
    # a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[N_dealias//2+1, 0].set(jnp.conj(a_hat_fft_pad_sym[N_dealias//2-1, 0]))

    # # Conjugate Symmetry with Slicing and Vectorization
    # # Calculate slice indices dynamically
    # slice_start = N_dealias // 2 - 1
    # slice_end = -slice_start

    # a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[slice_end, 1:].set(jnp.conj(jnp.flip(a_hat_fft_pad_sym[slice_start, 1:])))
    # a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[1:, slice_end].set(jnp.conj(jnp.flip(a_hat_fft_pad_sym[1:, slice_start])))

    ############ Alternatively equate the data to zero at wavenumber (N//2-1) to (N//2+1) to make it conjugate symmetric
    a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[N_dealias//2-1,:].set(0)
    a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[N_dealias//2+1,:].set(0)
    a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[:,N_dealias//2-1].set(0)
    a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[:,N_dealias//2+1].set(0)

    return a_hat_fft_pad_sym
