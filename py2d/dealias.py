import numpy as np
import jax.numpy as jnp
from jax import jit

from py2d.filter import coarse_spectral_filter_square, coarse_spectral_filter_square_jit
from py2d.util import fft2_to_rfft2, rfft2_to_fft2

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

        Nx, Ny = u.shape

        # Convert to spectral space for dealiasing
        u_hat_alias = np.fft.rfft2(u)
        v_hat_alias = np.fft.rfft2(v)
        # dealising
        uv_alias_hat = multiply_dealias_spectral(u_hat_alias, v_hat_alias)
        uv_alias = np.fft.irfft2(uv_alias_hat, s=[Nx,Ny])
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

    Nfine = u_dealias_hat.shape[0]

    u_dealias = np.fft.irfft2(u_dealias_hat, s=[Nfine,Nfine])
    v_dealias = np.fft.irfft2(v_dealias_hat, s=[Nfine,Nfine])

    u_dealias_v_dealias_hat = np.fft.rfft2(u_dealias * v_dealias)

    # Multiply on the dealise grid and coarse grain to alias grid
    uv_dealias_hat = coarse_spectral_filter_square(u_dealias_v_dealias_hat, Ncoarse)
    
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
    - u_pad: 2D numpy array in spectral space (if spectral=True) or
                 in physical space (if spectral=False) after padding.

    Note:
    - The u_hat_dealias needs to be conjugate symmetric for the inverse FFT to be real.
    - Padded dealising grid should be even in size. i.e. Ngrid_alias is multiple of 4
    """
    # Determine the original and dealiased grid sizes

    if spectral:
        u_hat = u
    else:
        u_hat = np.fft.rfft2(u)

    N_coarse = u_hat.shape[0]

    if  N_coarse % 2 == 0:
        N_pad = int(K * N_coarse)
    else:
        N_pad = int(K * (N_coarse-1)) + 1


    # Index of Nyquist wave number - fine grid
    kn_pad = N_pad//2
    # Index of Nyquist wave number - coarse grid
    kn_coarse =int(N_coarse//2)

    # Scale the spectral data to account for the increased grid size
    u_hat_scaled = (N_pad/N_coarse)**2 * u_hat

    # Initialize a 2D array of zeros with the dealiased shape
    u_hat_pad = np.zeros((N_pad, kn_pad+1), dtype=complex)

    # Compute indices for padding
    if N_coarse % 2 == 0:
        indx_pad = np.r_[0:kn_coarse+1, N_pad-kn_coarse+1:N_pad]
    else:
        indx_pad = np.r_[0:kn_coarse+1, N_pad-kn_coarse:N_pad]
    indy_pad = np.r_[0:kn_coarse+1]
    
    # Apply scaling and pad the spectral data with zeros
    u_hat_pad[np.ix_(indx_pad, indy_pad)] = u_hat_scaled

    # ################## Making the data conjugate symmetric ##################

    if N_pad % 2 == 0:

        # # ** Method#1: Making the array conjugate symmetric
        u_hat_pad[N_pad-N_coarse//2,:] = np.conj(u_hat_pad[N_coarse//2,:])

        # # ** Method #2: Making the Nyquist wavenumber 0
        # u_hat_pad[kn_coarse,:] = 0
        # u_hat_pad[:,kn_coarse] = 0
        pass

    else: # Odd grid size

        # ** Method#1: Do nothing - its already conjugate symmetric

        # # # ** Method #2: Making the Nyquist wavenumber 0
        # u_hat_pad[kn_coarse,:] = 0
        # u_hat_pad[kn_pad+(kn_pad-kn_coarse)+1,:] = 0
        # u_hat_pad[:,kn_coarse] = 0
        pass

    # ** Method#3:  take irfft and back:  works for both odd and even grid sizes
    # u = np.fft.irfft2(u_hat_pad, s=[N_dealias,N_dealias])
    # u_hat_pad = np.fft.rfft2(u)
        
    # return u_hat_dealias
    if spectral:
        return u_hat_pad
    else: 
        return np.fft.irfft2(u_hat_pad, s=[N_pad,N_pad])

# def padding_for_dealias(u, spectral=False, K=3/2):
#     """
#     Padding zeros to the high wavenumber in the
#     spectral space for a 2D square grid. This is required for handling
#     nonlinear terms in simulations of 2D Homogeneous Isotropic Turbulence.
#     Reference: https://math.jhu.edu/~feilu/notes/DealiasingFFT.pdf

#     Parameters:
#     - u: 2D numpy array representing the field in physical or spectral space.
#     - spectral: Boolean indicating if 'u' is already in spectral space. Default is False.
#     - K: Scaling factor to increase the grid size for dealiasing. Default is 3/2.

#     Returns:
#     - u_dealias: 2D numpy array in spectral space (if spectral=True) or
#                  in physical space (if spectral=False) after padding.

#     Note:
#     - The u_hat_dealias needs to be conjugate symmetric for the inverse FFT to be real.
#     - Padded dealising grid should be even in size. i.e. Ngrid_alias is multiple of 4
#     """
#     # Determine the original and dealiased grid sizes
#     N_alias = u.shape[0]
#     N_dealias = int(K * N_alias)

#     if np.mod(N_dealias,2) != 0:
#         raise ValueError("Padded dealising grid should be even in size. Dealiased grid size is ", N_dealias)

#     if spectral:
#         u_hat_alias = u
#     else:
#         # Compute the spectral coefficients of the input field
#         u_hat_alias = np.fft.fft2(u)

#     # Scale the spectral data to account for the increased grid size
#     u_hat_alias_scaled = K**2 * u_hat_alias

#     # Initialize a 2D array of zeros with the dealiased shape
#     u_hat_dealias = np.zeros((N_dealias, N_dealias), dtype=complex)

#     # Compute indices for padding
#     indvpad = np.r_[0:int(N_alias/2)+1, N_dealias-int(N_alias/2)+1:N_dealias]
    
#     # Apply scaling and pad the spectral data with zeros
#     u_hat_dealias[np.ix_(indvpad, indvpad)] = u_hat_alias_scaled

#     # Making the padded array conjugate symmetric
#     u_hat_dealias_pad = conjugate_symmetrize_padding(u_hat_dealias, K=K)

#     if spectral:
#         return u_hat_dealias_pad
#     else:
#         # Return in the appropriate space
#         return np.fft.ifft2(u_hat_dealias_pad).real
    
# def conjugate_symmetrize_padding(a_hat_fft_pad, K):

#     a_hat_fft_pad_sym = a_hat_fft_pad.copy()

#     # Nyquist wavenumber index after Padding - This is the Nyquist wavenumber of the dealiased grid
#     Ny = int(a_hat_fft_pad.shape[0])//2

#     # Nyquist wavenumber index before Padding - This is the Nyquist wavenumber of the original grid
#     Ny_old = int(Ny*1/K)

#     # ######################## Method #1 ############################ 
#     # # 0th Wavenumber is conjugate symmetric
#     # a_hat_fft_pad_sym[0,Ny+1:] = np.flip(a_hat_fft_pad[0,1:Ny]).conj()
#     # a_hat_fft_pad_sym[Ny+1:,0] = np.flip(a_hat_fft_pad[1:Ny,0]).conj()

#     # # Nyquist wavenumber - before padding (Ny_old) is conjugate symmetric
#     # # Padding creates this un-symmetricity
#     # a_hat_fft_pad_sym[Ny+(Ny-Ny_old),1:] = np.flip(a_hat_fft_pad[Ny-(Ny-Ny_old),1:]).conj()
#     # a_hat_fft_pad_sym[1:,Ny+(Ny-Ny_old)] = np.flip(a_hat_fft_pad[1:,Ny-(Ny-Ny_old)]).conj()

#     # # 0th wavenumber 0th wavenumbers has zero imaginary part
#     # a_hat_fft_pad_sym[0,0] = a_hat_fft_pad[0,0].real # (Kx=0, Ky=0)

#     # # Nyquist wavenumber of 0th wavenumber has zero imaginary part
#     # a_hat_fft_pad_sym[0,Ny//2] = a_hat_fft_pad[0,Ny//2].real # (Kx=0, Ky=Ny)
#     # a_hat_fft_pad_sym[Ny//2,0] = a_hat_fft_pad[Ny//2,0].real # (Kx=Ny, Ky=0)

#     # # Nyquist wavenumber of Nyquist wavenumber has zero imaginary part
#     # a_hat_fft_pad_sym[Ny,Ny] = a_hat_fft_pad[Ny,Ny].real # (Kx=Ny, Ky=Ny)

#     ####################### Method #2 ############################

#     # ############ Alternatively equate the data to zero at Ny_old wavenumber (Kx,Ky) = (Ny-(Ny-Ny_old),Ny+(Ny-Ny_old)) to make it conjugate symmetric
#     a_hat_fft_pad_sym[Ny_old,:] = 0
#     a_hat_fft_pad_sym[:,Ny_old] = 0

#     # # This is already zero - don't need to set it again
#     # a_hat_fft_pad_sym[Ny+(Ny-Ny_old),:] = 0
#     # a_hat_fft_pad_sym[:,Ny+(Ny-Ny_old)] = 0

#     return a_hat_fft_pad_sym

@jit
def multiply_dealias_physical_jit(a, b):
    """
    Multiply two fields and dealias the result, Input is in spectral space and returned in physical space.

    Parameters:
    - u_hat, v_hat: 2D numpy arrays representing the spectral coefficients of the fields.

    Returns:
    - Product of the two fields, dealiased and converted to physical space.
    """
    Nx, Ny = a.shape

    a_hat = jnp.fft.rfft2(a)
    b_hat = jnp.fft.rfft2(b)

    ab_dealias_hat = multiply_dealias_spectral_jit(a_hat, b_hat)

    ab_dealias = jnp.fft.irfft2(ab_dealias_hat, s=[Nx,Ny])
    
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

    Nfine = a_dealias_hat.shape[0]

    a_dealias = jnp.fft.irfft2(a_dealias_hat, s=[Nfine,Nfine])
    b_dealias = jnp.fft.irfft2(b_dealias_hat, s=[Nfine,Nfine])

    a_dealias_b_dealias_hat = jnp.fft.rfft2(a_dealias * b_dealias)

    # Multiply on the dealise grid and coarse grain to alias grid
    ab_dealias_hat = coarse_spectral_filter_square_jit(a_dealias_b_dealias_hat, Ncoarse)
    
    return ab_dealias_hat


def padding_for_dealias_spectral_jit(u_hat, K=3/2):
    """
    Pads zeros to the high wavenumbers in spectral space for dealiasing in 2D 
    simulations.

    This function is designed to be JAX-compatible and includes multiple methods
    to ensure conjugate symmetry for real-valued iFFT results.

    Args:
        u_hat: A JAX array representing the field in spectral space.
        spectral: Boolean indicating if 'u' is already in spectral space.
                  Default is False.
        K: Scaling factor to increase the grid size for dealiasing. Default is 3/2.

    Returns:
        u_hat_pad: A JAX array in spectral space (if spectral=True) or in physical 
               space (if spectral=False) after padding.
    """

    N_coarse = u_hat.shape[0]
    N_pad = int(K * N_coarse)

    kn_pad = N_pad // 2
    kn_coarse = N_coarse // 2

    u_hat_scaled = (N_pad / N_coarse) ** 2 * u_hat
    u_hat_pad = jnp.zeros((N_pad, kn_pad + 1), dtype=complex)


    u_hat_pad = u_hat_pad.at[:kn_coarse+1, :kn_coarse+1].set(u_hat_scaled[:kn_coarse+1, :kn_coarse+1])

    if N_pad % 2 == 0:
        u_hat_pad = u_hat_pad.at[N_pad-kn_coarse+1:,:kn_coarse+1].set(u_hat_scaled[N_coarse-kn_coarse+1:, :kn_coarse+1])

    else:
        u_hat_pad = u_hat_pad.at[N_pad-kn_coarse:,:kn_coarse+1].set(u_hat_scaled[N_coarse-kn_coarse:, :kn_coarse+1])

    # ################## Making the data conjugate symmetric ##################

    if N_pad % 2 == 0:

        # # Method #1: Direct manipulation
        # u_hat_pad = u_hat_pad.at[N_pad - N_coarse // 2, :].set(jnp.conj(u_hat_pad[N_coarse // 2, :]))

        # # # Method #2: Setting Nyquist wavenumbers to zero 
        u_hat_pad = u_hat_pad.at[kn_coarse, :].set(0) 
        u_hat_pad = u_hat_pad.at[:, kn_coarse].set(0)

    else:  # Odd grid size 

        # ** Method#1: Do nothing - its already conjugate symmetric

        # # # Method #2: Setting Nyquist wavenumbers to zero 
        u_hat_pad = u_hat_pad.at[kn_coarse, :].set(0)  
        u_hat_pad = u_hat_pad.at[kn_pad+(kn_pad-kn_coarse)+1, :].set(0)  
        u_hat_pad = u_hat_pad.at[:, kn_coarse].set(0) 
        pass

    # # # Method #3: iFFT and FFT roundtrip 
    # # u = jnp.fft.irfft2(u_hat_pad, s=[N_pad, N_pad])
    # # u_hat_pad = jnp.fft.rfft2(u)

    return u_hat_pad

# @jit
# def padding_for_dealias_spectral_jit(a_hat_alias, K=3/2):
#     """
#     Padding zeros to the high wavenumber in the
#     spectral space for a 2D square grid. This is required for handling
#     nonlinear terms in simulations of 2D Homogeneous Isotropic Turbulence.
#     Reference: https://math.jhu.edu/~feilu/notes/DealiasingFFT.pdf

#     Parameters:
#     - u: 2D numpy array representing the field in spectral space.
#     - spectral: Boolean indicating if 'u' is already in spectral space. Default is False.
#     - K: Scaling factor to increase the grid size for dealiasing. Default is 3/2.

#     Returns:
#     - u_dealias_hat: 2D numpy array in spectral space after padding.

#     Note:
#     - The u_hat_dealias needs to be conjugate symmetric for the inverse FFT to be real.
#     """
#     # Determine the original and dealiased grid sizes
#     N_alias = a_hat_alias.shape[0]
#     N_dealias = int(K * N_alias)

#     # Scale the spectral data to account for the increased grid size
#     a_hat_alias_scaled = K**2 * a_hat_alias

#     # ********** Jax Code ************

#     a_hat_dealias = jnp.zeros((N_dealias, N_dealias), dtype=complex)

#   # Extract the corners of the scaled array
#     utopleft = a_hat_alias_scaled[0:int(N_alias/2)+1, 0:int(N_alias/2)+1]
#     utopright = a_hat_alias_scaled[0:int(N_alias/2)+1, N_alias-int(N_alias/2)+1:N_alias]
#     ubottomleft = a_hat_alias_scaled[N_alias-int(N_alias/2)+1:N_alias, 0:int(N_alias/2)+1]
#     ubottomright = a_hat_alias_scaled[N_alias-int(N_alias/2)+1:N_alias, N_alias-int(N_alias/2)+1:N_alias]

#     # Since JAX arrays are immutable, use the .at[].set() method for updates
#     a_hat_dealias = a_hat_dealias.at[0:int(N_alias/2)+1, 0:int(N_alias/2)+1].set(utopleft)
#     a_hat_dealias = a_hat_dealias.at[0:int(N_alias/2)+1, N_dealias-int(N_alias/2)+1:N_dealias].set(utopright)
#     a_hat_dealias = a_hat_dealias.at[N_dealias-int(N_alias/2)+1:N_dealias, 0:int(N_alias/2)+1].set(ubottomleft)
#     a_hat_dealias = a_hat_dealias.at[N_dealias-int(N_alias/2)+1:N_dealias, N_dealias-int(N_alias/2)+1:N_dealias].set(ubottomright)

#     ######################## Making array conjugate symmetric ############################
#     # Making first row,col conjugate symmetric as well as the Nyquist +/- 1 row,col

#     a_hat_fft_pad_sym = a_hat_dealias.copy()

#     # Nyquist wavenumber index after Padding - This is the Nyquist wavenumber of the dealiased grid
#     Ny = int(a_hat_dealias.shape[0])//2

#     # Nyquist wavenumber index before Padding - This is the Nyquist wavenumber of the original grid
#     Ny_old = int(Ny*1/K)

#     # First Row and Column (using at.set)
#     # a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[0, N_dealias//2+1].set(jnp.conj(a_hat_fft_pad_sym[0, N_dealias//2-1]))
#     # a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[N_dealias//2+1, 0].set(jnp.conj(a_hat_fft_pad_sym[N_dealias//2-1, 0]))

#     # # Conjugate Symmetry with Slicing and Vectorization
#     # # Calculate slice indices dynamically
#     # slice_start = N_dealias // 2 - 1
#     # slice_end = -slice_start

#     # a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[slice_end, 1:].set(jnp.conj(jnp.flip(a_hat_fft_pad_sym[slice_start, 1:])))
#     # a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[1:, slice_end].set(jnp.conj(jnp.flip(a_hat_fft_pad_sym[1:, slice_start])))

#     # # ############ Alternatively equate the data to zero at wavenumber (N//2-1) to (N//2+1) to make it conjugate symmetric
#     a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[Ny_old,:].set(0)
#     a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[:,Ny_old].set(0)

#     # # This is already zero - don't need to set it again
#     # a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[Ny+(Ny-Ny_old),:].set(0)
#     # a_hat_fft_pad_sym = a_hat_fft_pad_sym.at[:,Ny+(Ny-Ny_old)].set(0)

#     # Return in the appropriate space
#     return a_hat_fft_pad_sym

