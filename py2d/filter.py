import numpy as np
import jax.numpy as jnp
from jax import jit

from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.util import fft2_to_rfft2, rfft2_to_fft2

def filter2D(U, filterType='gaussian', coarseGrainType='spectral', Delta=None, Ngrid=None, spectral=False):
    """
    Filters and coarse grains 2D square grids typically used in 2D Forced Homogeneous Isotropic Turbulence (2D-FHIT).
    
    Parameters
    ----------
    U : numpy.ndarray
        The 2D input data to be filtered and coarse grained.
    
    filterType : str, optional
        The type of filter to apply. It can be 'gaussian', 'box', 'boxSpectral', 
        'spectral' or 'spectral-circle', 'spectral-square', 'None'
        The default is 'gaussian'.
    
    coarseGrainType : str, optional
        The method of coarse graining to use. It can be 'spectral' or 'none'. 
        The default is 'spectral', which means coarse graining will be done in spectral space.
    
    Delta : float, optional
        The characteristic scale of the filter. If not provided, it will be computed from the provided Ngrid.
    
    Ngrid : list or tuple, optional
        The grid sizes in x and y directions. Must be provided if Delta is not provided.
    
    spectral : bool, optional
        If True, the input data U is considered to be already Fourier-transformed. 
        The default is False, in which case U is Fourier-transformed within the function.

    Returns
    -------
    numpy.ndarray
        The filtered and coarse-grained version of the input data U.
    """

    # Fourier transform the input data if not already done
    if not spectral:
        U_hat = np.fft.rfft2(U)
    else:
        U_hat = U

    # Get grid size in x and y directions
    NX_DNS = np.shape(U_hat)[0]
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Domain size
    
    # If Delta is not provided, compute it from Ngrid
    if Delta is None:
        if Ngrid is None:
            raise ValueError("Must provide either Delta or Ngrid")
        else:
            Delta = 2 * Lx / Ngrid[0]

    # Initialize wavenumbers for the DNS grid
    Kx_DNS, Ky_DNS, _, Ksq_DNS, _ = initialize_wavenumbers_rfft2(NX_DNS, NX_DNS, Lx, Ly, INDEXING='ij')

    # Apply filter to the data
    if filterType == 'gaussian':
        Gk = np.exp(-Ksq_DNS * (Delta ** 2) / 24)
        U_f_hat = Gk * U_hat

    elif filterType in ['box', 'boxSpectral']:
        Gkx = np.sinc(0.5 * Kx_DNS * Delta / np.pi)  # numpy's sinc includes pi factor
        Gky = np.sinc(0.5 * Ky_DNS * Delta / np.pi)
        Gkx[0, :] = 1.0
        Gky[:, 0] = 1.0
        Gk = Gkx * Gky
        U_f_hat = Gk * U_hat

    elif filterType == 'spectral' or filterType == 'spectral-circle':
        kc = Lx / (Delta)
        U_f_hat = spectral_filter_circle_same_size(U_hat, kc)

    elif filterType == 'spectral-square':
        kc = Lx / (Delta)
        U_f_hat = spectral_filter_square_same_size(U_hat, kc)

    elif filterType == None:
        U_f_hat = U_hat

    else:
        raise ValueError("Invalid filter type : " + filterType)

    # Apply coarse graining
    if coarseGrainType == 'spectral':
        U_f_c_hat = coarse_spectral_filter_square(U_f_hat, Ngrid[0])

    elif coarseGrainType == None:
        U_f_c_hat = U_f_hat

    else:
        raise ValueError("Invalid coarse grain type : " + coarseGrainType)

    # Inverse Fourier transform the result and return the real part
    if not spectral:
        return np.fft.irfft2(U_f_c_hat, s=(U_f_c_hat.shape[0], U_f_c_hat.shape[0]))
    else:
        return U_f_c_hat


def invertFilter2D(Uf, filterType='gaussian', Delta=None, spectral=False):
    """
    Invert filters (only for valid for invertible filters) used in 2D Turbulence.
    
    Parameters
    ----------
    U : numpy.ndarray
        The 2D input data to be filtered and coarse grained.
    
    filterType : str, optional
        The type of filter to apply. It can be 'gaussian', 'box'
        The default is 'gaussian'.
    
    
    Delta : float, optional
        The characteristic scale of the filter. If not provided, it will be computed from the provided Ngrid.
    
    spectral : bool, optional
        If True, the input data U is considered to be already Fourier-transformed. 
        The default is False, in which case U is Fourier-transformed within the function.

    Returns
    -------
    numpy.ndarray
        The inverted data
    """

    # Fourier transform the input data if not already done
    if not spectral:
        Uf_hat = np.fft.rfft2(Uf)
    else:
        Uf_hat = Uf

    # Get grid size in x and y directions
    Ngrid = np.shape(Uf_hat)[0]
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Domain size
    
    # If Delta is not provided, compute it from Ngrid
    if Delta is None:
        Delta = 2 * Lx / Ngrid

    # Initialize wavenumbers for the DNS grid
    Kx, Ky, _, Ksq, _ = initialize_wavenumbers_rfft2(Ngrid, Ngrid, Lx, Ly, INDEXING='ij')

    # Apply filter to the data
    if filterType == 'gaussian':
        Gk = np.exp(-Ksq * (Delta ** 2) / 24)

    elif filterType in ['box', 'boxSpectral']:
        Gkx = np.sinc(0.5 * Kx * Delta / np.pi)  # numpy's sinc includes pi factor
        Gky = np.sinc(0.5 * Ky * Delta / np.pi)
        Gkx[0, :] = 1.0
        Gky[:, 0] = 1.0
        Gkx[Ngrid//2, :] = 1.0 # these values are zero; avoiding division by zero in next steps
        Gky[:,Ngrid//2] = 1.0
        Gk = Gkx * Gky

    else:
        raise ValueError("Invalid filter type : " + filterType)
    
    U_hat = Uf_hat/Gk

    # Inverse Fourier transform the result and return the real part
    if not spectral:
        return np.fft.irfft2(U_hat, s=(U_hat.shape[0], U_hat.shape[0]))
    else:
        return U_hat
    

def spectral_filter_circle_same_size(q_hat, kc):
    '''
    A circular sharp spectral filter for 2D flow variables. The function takes a 2D square matrix at high resolution 
    and Ngrid_coarse for low resolution. The function coasre grains the data in spectral domain and 
    returns the low resolution data in the frequency domain.

    Parameters:
    q (numpy.ndarray): The input 2D square matrix.
    kc (int): The cutoff wavenumber.

    Returns:
    numpy.ndarray: The filtered data. The data is in the frequency domain. 
    '''
    NX_DNS = q_hat.shape[0]
    Lx, Ly = 2 * np.pi, 2 * np.pi

    _, _, Kabs_DNS, _, _ = initialize_wavenumbers_rfft2(NX_DNS, NX_DNS, Lx, Ly, INDEXING='ij')

    q_filtered_hat = np.where(Kabs_DNS < kc, q_hat, 0)
    return q_filtered_hat

def spectral_filter_square_same_size(q_hat, kc):
    '''
    A sharp spectral filter for 2D flow variables. The function takes a 2D square matrix at high resolution 
    The function applies a square sharp-spectral (cut-off filter) to the input data in the frequency domain.

    Parameters:
    q (numpy.ndarray): The input 2D square matrix.
    kc (int): The cutoff wavenumber.

    Returns:
    numpy.ndarray: The filtered data. The data is in the frequency domain. 
    '''
    NX_DNS = q_hat.shape[0]
    Lx, Ly = 2 * np.pi, 2 * np.pi

    Kx_DNS, Ky_DNS, _, _, _ = initialize_wavenumbers_rfft2(NX_DNS, NX_DNS, Lx, Ly, INDEXING='ij')

    Kx_DNS_abs = np.abs(Kx_DNS)
    Ky_DNS_abs = np.abs(Ky_DNS)

    q_filtered_hat = np.where((Kx_DNS_abs < kc) & (Ky_DNS_abs < kc), q_hat, 0)
    return q_filtered_hat

def coarse_spectral_filter_square(a_hat, NCoarse):
    """
    Apply a coarse spectral filter to the Fourier-transformed input on a 2D square grid.

    This function filters the Fourier space representation of some input data, effectively removing
    high-frequency information above a certain threshold determined by the number of effective
    large eddy simulation (LES) points, `NLES`. The filter is applied on a square grid in Fourier space.

    Parameters
    ----------
    a_hat : numpy.ndarray
        The 2D Fourier-transformed input data, expected to be a square grid.
    
    NLES : int
        The number of effective LES points, determining the cutoff for the spectral filter.
        Frequencies beyond half of this value will be cut off.

    Returns
    -------
    numpy.ndarray
        The filtered Fourier-transformed data.

    Note
    ----
    Works for both odd and even number of grid points
    """

    # Check if both original grid and coarse grid are even or odd
    if np.mod(a_hat.shape[0], 2) != np.mod(NCoarse, 2):
        print(a_hat.shape, NCoarse)
        raise ValueError("Both original grid and coarse grid should be even or odd")


    # Determine the size of the input array
    N = a_hat.shape[0]

    # Index of Nyquist wave number - fine grid
    kn_fine = N//2

    # Index of Nyquist wave number - coarse grid
    kn_coarse =int(NCoarse//2)

    # Compute indices for coarse grid
    indvpadx = np.r_[0:kn_coarse+1, kn_fine+1+(kn_fine-kn_coarse):N]
    indvpady = np.r_[0:kn_coarse+1]

    # Apply scaling and pad the spectral data with zeros
    u_hat_coarse = a_hat[np.ix_(indvpadx, indvpady)]*((NCoarse/N)**2)

    # ################## Making the data conjugate symmetric ##################

    if NCoarse % 2 == 0:

        # # # ** Method#1: Making the array conjugate symmetric
        u_hat_coarse[kn_coarse,0] = np.real(u_hat_coarse[kn_coarse,0])
        u_hat_coarse[0,kn_coarse] = np.real(u_hat_coarse[0,kn_coarse])
        u_hat_coarse[1:,kn_coarse] = 0.5*(u_hat_coarse[1:,kn_coarse] + np.conj(np.flip(u_hat_coarse[1:,kn_coarse])))

        # # # ** Method #2: Making the Nyquist wavenumber 0
        # u_hat_coarse[kn_coarse,:] = 0
        # u_hat_coarse[:,kn_coarse] = 0
        pass

    else: # Odd grid size

        # ** Method#1: Making the array conjugate symmetric
        u_hat_coarse[0,kn_coarse] = np.real(u_hat_coarse[0,kn_coarse])

        # # ** Method #2: Making the Nyquist wavenumber 0
        # u_hat_coarse[kn_coarse,:] = 0
        # u_hat_coarse[kn_coarse+1,:] = 0
        # u_hat_coarse[:,kn_coarse] = 0


    # ** Method#3:  take irfft and back:  works for both odd and even grid sizes: Does same thing as Method#1
    u = np.fft.irfft2(u_hat_coarse, s=[NCoarse,NCoarse])
    u_hat_coarse = np.fft.rfft2(u)

    return u_hat_coarse

# # Code deals with ifft's and fft's
# def coarse_spectral_filter_square_2DFHIT(a_hat, NCoarse):
#     """
#     Apply a coarse spectral filter to the Fourier-transformed input on a 2D square grid.

#     This function filters the Fourier space representation of some input data, effectively removing
#     high-frequency information above a certain threshold determined by the number of effective
#     large eddy simulation (LES) points, `NLES`. The filter is applied on a square grid in Fourier space.

#     Parameters
#     ----------
#     a_hat : numpy.ndarray
#         The 2D Fourier-transformed input data, expected to be a square grid.
    
#     NLES : int
#         The number of effective LES points, determining the cutoff for the spectral filter.
#         Frequencies beyond half of this value will be cut off.

#     Returns
#     -------
#     numpy.ndarray
#         The filtered Fourier-transformed data.
#     """

#     # Determine the size of the input array
#     N = a_hat.shape[0]
    
#     # Compute the cutoff point in Fourier space
#     dkcut= int(NCoarse/2)
    
#     # Define the start and end indices for the slice in Fourier space to keep
#     ids = int(N/2)-dkcut
#     ide = int(N/2)+dkcut
    
#     # Shift the zero-frequency component to the center, then normalize the Fourier-transformed data
#     a_hat_shift = np.fft.fftshift(a_hat)/(N**2)

#     # Apply the spectral filter by slicing the 2D array
#     wfiltered_hat_shift = a_hat_shift[ids:ide,ids:ide]

#     # Shift the zero-frequency component back to the original place and un-normalize the data
#     wfiltered_hat = np.fft.ifftshift(wfiltered_hat_shift)*(NCoarse**2)

#     # Ensure Nyquist wavenumber is its own complex conjugate
#     wfiltered_hat_sym = conjugate_symmetrize_coarse(wfiltered_hat)

#     # Return the filtered data
#     return wfiltered_hat_sym

# def conjugate_symmetrize_coarse(a_hat):
#     # Ensures Nyquist wavenumbers are complex conjugates 
#     # This function is to be used with square_spectral_coarse_grain

#     NCoarse = int(a_hat.shape[0])
#     a_hat_sym = a_hat.copy()

#     # # 0th Wavenumber is conjugate symmetric
#     a_hat_sym[0,NCoarse//2+1:] = np.flip(a_hat[0,1:NCoarse//2]).conj()
#     a_hat_sym[NCoarse//2+1:,0] = np.flip(a_hat[1:NCoarse//2,0]).conj()

#     # # Nyquist frequency is conjugate symmetric
#     a_hat_sym[NCoarse//2,NCoarse//2+1:] = np.flip(a_hat[NCoarse//2,1:NCoarse//2]).conj()
#     a_hat_sym[NCoarse//2+1:,NCoarse//2] = np.flip(a_hat[1:NCoarse//2,NCoarse//2]).conj()

#     # # 0th wavenumber 0th wavenumbers has zero imaginary part
#     a_hat_sym[0,0] = a_hat[0,0].real # (Kx=0, Ky=0)

#     # # Nyquist wavenumber of 0th wavenumber has zero imaginary part
#     a_hat_sym[0,NCoarse//2] = a_hat[0,NCoarse//2].real # (Kx=0, Ky=Ny/2)
#     a_hat_sym[NCoarse//2,0] = a_hat[NCoarse//2,0].real # (Kx=Nx/2, Ky=0)

#     # # Nyquist frequency of Nyquist frequency has zero imaginary part
#     a_hat_sym[NCoarse//2,NCoarse//2] = a_hat[NCoarse//2,NCoarse//2].real # (Kx=Nx/2, Ky=Ny/2)

# ##########################
#     # #Ensure Nyquist wavenumber is its own complex conjugate
#     # a_hat_sym[NCoarse//2,1:] = np.real((a_hat[NCoarse//2,1:] + np.conj(np.flip(a_hat[NCoarse//2,1:])))/2)
#     # a_hat_sym[1:,NCoarse//2] = np.real((a_hat[1:,NCoarse//2] + np.conj(np.flip(a_hat[1:,NCoarse//2])))/2)

#     # ##### Remove data at the nyquist frequency to make it conjugate frequency #####)
#     a_hat_sym[NCoarse//2, :] = 0
#     a_hat_sym[:,NCoarse//2] = 0

#     return a_hat_sym

# # JAX compatible
# def coarse_spectral_filter_square_2DFHIT_jit(a_hat, NCoarse):
#     """
#     Apply a coarse spectral filter to the Fourier-transformed input on a 2D square grid.

#     This function filters the Fourier space representation of some input data, effectively removing
#     high-frequency information above a certain threshold determined by the number of effective
#     large eddy simulation (LES) points, `NLES`. The filter is applied on a square grid in Fourier space.

#     Parameters
#     ----------
#     a_hat : numpy.ndarray
#         The 2D Fourier-transformed input data, expected to be a square grid.
    
#     NLES : int
#         The number of effective LES points, determining the cutoff for the spectral filter.
#         Frequencies beyond half of this value will be cut off.

#     Returns
#     -------
#     numpy.ndarray
#         The filtered Fourier-transformed data.
#     """

#     # Determine the size of the input array
#     N = a_hat.shape[0]
    
#     # Compute the cutoff point in Fourier space
#     dkcut= NCoarse//2
    
#     # Shift the zero-frequency component to the center, then normalize the Fourier-transformed data
#     a_hat_shift = jnp.fft.fftshift(a_hat)/(N**2)

#     # Define the start and end indices for the slice in Fourier space to keep
#     ids = N//2-dkcut
#     ide = N//2+dkcut

#     # Apply the spectral filter by slicing the 2D array
#     wfiltered_hat_shift = a_hat_shift[ids:ide,ids:ide]

#     # Shift the zero-frequency component back to the original place and un-normalize the data
#     wfiltered_hat = jnp.fft.ifftshift(wfiltered_hat_shift)*(NCoarse**2)

#     # Ensure Nyquist wavenumber is its own complex conjugate
#     wfiltered_hat_sym = conjugate_symmetrize_coarse_jit(wfiltered_hat)

#     # Return the filtered data
#     return wfiltered_hat_sym

# @jit
# def conjugate_symmetrize_coarse_jit(a_hat_coarse):
#     # Ensures Nyquist wavenumbers are complex conjugates 
#     # This function is to be used with square_spectral_coarse_grain

#     NCoarse = int(a_hat_coarse.shape[0])
#     a_hat_coarse_sym = a_hat_coarse.copy()

#     # #Ensure Nyquist wavenumber is its own complex conjugate
#     # a_hat_coarse_sym = a_hat_coarse_sym.at[NCoarse//2, 0].set(jnp.real(a_hat_coarse_sym[NCoarse//2, 0]))
#     # a_hat_coarse_sym = a_hat_coarse_sym.at[0, NCoarse//2].set(jnp.real(a_hat_coarse_sym[0, NCoarse//2]))
#     # a_hat_coarse_sym = a_hat_coarse_sym.at[NCoarse//2,1:].set((a_hat_coarse_sym[NCoarse//2,1:] + jnp.conj(jnp.flip(a_hat_coarse_sym[NCoarse//2,1:])))/2)
#     # a_hat_coarse_sym = a_hat_coarse_sym.at[1:,NCoarse//2].set((a_hat_coarse_sym[1:,NCoarse//2] + jnp.conj(jnp.flip(a_hat_coarse_sym[1:,NCoarse//2])))/2)

#     ##### Alternatively remove the data (make it zero) at the Nyquist frequency to make it conjugate symmetric #####
#     a_hat_coarse_sym = a_hat_coarse_sym.at[NCoarse//2, :].set(0)
#     a_hat_coarse_sym = a_hat_coarse_sym.at[:, NCoarse//2].set(0)
    
#     return a_hat_coarse_sym

# @jit
def filter2D_gaussian_spectral_jit(U_hat, Ksq, Delta):

    Gk = jnp.exp(-Ksq * (Delta ** 2) / 24)
    U_f_hat = Gk * U_hat

    return U_f_hat

@jit
def filter2D_box_spectral_jit(U_hat, Kx, Ky, Delta):

    Gkx = jnp.sinc(0.5 * Kx * Delta / jnp.pi)
    Gky = jnp.sinc(0.5 * Ky * Delta / jnp.pi)

    Gkx = Gkx.at[0, :].set(1.0)
    Gky = Gky.at[:, 0].set(1.0)

    Gk = Gkx * Gky

    U_f_hat = Gk * U_hat

    return U_f_hat

# @jit
def invertFilter2D_gaussian_spectral_jit(Uf_hat, Ksq, Delta):
    Gk = jnp.exp(-Ksq * (Delta ** 2) / 24)
    U_hat = Uf_hat/Gk

    return U_hat

@jit
def invertFilter2D_box_spectral_jit(Uf_hat, Kx, Ky, Delta):
    Gkx = jnp.sinc(0.5 * Kx * Delta / jnp.pi)  # jnp.sinc includes pi factor
    Gky = jnp.sinc(0.5 * Ky * Delta / jnp.pi)
    
    Gkx = Gkx.at[0, :].set(1.0)
    Gky = Gky.at[:, 0].set(1.0)
    
    Ngrid = Uf_hat.shape[0]
    Gkx = Gkx.at[Ngrid//2, :].set(1.0)  # avoiding division by zero in next steps
    Gky = Gky.at[:, Ngrid//2].set(1.0)
    
    Gk = Gkx * Gky

    U_hat = Uf_hat / Gk

    return U_hat

@jit
def spectral_filter_square_same_size_jit(q_hat, kc):
    '''
    A sharp spectral filter for 2D flow variables. The function takes a 2D square matrix at high resolution 
    The function applies a square sharp-spectral (cut-off filter) to the input data in the frequency domain.

    Parameters:
    q (numpy.ndarray): The input 2D square matrix.
    kc (int): The cutoff wavenumber.

    Returns:
    numpy.ndarray: The filtered data. The data is in the frequency domain. 
    '''
    NX_DNS = q_hat.shape[0]
    Lx, Ly = 2 * jnp.pi, 2 * jnp.pi

    # Create an array of the discrete Fourier Transform sample frequencies in x,y-direction
    kx = 2 * jnp.pi * jnp.fft.fftfreq(NX_DNS, d=Lx/NX_DNS)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(NX_DNS, d=Ly/NX_DNS)

    # Return coordinate grids (2D arrays) for the x and y wavenumbers
    (Kx_DNS, Ky_DNS) = jnp.meshgrid(kx, ky, indexing='ij')

    Kx_DNS_abs = jnp.abs(fft2_to_rfft2(Kx_DNS))
    Ky_DNS_abs = jnp.abs(fft2_to_rfft2(Ky_DNS))

    q_filtered_hat = jnp.where((Kx_DNS_abs < kc) & (Ky_DNS_abs < kc), q_hat, 0)
    return q_filtered_hat

def coarse_spectral_filter_square_jit(a_hat, NCoarse):
    """
    Apply a coarse spectral filter to the Fourier-transformed input on a 2D square grid.

    This function filters the Fourier space representation of input data, removing
    high-frequency information above a threshold determined by the number of effective
    large eddy simulation (LES) points, `NCoarse`. 

    Args:
        a_hat: JAX array. The 2D Fourier-transformed input data (square grid).
        NLES: int. The number of effective LES points, defining the spectral filter cutoff.

    Returns:
        JAX array. The filtered Fourier-transformed data.

    Note:
        Works for both odd and even number of grid points
    """

    # Check if shapes are compatible
    # if jnp.mod(a_hat.shape[0], 2) != jnp.mod(NCoarse, 2):
    #     raise ValueError("Both original grid and coarse grid should be even or odd")

    N = a_hat.shape[0]
    kn_fine = N // 2
    kn_coarse = NCoarse // 2

    # Apply scaling coarse graining
    u_hat_coarse = jnp.zeros((NCoarse, NCoarse//2+1), dtype=complex)
    u_hat_coarse = u_hat_coarse.at[:kn_coarse+1, :].set(a_hat[:kn_coarse+1, :kn_coarse+1] )
    u_hat_coarse = u_hat_coarse.at[kn_coarse+1:, :].set(a_hat[kn_fine+1+(kn_fine-kn_coarse):, :kn_coarse+1] )
    u_hat_coarse = u_hat_coarse * ((NCoarse / N) ** 2)

    # Enforce conjugate symmetry
    if NCoarse % 2 == 0:
        # # Method #1: Making the array conjugate symmetric
        # u_hat_coarse = u_hat_coarse.at[kn_coarse, 0].set(jnp.real(u_hat_coarse[kn_coarse, 0]))
        # u_hat_coarse = u_hat_coarse.at[0, kn_coarse].set(jnp.real(u_hat_coarse[0, kn_coarse]))
        # u_hat_coarse = u_hat_coarse.at[1:, kn_coarse].set(
        #     0.5 * (u_hat_coarse[1:, kn_coarse] + jnp.conj(jnp.flip(u_hat_coarse[1:, kn_coarse])))
        # )

        # Method #2: Making the Nyquist wavenumber 0
        u_hat_coarse = u_hat_coarse.at[kn_coarse,:].set(0)
        u_hat_coarse = u_hat_coarse.at[:,kn_coarse].set(0)

    else:  # Odd grid size
        # Method #1: Making the array conjugate symmetric
        # u_hat_coarse = u_hat_coarse.at[0,kn_coarse].set(jnp.real(u_hat_coarse[0,kn_coarse]))

        # # Method #2: Making the Nyquist wavenumber 0
        u_hat_coarse = u_hat_coarse.at[kn_coarse, :].set(0.0)  
        u_hat_coarse = u_hat_coarse.at[kn_coarse + 1, :].set(0.0)  
        u_hat_coarse = u_hat_coarse.at[:, kn_coarse].set(0.0)

    # # Method #3:  take irfft and back:  works for both odd and even grid sizes 
    # u = jnp.fft.irfft2(u_hat_coarse, s=[NCoarse,NCoarse])
    # u_hat_coarse = jnp.fft.rfft2(u)

    return u_hat_coarse

