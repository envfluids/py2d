import numpy as np
from py2d.initialize import initialize_wavenumbers_2DFHIT

def filter2D_2DFHIT(U, filterType='gaussian', coarseGrainType='spectral', Delta=None, Ngrid=None, spectral=False):
    """
    Filters and coarse grains 2D square grids typically used in 2D Forced Homogeneous Isotropic Turbulence (2D-FHIT).
    
    Parameters
    ----------
    U : numpy.ndarray
        The 2D input data to be filtered and coarse grained.
    
    filterType : str, optional
        The type of filter to apply. It can be 'gaussian', 'box', 'boxSpectral' or 'spectral'. 
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
    
    # If Delta is not provided, compute it from Ngrid
    if Delta is None:
        if Ngrid is None:
            raise ValueError("Must provide either Delta or Ngrid")
        else:
            Delta = 2 * np.pi / Ngrid[0]

    # Fourier transform the input data if not already done
    if not spectral:
        U_hat = np.fft.fft2(U)
    else:
        U_hat = U

    # Get grid size in x and y directions
    NX_DNS, NY_DNS = np.shape(U_hat)
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Domain size

    # Initialize wavenumbers for the DNS grid
    Kx_DNS, Ky_DNS, _, Ksq_DNS, _ = initialize_wavenumbers_2DFHIT(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

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

    elif filterType == 'spectral':
        kc = Lx / (Delta)
        U_f_hat = spectral_filter_circle_same_size_2DFHIT(U_hat, kc)

    # Apply coarse graining
    if coarseGrainType == 'spectral':
        U_f_c_hat = coarse_spectral_filter_square_2DFHIT(U_f_hat, Ngrid)

    elif coarseGrainType == None:
        U_f_c_hat = U_f_hat

    # Inverse Fourier transform the result and return the real part
    if not spectral:
        return np.real(np.fft.ifft2(U_f_c_hat))
    else:
        return U_f_c_hat


def spectral_filter_circle_same_size_2DFHIT(q_hat, kc):
    '''
    A sharp spectral filter for 2D flow variables. The function takes a 2D square matrix at high resolution 
    and Ngrid_coarse for low resolution. The function coasre grains the data in spectral domain and 
    returns the low resolution data in the frequency domain.

    Parameters:
    q (numpy.ndarray): The input 2D square matrix.
    kc (int): The cutoff wavenumber.

    Returns:
    numpy.ndarray: The filtered data. The data is in the frequency domain. 
    '''
    NX_DNS, NY_DNS = q_hat.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi

    _, _, Kabs_DNS, _, _ = initialize_wavenumbers_2DFHIT(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    q_filtered_hat = np.where(Kabs_DNS < kc, q_hat, 0)
    return q_filtered_hat


def coarse_spectral_filter_square_2DFHIT(a_hat, NCoarse):
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
    """

    # Determine the size of the input array
    N = np.shape(a_hat)[0]
    
    # Compute the cutoff point in Fourier space
    dkcut= int(NCoarse/2)
    
    # Define the start and end indices for the slice in Fourier space to keep
    ids = int(N/2)-dkcut
    ide = int(N/2)+dkcut
    
    # Shift the zero-frequency component to the center, then normalize the Fourier-transformed data
    a_hat_shift = np.fft.fftshift(a_hat)/(N**2)

    # Apply the spectral filter by slicing the 2D array
    wfiltered_hat_shift = a_hat_shift[ids:ide,ids:ide]

    # Shift the zero-frequency component back to the original place and un-normalize the data
    wfiltered_hat = np.fft.ifftshift(wfiltered_hat_shift)*(NCoarse**2)

    # Return the filtered data
    return wfiltered_hat