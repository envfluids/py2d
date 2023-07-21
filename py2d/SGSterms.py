import numpy as np

from py2d.convert import Omega2Psi_2DFHIT, Psi2UV_2DFHIT
from py2d.derivative import derivative_2DFHIT
from py2d.initialize import initialize_wavenumbers_2DFHIT
from py2d.filter import filter2D_2DFHIT

filterType = 'gaussian'
coarseGrainType = 'spectral'

def Tau(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None):
    """
    Function to calculate subgrid scale stress components Tau11, Tau12, Tau22 based on the given vorticity field Omega_DNS.
    This involves filtering the velocity field and its product and then subtracting the filtered product from the product of the filters.

    Parameters:
    Omega_DNS (numpy.ndarray): 2D vorticity field.
    filterType (str): Type of filter to use ('gaussian', 'box', 'boxSpectral' etc.).
    coarseGrainType (str): Type of coarse graining to use ('spectral' etc.).
    Delta (float): filter width: positive real number
    N_LES (numpy.ndarray): size of coarse (LES) grid: 2x1 array

    Returns:
    Tau11, Tau12, Tau22 (numpy.ndarray): Subgrid scale stress components.
    """
    # Getting the shape of the vorticity field
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi

    # Initialize wavenumbers for the Fourier space operations
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_2DFHIT(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Convert vorticity field to stream function field
    # The stream function is defined as the inverse Fourier transform of the vorticity divided by the square of the wavenumber
    Psi_DNS = Omega2Psi_2DFHIT(Omega=Omega_DNS, invKsq=invKsq_DNS)
    
    # Convert stream function field to velocity fields
    # The velocity components are defined as the partial derivatives of the stream function
    U_DNS, V_DNS = Psi2UV_2DFHIT(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity fields
    # Filtering is done to remove small scales from the data
    U_f = filter2D_2DFHIT(U_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    V_f = filter2D_2DFHIT(V_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Filter the products of the fields
    # This involves multiplying the fields with each other and then applying the filter
    UU_f = filter2D_2DFHIT(U_DNS*U_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VV_f = filter2D_2DFHIT(V_DNS*V_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UV_f = filter2D_2DFHIT(U_DNS*V_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Subtract the filtered product from the product of the filters to get subgrid scale stress components
    # These represent the effect of the small scales on the large scales
    Tau11 = UU_f - U_f*U_f
    Tau12 = UV_f - U_f*V_f
    Tau22 = VV_f - V_f*V_f

    return Tau11, Tau12, Tau22


def Sigma(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None):
    """
    Function to calculate Sigma vector components Sigma1, Sigma2 based on the given vorticity field Omega_DNS.
    This involves filtering the velocity field and its product with vorticity and then subtracting the filtered product from the product of the filters.

    Parameters:
    Omega_DNS (numpy.ndarray): 2D vorticity field.
    filterType (str): Type of filter to use ('gaussian', 'box', 'boxSpectral' etc.).
    coarseGrainType (str): Type of coarse graining to use ('spectral' etc.).
    Delta (float): filter width: positive real number
    N_LES (numpy.ndarray): size of coarse (LES) grid: 2x1 array

    Returns:
    Sigma1, Sigma2 (numpy.ndarray): Sigma components.
    """

    # Get the shape of the vorticity field
    NX_DNS, NY_DNS = Omega_DNS.shape

    # Define the domain length in both directions
    Lx, Ly = 2 * np.pi, 2 * np.pi

    # Initialize the wave numbers for the DNS grid
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_2DFHIT(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Convert the vorticity field to a stream function field
    Psi_DNS = Omega2Psi_2DFHIT(Omega=Omega_DNS, invKsq=invKsq_DNS)
    
    # Convert the stream function field to the velocity fields
    U_DNS, V_DNS = Psi2UV_2DFHIT(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity fields
    U_f = filter2D_2DFHIT(U_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    V_f = filter2D_2DFHIT(V_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    
    # Filter the vorticity field
    Omega_f = filter2D_2DFHIT(Omega_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Filter the products of the velocity fields and the vorticity field
    UOmega_f = filter2D_2DFHIT(U_DNS*Omega_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VOmega_f = filter2D_2DFHIT(V_DNS*Omega_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Compute the Sigma components
    Sigma1 = UOmega_f - U_f*Omega_f
    Sigma2 = VOmega_f - V_f*Omega_f

    # Return the Sigma components
    return Sigma1, Sigma2

def PiUV(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None):
    """
    Function to calculate the components of vector PiUV: PiUV1, PiUV2 based on the given vorticity field Omega_DNS.
    This involves calculating the divergence of the subgrid scale stress tensor in conservative form

    Parameters:
    Omega_DNS (numpy.ndarray): 2D vorticity field.
    filterType (str): Type of filter to use ('gaussian', etc.).
    Delta (float): filter width: positive real number
    N_LES (numpy.ndarray): size of coarse (LES) grid: 2x1 array

    Returns:
    PiUV1, PiUV2 (numpy.ndarray): PiUV components.
    """
    # Get the shape of the DNS field
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi

    # Initialize the wavenumbers for the DNS and LES fields
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_2DFHIT(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')
    Kx_LES, Ky_LES, _, _, invKsq_LES = initialize_wavenumbers_2DFHIT(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Convert the vorticity field to a stream function and then to velocity components
    Psi_DNS = Omega2Psi_2DFHIT(Omega=Omega_DNS, invKsq=invKsq_DNS)
    U_DNS, V_DNS = Psi2UV_2DFHIT(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Calculate the derivatives of the product of the DNS velocity components
    UUx = derivative_2DFHIT(U_DNS*U_DNS, [1,0], Kx_DNS, Ky_DNS)
    UVy = derivative_2DFHIT(U_DNS*V_DNS, [0,1], Kx_DNS, Ky_DNS)
    UVx = derivative_2DFHIT(U_DNS*V_DNS, [1,0], Kx_DNS, Ky_DNS)
    VVy = derivative_2DFHIT(V_DNS*V_DNS, [0,1], Kx_DNS, Ky_DNS)

    # Filter the derivatives of the product of the DNS velocity components
    UUx_f = filter2D_2DFHIT(UUx, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UVy_f = filter2D_2DFHIT(UVy, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UVx_f = filter2D_2DFHIT(UVx, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VVy_f = filter2D_2DFHIT(VVy, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Filter the DNS velocity components
    U_f = filter2D_2DFHIT(U_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    V_f = filter2D_2DFHIT(V_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Calculate the derivatives of the product of the filtered velocity components
    U_f_U_f_x = derivative_2DFHIT(U_f*U_f, [1,0], Kx_LES, Ky_LES)
    U_f_V_f_y = derivative_2DFHIT(U_f*V_f, [0,1], Kx_LES, Ky_LES)
    U_f_V_f_x = derivative_2DFHIT(U_f*V_f, [1,0], Kx_LES, Ky_LES)
    V_f_V_f_y = derivative_2DFHIT(V_f*V_f, [0,1], Kx_LES, Ky_LES)

    # Compute the final components PiUV1 and PiUV2
    PiUV1 = UUx_f + UVy_f - (U_f_U_f_x + U_f_V_f_y)
    PiUV2 = UVx_f + VVy_f - (U_f_V_f_x + V_f_V_f_y)

    return PiUV1, PiUV2

def PiOmega(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None):
    """
    Calculate PiOmega based on the given vorticity field (Omega_DNS). This involves calculating the divergence 
    of the Sigma vector in both conservative and non-conservative forms.

    Parameters:
    ------
    Omega_DNS (numpy.ndarray): 2D vorticity field.
    filterType (str): Type of filter to use ('gaussian', etc.).
    Delta (float): Filter width - positive real number.
    N_LES (numpy.ndarray): Size of coarse (LES) grid - 2x1 array.

    Returns:
    ------
    PiOmega (numpy.ndarray): PiOmega array.

    Notes:
    ------
    Non conservative form:
    PiOmega = \bar{N(Omega,Psi)} - N(\bar{Omega},\bar{Psi})
    Jacobian N(Omega,Psi) = Omega_x * Psi_y - Omega_y * Psi_x
    """
    
    # Define the shape and domain size of the grid
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi
    
    # Initialize the wavenumbers for the DNS and LES grids
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_2DFHIT(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')
    Kx_LES, Ky_LES, _, _, invKsq_LES = initialize_wavenumbers_2DFHIT(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Convert the vorticity field to a stream function and then to velocity components
    Psi_DNS = Omega2Psi_2DFHIT(Omega=Omega_DNS, invKsq=invKsq_DNS)
    U_DNS, V_DNS = Psi2UV_2DFHIT(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)
    
    # Compute convective terms in DNS and filtered fields
    UOmega_x = derivative_2DFHIT(U_DNS*Omega_DNS, [1,0], Kx_DNS, Ky_DNS)
    VOmega_y = derivative_2DFHIT(V_DNS*Omega_DNS, [0,1], Kx_DNS, Ky_DNS)
    
    # Apply the filter to the computed convective terms
    UOmega_x_f = filter2D_2DFHIT(UOmega_x, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VOmega_y_f = filter2D_2DFHIT(VOmega_y, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Filter the DNS velocity and vorticity fields
    U_f = filter2D_2DFHIT(U_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    V_f = filter2D_2DFHIT(V_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omega_f = filter2D_2DFHIT(Omega_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Compute convective terms in filtered fields
    U_f_Omega_f_x = derivative_2DFHIT(U_f*Omega_f, [1,0], Kx_LES, Ky_LES)
    V_f_Omega_f_y = derivative_2DFHIT(V_f*Omega_f, [0,1], Kx_LES, Ky_LES)
    
    print(UOmega_x_f.dtype, VOmega_y_f.dtype, U_f_Omega_f_x.dtype, V_f_Omega_f_y.dtype)

    # Compute PiOmega using the difference between filtered and non-filtered convective terms
    PiOmega = UOmega_x_f + VOmega_y_f - (U_f_Omega_f_x + V_f_Omega_f_y)

    return PiOmega
