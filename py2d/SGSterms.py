import numpy as np

from py2d.convert import Omega2Psi, Psi2UV
from py2d.derivative import derivative
from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.filter import filter2D
from py2d.dealias import multiply_dealias


def Tau(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculate subgrid scale (SGS) stress components (Tau11, Tau12, Tau22) based on the given 2D vorticity field (Omega_DNS).
    This involves filtering the velocity field derived from the vorticity and then computing the difference between the filtered
    product of the velocity components and the product of the filtered velocity components.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D vorticity field.
    - filterType (str): Type of filter to use ('gaussian', 'box', 'boxSpectral', etc.).
    - coarseGrainType (str): Type of coarse graining to use ('spectral', etc.).
    - Delta (float): Filter width, a positive real number indicating the scale of filtering.
    - N_LES (numpy.ndarray): Size of the coarse (LES) grid as a 2-element array.
    - dealias (bool): Flag indicating whether to apply dealiasing in the multiplication of fields.

    Returns:
    - Tuple of numpy.ndarrays: Subgrid scale stress components (Tau11, Tau12, Tau22).
    """
    # Extract the dimensions of the DNS grid
    NX_DNS, NY_DNS = Omega_DNS.shape
    # Domain size for periodic boundary conditions
    Lx, Ly = 2 * np.pi, 2 * np.pi

    # Initialize wavenumbers for spectral operations based on the DNS grid dimensions
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Compute the stream function from the vorticity field using the inverse Laplacian in spectral space
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    
    # Derive the velocity components from the stream function by taking its spatial derivatives
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Apply the specified filter to the velocity fields to obtain their coarse-grained representations
    Uf_c = filter2D(U_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vf_c = filter2D(V_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Compute the product of the filtered velocity fields
    Uf_c_Uf_c = multiply_dealias(Uf_c, Uf_c, dealias=dealias)
    Vf_c_Vf_c = multiply_dealias(Vf_c, Vf_c, dealias=dealias)
    Uf_c_Vf_c = multiply_dealias(Uf_c, Vf_c, dealias=dealias)

    # Compute the product of the unfiltered velocity fields
    UU = multiply_dealias(U_DNS, U_DNS, dealias=dealias)
    VV = multiply_dealias(V_DNS, V_DNS, dealias=dealias)
    UV = multiply_dealias(U_DNS, V_DNS, dealias=dealias)

    # Filter the products of the velocity fields to simulate the effect of the resolved scales on the SGS stresses
    UU_f_c = filter2D(UU, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VV_f_c = filter2D(VV, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UV_f_c = filter2D(UV, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Calculate the SGS stress components by subtracting the product of the filtered velocities from the filtered product
    Tau11 = UU_f_c - Uf_c_Uf_c
    Tau12 = UV_f_c - Uf_c_Vf_c
    Tau22 = VV_f_c - Vf_c_Vf_c

    return Tau11, Tau12, Tau22


def Sigma(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculate the Sigma vector components (Sigma1, Sigma2) based on the given 2D vorticity field (Omega_DNS).
    This process involves filtering the velocity field derived from the vorticity, filtering the product of the velocity
    field with the vorticity, and then computing the difference between the filtered product and the product of the
    filtered fields.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D vorticity field.
    - filterType (str): Type of filter to use ('gaussian', 'box', 'boxSpectral', etc.).
    - coarseGrainType (str): Type of coarse graining to use ('spectral', etc.).
    - Delta (float): Filter width, a positive real number indicating the scale of filtering.
    - N_LES (numpy.ndarray): Size of the coarse (LES) grid as a 2-element array.
    - dealias (bool): Flag indicating whether to apply dealiasing in the multiplication of fields.

    Returns:
    - Sigma1, Sigma2 (numpy.ndarray): Sigma vector components.
    """

    # Extract the dimensions of the DNS grid to define the spectral domain
    NX_DNS, NY_DNS = Omega_DNS.shape

    # Domain size for periodic boundary conditions, assuming a square domain
    Lx, Ly = 2 * np.pi, 2 * np.pi

    # Initialize the wavenumbers for spectral operations based on the DNS grid dimensions
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Derive the stream function from the vorticity using the inverse Laplacian in the spectral domain
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    
    # Calculate the velocity components from the stream function by taking spatial derivatives
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Apply the specified filter to the velocity fields to obtain their coarse-grained representations
    Uf_c = filter2D(U_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vf_c = filter2D(V_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omegaf_c = filter2D(Omega_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Compute the product of the unfiltered velocity fields with the unfiltered vorticity field
    UOmega = multiply_dealias(U_DNS, Omega_DNS, dealias=dealias)
    VOmega = multiply_dealias(V_DNS, Omega_DNS, dealias=dealias)

    # Filter the products of the velocity fields with the vorticity field
    UOmega_f_c = filter2D(UOmega, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VOmega_f_c = filter2D(VOmega, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Compute the product of the filtered velocity fields with the filtered vorticity field
    Uf_c_Omegaf_c = multiply_dealias(Uf_c, Omegaf_c, dealias=dealias)
    Vf_c_Omegaf_c = multiply_dealias(Vf_c, Omegaf_c, dealias=dealias)

    # Calculate the Sigma components by subtracting the product of the filtered fields from the filtered product
    Sigma1 = UOmega_f_c - Uf_c_Omegaf_c
    Sigma2 = VOmega_f_c - Vf_c_Omegaf_c

    # Return the calculated Sigma vector components
    return Sigma1, Sigma2


def PiUV(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculate the components of the PiUV vector (PiUV1, PiUV2) based on a given 2D vorticity field (Omega_DNS).
    This process involves calculating the divergence of the subgrid scale (SGS) stress tensor in conservative form,
    which is derived from the filtered velocity field and its derivatives.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D vorticity field.
    - filterType (str): Type of filter to use (e.g., 'gaussian').
    - coarseGrainType (str): Type of coarse graining to use (e.g., 'spectral').
    - Delta (float): Filter width, a positive real number indicating the scale of filtering.
    - N_LES (numpy.ndarray): Size of the coarse (LES) grid as a 2-element array.
    - dealias (bool): Flag indicating whether to apply dealiasing in the multiplication of fields.

    Returns:
    - PiUV1, PiUV2 (numpy.ndarray): Components of the PiUV vector.
    """

    # Extract the dimensions of the DNS grid to define the spectral domain
    NX_DNS, NY_DNS = Omega_DNS.shape

    # Domain size for periodic boundary conditions, assuming a square domain
    Lx, Ly = 2 * np.pi, 2 * np.pi

    # Determine the LES grid dimensions based on the coarseGrainType
    NX_LES, NY_LES = (NX_DNS, NY_DNS) if coarseGrainType in [None, 'physical'] else (N_LES[0], N_LES[1])

    # Initialize the wavenumbers for spectral operations for both DNS and LES grids
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')
    Kx_LES, Ky_LES, _, _, invKsq_LES = initialize_wavenumbers_rfft2(NX_LES, NY_LES, Lx, Ly, INDEXING='ij')

    # Convert the vorticity field to a stream function, then to velocity components
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Compute the product of the DNS velocity components
    UU = multiply_dealias(U_DNS, U_DNS, dealias=dealias)
    VV = multiply_dealias(V_DNS, V_DNS, dealias=dealias)
    UV = multiply_dealias(U_DNS, V_DNS, dealias=dealias)

    # Calculate the spatial derivatives of the velocity component products
    UUx = derivative(UU, [1,0], Kx_DNS, Ky_DNS)
    UVy = derivative(UV, [0,1], Kx_DNS, Ky_DNS)
    UVx = derivative(UV, [1,0], Kx_DNS, Ky_DNS)
    VVy = derivative(VV, [0,1], Kx_DNS, Ky_DNS)

    # Filter the derivatives to obtain their coarse-grained representations
    UUx_f_c = filter2D(UUx, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UVy_f_c = filter2D(UVy, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UVx_f_c = filter2D(UVx, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VVy_f_c = filter2D(VVy, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Filter the DNS velocity components themselves
    Uf_c = filter2D(U_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vf_c = filter2D(V_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Compute the product of the filtered velocity components
    Uf_c_Uf_c = multiply_dealias(Uf_c, Uf_c, dealias=dealias)
    Vf_c_Vf_c = multiply_dealias(Vf_c, Vf_c, dealias=dealias)
    Uf_c_Vf_c = multiply_dealias(Uf_c, Vf_c, dealias=dealias)

    # Calculate the spatial derivatives of the product of the filtered velocity components
    Uf_Uf_x = derivative(Uf_c_Uf_c, [1,0], Kx_LES, Ky_LES)
    Uf_Vf_y = derivative(Uf_c_Vf_c, [0,1], Kx_LES, Ky_LES)
    Uf_Vf_x = derivative(Uf_c_Vf_c, [1,0], Kx_LES, Ky_LES)
    Vf_Vf_y = derivative(Vf_c_Vf_c, [0,1], Kx_LES, Ky_LES)

    # Compute the PiUV components by subtracting the filtered derivatives of the product of the DNS velocity components
    # from the derivatives of the product of the filtered velocity components
    PiUV1 = UUx_f_c + UVy_f_c - (Uf_Uf_x + Uf_Vf_y)
    PiUV2 = UVx_f_c + VVy_f_c - (Uf_Vf_x + Vf_Vf_y)

    # Return the calculated PiUV vector components
    return PiUV1, PiUV2

# PiOmega
def PiOmega(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculate the PiOmega scalar field based on a given 2D vorticity field (Omega_DNS). This involves
    calculating the divergence of the Sigma vector, representing the interaction between the velocity
    field and vorticity, in both conservative and non-conservative forms.

    Parameters:
    ------
    Omega_DNS (numpy.ndarray): 2D vorticity field.
    filterType (str): Type of filter to apply ('gaussian', etc.).
    coarseGrainType (str): Determines the method for coarse graining ('spectral', etc.).
    Delta (float): Filter width, a positive real number indicating the scale of filtering.
    N_LES (numpy.ndarray): Size of the coarse (LES) grid, specified as a 2-element array.
    dealias (bool): Indicates whether dealiasing should be applied during field multiplication.

    Returns:
    ------
    PiOmega (numpy.ndarray): The calculated PiOmega field, representing the divergence of the
                             subgrid-scale interaction term in the filtered velocity and vorticity fields.

    Notes:
    ------
    The non-conservative form of PiOmega is calculated as follows:
    PiOmega = \bar{N}(Omega, Psi) - N(\bar{Omega}, \bar{Psi}),
    where N(Omega, Psi) is the Jacobian representing the interaction between vorticity and stream function,
    given by Omega_x * Psi_y - Omega_y * Psi_x.
    """
    
    # Extract the dimensions of the DNS grid and define the domain size
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Assuming a square domain with periodic boundary conditions

    # Determine the LES grid dimensions based on the coarse graining type
    if coarseGrainType in [None, 'physical']:
        NX_LES, NY_LES = NX_DNS, NY_DNS
    else:
        NX_LES, NY_LES = int(N_LES[0]), int(N_LES[1])
    
    # Initialize wavenumbers for spectral domain operations for both DNS and LES grids
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')
    Kx_LES, Ky_LES, _, _, invKsq_LES = initialize_wavenumbers_rfft2(NX_LES, NY_LES, Lx, Ly, INDEXING='ij')

    # Convert the vorticity field to a stream function, then derive velocity components
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Compute the product of velocity components with vorticity and their spatial derivatives
    UOmega = multiply_dealias(U_DNS, Omega_DNS, dealias=dealias)
    VOmega = multiply_dealias(V_DNS, Omega_DNS, dealias=dealias)

    UOmegax = derivative(UOmega, [1,0], Kx_DNS, Ky_DNS)
    VOmegay = derivative(VOmega, [0,1], Kx_DNS, Ky_DNS)
    
    # Apply the filter to the computed spatial derivatives
    UOmegax_f_c = filter2D(UOmegax, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VOmegay_f_c = filter2D(VOmegay, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Filter the DNS velocity and vorticity fields to obtain their coarse-grained representations
    Uf_c = filter2D(U_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vf_c = filter2D(V_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omegaf_c = filter2D(Omega_DNS, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Compute the product of the filtered velocity components with the filtered vorticity and their spatial derivatives
    Uf_c_Omegaf_c = multiply_dealias(Uf_c, Omegaf_c, dealias=dealias)
    Vf_c_Omegaf_c = multiply_dealias(Vf_c, Omegaf_c, dealias=dealias)

    Uf_c_Omegaf_c_x = derivative(Uf_c_Omegaf_c, [1,0], Kx_LES, Ky_LES)
    Vf_c_Omegaf_c_y = derivative(Vf_c_Omegaf_c, [0,1], Kx_LES, Ky_LES)
    
    # Compute PiOmega by calculating the divergence of the filtered convective terms and subtracting it from the divergence of the non-filtered convective terms
    PiOmega = UOmegax_f_c + VOmegay_f_c - (Uf_c_Omegaf_c_x + Vf_c_Omegaf_c_y)

    return PiOmega


############################# Germano Decompostion of SGS Terms #############################
# Citation: Germano, M. (1986). A proposal for a redefinition of the turbulent stresses in the filtered Navier–Stokes equations. The Physics of fluids, 29(7), 2323-2324.
# Pope, S. B. (2000). Turbulent flows. Cambridge university press. Section 13.2.2 

# Tau - Leonard
def TauLeonard(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculates the Leonard stress components (Tau11Leonard, Tau12Leonard, Tau22Leonard) for turbulence modeling in fluid dynamics.
    This function operates on a given 2D vorticity field (Omega_DNS) by filtering the velocity field derived from the vorticity,
    computing the product of the filtered velocities, and then applying a secondary filter to both the velocity fields and their products.
    The Leonard stress components are obtained by subtracting the filtered product of the filtered velocities from the filtered product of the velocities.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D array representing the vorticity field.
    - filterType (str, optional): Specifies the type of filter to apply. Default is 'gaussian'. Other types could be 'box', 'boxSpectral', etc.
    - coarseGrainType (str, optional): Specifies the method for coarse graining, with 'spectral' as the default method.
    - Delta (float, optional): The width of the filter. Must be a positive real number.
    - N_LES (numpy.ndarray, optional): Specifies the size of the coarse-grained (LES) grid as a 2-element array.
    - dealias (bool, optional): Flag to enable or disable dealiasing in the multiplication of velocity fields. Default is True.

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The Leonard stress components (Tau11Leonard, Tau12Leonard, Tau22Leonard).

    Note:
    - The Leonard stress components represent interaction within large scales.
    """
    # Getting the shape of the vorticity field
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Assuming a domain size, could be parameters

    # Initialize wavenumbers for Fourier space operations
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Convert vorticity field to stream function field
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    
    # Convert stream function to velocity fields
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity fields
    Uf = filter2D(U_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Vf = filter2D(V_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)

    # Apply coarse graining to the filtered velocity fields
    Uf_f_c = filter2D(Uf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vf_f_c = filter2D(Vf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Compute the product of the filtered velocities and apply dealiasing if enabled
    Uf_f_c_Uf_f_c = multiply_dealias(Uf_f_c, Uf_f_c, dealias=dealias)
    Vf_f_c_Vf_f_c = multiply_dealias(Vf_f_c, Vf_f_c, dealias=dealias)
    Uf_f_c_Vf_f_c = multiply_dealias(Uf_f_c, Vf_f_c, dealias=dealias)

    # Compute the product of the velocities before filtering and apply dealiasing
    UfUf = multiply_dealias(Uf, Uf, dealias=dealias)
    VfVf = multiply_dealias(Vf, Vf, dealias=dealias)
    UfVf = multiply_dealias(Uf, Vf, dealias=dealias)

    # Filter the products of the velocities
    UfUf_f_c = filter2D(UfUf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VfVf_f_c = filter2D(VfVf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UfVf_f_c = filter2D(UfVf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Calculate the Leonard stress components by subtracting the filtered product of the filtered velocities from the filtered product of the velocities
    Tau11Leonard = UfUf_f_c - Uf_f_c_Uf_f_c
    Tau12Leonard = UfVf_f_c - Uf_f_c_Vf_f_c
    Tau22Leonard = VfVf_f_c - Vf_f_c_Vf_f_c

    return Tau11Leonard, Tau12Leonard, Tau22Leonard

# Tau - Cross
def TauCross(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculates the cross stress components (Tau11Cross, Tau12Cross, Tau22Cross) for turbulence modeling in fluid dynamics.
    This function derives velocity fields from a given 2D vorticity field (Omega_DNS), applies a filter to these fields to separate
    the large-scale (filtered) and small-scale (difference between original and filtered) components, and then computes the interactions
    between these scales. The cross stress components are calculated by filtering the product of large-scale and small-scale velocity components
    and subtracting the product of their individually filtered components.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D array representing the vorticity field.
    - filterType (str, optional): Specifies the type of filter to apply. Default is 'gaussian'. Other types could be 'box', 'boxSpectral', etc.
    - coarseGrainType (str, optional): Specifies the method for coarse graining, with 'spectral' as the default method.
    - Delta (float, optional): The width of the filter. Must be a positive real number.
    - N_LES (numpy.ndarray, optional): Specifies the size of the coarse-grained (LES) grid as a 2-element array.
    - dealias (bool, optional): Flag to enable or disable dealiasing in the multiplication of velocity fields. Default is True.

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The cross stress components (Tau11Cross, Tau12Cross, Tau22Cross).

    Note:
    - The cross stress components quantify the interactions between the resolved (large-scale) and unresolved (small-scale) motions,
      which are crucial for understanding energy transfer in turbulent flows.
    """
    # Getting the shape of the vorticity field
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Domain size, could be parameters

    # Initialize wavenumbers for Fourier space operations
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Convert vorticity field to stream function field
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    
    # Convert stream function to velocity fields
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity fields to obtain large-scale components
    Uf = filter2D(U_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Vf = filter2D(V_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)

    # Calculate small-scale components by subtracting filtered velocities from original velocities
    Ud = U_DNS - Uf
    Vd = V_DNS - Vf

    # Compute the interaction between large-scale and small-scale components
    UfUd = multiply_dealias(Uf, Ud, dealias=dealias)
    VfVd = multiply_dealias(Vf, Vd, dealias=dealias)
    UfVd = multiply_dealias(Uf, Vd, dealias=dealias)
    VfUd = multiply_dealias(Vf, Ud, dealias=dealias)

    # Filter the interactions
    UfUd_f_c = filter2D(UfUd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VfVd_f_c = filter2D(VfVd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UfVd_f_c = filter2D(UfVd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VfUd_f_c = filter2D(VfUd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    # Filter the large-scale and small-scale components individually
    Uf_f_c = filter2D(Uf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vf_f_c = filter2D(Vf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Ud_f_c = filter2D(Ud, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vd_f_c = filter2D(Vd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Uf_f_c_Ud_f_c = multiply_dealias(Uf_f_c, Ud_f_c, dealias=dealias)
    Vf_f_c_Vd_f_c = multiply_dealias(Vf_f_c, Vd_f_c, dealias=dealias)
    Uf_f_c_Vd_f_c = multiply_dealias(Uf_f_c, Vd_f_c, dealias=dealias)
    Vf_f_c_Ud_f_c = multiply_dealias(Vf_f_c, Ud_f_c, dealias=dealias)

    # Compute the cross stress components by subtracting the product of individually filtered components from the filtered interactions
    Tau11Cross = 2 * (UfUd_f_c - Uf_f_c_Ud_f_c)
    Tau12Cross = UfVd_f_c + VfUd_f_c - Uf_f_c_Vd_f_c - Vf_f_c_Ud_f_c
    Tau22Cross = 2 * (VfVd_f_c - Vf_f_c_Vd_f_c)

    return Tau11Cross, Tau12Cross, Tau22Cross


# Tau - Reynolds
def TauReynolds(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculates the Reynolds stress components (Tau11Reynolds, Tau12Reynolds, Tau22Reynolds) for turbulence modeling in fluid dynamics.
    This function operates on a given 2D vorticity field (Omega_DNS) by filtering the velocity field derived from the vorticity,
    computing the product of the filtered velocities, and then applying a secondary filter to both the velocity fields and their products.
    The Reynolds stress components are obtained by subtracting the product of the filtered velocities from the filtered product of the velocities.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D array representing the vorticity field.
    - filterType (str, optional): Specifies the type of filter to apply. Default is 'gaussian'. Other types could be 'box', 'boxSpectral', etc.
    - coarseGrainType (str, optional): Specifies the method for coarse graining, with 'spectral' as the default method.
    - Delta (float, optional): The width of the filter. Must be a positive real number.
    - N_LES (numpy.ndarray, optional): Specifies the size of the coarse-grained (LES) grid as a 2-element array.
    - dealias (bool, optional): Flag to enable or disable dealiasing in the multiplication of velocity fields. Default is True.

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The Reynolds stress components (Tau11Reynolds, Tau12Reynolds, Tau22Reynolds).

    Note:
    - The Reynolds stress components represent interaction within small scales.
    """
    # Getting the shape of the vorticity field
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Assuming a domain size, could be parameters

    # Initialize wavenumbers for Fourier space operations
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Convert vorticity field to stream function field
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)

    # Convert stream function to velocity fields
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity fields
    Uf = filter2D(U_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Vf = filter2D(V_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)

    Ud = U_DNS - Uf
    Vd = V_DNS - Vf

    UdUd = multiply_dealias(Ud, Ud, dealias=dealias)
    VdVd = multiply_dealias(Vd, Vd, dealias=dealias)
    UdVd = multiply_dealias(Ud, Vd, dealias=dealias)

    UdUd_f_c = filter2D(UdUd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VdVd_f_c = filter2D(VdVd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UdVd_f_c = filter2D(UdVd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Udf_c = filter2D(Ud, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vdf_c = filter2D(Vd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Udf_c_Udf_c = multiply_dealias(Udf_c, Udf_c, dealias=dealias)
    Vdf_c_Vdf_c = multiply_dealias(Vdf_c, Vdf_c, dealias=dealias)
    Udf_c_Vdf_c = multiply_dealias(Udf_c, Vdf_c, dealias=dealias)

    Tau11Reynolds = UdUd_f_c - Udf_c_Udf_c
    Tau12Reynolds = UdVd_f_c - Udf_c_Vdf_c
    Tau22Reynolds = VdVd_f_c - Vdf_c_Vdf_c

    return Tau11Reynolds, Tau12Reynolds, Tau22Reynolds


# Sigma - Leonard
def SigmaLeonard(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculates the Leonard stress components (Sigma1Leonard, Sigma2Leonard) for turbulence modeling in fluid dynamics.
    This function operates on a given 2D vorticity field (Omega_DNS) by filtering the velocity field derived from the vorticity,
    computing the product of the filtered velocities, and then applying a secondary filter to both the velocity fields and their products.
    The Leonard stress components are obtained by subtracting the filtered product of the filtered velocities from the filtered product of the velocities.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D array representing the vorticity field.
    - filterType (str, optional): Specifies the type of filter to apply. Default is 'gaussian'. Other types could be 'box', 'boxSpectral', etc.
    - coarseGrainType (str, optional): Specifies the method for coarse graining, with 'spectral' as the default method.
    - Delta (float, optional): The width of the filter. Must be a positive real number.
    - N_LES (numpy.ndarray, optional): Specifies the size of the coarse-grained (LES) grid as a 2-element array.
    - dealias (bool, optional): Flag to enable or disable dealiasing in the multiplication of velocity fields. Default is True.

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray]: The Leonard stress components (Sigma1Leonard, Sigma2Leonard).

    Note:
    - The Leonard stress components represent interaction within large scales and are crucial for capturing the effects of resolved scales on unresolved scales.
    """
    
    # Extract the dimensions of the DNS grid and define the domain size for periodic boundary conditions
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Assuming a square domain

    # Initialize wavenumbers for Fourier space operations based on the DNS grid dimensions
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Convert the vorticity field to a stream function, then derive the velocity components
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity and vorticity fields to obtain their coarse-grained representations
    Uf = filter2D(U_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Vf = filter2D(V_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Omegaf = filter2D(Omega_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)

    # Compute the product of the filtered velocity fields with the filtered vorticity
    UfOmegaf = multiply_dealias(Uf, Omegaf, dealias=dealias)
    VfOmegaf = multiply_dealias(Vf, Omegaf, dealias=dealias)

    # Apply a secondary filter to the products of the filtered velocity fields and vorticity
    UfOmegaf_f_c = filter2D(UfOmegaf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VfOmegaf_f_c = filter2D(VfOmegaf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Uf_f_c = filter2D(Uf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vf_f_c = filter2D(Vf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omegaf_f_c = filter2D(Omegaf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Uf_f_c_Omegad_f_c = multiply_dealias(Uf_f_c, Omegaf_f_c, dealias=dealias)
    Vf_f_c_Omegad_f_c = multiply_dealias(Vf_f_c, Omegaf_f_c, dealias=dealias)

    # Compute the Leonard stress components by subtracting the product of the secondary filtered velocities from the secondary filtered product of the velocities
    Sigma1Leonard = UfOmegaf_f_c - Uf_f_c_Omegad_f_c
    Sigma2Leonard = VfOmegaf_f_c - Vf_f_c_Omegad_f_c

    return Sigma1Leonard, Sigma2Leonard


# Sigma - Cross
def SigmaCross(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculates the cross stress components (Sigma1Cross, Sigma2Cross) for turbulence modeling in fluid dynamics.
    This function derives velocity fields from a given 2D vorticity field (Omega_DNS), applies a filter to these fields to separate
    the large-scale (filtered) and small-scale (difference between original and filtered) components, and then computes the interactions
    between these scales. The cross stress components are calculated by filtering the product of large-scale and small-scale velocity components
    and subtracting the product of their individually filtered components.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D array representing the vorticity field.
    - filterType (str, optional): Specifies the type of filter to apply. Default is 'gaussian'. Other types could be 'box', 'boxSpectral', etc.
    - coarseGrainType (str, optional): Specifies the method for coarse graining, with 'spectral' as the default method.
    - Delta (float, optional): The width of the filter. Must be a positive real number.
    - N_LES (numpy.ndarray, optional): Specifies the size of the coarse-grained (LES) grid as a 2-element array.
    - dealias (bool, optional): Flag to enable or disable dealiasing in the multiplication of velocity fields. Default is True.

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray]: The cross stress components (Sigma1Cross, Sigma2Cross).

    Note:
    - The cross stress components quantify the interactions between the resolved (large-scale) and unresolved (small-scale) motions,
      which are crucial for understanding energy transfer in turbulent flows.
    """
    # Getting the shape of the vorticity field
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Domain size, could be parameters

    # Initialize wavenumbers for Fourier space operations
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Convert vorticity field to stream function field
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    
    # Calculate the velocity components from the stream function by taking spatial derivatives
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity fields
    Uf = filter2D(U_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Vf = filter2D(V_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Omegaf = filter2D(Omega_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)

    Ud = U_DNS - Uf
    Vd = V_DNS - Vf
    Omegad = Omega_DNS - Omegaf

    # Apply the specified filter to the velocity fields to obtain their coarse-grained representations
    Uf_f_c = filter2D(Uf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vf_f_c = filter2D(Vf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omegaf_f_c = filter2D(Omegaf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Ud_f_c = filter2D(Ud, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vd_f_c = filter2D(Vd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omegad_f_c = filter2D(Omegad, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    UfOmegad = multiply_dealias(Uf, Omegad, dealias=dealias)
    VfOmegad = multiply_dealias(Vf, Omegad, dealias=dealias)
    UdOmegaf = multiply_dealias(Ud, Omegaf, dealias=dealias)
    VdOmegaf = multiply_dealias(Vd, Omegaf, dealias=dealias)

    UfOmegad_f_c = filter2D(UfOmegad, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VfOmegad_f_c = filter2D(VfOmegad, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UdOmegaf_f_c = filter2D(UdOmegaf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VdOmegaf_f_c = filter2D(VdOmegaf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Uf_f_c_Omegad_f_c = multiply_dealias(Uf_f_c, Omegad_f_c, dealias=dealias)
    Vf_f_c_Omegad_f_c = multiply_dealias(Vf_f_c, Omegad_f_c, dealias=dealias)
    Ud_f_c_Omegaf_f_c = multiply_dealias(Ud_f_c, Omegaf_f_c, dealias=dealias)
    Vd_f_c_Omegaf_f_c = multiply_dealias(Vd_f_c, Omegaf_f_c, dealias=dealias)

    Sigma1Cross = UfOmegad_f_c + UdOmegaf_f_c - Uf_f_c_Omegad_f_c - Ud_f_c_Omegaf_f_c
    Sigma2Cross = VfOmegad_f_c + VdOmegaf_f_c - Vf_f_c_Omegad_f_c - Vd_f_c_Omegaf_f_c

    return Sigma1Cross, Sigma2Cross

# Sigma - Reynolds
def SigmaReynolds(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculates the Reynolds stress components (Sigma1Reynolds, Sigma2Reynolds) for turbulence modeling in fluid dynamics.
    This function operates on a given 2D vorticity field (Omega_DNS) by filtering the velocity field derived from the vorticity,
    computing the product of the filtered velocities, and then applying a secondary filter to both the velocity fields and their products.
    The Reynolds stress components are obtained by subtracting the product of the filtered velocities from the filtered product of the velocities.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D array representing the vorticity field.
    - filterType (str, optional): Specifies the type of filter to apply. Default is 'gaussian'. Other types could be 'box', 'boxSpectral', etc.
    - coarseGrainType (str, optional): Specifies the method for coarse graining, with 'spectral' as the default method.
    - Delta (float, optional): The width of the filter. Must be a positive real number.
    - N_LES (numpy.ndarray, optional): Specifies the size of the coarse-grained (LES) grid as a 2-element array.
    - dealias (bool, optional): Flag to enable or disable dealiasing in the multiplication of velocity fields. Default is True.

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray]: The Reynolds stress components (Sigma1Reynolds, Sigma2Reynolds).

    Note:
    - The Reynolds stress components represent interaction within small scales.
    """
    # Getting the shape of the vorticity field
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Assuming a domain size, could be parameters

    # Initialize wavenumbers for Fourier space operations
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    # Convert vorticity field to stream function field
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    
    # Calculate the velocity components from the stream function by taking spatial derivatives
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity fields
    Uf = filter2D(U_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Vf = filter2D(V_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Omegaf = filter2D(Omega_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)

    Ud = U_DNS - Uf
    Vd = V_DNS - Vf
    Omegad = Omega_DNS - Omegaf

    Ud_f_c = filter2D(Ud, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vd_f_c = filter2D(Vd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omegad_f_c = filter2D(Omegad, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Ud_f_c_Omegad_f_c = multiply_dealias(Ud_f_c, Omegad_f_c, dealias=dealias)
    Vd_f_c_Omegad_f_c = multiply_dealias(Vd_f_c, Omegad_f_c, dealias=dealias)

    UdOmegad = multiply_dealias(Ud, Omegad, dealias=dealias)
    VdOmegad = multiply_dealias(Vd, Omegad, dealias=dealias)

    UdOmegad_f_c = filter2D(UdOmegad, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VdOmegad_f_c = filter2D(VdOmegad, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Sigma1Reynolds = UdOmegad_f_c - Ud_f_c_Omegad_f_c
    Sigma2Reynolds = VdOmegad_f_c - Vd_f_c_Omegad_f_c

    return Sigma1Reynolds, Sigma2Reynolds

# PiOmega - Leonard
def PiOmegaLeonard(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculates the Leonard stress components (PiOmega) for turbulence modeling in fluid dynamics.
    This function operates on a given 2D vorticity field (Omega_DNS) by filtering the velocity field derived from the vorticity,
    computing the product of the filtered velocities, and then applying a secondary filter to both the velocity fields and their products.
    The PiOmega term is obtained by subtracting the product of the filtered velocities from the filtered product of the velocities.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D array representing the vorticity field.
    - filterType (str, optional): Specifies the type of filter to apply. Default is 'gaussian'. Other types could be 'box', 'boxSpectral', etc.
    - coarseGrainType (str, optional): Specifies the method for coarse graining, with 'spectral' as the default method.
    - Delta (float, optional): The width of the filter. Must be a positive real number.
    - N_LES (numpy.ndarray, optional): Specifies the size of the coarse-grained (LES) grid as a 2-element array.
    - dealias (bool, optional): Flag to enable or disable dealiasing in the multiplication of velocity fields. Default is True.

    Returns:
    - numpy.ndarray: The PiOmega term.

    Note:
    - The PiOmega term represents the interaction between the vorticity and velocity fields and is crucial for capturing the effects of resolved scales on unresolved scales.
    """
    # Extract the dimensions of the DNS grid and define the domain size for periodic boundary conditions
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Assuming a square domain

    # Determine the LES grid dimensions based on the coarse graining type
    NX_LES, NY_LES = (NX_DNS, NY_DNS) if coarseGrainType in [None, 'physical'] else N_LES

    # Initialize wavenumbers for Fourier space operations based on the DNS grid dimensions
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')
    Kx_LES, Ky_LES, _, _, invKsq_LES = initialize_wavenumbers_rfft2(NX_LES, NY_LES, Lx, Ly, INDEXING='ij')

    # Convert the vorticity field to a stream function, then derive the velocity components
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity and vorticity fields to obtain their coarse-grained representations
    Uf = filter2D(U_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Vf = filter2D(V_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Omegaf = filter2D(Omega_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)

    # Compute the product of the filtered velocity fields with the filtered vorticity
    UfOmegaf = multiply_dealias(Uf, Omegaf, dealias=dealias)
    VfOmegaf = multiply_dealias(Vf, Omegaf, dealias=dealias)

    UfOmegaf_x = derivative(UfOmegaf, [1,0], Kx_DNS, Ky_DNS)
    VfOmegaf_y = derivative(VfOmegaf, [0,1], Kx_DNS, Ky_DNS)

    # Apply a secondary filter to the products of the filtered velocity fields and vorticity
    UfOmegaf_x_f_c = filter2D(UfOmegaf_x, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VfOmegaf_y_f_c = filter2D(VfOmegaf_y, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Uf_f_c = filter2D(Uf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vf_f_c = filter2D(Vf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omegaf_f_c = filter2D(Omegaf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Uf_f_c_Omegaf_f_c = multiply_dealias(Uf_f_c, Omegaf_f_c, dealias=dealias)
    Vf_f_c_Omegaf_f_c = multiply_dealias(Vf_f_c, Omegaf_f_c, dealias=dealias)

    Uf_f_c_Omegaf_f_c_x = derivative(Uf_f_c_Omegaf_f_c, [1,0], Kx_LES, Ky_LES)
    Vf_f_c_Omegad_f_c_y = derivative(Vf_f_c_Omegaf_f_c, [0,1], Kx_LES, Ky_LES)

    # Compute the Leonard stress components by subtracting the product of the secondary filtered velocities from the secondary filtered product of the velocities
    PiOmegaLeonard = UfOmegaf_x_f_c + VfOmegaf_y_f_c - Uf_f_c_Omegaf_f_c_x - Vf_f_c_Omegad_f_c_y

    return PiOmegaLeonard

# PiOmega - Cross
def PiOmegaCross(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculates the cross stress components (PiOmega) for turbulence modeling in fluid dynamics.
    This function derives velocity fields from a given 2D vorticity field (Omega_DNS), applies a filter to these fields to separate
    the large-scale (filtered) and small-scale (difference between original and filtered) components, and then computes the interactions
    between these scales. The PiOmega term is calculated by filtering the product of large-scale and small-scale velocity components
    and subtracting the product of their individually filtered components.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D array representing the vorticity field.
    - filterType (str, optional): Specifies the type of filter to apply. Default is 'gaussian'. Other types could be 'box', 'boxSpectral', etc.
    - coarseGrainType (str, optional): Specifies the method for coarse graining, with 'spectral' as the default method.
    - Delta (float, optional): The width of the filter. Must be a positive real number.
    - N_LES (numpy.ndarray, optional): Specifies the size of the coarse-grained (LES) grid as a 2-element array.
    - dealias (bool, optional): Flag to enable or disable dealiasing in the multiplication of velocity fields. Default is True.

    Returns:
    - numpy.ndarray: The PiOmega term.

    Note:
    - The PiOmega term represents the interaction between the vorticity and velocity fields and is crucial for capturing the effects of resolved scales on unresolved scales.
    """
    # Extract the dimensions of the DNS grid and define the domain size for periodic boundary conditions
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Assuming a square domain

    # Determine the LES grid dimensions based on the coarse graining type
    NX_LES, NY_LES = (NX_DNS, NY_DNS) if coarseGrainType in [None, 'physical'] else N_LES

    # Initialize wavenumbers for Fourier space operations based on the DNS grid dimensions
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')
    Kx_LES, Ky_LES, _, _, invKsq_LES = initialize_wavenumbers_rfft2(NX_LES, NY_LES, Lx, Ly, INDEXING='ij')

    # Convert the vorticity field to a stream function, then derive the velocity components
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity and vorticity fields to obtain their coarse-grained representations
    Uf = filter2D(U_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Vf = filter2D(V_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Omegaf = filter2D(Omega_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)

    Ud = U_DNS - Uf
    Vd = V_DNS - Vf
    Omegad = Omega_DNS - Omegaf

    # Apply the specified filter to the velocity fields to obtain their coarse-grained representations
    Uf_f_c = filter2D(Uf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vf_f_c = filter2D(Vf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omegaf_f_c = filter2D(Omegaf, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Ud_f_c = filter2D(Ud, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vd_f_c = filter2D(Vd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omegad_f_c = filter2D(Omegad, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    UfOmegad = multiply_dealias(Uf, Omegad, dealias=dealias)
    VfOmegad = multiply_dealias(Vf, Omegad, dealias=dealias)
    UdOmegaf = multiply_dealias(Ud, Omegaf, dealias=dealias)
    VdOmegaf = multiply_dealias(Vd, Omegaf, dealias=dealias)

    UfOmegad_x = derivative(UfOmegad, [1,0], Kx_DNS, Ky_DNS)
    VfOmegad_y = derivative(VfOmegad, [0,1], Kx_DNS, Ky_DNS)
    UdOmegaf_x = derivative(UdOmegaf, [1,0], Kx_DNS, Ky_DNS)
    VdOmegaf_y = derivative(VdOmegaf, [0,1], Kx_DNS, Ky_DNS)

    UfOmegad_x_f_c = filter2D(UfOmegad_x, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VfOmegad_y_f_c = filter2D(VfOmegad_y, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    UdOmegaf_x_f_c = filter2D(UdOmegaf_x, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VdOmegaf_y_f_c = filter2D(VdOmegaf_y, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Uf_f_c_Omegad_f_c = multiply_dealias(Uf_f_c, Omegad_f_c, dealias=dealias)
    Vf_f_c_Omegad_f_c = multiply_dealias(Vf_f_c, Omegad_f_c, dealias=dealias)
    Ud_f_c_Omegaf_f_c = multiply_dealias(Ud_f_c, Omegaf_f_c, dealias=dealias)
    Vd_f_c_Omegaf_f_c = multiply_dealias(Vd_f_c, Omegaf_f_c, dealias=dealias)

    Uf_f_c_Omegad_f_c_x = derivative(Uf_f_c_Omegad_f_c, [1,0], Kx_LES, Ky_LES)
    Vf_f_c_Omegad_f_c_y = derivative(Vf_f_c_Omegad_f_c, [0,1], Kx_LES, Ky_LES)
    Ud_f_c_Omegaf_f_c_x = derivative(Ud_f_c_Omegaf_f_c, [1,0], Kx_LES, Ky_LES)
    Vd_f_c_Omegaf_f_c_y = derivative(Vd_f_c_Omegaf_f_c, [0,1], Kx_LES, Ky_LES)

    PiOmegaCross = UfOmegad_x_f_c + UdOmegaf_x_f_c + VfOmegad_y_f_c + VdOmegaf_y_f_c - Uf_f_c_Omegad_f_c_x - Ud_f_c_Omegaf_f_c_x - Vf_f_c_Omegad_f_c_y - Vd_f_c_Omegaf_f_c_y
    
    return PiOmegaCross

# PiOmega - Reynolds
def PiOmegaReynolds(Omega_DNS, filterType='gaussian', coarseGrainType='spectral', Delta=None, N_LES=None, dealias=True):
    """
    Calculates the Reynolds stress components (PiOmega) for turbulence modeling in fluid dynamics.
    This function operates on a given 2D vorticity field (Omega_DNS) by filtering the velocity field derived from the vorticity,
    computing the product of the filtered velocities, and then applying a secondary filter to both the velocity fields and their products.
    The PiOmega term is obtained by subtracting the product of the filtered velocities from the filtered product of the velocities.

    Parameters:
    - Omega_DNS (numpy.ndarray): 2D array representing the vorticity field.
    - filterType (str, optional): Specifies the type of filter to apply. Default is 'gaussian'. Other types could be 'box', 'boxSpectral', etc.
    - coarseGrainType (str, optional): Specifies the method for coarse graining, with 'spectral' as the default method.
    - Delta (float, optional): The width of the filter. Must be a positive real number.
    - N_LES (numpy.ndarray, optional): Specifies the size of the coarse-grained (LES) grid as a 2-element array.
    - dealias (bool, optional): Flag to enable or disable dealiasing in the multiplication of velocity fields. Default is True.

    Returns:
    - numpy.ndarray: The PiOmega term.

    Note:
    - The PiOmega term represents the interaction between the vorticity and velocity fields and is crucial for capturing the effects of resolved scales on unresolved scales.
    """
    # Extract the dimensions of the DNS grid and define the domain size
    NX_DNS, NY_DNS = Omega_DNS.shape
    Lx, Ly = 2 * np.pi, 2 * np.pi  # Assuming a square domain with periodic boundary conditions

    # Determine the LES grid dimensions based on the coarse graining type
    NX_LES, NY_LES = (NX_DNS, NY_DNS) if coarseGrainType in [None, 'physical'] else N_LES
    
    # Initialize wavenumbers for spectral domain operations for both DNS and LES grids
    Kx_DNS, Ky_DNS, _, _, invKsq_DNS = initialize_wavenumbers_rfft2(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')
    Kx_LES, Ky_LES, _, _, invKsq_LES = initialize_wavenumbers_rfft2(NX_LES, NY_LES, Lx, Ly, INDEXING='ij')

    # Convert the vorticity field to a stream function, then derive velocity components
    Psi_DNS = Omega2Psi(Omega=Omega_DNS, invKsq=invKsq_DNS)
    U_DNS, V_DNS = Psi2UV(Psi=Psi_DNS, Kx=Kx_DNS, Ky=Ky_DNS)

    # Filter the velocity fields
    Uf = filter2D(U_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Vf = filter2D(V_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)
    Omegaf = filter2D(Omega_DNS, filterType=filterType, coarseGrainType=None, Delta=Delta, Ngrid=N_LES)

    Ud = U_DNS - Uf
    Vd = V_DNS - Vf
    Omegad = Omega_DNS - Omegaf

    Ud_f_c = filter2D(Ud, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Vd_f_c = filter2D(Vd, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    Omegad_f_c = filter2D(Omegad, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    Ud_f_c_Omegad_f_c = multiply_dealias(Ud_f_c, Omegad_f_c, dealias=dealias)
    Vd_f_c_Omegad_f_c = multiply_dealias(Vd_f_c, Omegad_f_c, dealias=dealias)

    Ud_f_c_Omegad_f_c_x = derivative(Ud_f_c_Omegad_f_c, [1,0], Kx_LES, Ky_LES)
    Vd_f_c_Omegad_f_c_y = derivative(Vd_f_c_Omegad_f_c, [0,1], Kx_LES, Ky_LES)

    UdOmegad = multiply_dealias(Ud, Omegad, dealias=dealias)
    VdOmegad = multiply_dealias(Vd, Omegad, dealias=dealias)

    UdOmegad_x = derivative(UdOmegad, [1,0], Kx_DNS, Ky_DNS)
    VdOmegad_y = derivative(VdOmegad, [0,1], Kx_DNS, Ky_DNS)

    UdOmegad_x_f_c = filter2D(UdOmegad_x, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)
    VdOmegad_y_f_c = filter2D(VdOmegad_y, filterType=filterType, coarseGrainType=coarseGrainType, Delta=Delta, Ngrid=N_LES)

    PiOmegaReynolds = UdOmegad_x_f_c - Ud_f_c_Omegad_f_c_x + VdOmegad_y_f_c - Vd_f_c_Omegad_f_c_y

    return PiOmegaReynolds