import numpy as np
from py2d.initialize import initialize_wavenumbers_2DFHIT

def filter2D_2DFHIT(U, filterType='gaussian', coarseGrainType='spectral', Delta=None, Ngrid=None):
    """
    Filters and coarse grains 2D square grids
    Use Case: 2D Forced Homogenous Isotropic Tubulence (2D-FHIT)
    """

    if Delta is None:
        if Ngrid is None:
            raise ValueError("Must provide either Delta or Ngrid")
        else:
            Delta = 2*np.pi/Ngrid[0]

    NX_DNS, NY_DNS = np.shape(U)
    Lx, Ly = 2*np.pi, 2*np.pi # Domain size

    # Get the wavenumbers for DNS grid
    _, _, _, Ksq_DNS, _ = initialize_wavenumbers_2DFHIT(NX_DNS, NY_DNS, Lx, Ly, INDEXING='ij')

    U_hat = np.fft.fft2(U)

    if filterType == 'gaussian':
        Gk = np.exp(-Ksq_DNS*(Delta**2)/24)
        U_f_hat = Gk*U_hat

    return np.real(np.fft.ifft2(U_f_hat))