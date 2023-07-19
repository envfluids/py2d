import numpy as np
      
def initialize_perturbation(NX, Kx, Ky):
    # -------------- Initialization using pertubration --------------
    kp = 10.0
    A  = 4*np.power(kp,(-5))/(3*np.pi)
    absK = np.sqrt(Kx*Kx+Ky*Ky)
    Ek = A*np.power(absK,4)*np.exp(-np.power(absK/kp,2))
    coef1 = np.random.uniform(0,1,[NX//2+1,NX//2+1])*np.pi*2
    coef2 = np.random.uniform(0,1,[NX//2+1,NX//2+1])*np.pi*2

    perturb = np.zeros([NX,NX])
    perturb[0:NX//2+1, 0:NX//2+1] = coef1[0:NX//2+1, 0:NX//2+1]+coef2[0:NX//2+1, 0:NX//2+1]
    perturb[NX//2+1:, 0:NX//2+1] = coef2[NX//2-1:0:-1, 0:NX//2+1] - coef1[NX//2-1:0:-1, 0:NX//2+1]
    perturb[0:NX//2+1, NX//2+1:] = coef1[0:NX//2+1, NX//2-1:0:-1] - coef2[0:NX//2+1, NX//2-1:0:-1]
    perturb[NX//2+1:, NX//2+1:] = -(coef1[NX//2-1:0:-1, NX//2-1:0:-1] + coef2[NX//2-1:0:-1, NX//2-1:0:-1])
    perturb = np.exp(1j*perturb)

    w1_hat = np.sqrt(absK/np.pi*Ek)*perturb*np.power(NX,2)

    psi_hat = -w1_hat*invKsq
    psiPrevious_hat = psi_hat.astype(np.complex128)
    psiCurrent_hat = psi_hat.astype(np.complex128)
    
    return w1_hat, psi_hat, psiPrevious_hat, psiCurrent_hat
    
    
#def initialize_resume(NX, Kx, Ky):

def initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly, INDEXING='ij'):
    '''
    Initialize the wavenumbers for 2D Forced Homogeneous Isotropic Turbulence (2D-FHIT).
    
    Parameters:
    -----------
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    Lx : float
        Length of the domain in the x-direction.
    Ly : float
        Length of the domain in the y-direction.

    Returns:
    --------
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.
    '''

    # Compute the wavenumbers in the x-direction based on the FFT frequencies.
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)

    # Compute the wavenumbers in the y-direction based on the FFT frequencies.
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)

    # Create 2D arrays of the wavenumbers using a meshgrid. 
    (Kx, Ky) = np.meshgrid(kx, ky, indexing=INDEXING)

    # Calculate the squared magnitudes of the wavenumbers. 
    Ksq = Kx ** 2 + Ky ** 2

    # Set the zero wavenumber to a large value to avoid division by zero.
    Ksq[0,0] = 1e16

    # Return the wavenumbers in the x and y direction and their squared magnitudes. 
    return Kx, Ky, Ksq


def gridgen(Lx, Ly, Nx, Ny, INDEXING='ij'):
    '''
    Generate a 2D grid.

    Parameters:
    -----------
    Lx : float
        Length of the domain in the x-direction.
    NX : int
        Number of grid points in the x and y-directions.
    INDEXING : str, optional
        Convention to use for indexing. Default is 'ij' (matrix indexing).

    Returns:
    --------
    Lx : float
        Length of the domain in the x-direction.
    Lx : float
        Length of the domain in the y-direction (same as x-direction as grid is square).
    X : numpy.ndarray
        2D array of x-coordinates.
    Y : numpy.ndarray
        2D array of y-coordinates.
    dx : float
        Size of grid spacing in the x-direction.
    dx : float
        Size of grid spacing in the y-direction (same as x-direction as grid is square).
    '''

    # Calculate the size of the grid spacing
    dx = Lx / Nx
    dy = Ly / Ny

    # Create an array of x-coordinates, ranging from 0 to (Lx - dx)
    x = np.linspace(0, Lx - dx, num=Nx)
    y = np.linspace(0, Lx - dx, num=Ny)

    # Create 2D arrays of the x and y-coordinates using a meshgrid.
    X, Y = np.meshgrid(x, y, indexing=INDEXING)

    # Return the lengths of the domain, the x and y-coordinates, and the size of the grid spacing.
    return Lx, Ly, X, Y, dx, dy
