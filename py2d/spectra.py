import numpy as np

from py2d.initialize import initialize_wavenumbers_2DFHIT
from py2d.derivative import derivative_2DFHIT

def spectrum_angled_average_2DFHIT(A, spectral = False):
    '''
    Compute the radially/angle-averaged spectrum of a 2D square matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The input 2D square matrix. If `spectral` is False, `A` is in the physical domain; otherwise, it is in the spectral domain.
    spectral : bool, optional
        Whether `A` is in the spectral domain. Default is False.

    Returns
    -------
    spectrum : numpy.ndarray
        The radially/angle-averaged spectrum of `A`.
    wavenumbers : numpy.ndarray
        The corresponding wavenumbers.

    Raises
    ------
    ValueError
        If `A` is not a 2D square matrix or `spectral` is not a boolean or `A` contains non-numeric values.

    Notes
    -----
    To calculate angled-averaged spectrum of absolute value of a complex-valued matrix, input np.sqrt(np.conj(A_hat)*A_hat) as A.
    np.abs(A_hat) is not the same as np.sqrt(np.conj(A_hat)*A_hat) for complex-valued A_hat.
    np.abs(A_hat) calculates the magnitude (sqrt(a^2 + b^2)) of the complex-valued A_hat, 
    while np.sqrt(np.conj(A_hat)*A_hat) calculates the absolute value (sqrt(a^2 + (ib)^2)) of A_hat where A_hat = a + ib.
    '''

    # Check if input 'A' is a 2D square matrix
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('Input is not a 2D square matrix. Please input a 2D square matrix')
    # Check if 'spectral' is a boolean value
    if not isinstance(spectral, bool):
        raise ValueError('Invalid input for spectral. It should be a boolean value')
    # Check if input 'A' contains non-numeric values
    if not np.issubdtype(A.dtype, np.number):
        raise ValueError('Input contains non-numeric values')

    # Calculate the number of grid points in one dimension
    nx,ny = A.shape
    # Set the size of the domain
    Lx, Ly = 2 * np.pi, 2 * np.pi

    _, _, Kabs, _, _ = initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly, INDEXING='ij')

    # Compute the 2D FFT of 'A' if it is in the physical domain
    if not spectral:
        A_hat = np.fft.fft2(A)
    else:  # If 'A' is already in the spectral domain, copy it to 'spectral_A'
        A_hat = A
    # Normalize the spectrum by the total number of grid points
    A_hat = A_hat / nx ** 2

    # Calculate the maximum wavenumber to be considered in the average
    kxMax = round(nx / 2)

    # Initialize the output array with zeros
    A_angled_average_spectra = np.zeros(kxMax + 1)
    # The zeroth wavenumber is just the first element of the spectrum
    A_angled_average_spectra[0] = A_hat[0,0]

    # Compute the angle-averaged spectrum for wavenumbers 1 to kxMax
    for k in range(1, kxMax + 1):
        tempInd = (Kabs >= (k - 0.5)) & (Kabs < (k + 0.5))
        A_angled_average_spectra[k] = np.sum(A_hat[tempInd])

    # Generate the array of wavenumbers corresponding to the computed spectrum
    wavenumbers = np.linspace(0, kxMax, len(A_angled_average_spectra))

    return A_angled_average_spectra, wavenumbers


def TKE_angled_average_2DFHIT(Psi, Omega, spectral=False):
    '''
    Calculate the energy spectrum and its angled average.

    Parameters
    ----------
    Psi : numpy.ndarray
        The input 2D square matrix of stream function. If `spectral` is False, `Psi` is in the physical domain; otherwise, it is in the spectral domain.
    Omega : numpy.ndarray
        The input 2D square matrix of vorticity. If `spectral` is False, `Omega` is in the physical domain; otherwise, it is in the spectral domain.
    spectral : bool, optional
        Whether `Psi` and `Omega` are in the spectral domain. Default is False.

    Returns
    -------
    TKE_angled_average : numpy.ndarray
        The radially/angle-averaged energy spectrum of `E_hat`.
    kkx : numpy.ndarray
        The corresponding wavenumbers.
    '''

    # Get the size of the input square matrix.
    NX = Psi.shape[0]

    # If the data is in physical space, convert it to spectral space using 2D FFT. 
    if not spectral:
        Psi_hat = np.fft.fft2(Psi)
        Omega_hat = np.fft.fft2(Omega)
    else:  # If the data is already in the spectral space, we don't need to transform it.
        Psi_hat = Psi
        Omega_hat = Omega

    # Compute the kinetic energy in the spectral space.
    E_hat = 0.5 * (np.conj(Psi_hat) * Omega_hat) / NX ** 2

    # Perform an angle-averaged spectrum calculation on the computed kinetic energy.
    TKE_angled_average, kkx = spectrum_angled_average_2DFHIT(E_hat, spectral=True)

    # Return the angle-averaged kinetic energy spectrum and the corresponding wavenumbers.
    return TKE_angled_average, kkx


def enstrophy_angled_average_2DFHIT(Omega, spectral=False):
    '''
    Calculate the enstrophy spectrum and its angled average.

    Parameters
    ----------
    Omega : numpy.ndarray
        The input 2D square matrix of vorticity. If `spectral` is False, `Omega` is in the physical domain; otherwise, it is in the spectral domain.
    spectral : bool, optional
        Whether `Omega` is in the spectral domain. Default is False.

    Returns
    -------
    Z_angled_average_spectra : numpy.ndarray
        The radially/angle-averaged enstrophy spectrum of `Z_hat`.
    kkx : numpy.ndarray
        The corresponding wavenumbers.
    '''

    # Get the size of the input square matrix.
    NX = Omega.shape[0]

    # If the data is in physical space, convert it to spectral space using 2D FFT. 
    if not spectral:
        Omega_hat = np.fft.fft2(Omega)
    else:  # If the data is already in the spectral space, we don't need to transform it.
        Omega_hat = Omega

    # Compute the enstrophy in the spectral space.
    Z_hat = 0.5 * (np.conj(Omega_hat) * Omega_hat) / NX ** 2

    # Perform an angle-averaged spectrum calculation on the computed enstrophy.
    Z_angled_average_spectra, kkx = spectrum_angled_average_2DFHIT(Z_hat, spectral=True)

    # Return the angle-averaged enstrophy spectrum and the corresponding wavenumbers.
    return Z_angled_average_spectra, kkx

def energyTransfer_spectra_2DFHIT(Kx, Ky, U=None, V=None, Tau11=None, Tau12=None, Tau22=None, Psi=None, PiOmega=None, method='Tau', spectral=False):
    '''
    Compute the energy transfer spectra using 2D Forced Homogeneous Isotropic Turbulence (2D-FHIT)
    PTau > 0: dissipation
    PTau < 0: backscatter

    Parameters
    ----------
    Kx, Ky : numpy.ndarray
        The 2D wavenumber arrays
    U : numpy.ndarray, optional
        The U velocity field (2D array)
    V : numpy.ndarray, optional
        The V velocity field (2D array)
    Tau11 : numpy.ndarray, optional
        The Tau11 field (2D array)
    Tau12 : numpy.ndarray, optional
        The Tau12 field (2D array)
    Tau22 : numpy.ndarray, optional
        The Tau22 field (2D array)
    Psi : numpy.ndarray, optional
        The Psi field (2D array)
    PiOmega : numpy.ndarray, optional
        The PiOmega field (2D array)
    method : str, optional
        The method to calculate Ptau_hat. 'Tau' (default) or 'PiOmega'
    spectral : bool, optional
        Determines whether the input is in the 'physical' space (default=False) or 'spectral' space

    Returns
    -------
    spectra : numpy.ndarray
        The computed energy transfer spectra
    wavenumber : numpy.ndarray
        The corresponding wavenumbers

    This function computes the energy transfer spectra of the given 2D velocity fields
    using the 2D Forced Homogeneous Isotropic turbulence. The inputs can be in the physical or spectral space.
    '''

    # If the method is 'Tau', calculate the energy transfer using Tau11, Tau12, Tau22, U, and V.
    if method == 'Tau':
        # Check if the necessary parameters are provided.
        if U is None or V is None or Tau11 is None or Tau12 is None or Tau22 is None:
            raise ValueError("U, V, Tau11, Tau12, Tau22 must be provided to calculate energy transfer using 'Tau' method")

        # Calculate the derivatives of U with respect to x and y, and the derivative of V with respect to x.
        Ux = derivative_2DFHIT(U, [1, 0], Kx=Kx, Ky=Ky, spectral=spectral)
        Uy = derivative_2DFHIT(U, [0, 1], Kx=Kx,Ky=Ky, spectral=spectral)
        Vx = derivative_2DFHIT(V, [1, 0], Kx=Kx, Ky=Ky, spectral=spectral)

        # If the data is in physical space, convert it to spectral space using 2D FFT. 
        if spectral == False:
            Tau11_hat = np.fft.fft2(Tau11)
            Tau12_hat = np.fft.fft2(Tau12)
            Tau22_hat = np.fft.fft2(Tau22)
            U1x_hat = np.fft.fft2(Ux)
            U1y_hat = np.fft.fft2(Uy)
            V1x_hat = np.fft.fft2(Vx)
        else:  # If the data is already in the spectral space, we don't need to transform it.
            Tau11_hat = Tau11
            Tau12_hat = Tau12
            Tau22_hat = Tau22
            U1x_hat = Ux
            U1y_hat = Uy
            V1x_hat = Vx

        # Assuming that U is a square matrix, derive the size of the LES grid.
        N_LES = U.shape[0]

        # Compute the energy transfer function in spectral space.
        Ptau_hat = (-np.conj(Tau11_hat)*U1x_hat + np.conj(Tau22_hat)*U1x_hat - np.conj(Tau12_hat)*U1y_hat - np.conj(Tau12_hat)*V1x_hat)/(N_LES*N_LES)

    # If the method is 'PiOmega', calculate the energy transfer using Psi and PiOmega.
    elif method == 'PiOmega':
        # Check if the necessary parameters are provided.
        if Psi is None or PiOmega is None:
            raise ValueError("Psi, PiOmega must be provided to calculate energy transfer using 'PiOmega' method")

        # If the data is in physical space, convert it to spectral space using 2D FFT. 
        if spectral == False:
            Psi1_hat = np.fft.fft2(Psi)
            PiOmega_hat = np.fft.fft2(PiOmega)
        else:  # If the data is already in the spectral space, we don't need to transform it.
            Psi1_hat = Psi
            PiOmega_hat = PiOmega

        # Assuming that Psi is a square matrix, derive the size of the LES grid.
        N_LES = Psi.shape[0]

        # Compute the energy transfer function in spectral space.
        Ptau_hat = (np.conj(PiOmega_hat)*Psi1_hat)/(N_LES*N_LES)

    else:  # If an unsupported method is provided, raise an error.
        raise ValueError("Invalid method. Choose either 'Tau' or 'PiOmega'")

    # Perform an angle-averaged spectrum calculation on the computed energy transfer function.
    spectra, wavenumber = spectrum_angled_average_2DFHIT(Ptau_hat, spectral=True)

    # Return the real part of the computed spectrum along with the corresponding wavenumbers.
    return np.real(spectra), wavenumber


def enstrophyTransfer_spectra_2DFHIT(Kx, Ky, Omega=None, Sigma1=None, Sigma2=None, PiOmega=None, method='Sigma', spectral=False):
    '''
    Compute the enstrophy transfer spectra using 2D Forced Homogeneous Isotropic Turbulence (2D-FHIT)
    PZ > 0: dissipation
    PZ < 0: backscatter

    Parameters
    ----------
    Kx, Ky : numpy.ndarray
        The 2D wavenumber arrays
    Omega : numpy.ndarray, optional
        The vorticity field (2D array)
    Sigma1 : numpy.ndarray, optional
        The Sigma1 field (2D array)
    Sigma2 : numpy.ndarray, optional
        The Sigma2 field (2D array)
    PiOmega : numpy.ndarray, optional
        The PiOmega field (2D array)
    method : str, optional
        The method to calculate Pz_hat. 'Sigma' (default) or 'PiOmega'
    spectral : bool, optional
        Determines whether the input is in the 'physical' space (default=False) or 'spectral' space

    Returns
    -------
    spectra : numpy.ndarray
        The computed enstrophy transfer spectra
    wavenumber : numpy.ndarray
        The corresponding wavenumbers

    This function computes the enstrophy transfer spectra of the given 2D velocity fields
    using the 2D Forced Homogeneous Isotropic Turbulence. The inputs can be in the physical or spectral space.
    '''

    # Assuming that Omega is a square matrix, the size of the LES grid is derived from Omega.
    N_LES = Omega.shape[0] 
    
    # For the 'Sigma' method, compute the transfer spectra using Omega, Sigma1, and Sigma2.
    if method == 'Sigma':
        # Ensuring that the required arrays are provided.
        if Omega is None or Sigma1 is None or Sigma2 is None:
            raise ValueError("Omega, Sigma1, Sigma2 must be provided to calculate enstrophy transfer using 'Sigma' method")

        # Calculating the derivatives of Omega in x and y direction.
        Omegax = derivative_2DFHIT(Omega, [1, 0], Kx=Kx, Ky=Ky, spectral=spectral)
        Omegay = derivative_2DFHIT(Omega, [0, 1], Kx=Kx, Ky=Ky, spectral=spectral)

        # If the input is in the physical space, we need to convert it to spectral space using 2D FFT.
        if not spectral:
            Omegax_hat = np.fft.fft2(Omegax)
            Omegay_hat = np.fft.fft2(Omegay)
            Sigma1_hat = np.fft.fft2(Sigma1)
            Sigma2_hat = np.fft.fft2(Sigma2)
        else:  # If the input is already in the spectral space, we don't need to transform it.
            Omegax_hat = Omegax
            Omegay_hat = Omegay
            Sigma1_hat = Sigma1
            Sigma2_hat = Sigma2

        # Computing the enstrophy transfer function in spectral space.
        Pz_hat = -(np.conj(Sigma1_hat) * Omegax_hat + np.conj(Sigma2_hat) * Omegay_hat) / (N_LES * N_LES)

    # For the 'PiOmega' method, compute the transfer spectra using Omega and PiOmega.
    elif method == 'PiOmega':
        # Ensuring that the required arrays are provided.
        if Omega is None or PiOmega is None:
            raise ValueError("Omega, PiOmega must be provided to calculate enstrophy transfer using 'PiOmega' method")

        # If the input is in the physical space, we need to convert it to spectral space using 2D FFT.
        if not spectral:
            Omega_hat = np.fft.fft2(Omega)
            PiOmega_hat = np.fft.fft2(PiOmega)
        else:  # If the input is already in the spectral space, we don't need to transform it.
            Omega_hat = Omega
            PiOmega_hat = PiOmega

        # Computing the enstrophy transfer function in spectral space. Dividing by N_LES**2 is scale the fourier transform.
        Pz_hat = (np.conj(PiOmega_hat) * Omega_hat) / (N_LES * N_LES)

    else:  # If an unsupported method is provided, raise an error.
        raise ValueError("Invalid method. Choose either 'Sigma' or 'PiOmega'")

    # Performing an angle-averaged spectrum calculation on the computed enstrophy transfer function.
    spectra, wavenumber = spectrum_angled_average_2DFHIT(Pz_hat, spectral=True)

    # Returning the real part of the computed spectrum along with the corresponding wavenumbers.
    return np.real(spectra), wavenumber
