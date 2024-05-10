import numpy as np

def derivative(T, order, Kx, Ky, spectral=False):
    """
    Calculate spatial derivatives for 2D_FHIT in spectral space.
    Boundary conditions are periodic in x and y spatial dimensions

    Parameters:
    ----------
    T : numpy.ndarray
        Input flow field. If `spectral` is False, T is in the physical domain; otherwise, it is in the spectral domain.
    order : list
        Array of order of derivatives in x and y spatial dimensions. Expects a list of two integers >=0.
    Kx, Ky : numpy.ndarray
        Precomputed wavenumbers in the x and y dimensions.
    spectral : bool, optional
        Whether T is in the spectral domain. Default is False.

    Returns:
    -------
    Tderivative or Tderivative_hat : numpy.ndarray
        The derivative of the flow field T. If `spectral` is False, this is in the physical domain; otherwise, it's in the spectral domain.
    """

    # If the input data is not in the spectral domain, we transform it
    if spectral == False:
        Tderivative = derivative_physical(T, order, Kx, Ky)
        return Tderivative
    else:  # if it is already in the spectral domain, we do nothing
        T_hat = T
        Tderivative_hat = derivative_spectral(T_hat, order, Kx, Ky)
        return Tderivative_hat

    
def derivative_spectral(T_hat, order, Kx, Ky):
    """
    Calculate spatial derivatives for 2D_FHIT in spectral space.
    Boundary conditions are periodic in x and y spatial dimensions

    Parameters:
    ----------
    T_hat : numpy.ndarray
        Input flow field. T_hat is in the spectral domain.
    order : list
        Array of order of derivatives in x and y spatial dimensions. Expects a list of two integers >=0.
    Kx, Ky : numpy.ndarray
        Precomputed wavenumbers in the x and y dimensions.

    Returns:
    -------
    Tderivative_hat : numpy.ndarray
        The derivative of the flow field T_hat in the spectral domain.
    """

    # Orders of derivatives in x and y dimensions
    orderX = order[0]
    orderY = order[1]

    # Calculating derivatives in spectral space using the Fourier derivative theorem
    Tderivative_hat = ((1j*Kx)**orderX) * ((1j*Ky)**orderY) * T_hat

    return Tderivative_hat

def derivative_physical(T, order, Kx, Ky, ):
    """
    Calculate spatial derivatives for 2D_FHIT in physical space using spectral methods.
    Boundary conditions are periodic in x and y spatial dimensions

    Parameters:
    ----------
    T : numpy.ndarray
        Input flow field. T is in the physical domain.
    order : list
        Array of order of derivatives in x and y spatial dimensions. Expects a list of two integers >=0.
    Kx, Ky : numpy.ndarray
        Precomputed wavenumbers in the x and y dimensions.

    Returns:
    -------
    Tderivative : numpy.ndarray
        The derivative of the flow field T in the physical domain.
    """

    Nx, Ny = T.shape

    # We transform data to spectral domain
    T_hat = np.fft.rfft2(T)

    # Orders of derivatives in x and y dimensions
    Tderivative_hat = derivative_spectral(T_hat, order, Kx, Ky)

    # Transform the result back into the physical domain
    Tderivative = np.fft.irfft2(Tderivative_hat, s=[Nx, Ny])

    
    
    return Tderivative

    


