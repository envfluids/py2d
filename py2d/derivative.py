import numpy as np

def derivative_2D_FHIT(T, order, Kx, Ky, spectral=False):
    """
    Calculate spatial derivatives for 2D_FHIT in spectral space.
    Boundary conditions are periodic in x and y spatial dimensions
    Length of domain 2*pi

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
        T_hat = np.fft.fft2(T)
    else:  # if it is already in the spectral domain, we do nothing
        T_hat = T

    # Orders of derivatives in x and y dimensions
    orderX = order[0]
    orderY = order[1]

    # Calculating derivatives in spectral space using the Fourier derivative theorem
    Tderivative_hat = ((1j*Kx)**orderX) * ((1j*Ky)**orderY) * T_hat

    # If the input data was not in the spectral domain, we transform the result back into the physical domain
    if spectral == False:
        Tderivative = np.real(np.fft.ifft2(Tderivative_hat))
        return Tderivative
    else:  # if it was in the spectral domain, we leave the result in the spectral domain
        return Tderivative_hat

