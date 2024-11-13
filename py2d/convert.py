# ----------------------------------------------------------------------
# Created : Karan Jakhar May 2023
# ----------------------------------------------------------------------

import numpy as np

def Omega2Psi(Omega, invKsq, spectral=False):
    """
    Calculate the stream function from vorticity.

    This function calculates the stream function (Psi) from vorticity (Omega) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    both physical and spectral space calculations.

    Parameters:
    -----------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical or spectral space, depending on the 'spectral' flag.
    invKsq : numpy.ndarray
        2D array of the inverse of square of the wavenumber magnitudes.
    spectral : bool, optional
        If True, assumes input vorticity is in spectral space and returns stream function in
        spectral space. If False (default), assumes input vorticity is in physical space and
        returns stream function in physical space.

    Returns:
    --------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.

    Notes:
    ------
    Ksq[0,0] is set to a large value to avoid division by zero.
    Ksq[0,0] maybe me set to 0, causing division by zero.
    Psi[0,0] can be set to 0, to avoid any nan values.
    """
    # Check if the 'spectral' flag is set to False. If it is, transform the vorticity from physical space to spectral space using a 2D Fast Fourier Transform.
    if not spectral:
        Psi = Omega2Psi_physical(Omega, invKsq)
        return Psi
    # If the 'spectral' flag is set to True, assume that the input vorticity is already in spectral space.
    else:
        Omega_hat = Omega
        Psi_hat = Omega2Psi_spectral(Omega_hat, invKsq)
        return Psi_hat


def Psi2Omega(Psi, Ksq, spectral=False):
    """
    Calculate the vorticity from the stream function.

    This function calculates the vorticity (Omega) from the stream function (Psi) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    both physical and spectral space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.
    spectral : bool, optional
        If True, assumes input stream function is in spectral space and returns vorticity in
        spectral space. If False (default), assumes input stream function is in physical space and
        returns vorticity in physical space.

    Returns:
    --------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical or spectral space, depending on the 'spectral' flag.

    """
    # Check if the 'spectral' flag is set to False. If it is, transform the stream function from physical space to spectral space using a 2D Fast Fourier Transform.
    if not spectral:
        Omega = Psi2Omega_physical(Psi, Ksq)
        return Omega
    # If the 'spectral' flag is set to True, assume that the input stream function is already in spectral space.
    else:
        Psi_hat = Psi
        Omega_hat = Psi2Omega_spectral(Psi_hat, Ksq)
        return Omega_hat
    

def Omega2UV(Omega, Kx, Ky, invKsq, spectral = False):
    """
    Calculate the velocity components U and V from the vorticity.
    
    This function calculates the velocity components U and V from the vorticity (Omega) using the relationships
    that vorticity is the negative Laplacian of the stream function. 
    Further the velocity components U and V from the stream function (Psi) using the relationships
        U = d(Psi)/dy
        V = -d(Psi)/dx
    The function can handle both physical and spectral space calculations.
    
    Parameters:
    -----------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    invKsq : numpy.ndarray
        2D array of the inverse of square of the wavenumber magnitudes.
    spectral : bool, optional
        If True, assumes input vorticity is in spectral space and returns velocity components
        in spectral space. If False (default), assumes input vorticity is in physical space and
        returns velocity components in physical space.

    Returns:
    --------
    U, V : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in physical or spectral space, depending on the 'spectral' flag.
    """

    if not spectral:
        U, V = Omega2UV_physical(Omega, Kx, Ky, invKsq)
        return U, V
    else:
        Omega_hat = Omega
        U_hat, V_hat = Omega2UV_spectral(Omega_hat, Kx, Ky, invKsq)
        return U_hat, V_hat

def UV2Omega(U, V, Kx, Ky, spectral = False):
    """
    Calculate the vorticity from the velocity components.
    
    This function calculates the vorticity (Omega) from the velocity components U and V using the relationships
        Omega = (1j*Kx)*V - (1j*Ky)*U
    The function can handle both physical and spectral space calculations.
    
    Parameters:
    -----------
    U : numpy.ndarray
        Velocity component U (2D array) in physical or spectral space, depending on the 'spectral' flag.
    V : numpy.ndarray
        Velocity component V (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    spectral : bool, optional
        If True, assumes input velocity components are in spectral space and returns vorticity
        in spectral space. If False (default), assumes input velocity components are in physical space
        and returns vorticity in physical space.

    Returns:
    --------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical or spectral space, depending on the 'spectral' flag.
    """
    if not spectral:
        Omega = UV2Omega_physical(U, V, Kx, Ky)
        return Omega
    else:
        U_hat = U
        V_hat = V
        Omega_hat = UV2Omega_spectral(U_hat, V_hat, Kx, Ky)
        return Omega_hat

def Psi2UV(Psi, Kx, Ky, spectral = False):
    """
    Calculate the velocity components U and V from the stream function.

    This function calculates the velocity components U and V from the stream function (Psi) 
    using the relationships:
        U = d(Psi)/dy
        V = -d(Psi)/dx
    Depending on the 'spectral' flag, the function can handle both physical and spectral space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    spectral : bool, optional
        If True, assumes input stream function is in spectral space and returns velocity components
        in spectral space. If False (default), assumes input stream function is in physical space and
        returns velocity components in physical space.

    Returns:
    --------
    U, V : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in physical or spectral space, depending on the 'spectral' flag.

    """

    # If the 'spectral' flag is False, perform a 2D Fast Fourier Transform on the input stream function
    # to transform it into spectral space
    if not spectral:
        U, V = Psi2UV_physical(Psi, Kx, Ky)
        return U, V
    else:
        Psi_hat = Psi
        U_hat, V_hat = Psi2UV_spectral(Psi_hat, Kx, Ky)
        return U_hat, V_hat


def Tau2PiOmega(Tau11, Tau12, Tau22, Kx, Ky, spectral=False):
    """
    Calculate PiOmega, the curl of the divergence of Tau, where Tau is a 2D symmetric tensor.

    Parameters:
    -----------
    Tau11 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, 
        depending on the 'spectral' flag.
    Tau12 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, 
        depending on the 'spectral' flag.
    Tau22 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, 
        depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    spectral : bool, optional
        If True, assumes input Tau elements are in spectral space and returns PiOmega in spectral space.
        If False (default), assumes input Tau elements are in physical space and returns PiOmega in physical space.

    Returns:
    --------
    PiOmega : numpy.ndarray
        PiOmega (2D array) in physical or spectral space, depending on the 'spectral' flag.

    """
    # Transform Tau elements to spectral space via 2D Fast Fourier Transform if 'spectral' flag is False
    if not spectral:
        PiOmega = Tau2PiOmega_physical(Tau11, Tau12, Tau22, Kx, Ky)
        return PiOmega

    else:
        Tau11_hat = Tau11
        Tau12_hat = Tau12
        Tau22_hat = Tau22
        PiOmega_hat = Tau2PiOmega_spectral(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky)
        return PiOmega_hat
    

def Tau2PiUV(Tau11, Tau12, Tau22, Kx, Ky, spectral=False):
    """
    Calculate PiUV, the divergence of Tau, where Tau is a 2D symmetric tensor.

    Parameters:
    -----------
    Tau11 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, 
        depending on the 'spectral' flag.
    Tau12 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, 
        depending on the 'spectral' flag.
    Tau22 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, 
        depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    spectral : bool, optional
        If True, assumes input Tau elements are in spectral space and returns PiOmega in spectral space.
        If False (default), assumes input Tau elements are in physical space and returns PiOmega in physical space.

    Returns:
    --------
    PiOmega : numpy.ndarray
        PiOmega (2D array) in physical or spectral space, depending on the 'spectral' flag.

    Notes:
    ------
    This function serves as a wrapper for Tau2PiUV_spectral() and Tau2PiUV_physical().
    Depending on the 'spectral' flag, it selects the appropriate function and computes PiUV.
    """
    if not spectral:
        # If 'spectral' flag is False, compute PiUV in physical space
        PiUV1, PiUV2 = Tau2PiUV_physical(Tau11, Tau12, Tau22, Kx, Ky)
        return PiUV1, PiUV2
    else:
        # If 'spectral' flag is True, compute PiUV in spectral space
        Tau11_hat = Tau11
        Tau12_hat = Tau12
        Tau22_hat = Tau22
        PiUV1_hat, PiUV2_hat = Tau2PiUV_spectral(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky)
        return PiUV1_hat, PiUV2_hat


def PiUV2PiOmega(PiUV1, PiUV2, Kx, Ky, spectral = False):
    """
    Compute the Curl of PiUV, referred to as PiOmega, in either physical or spectral space.

    Parameters:
    -----------
    PiUV1, PiUV2 : numpy.ndarray
        The components of PiUV.
    Kx, Ky : numpy.ndarray
        The wavenumbers in the x and y directions, respectively.
    spectral : bool, optional
        If True, assumes input Sigma elements are in spectral space and returns PiOmega in spectral space.
        If False (default), assumes input Sigma elements are in physical space and returns PiOmega in physical space.

    Returns:
    --------
    PiOmega or PiOmega_hat : numpy.ndarray
        The divergence of Sigma, referred to as PiOmega, in either physical (PiOmega) or spectral (PiOmega_hat) space 
        depending on the 'spectral' flag.
    
    Notes:
    ------
    This function serves as a wrapper for PiUV2PiOmega_spectral() and PiUV2PiOmega_physical().
    Depending on the 'spectral' flag, it selects the appropriate function and computes PiOmega.
    """
    if not spectral:
        # If 'spectral' flag is False, compute PiOmega in physical space
        PiOmega = PiUV2PiOmega_physical(PiUV1, PiUV2, Kx, Ky)
        return PiOmega
    else:
        # If 'spectral' flag is True, compute PiOmega in spectral space
        PiUV1_hat = PiUV1
        PiUV2_hat = PiUV2
        PiOmega_hat = PiUV2PiOmega_spectral(PiUV1_hat, PiUV2_hat, Kx, Ky)
        return PiOmega_hat


def Sigma2PiOmega(Sigma1, Sigma2, Kx, Ky, spectral = False):
    """
    Compute the divergence of Sigma, referred to as PiOmega, in either physical or spectral space.

    Parameters:
    -----------
    Sigma1, Sigma2 : numpy.ndarray
        The components of Sigma.
    Kx, Ky : numpy.ndarray
        The wavenumbers in the x and y directions, respectively.
    spectral : bool, optional
        If True, assumes input Sigma elements are in spectral space and returns PiOmega in spectral space.
        If False (default), assumes input Sigma elements are in physical space and returns PiOmega in physical space.

    Returns:
    --------
    PiOmega or PiOmega_hat : numpy.ndarray
        The divergence of Sigma, referred to as PiOmega, in either physical (PiOmega) or spectral (PiOmega_hat) space 
        depending on the 'spectral' flag.
    
    Notes:
    ------
    This function serves as a wrapper for Sigma2PiOmega_spectral() and Sigma2PiOmega_physical().
    Depending on the 'spectral' flag, it selects the appropriate function and computes PiOmega.
    """
    if not spectral:
        # If 'spectral' flag is False, compute PiOmega in physical space
        PiOmega = Sigma2PiOmega_physical(Sigma1, Sigma2, Kx, Ky)
        return PiOmega
    else:
        # If 'spectral' flag is True, compute PiOmega in spectral space
        Sigma1_hat = Sigma1
        Sigma2_hat = Sigma2
        PiOmega_hat = Sigma2PiOmega_spectral(Sigma1_hat, Sigma2_hat, Kx, Ky)
        return PiOmega_hat


def strain_rate(Psi, Kx, Ky, spectral=False):
    """
    Calculate the Strain rate components S11, S12, and S22 from the stream function.

    This function calculates the velocity components U and V from the stream function (Psi)
    using the relationship between them. The function can handle both physical and spectral
    space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.

    Returns:
    --------
    S11, S12, S22 : tuple of numpy.ndarray
        Strain rate components S11, S12, and S22 (2D arrays) in physical or spectral space, 
        depending on the 'spectral' flag.

    Notes:
    ------
    This assumes that continuity equation is valid, i.e., the stream function is divergence-free.
    Here, Ux = -Vy
    """

    # Transform Psi to spectral space using 2D Fast Fourier Transform if 'spectral' flag is False
    if not spectral:
        S11, S12, S22 = strain_rate_physical(Psi, Kx, Ky)
        return S11, S12, S22
    else:
        Psi_hat = Psi
        S11_hat, S12_hat, S22_hat = strain_rate_spectral(Psi_hat, Kx, Ky)
        return S11_hat, S12_hat, S22_hat


############################################################################################################
#  Rewriting the functions to be used JAX code

def Omega2Psi_spectral(Omega_hat, invKsq):
    """
    Calculate the stream function from vorticity in spectral space

    This function calculates the stream function (Psi) from vorticity (Omega) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    spectral space calculations.

    Parameters:
    -----------
    Omega_hat : numpy.ndarray
        Vorticity spectral space.
    invKsq : numpy.ndarray
        2D array of the inverse of square of the wavenumber magnitudes.

    Returns:
    --------
    Psi_hat : numpy.ndarray
        spectral space

    Notes:
    ------
    Ksq[0,0] is set to a large value to avoid division by zero.
    Ksq[0,0] maybe me set to 0, causing division by zero.
    Psi[0,0] can be set to 0, to avoid any nan values.
    """

    # Compute the Laplacian of the stream function in spectral space by taking the negative of the vorticity in spectral space.
    lap_Psi_hat = -Omega_hat
    # Divide the Laplacian of the stream function by the negative of the square of the wavenumber magnitudes (1/Ksq = inKsq) to compute the stream function in spectral space.
    Psi_hat = lap_Psi_hat * (-invKsq)

    # Return the stream function in spectral space.
    return Psi_hat

def Omega2Psi_physical(Omega, invKsq):
    """
    Calculate the stream function from vorticity in physical space

    This function calculates the stream function (Psi) from vorticity (Omega) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle physical space calculations.

    Parameters:
    -----------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical space
    invKsq : numpy.ndarray
        2D array of the inverse of square of the wavenumber magnitudes.

    Returns:
    --------
    Psi : numpy.ndarray
        Stream function (2D array) in physical space.

    """
    Nx, Ny = Omega.shape

    # Transform the vorticity from physical space to spectral space using a 2D Fast Fourier Transform.
    Omega_hat = np.fft.rfft2(Omega)

    # Compute the stream function in spectral space using the Omega2Psi_spectral function.
    Psi_hat = Omega2Psi_spectral(Omega_hat, invKsq)

    # Transform the stream function from spectral space back to physical space using an inverse 2D Fast Fourier Transform before returning it.
    return np.fft.irfft2(Psi_hat, s=[Nx,Ny])

############################################################################################################

def Psi2Omega_spectral(Psi_hat, Ksq):
    """
    Calculate the vorticity from the stream function in spectral space

    This function calculates the vorticity (Omega) from the stream function (Psi) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    spectral space calculations.

    Parameters:
    -----------
    Psi_hat : numpy.ndarray
        Stream function (2D array) in  spectral space.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.

    Returns:
    --------
    Omega_hat : numpy.ndarray
        Vorticity (2D array) in  spectral space.

    """
    # Compute the Laplacian of the stream function in spectral space by multiplying the stream function in spectral space by the negative of the square of the wavenumber magnitudes.
    lap_Psi_hat = (-Ksq) * Psi_hat
    # Compute the vorticity in spectral space by taking the negative of the Laplacian of the stream function in spectral space.
    Omega_hat = -lap_Psi_hat

    # Return the vorticity in spectral space.
    return Omega_hat

def Psi2Omega_physical(Psi, Ksq):
    """
    Calculate the vorticity from the stream function in physical space.

    This function calculates the vorticity (Omega) from the stream function (Psi) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    physical space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical space
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.

    Returns:
    --------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical space.

    """
    Nx, Ny = Psi.shape

    # Transform the stream function from physical space to spectral space using a 2D Fast Fourier Transform.
    Psi_hat = np.fft.rfft2(Psi)

    # Compute the vorticity in spectral space using the Psi2Omega_spectral function.
    Omega_hat = Psi2Omega_spectral(Psi_hat, Ksq)

    # Transform the vorticity from spectral space back to physical space using an inverse 2D Fast Fourier Transform, then take the real part (to remove any residual imaginary parts due to numerical error) before returning it.
    return np.real(np.fft.irfft2(Omega_hat, s=[Nx,Ny]))

############################################################################################################

def Psi2UV_spectral(Psi_hat, Kx, Ky):
    """
    Calculate the velocity components U and V from the stream function in spectral space

    This function calculates the velocity components U and V from the stream function (Psi) 
    using the relationships:
        U = d(Psi)/dy
        V = -d(Psi)/dx
    The function can handle spectral space calculations.

    Parameters:
    -----------
    Psi_hat : numpy.ndarray
        Stream function (2D array) in spectral space
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.

    Returns:
    --------
    U_hat, V_hat : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in spectral space.

    """

    # Calculate the Fourier coefficient of U (velocity in y-direction)
    # using the relationship U = d(Psi)/dy
    # In Fourier space, differentiation corresponds to multiplication by an imaginary unit and the wavenumber
    U_hat = (1.j) * Ky * Psi_hat

    # Calculate the Fourier coefficient of V (velocity in x-direction)
    # using the relationship V = -d(Psi)/dx
    # In Fourier space, differentiation corresponds to multiplication by an imaginary unit and the wavenumber
    V_hat = -(1.j) * Kx * Psi_hat

    return U_hat, V_hat

def Psi2UV_physical(Psi, Kx, Ky):
    """
    Calculate the velocity components U and V from the stream function in physical space

    This function calculates the velocity components U and V from the stream function (Psi) 
    using the relationships:
        U = d(Psi)/dy
        V = -d(Psi)/dx
    The function can handle both physical space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical space.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.

    Returns:
    --------
    U, V : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in physical space.

    """
    Nx, Ny = Psi.shape

    # Perform a 2D Fast Fourier Transform on the input stream function to transform it into spectral space
    Psi_hat = np.fft.rfft2(Psi)

    # Calculate the Fourier coefficients of the velocity components U and V in spectral space using the Psi2UV_spectral function
    U_hat, V_hat = Psi2UV_spectral(Psi_hat, Kx, Ky)

    # Perform an inverse 2D Fast Fourier Transform on the Fourier coefficients of the velocity components to transform them into physical space
    return np.fft.irfft2(U_hat, s=[Nx,Ny]), np.fft.irfft2(V_hat, s=[Nx,Ny])

############################################################################################################

def Tau2PiOmega_spectral(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky):
    """
    Calculate PiOmega, the curl of the divergence of Tau, where Tau is a 2D symmetric tensor in spectral space.

    Parameters:
    -----------
    Tau11_hat : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in spectral space
    Tau12_hat : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in spectral space
    Tau22_hat : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in spectral space
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.

    Returns:
    --------
    PiOmega_hat : numpy.ndarray
        PiOmega (2D array) in spectral space.

    """
    # Calculate PiOmega in spectral space using the given mathematical relationships
    PiOmega_hat = Kx * Ky * (Tau11_hat - Tau22_hat) - (Kx * Kx - Ky * Ky) * Tau12_hat

    return PiOmega_hat


def Tau2PiOmega_physical(Tau11, Tau12, Tau22, Kx, Ky):
    """
    Calculate PiOmega, the curl of the divergence of Tau, where Tau is a 2D symmetric tensor in physical space

    Parameters:
    -----------
    Tau11 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical space.
    Tau12 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical space. 
    Tau22 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical space. 
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.

    Returns:
    --------
    PiOmega : numpy.ndarray
        PiOmega (2D array) in physical space.

    """
    Nx, Ny = Tau11.shape

    # Transform Tau elements to spectral space via 2D Fast Fourier Transform 
    Tau11_hat = np.fft.rfft2(Tau11)
    Tau12_hat = np.fft.rfft2(Tau12)
    Tau22_hat = np.fft.rfft2(Tau22)

    # Calculate PiOmega in spectral space using the given mathematical relationships
    PiOmega_hat = Tau2PiOmega_spectral(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky)
    
    # Transform PiOmega back to physical space using inverse 2D Fast Fourier Transform
    return np.fft.irfft2(PiOmega_hat, s=[Nx,Ny])

############################################################################################################

def Omega2UV_spectral(Omega_hat, Kx, Ky, invKsq):
    """
    Calculate the velocity components U and V from the vorticity in spectral space.
    
    Parameters:
    -----------
    Omega_hat : numpy.ndarray
        Vorticity (2D array) in spectral space.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    invKsq : numpy.ndarray
        2D array of the inverse of square of the wavenumber magnitudes.
        
    Returns:
    --------
    U_hat, V_hat : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in spectral space.
        
    """

    Psi_hat = Omega2Psi_spectral(Omega_hat, invKsq)
    U_hat, V_hat = Psi2UV_spectral(Psi_hat, Kx, Ky)

    return U_hat, V_hat

def Omega2UV_physical(Omega, Kx, Ky, invKsq):
    """
    Calculate the velocity components U and V from the vorticity in physical space.
    
    Parameters:
    -----------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical space.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    invKsq : numpy.ndarray
        2D array of the inverse of square of the wavenumber magnitudes.
        
    Returns:
    --------
    U, V : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in physical space.
        
    """
    Nx, Ny = Omega.shape

    Omega_hat = np.fft.rfft2(Omega)
    U_hat, V_hat = Omega2UV_spectral(Omega_hat, Kx, Ky, invKsq)

    return np.fft.irfft2(U_hat, s=[Nx,Ny]), np.fft.irfft2(V_hat, s=[Nx,Ny])

############################################################################################################

def UV2Omega_spectral(U_hat, V_hat, Kx, Ky):
    """
    Calculate the vorticity from the velocity components in spectral space.
    
    Parameters:
    -----------
    U_hat : numpy.ndarray
        Velocity component U (2D array) in spectral space.
    V_hat : numpy.ndarray
        Velocity component V (2D array) in spectral space.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
        
    Returns:
    --------
    Omega_hat : numpy.ndarray
        Vorticity (2D array) in spectral space.

    Notes:
    ------
    The function computes the vorticity in spectral space using the relationships:
    Omega_hat = (1j*Kx)*V_hat - (1j*Ky)*U_hat
        
    """

    # Calculate the Fourier coefficient of the vorticity in spectral space using the given mathematical relationships
    Omega_hat = (1j*Kx)*V_hat - (1j*Ky)*U_hat

    return Omega_hat

def UV2Omega_physical(U, V, Kx, Ky):
    """
    Calculate the vorticity from the velocity components in physical space.
    
    Parameters:
    -----------
    U : numpy.ndarray
        Velocity component U (2D array) in physical space.
    V : numpy.ndarray
        Velocity component V (2D array) in physical space.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
        
    Returns:
    --------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical space.

    Notes:
    ------
    The function first transforms the physical velocity components U and V to spectral space, 
    then calculates the spectral vorticity components using UV2Omega_spectral() function,
    and finally transforms the spectral vorticity components back to physical space.
    """
    Nx, Ny = U.shape

    # Transform the physical velocity components U and V to spectral space via 2D Fast Fourier Transform
    U_hat = np.fft.rfft2(U)
    V_hat = np.fft.rfft2(V)

    # Compute the vorticity in spectral space using the UV2Omega_spectral function
    Omega_hat = UV2Omega_spectral(U_hat, V_hat, Kx, Ky)

    # Transform the spectral vorticity back to physical space using inverse 2D Fast Fourier Transform
    return np.fft.irfft2(Omega_hat, s=[Nx,Ny])

############################################################################################################

def Tau2PiUV_spectral(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky):
    """
    Compute the divergence of Tau, where Tau is a 2D symmetric tensor in spectral space, and return PiUV in spectral space.
    
    Parameters:
    -----------
    Tau11_hat, Tau12_hat, Tau22_hat : numpy.ndarray
        The spectral representations of the components of the 2D symmetric tensor Tau.
    Kx, Ky : numpy.ndarray
        The wavenumbers in the x and y directions, respectively.
        
    Returns:
    --------
    PiUV1_hat, PiUV2_hat : numpy.ndarray
        The spectral representations of the components of the divergence of Tau, denoted as PiUV.

    Notes:
    ------
    The function computes the spectral representations of the components of the divergence of Tau, PiUV1 and PiUV2, 
    using the relationships:
    PiUV1_hat = (1j*Kx)*Tau11_hat + (1j*Ky)*Tau12_hat
    PiUV2_hat = (1j*Kx)*Tau12_hat + (1j*Ky)*Tau22_hat
    """
    
    # Compute PiUV1_hat and PiUV2_hat using the relationship in spectral space
    PiUV1_hat = (1j*Kx)*Tau11_hat + (1j*Ky)*Tau12_hat
    PiUV2_hat = (1j*Kx)*Tau12_hat + (1j*Ky)*Tau22_hat

    return PiUV1_hat, PiUV2_hat

def Tau2PiUV_physical(Tau11, Tau12, Tau22, Kx, Ky):
    """
    Compute the divergence of Tau in physical space, where Tau is a 2D symmetric tensor, 
    and return PiUV in physical space.
    
    Parameters:
    -----------
    Tau11, Tau12, Tau22 : numpy.ndarray
        The physical representations of the components of the 2D symmetric tensor Tau.
    Kx, Ky : numpy.ndarray
        The wavenumbers in the x and y directions, respectively.
        
    Returns:
    --------
    PiUV1, PiUV2 : numpy.ndarray
        The components of the divergence of Tau, denoted as PiUV, in physical space.

    Notes:
    ------
    The function first converts the physical Tau tensor components to spectral space, 
    then calculates the spectral PiUV components using Tau2PiUV_spectral() function,
    and finally transforms the spectral PiUV components back to physical space.
    """
    Nx, Ny = Tau11.shape

    # Transform the physical components of Tau to spectral space
    Tau11_hat = np.fft.rfft2(Tau11)
    Tau12_hat = np.fft.rfft2(Tau12)
    Tau22_hat = np.fft.rfft2(Tau22)

    # Compute PiUV1_hat and PiUV2_hat using the spectral space function
    PiUV1_hat, PiUV2_hat = Tau2PiUV_spectral(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky)

    # Transform the spectral components of PiUV back to physical space
    PiUV1, PiUV2 = np.fft.irfft2(PiUV1_hat, s=[Nx,Ny]), np.fft.irfft2(PiUV2_hat, s=[Nx,Ny])

    return PiUV1, PiUV2

############################################################################################################

def Sigma2PiOmega_spectral(Sigma1_hat, Sigma2_hat, Kx, Ky):
    """
    Compute the divergence of Sigma, referred to as PiOmega, in spectral space.

    Parameters:
    -----------
    Sigma1_hat, Sigma2_hat : numpy.ndarray
        The spectral representations of the components of Sigma.
    Kx, Ky : numpy.ndarray
        The wavenumbers in the x and y directions, respectively.

    Returns:
    --------
    PiOmega_hat : numpy.ndarray
        The spectral representation of PiOmega, the divergence of Sigma.
    """
    # Calculate PiOmega in spectral space using the given mathematical relationships
    PiOmega_hat = (1j*Kx)*Sigma1_hat + (1j*Ky)*Sigma2_hat
    return PiOmega_hat


def Sigma2PiOmega_physical(Sigma1, Sigma2, Kx, Ky):
    """
    Compute the divergence of Sigma, referred to as PiOmega, in physical space.

    Parameters:
    -----------
    Sigma1, Sigma2 : numpy.ndarray
        The physical representations of the components of Sigma.
    Kx, Ky : numpy.ndarray
        The wavenumbers in the x and y directions, respectively.
    
    Returns:
    --------
    PiOmega : numpy.ndarray
        The physical representation of PiOmega, the divergence of Sigma.
    """
    Nx, Ny = Sigma1.shape

    # Transform Sigma elements to spectral space via 2D Fast Fourier Transform
    Sigma1_hat, Sigma2_hat = np.fft.rfft2(Sigma1), np.fft.rfft2(Sigma2)
    # Compute PiOmega in spectral space
    PiOmega_hat = Sigma2PiOmega_spectral(Sigma1_hat, Sigma2_hat, Kx, Ky)
    # Convert the result back to physical space
    PiOmega = np.fft.irfft2(PiOmega_hat, s=[Nx,Ny])
    return PiOmega

############################################################################################################

def PiUV2PiOmega_spectral(PiUV1_hat, PiUV2_hat, Kx, Ky):
    """
    Compute the Curl of PiUV, referred to as PiOmega, in spectral space.

    Parameters:
    -----------
    PiUV1_hat, PiUV2_hat : numpy.ndarray
        The spectral representations of the components of Sigma.
    Kx, Ky : numpy.ndarray
        The wavenumbers in the x and y directions, respectively.

    Returns:
    --------
    PiOmega_hat : numpy.ndarray
        The spectral representation of PiOmega, the Curl of PiUV.
    """
    # Calculate PiOmega in spectral space using the given mathematical relationships
    PiOmega_hat = (1j*Ky)*PiUV1_hat - (1j*Kx)*PiUV2_hat
    return PiOmega_hat


def PiUV2PiOmega_physical(PiUV1, PiUV2, Kx, Ky):
    """
    Compute the Curl of PiUV, referred to as PiOmega, in physical space.

    Parameters:
    -----------
    PiUV1, PiUV2 : numpy.ndarray
        The physical representations of the components of PiUV.
    Kx, Ky : numpy.ndarray
        The wavenumbers in the x and y directions, respectively.
    
    Returns:
    --------
    PiOmega : numpy.ndarray
        The physical representation of PiOmega, the divergence of Sigma.
    """
    Nx, Ny = PiUV1.shape

    # Transform Sigma elements to spectral space via 2D Fast Fourier Transform
    PiVU1_hat, PiUV2_hat = np.fft.rfft2(PiUV1), np.fft.rfft2(PiUV2)
    # Compute PiOmega in spectral space
    PiOmega_hat = PiUV2PiOmega_spectral(PiVU1_hat, PiUV2_hat, Kx, Ky)
    # Convert the result back to physical space
    PiOmega = np.fft.irfft2(PiOmega_hat, s=[Nx,Ny])
    return PiOmega

############################################################################################################

def strain_rate_spectral(Psi_hat, Kx, Ky):
    """
    Calculate the Strain rate components S11_hat, S12_hat, and S22_hat from the stream function in spectral space

    This function calculates the velocity components U and V from the stream function (Psi)
    using the relationship between them. The function can handle  spectral space calculations.

    Parameters:
    -----------
    Psi_hat : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.

    Returns:
    --------
    S11_hat, S12_hat, S22_hat : tuple of numpy.ndarray
        Strain rate components S11_hat, S12_hat, and S22_hat spectral space.

    Notes:
    ------
    This assumes that continuity equation is valid, i.e., the stream function is divergence-free.
    Here, Ux = -Vy
    """

    # Calculate the Fourier coefficients of the strain rate components using given mathematical relationships
    Ux_hat = (1j*Kx) * (1j*Ky) * Psi_hat
    Vx_hat = (1j*Kx) * (-1j*Kx) * Psi_hat
    Uy_hat = (1j*Ky) * (1j*Ky) * Psi_hat
    S11_hat = Ux_hat
    S12_hat = 0.5 * (Uy_hat + Vx_hat)
    S22_hat = -Ux_hat

    return S11_hat, S12_hat, S22_hat

def strain_rate_physical(Psi, Kx, Ky):
    """
    Calculate the Strain rate components S11, S12, and S22 from the stream function in physical space

    This function calculates the velocity components U and V from the stream function (Psi)
    using the relationship between them. The function can handle physical space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.

    Returns:
    --------
    S11, S12, S22 : tuple of numpy.ndarray
        Strain rate components S11, S12, and S22 (2D arrays) in physical space

    Notes:
    ------
    This assumes that continuity equation is valid, i.e., the stream function is divergence-free.
    Here, Ux = -Vy
    """
    Nx, Ny = Psi.shape

    # Transform Psi to spectral space using 2D Fast Fourier Transform
    Psi_hat = np.fft.rfft2(Psi)

    # Calculate the Fourier coefficients of the strain rate components using given mathematical relationships
    S11_hat, S12_hat, S22_hat = strain_rate_spectral(Psi_hat, Kx, Ky)

    # Transform the strain rate components back to physical space using inverse 2D Fast Fourier Transform
    return np.fft.irfft2(S11_hat, s=[Nx,Ny]), np.fft.irfft2(S12_hat, s=[Nx,Ny]), np.fft.irfft2(S22_hat, s=[Nx,Ny])

def Tau2AnisotropicTau(Tau11, Tau12, Tau22):
    """
    Calculate the anisotropic (symmetric) part of the 2D tensor Tau.

    Parameters:
    -----------
    Tau11 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical space.
    Tau12 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical space.
    Tau22 : numpy.ndarray
        Element (2D array) of the 2D symmetric tensor Tau in physical space.

    Returns:
    --------
    AnisotropicTau11, AnisotropicTau12, AnisotropicTau22 : tuple of numpy.ndarray
        Anisotropic components of the 2D symmetric tensor Tau in physical space.

    Notes:
    ------
    The anisotropic part of the 2D symmetric tensor Tau is calculated as:
    AnisotropicTau11 = Tau11 - 1/2*(Tau11 + Tau22)
    AnisotropicTau12 = Tau12
    AnisotropicTau22 = Tau22 - 1/2*(Tau11 + Tau22)
    """
    # Calculate the anisotropic part of the 2D symmetric tensor Tau
    AnisotropicTau11 = 0.5*(Tau11 - Tau22) # Tau11 - 0.5 * (Tau11 + Tau22)
    AnisotropicTau12 = Tau12
    AnisotropicTau22 = 0.5*(Tau11 - Tau22) # Tau22 - 0.5 * (Tau11 + Tau22)

    return AnisotropicTau11, AnisotropicTau12, AnisotropicTau22

############################################################################################################