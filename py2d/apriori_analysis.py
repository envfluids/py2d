from py2d.derivative import derivative_2DFHIT

def energyTransfer_2DFHIT(U, V, Tau11, Tau12, Tau22, Kx, Ky):
    """
    Energy transfer of 2D_FHIT using SGS stress
    Input is single snapshot (N x N matrix)

    Inputs:
    U,V: Velocities
    Tau11, Tau12, Tau22: SGS stress

    Output: 
    PTau: energy transfer
    """

    Ux = derivative_2DFHIT(U, [1,0], Kx=Kx, Ky=Ky, spectral=False)
    Uy = derivative_2DFHIT(U, [0,1], Kx=Kx, Ky=Ky, spectral=False)
    Vx = derivative_2DFHIT(V, [1,0], Kx=Kx, Ky=Ky, spectral=False)

    PTau = -(Tau11-Tau22)*Ux - Tau12*(Uy+Vx)

    return PTau

def enstrophyTransfer_2D_FHIT(Omega, Sigma1, Sigma2, Kx, Ky):
    """
    Enstrophy transfer of 2D_FHIT using SGS vorticity stress

    Inputs:
    Omega: Vorticity
    Sigma1, Sigma2: SGS vorticity stress

    Output: 
    PZ: enstrophy transfer
    """

    Omegax = derivative_2DFHIT(Omega, [1,0], Kx=Kx, Ky=Ky, spectral=False)
    Omegay = derivative_2DFHIT(Omega, [0,1], Kx=Kx, Ky=Ky, spectral=False)

    PZ = -Sigma1*Omegax - Sigma2*Omegay

    return PZ

