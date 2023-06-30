from jax import jit

@jit
def derivative_2DFHIT(T_hat, order, Kx, Ky):
    """
    Calculate spatial derivatives for 2D_FHIT in spectral space.
    Boundary conditions are periodic in x and y spatial dimensions
    Length of domain 2*pi

    Input:
    T_hat: Input flow
     field in spectral space: Square Matrix NxN
    order [orderX, orderY]: Array of order of derivatives in x and y spatial dimensions: [Interger (>=0), Integer (>=0)] 
    Kx, Ky: Kx and Ky values calculated beforehand.

    Output:
    Tderivative_hat: derivative of the flow field T in spectral space: Square Matrix NxN
    """
    orderX = order[0]
    orderY = order[1]
    Tderivative_hat = ((1j*Kx)**orderX) * ((1j*Ky)**orderY) * T_hat

    return Tderivative_hat

