import numpy as np 

def structure_fxn_scalar(data, orderMax=8, Lx=2*np.pi, Ly=2*np.pi):
    """Calculates the scalar velocity structure function up to a specified order.

    Args:
        data (np.ndarray): A 2D array representing the scalar field.
        orderMax (int, optional): The maximum order of the structure function to calculate. 
                                  Defaults to 8.
        Lx (float, optional): Physical size of the domain in the x-direction. Defaults to 2*pi.
        Ly (float, optional): Physical size of the domain in the y-direction. Defaults to 2*pi.

    Returns:
        tuple: A tuple containing:
            * np.ndarray: A 3D array containing the structure function values. 
                          The dimensions are (lx, ly, order).
            * np.ndarray: Array of physical separation distances (lx, ly)
                          The dimensions are (lx, ly, order).

    Note:
        Reference: https://joss.theoj.org/papers/10.21105/joss.02185
    """

    Nx, Ny = data.shape
    dx = Lx/Nx    # Grid spacing in x-direction
    dy = Ly/Ny    # Grid spacing in y-direction

    # Initialize the structure function aarrays
    structure_fxn = np.zeros((Nx//2, Ny//2, orderMax))
    l_array = np.zeros((Nx//2, Ny//2, 2))

    # Iterate over separation distances (indices)
    for lx_ind in range(1, Nx//2 + 1):
        for ly_ind in range(1, Ny//2 + 1):
            ds = cal_structure_fxn_diff(data, lx_ind, ly_ind)

            # Calculate structure functions for different orders
            for order in range(1, orderMax + 1):
                structure_fxn[lx_ind-1, ly_ind-1, order-1] = np.mean(ds**order)  # Zero-based indexing

            l_array[lx_ind-1, ly_ind-1, 0] = lx_ind*dx
            l_array[lx_ind-1, ly_ind-1, 1] = ly_ind*dy

    return structure_fxn, l_array

def structure_fxn_2Dvector(dataA, dataB, orderMax=8, Lx=2*np.pi, Ly=2*np.pi):
    """Calculates longitudinal and transverse velocity structure functions for a 2D vector field.

    Args:
        dataA (np.ndarray): A 2D array representing the first component of the vector field.
        dataB (np.ndarray): A 2D array representing the second component of the vector field.
        orderMax (int, optional): The maximum order of the structure function to calculate. 
                                  Defaults to 8.
        Lx (float, optional): Physical size of the domain in the x-direction. Defaults to 2*pi.
        Ly (float, optional): Physical size of the domain in the y-direction. Defaults to 2*pi.

    Returns:
        tuple: A tuple containing:
            * np.ndarray: Longitudinal velocity structure function (3D array)
            * np.ndarray: Transverse velocity structure function (3D array)
            * np.ndarray: Array of physical separation distances (lx, ly, 2)

    Note:
        Reference: https://joss.theoj.org/papers/10.21105/joss.02185
    """

    Nx, Ny = dataA.shape

    dx = Lx/Nx    # Grid spacing in x-direction
    dy = Ly/Ny    # Grid spacing in y-direction

    # Initialize the structure function arrays
    structure_fxn_longitudinal = np.zeros((Nx//2, Ny//2, orderMax))
    structure_fxn_transverse = np.zeros((Nx//2, Ny//2, orderMax))
    l_array = np.zeros((Nx//2, Ny//2, 2))

    # Iterate over separation distances (indices)
    for lx_ind in range(1, Nx//2 + 1):
        for ly_ind in range(1, Ny//2 + 1):
            du = cal_structure_fxn_diff(dataA, lx_ind, ly_ind)
            dv = cal_structure_fxn_diff(dataB, lx_ind, ly_ind)

            # Physical separation distances 
            lx = lx_ind * dx 
            ly = ly_ind * dy

            # Unit vector calculation
            l_norm = np.sqrt(lx**2 + ly**2) 
            lx_unit = lx / l_norm
            ly_unit = ly / l_norm

            # Projection along and perpendicular to separation vector
            diff_parallel = du*lx_unit + dv*ly_unit
            diff_perp = np.sqrt((du - diff_parallel*lx_unit)**2 + (dv - diff_parallel*ly_unit)**2)

            # Calculate structure functions for different orders
            for order in range(1, orderMax + 1):
                structure_fxn_longitudinal[lx_ind-1, ly_ind-1, order-1] = np.mean(diff_parallel**order)
                structure_fxn_transverse[lx_ind-1, ly_ind-1, order-1] = np.mean(diff_perp**order)

            l_array[lx_ind-1, ly_ind-1, 0] = lx
            l_array[lx_ind-1, ly_ind-1, 1] = ly

    return structure_fxn_longitudinal, structure_fxn_transverse, l_array

def cal_structure_fxn_diff(data, lx_ind, ly_ind):
    """Calculates the difference for the velocity structure function calculation.

    Args:
        data (np.ndarray): The 2D array representing the field.
        lx (int): The separation distance in the x-direction.
        ly (int): The separation distance in the y-direction.
        Nx (int): The number of grid points in the x-direction.
        Ny (int): The number of grid points in the y-direction.

    Returns:
        np.ndarray: The difference array used for calculating the structure function.
    """

    Nx, Ny = data.shape

    # Calculate the velocity difference u(x+r) - u(x).  
    data_diff = data[lx_ind:Nx//2 + lx_ind, ly_ind:Ny//2 + ly_ind] - data[:Nx//2, :Ny//2]

    return data_diff