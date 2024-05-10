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

def angled_average_2D(A, l_array, kmax = 'grid'):
    '''Calculates the angle-averaged  2D scalar matrix'''

    # Check if input 'A' is a 2D square matrix
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('Input is not a 2D square matrix. Please input a 2D square matrix')
    # Check if input 'A' contains non-numeric values
    if not np.issubdtype(A.dtype, np.number):
        raise ValueError('Input contains non-numeric values')

    lx = l_array[:,:,0]
    ly = l_array[:,:,1]
    labs = np.sqrt(lx**2 + ly**2)
    dlx = lx[1,0] - lx[0,0]

    # Calculate the maximum wavenumber to be considered in the average
    if kmax == 'grid':
        l = np.arange(np.min(lx), np.max(lx)+dlx, dlx)
    elif kmax == 'diagonal':
        l = np.arange(np.min(labs), np.max(labs), dlx)

    # Initialize the output array with zeros
    A_angled_average = np.zeros(np.shape(l))

    # Compute the angle-averaged 
    for l_ind, l_value in enumerate(l):
        l_tempInd = (labs > (l_value - dlx/2)) & (labs <= (l_value + dlx/2))
        A_angled_average[l_ind] = np.sum(A[l_tempInd])

    return A_angled_average, l


def angled_average_struc_fxn_2D(structure_fx, l_array, kmax='grid'):
    """Calculating the spectra of the whole structure function or flatness containing structure functions upto orderMax"""
    
    orderMax = structure_fx.shape[2]

    angled_average_structure_fxn_list = []
    for count in range(orderMax):
        structure_fxn_temp, l_struc_fxn = angled_average_2D(structure_fx[:,:,count], l_array, kmax=kmax)
        angled_average_structure_fxn_list.append(structure_fxn_temp)

    angled_average_structure_fxn = np.array(angled_average_structure_fxn_list).T

    return angled_average_structure_fxn, l_struc_fxn



def flatness(structure_fxn_arr):
    """Calculates the flatness of the structure function. The structure function array should be output of 

    Args:
        structure_fxn (np.ndarray): The structure function values 
        orderMax (int, optional): The maximum order of the structure function. Defaults to 8.

    Returns:
        np.ndarray: The flatness values.
    """

    flatness_arr = np.zeros(structure_fxn_arr.shape)
    for order in range(1,structure_fxn_arr.shape[2]+1):
        flatness_arr[:,:,order-1] = structure_fxn_arr[:,:,order-1] ** order/ (structure_fxn_arr[:,:,1]**(order/2))

    return flatness_arr