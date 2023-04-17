import numpy as np

def boundary(alpha, beta, N, dx, type):
    '''
    This function returns the boundary condition matrix and vector for a given boundary condition type

    Parameters
    ----------
    alpha : float
        the left boundary
    beta : float
        the right boundary
    N : int
        the number of grid points
    dx : float
        the step size
    type : string
        the type of boundary condition
        options: 'DD', 'DN', 'DR', 'ND', 'NN', 'NR','RD', 'RN', 'RR'

    Returns
    -------
    A : array
        the matrix
    b : array
        the boundary condition vector

    '''
    # check the type of boundary condition is valid
    if type not in ['DD', 'DN', 'DR', 'ND', 'NN', 'NR','RD', 'RN', 'RR']:
        raise ValueError('Invalid boundary condition type')
    # make basic A tri-diagonal matrix
    A = np.zeros((N-1, N-1))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:,1:], 1)
    np.fill_diagonal(A, -2)

    # check which type of boundary condition for the first point
    if type[0] == 'D':
        b = np.zeros(N-1)
        b[0] = alpha
    elif type[0] == 'N':
        b = np.zeros(N-1)
        b[0] = 2*alpha*dx
        A[0, 1] = 2
    elif type[0] == 'R':
        b = np.zeros(N-1)
        b[0] = 2*alpha*dx
        A[0, 1] = -2*(1+alpha*dx)/dx

    # check which type of boundary condition for the last point
    if type[1] == 'D':
        b[-1] = beta
    elif type[1] == 'N':
        b[-1] = 2*beta*dx
        A[-2, -1] = 2
    elif type[1] == 'R':
        b[-1] = 2*beta*dx
        A[-2, -1] = -2*(1+beta*dx)/dx

    return A, b