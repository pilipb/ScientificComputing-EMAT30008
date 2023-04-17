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
        the matrix size (N-1, N-1)
    b : array
        the boundary condition vector, len = N-1

    '''
    # check the type of boundary condition is valid
    if type not in ['DD', 'DN', 'DR', 'ND', 'NN', 'NR','RD', 'RN', 'RR']:
        raise ValueError('Invalid boundary condition type')
    
    # make basic A tri-diagonal matrix
    A = np.zeros((N-1, N-1))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:,1:], 1)
    np.fill_diagonal(A, -2)

    # make the b vector
    b = np.zeros(N-1)

    # check which type of boundary condition for the first point
    if type[0] == 'D':
        b[0] = alpha
    elif type[0] == 'N':
        b[0] = 2*alpha*dx
        A[0, 1] = 2
    elif type[0] == 'R':
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


def tdma(A, b):
    '''
    Thomas algorithm or Tridiagonal matrix algorithm (TDMA) for solving
    A.x = b
    where A is a tridiagonal matrix

    
    Parameters:
    -----------------
    A: array
        the A matrix
    b: array
        the b vector

    Returns:
    -----------------
    x: array
        the solution of the linear system

    '''
    # define the size of the matrix
    N = len(b)
    c = np.zeros(N)
    d = np.zeros(N)
    x = np.zeros(N)

    # solve the linear system
    c[0] = A[0, 1]/A[0, 0]
    d[0] = b[0]/A[0, 0]

    # loop over the matrix
    for i in range(1, N-1):
        # solve the linear system
        c[i] = A[i, i+1]/(A[i, i] - A[i, i-1]*c[i-1])
        d[i] = (b[i] - A[i, i-1]*d[i-1])/(A[i, i] - A[i, i-1]*c[i-1])

    # solve the last linear system
    d[-1] = (b[-1] - A[-1, -2]*d[-2])/(A[-1, -1] - A[-1, -2]*c[-2])

    # solve the linear system backwards
    x[-1] = d[-1]
    for i in range(N-2, -1, -1):
        x[i] = d[i] - c[i]*x[i+1]

    return x