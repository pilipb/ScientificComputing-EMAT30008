import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from numpy.linalg import solve

def bvp_solver(q, D, alpha, beta, a, b, N, method='root', boundary='Dirichlet'):
    '''
    solving poisson equation using finite difference method
    Dirichlet boundary condition

    in domain:
    a <= x <= b
    boundary condition:
    u(a) = alpha
    u(b) = beta

    equation:
    u''(x) + q(x) = 0

    Parameters:
    -----------------
    q: function
        the function q(x)
    D: float
        D*u''(x) + q(x) = 0
    alpha: float
        the value of u(a)
    beta: float
        the value of u(b)
    a: float
        the left edge of the domain
    b: float
        the right edge of the domain
    N: int
        the number of grid points
    method: string
        the method to solve the linear system: 'root' for SciPy or 'solve' for Numpy
    boundary: string
        the type of boundary condition: 'Dirichlet' or 'Neumann' or 'Robin'

    Returns:
    -----------------
    u: array
        the solution of the problem
    xi: array
        the grid points


    '''

    # define the grid points
    xi = np.linspace(a, b, N)

    # define the step size
    dx = (b - a)/N

    # define the matrix A
    A_mat = np.zeros((N, N))
    A_mat[0, 0] = 1
    A_mat[-1, -1] = 1
    for i in range(1, N-1):
        A_mat[i, i-1] = -1/(dx**2)
        A_mat[i, i] = 2/(dx**2) + D
        A_mat[i, i+1] = -1/(dx**2)

    # define the vector b
    b_mat = np.zeros(N)
    b_mat[0] = alpha
    b_mat[-1] = beta
    for i in range(1, N-1):
        b_mat[i] = q(xi[i])

    # solve the linear system
    if method == 'root': # use SciPy
        u = root(lambda x: A_mat.dot(x) - b_mat, np.zeros(N)).x

    elif method == 'solve': # use Numpy
        u = solve(A_mat, b_mat)

    return u, xi

if __name__ == '__main__':

    # define the function q(x)
    q = lambda x: x**2

    # define the parameters
    D = 1
    alpha = 0
    beta = 0
    a = 0
    b = 1
    N = 100

    # define the method and boundary condition
    method = 'root'
    boundary = 'Dirichlet'

    # solve the problem
    u, xi = bvp_solver(q, D, alpha, beta, a, b, N, method=method, boundary=boundary)

    # plot the solution
    plt.plot(xi, u, 'o-')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('2nd Order BVP with source term q(x) = x**2, (%s Boundary Condition and %s Method)' % (boundary, method))
    plt.show()











