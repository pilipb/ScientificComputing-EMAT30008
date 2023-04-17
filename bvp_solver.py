import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from numpy.linalg import solve
from helpers import *

def bvp_solver(q, a, b,N , *args , D = 1.0, alpha = None,beta = None, method='root', bound_type='DD'):
    '''
    solving poisson equation using finite difference method
    Dirichlet boundary condition

    in domain:
    a <= x <= b
    boundary condition:
    u(a) = alpha
    u(b) = beta

    equation:
    D*u''(x) + q(x) = 0

    Parameters:
    -----------------
    q: function
        the function q(x, *args)
    a: float
        the left edge of the domain
    b: float
        the right edge of the domain
    N: int
        the number of grid points
    args: tuple
        the arguments for the function q(x)
    D: float
        the diffusion coefficient
    alpha: float
        D,N,R (a) = alpha, left boundary condition 
    beta: float or None
        D,N,R (b) = beta, right boundary condition
    method: string
        the method to solve the linear system: 'root' for SciPy or 'solve' for Numpy
    boundary: string
        the type of boundary condition: 'DD' for Dirichlet, Dirichlet, 'DN' for Dirichlet, Neumann, 'DR' for Dirichlet, Robin


    Returns:
    -----------------
    u: array
        the solution of the problem
    xi: array
        the grid points


    '''

    # create the grid
    x = np.linspace(a, b, N+1)
    x_int = x[1:-1] # interior points
    dx = (b-a)/N

    # define an initial guess for the solution
    u = np.zeros(N-1)

    # use boundary functions to define the boundary conditions (for interior points)
    A_mat, b_vec = boundary(alpha, beta, N, dx, bound_type)

    # define the vector q as a function of u but length N-1 
    def mak_q(u): 
        if callable(q):
            # make a vector of length x_int, and width 1 with q(x,u,args) for each x in x_int
            q_vec = q(x_int, u, *args)
            return q_vec
        else:
            return q
        
    # solve the linear system
    if method == 'scipy': # use SciPy root - use when q is a function of u
        # define the function f(u) = 0
        def f(u):
            return -D*A_mat@u + b_vec + mak_q(u)
        # solve the system
        sol = root(f, u)

        if not sol.success:
            raise ValueError('The solution did not converge')
        
        u = sol.x
        
    elif method == 'numpy': # use Numpy solve - use when q linear
        if callable(q):
            raise ValueError('q must be linear when using Numpy solve')
        q_vec = mak_q(u)
        u = solve(-D*A_mat, b_vec + q_vec)
    elif method == 'tdma': # use Thomas algorithm - use when the matrix is tridiagonal and q linear
        if callable(q):
            raise ValueError('q must be linear when using Thomas algorithm')
        q_vec = mak_q(u)
        u = tdma(-D*A_mat, b_vec + mak_q(u))

    # add the boundary conditions to the solution
    u = np.concatenate(([alpha], u, [beta]))

    return u, x


if __name__ == '__main__':

    # define the function q(x)
    def q(x, u, args):
        myu = args
        return np.exp(myu*u)

    # define the parameters
    D = 1
    alpha = 0
    beta = 0
    a = 0
    b = 1
    N = 10

    # define the method and boundary condition
    method = 'scipy'
    bound_type = 'DD'

    # solve for myu in [0, 4]
    myus = np.linspace(0, 4, 10)
    for myu in myus:

        # solve the problem
        u, xi = bvp_solver(q, a, b, N, myu,D=D, alpha=alpha, beta=beta, method=method, bound_type=bound_type)

        # plot the solution
        plt.plot(xi, u, 'o-', label='myu = %.2f' % myu)

    
    # # exact solution for source term q(x) = 1
    # def exact(x):
    #     return (-1/(2*D))*(x - a) * (x - b) + ((beta - alpha)/(b - a))*(x- a) + alpha

    # plot the exact solution
    # plt.plot(xi, exact(xi), '-', label='approximation by q(x) = 1')
    # plt.title('Bratu problem with q(x) = exp(myu*u) where myu = %f' % myu)

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()











