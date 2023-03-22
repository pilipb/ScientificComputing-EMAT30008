import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from numpy.linalg import solve

def bvp_solver(q, a, b,N , *args , D = 1.0, alpha = None,beta = None,gamma = None,delta = None,  method='root', boundary='Dirichlet'):
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
        the function q(x)
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
        u(a) = alpha, left boundary condition for Dirichlet
    beta: float or None
        u(b) = beta, right boundary condition for Dirichlet
    gamma: float or None
        u'(a) = gamma, right boundary condition for Neumann
    delta: float or None
        u'(b) = delta, right boundary condition for Neumann
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

    # define the grid points
    xi = np.linspace(a, b, N)

    # define the step size
    dx = (b - a)/N

    # define the matrix A
    A_mat = np.zeros((N, N))
    A_mat[0, 0] = -2/(dx**2)
    A_mat[-1, -1] = -2/(dx**2)
    for i in range(1, N-1):
        A_mat[i, i-1] = 1/(dx**2)
        A_mat[i, i] = -2/(dx**2) 
        A_mat[i, i+1] = 1/(dx**2)

    # define an initial guess for the solution
    u = np.zeros(N)

    # define the vector b
    b_vec = np.zeros(N)

    # modify the matrix A and vector b according to the boundary condition

    # list of boundary conditions
    boundary_list = ['D', 'N', 'R']

    # check the boundary condition
    if boundary[0] not in boundary_list or boundary[1] not in boundary_list:
        raise ValueError('The boundary condition must be D, N or R')
    
    # check that the corresponding boundary condition is given
    if boundary[0] == 'D' and alpha == None:
        raise ValueError('The boundary condition is Dirichlet, but alpha is not given')
    elif boundary[1] == 'D' and beta == None:
        raise ValueError('The boundary condition is Dirichlet, but beta is not given')
    elif boundary[1] == 'N' and gamma == None:
        raise ValueError('The boundary condition is Neumann, but gamma is not given')
    elif boundary[1] == 'R' and delta == None:
        raise ValueError('The boundary condition is Robin, but delta is not given')
    
    
    # modify the matrix A and vector b according to the boundary condition for the first point
    if boundary[0] == 'D':
        b_vec[0] = alpha


    # modify the matrix A and vector b according to the boundary condition for the last point
    if boundary[1] == 'D':
        b_vec[-1] = beta
    elif boundary[1] == 'N':
        b_vec[-1] = 2* gamma * dx
        # modify the matrix A
        A_mat[-2, -1] = 2
    elif boundary[1] == 'R':
        b_vec[-1] = 2 * delta * dx
        # modify the matrix A
        A_mat[-2, -1] = -2*(1 + delta*dx)


    # define the vector q as a function of u
    def mak_q(q, u, args):
        q_vec = np.zeros(N)
        for i in range(N):
            if len(args) != 0:
                q_vec[i] = q(xi[i], u[i], args)
            else:
                q_vec[i] = q(xi[i], u[i])
        return q_vec
    
    # create a Thomas algorithm solver
    def tdma(a, b):
        '''
        Thomas algorithm or Tridiagonal matrix algorithm (TDMA) for solving
        A.x = b
        where A is a tridiagonal matrix

        
        Parameters:
        -----------------
        a: array
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
        c[0] = a[0, 1]/a[0, 0]
        d[0] = b[0]/a[0, 0]

        # loop over the matrix
        for i in range(1, N-1):
            # solve the linear system
            c[i] = a[i, i+1]/(a[i, i] - a[i, i-1]*c[i-1])
            d[i] = (b[i] - a[i, i-1]*d[i-1])/(a[i, i] - a[i, i-1]*c[i-1])

        # solve the last linear system
        d[-1] = (b[-1] - a[-1, -2]*d[-2])/(a[-1, -1] - a[-1, -2]*c[-2])

        # solve the linear system backwards
        x[-1] = d[-1]
        for i in range(N-2, -1, -1):
            x[i] = d[i] - c[i]*x[i+1]

        return x

    # solve the linear system
    if method == 'scipy': # use SciPy root - use when q is a function of u
        u = root(lambda u: D*A_mat.dot(u) +  b_vec + mak_q(q,u,args), u).x
    elif method == 'numpy': # use Numpy solve - use when q linear
        u = solve(-D*A_mat, b_vec + mak_q(q, u, args))
    elif method == 'tdma': # use Thomas algorithm - use when the matrix is tridiagonal
        u = tdma(-D*A_mat, b_vec + mak_q(q, u, args))

    return u, xi


    
    



if __name__ == '__main__':

    # define the function q(x)
    def q(x, u, args):
        myu = args[0]
        return np.exp(myu*u)

    # define the parameters
    D = 1
    alpha = 0
    beta = 1
    a = 0
    b = 1
    N = 10
    # myu = 4

    # define the method and boundary condition
    method = 'scipy'
    boundary = 'DD'

    # solve for myu in [0, 4]
    myus = np.linspace(0, 4, 10)
    for myu in myus:


        # solve the problem
        u, xi = bvp_solver(q, a, b, N, myu,D=D, alpha=alpha, beta=beta,   method=method, boundary=boundary)

        # plot the solution
        plt.plot(xi, u, 'o-', label='myu = %.2f' % myu)

    
    # exact solution for source term q(x) = 1
    def exact(x):
        return (-1/(2*D))*(x - a) * (x - b) + ((beta - alpha)/(b - a))*(x- a) + alpha

    # plot the exact solution
    # plt.plot(xi, exact(xi), '-', label='approximation by q(x) = 1')
    # plt.title('Bratu problem with q(x) = exp(myu*u) where myu = %f' % myu)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()











