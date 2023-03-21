import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from numpy.linalg import solve

def bvp_solver(q, a, b, alpha ,beta ,N ,  *args , D = 1.0, method='root', boundary='Dirichlet'):
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
    D: float
        D*u''(x) + q(x) = 0
    a: float
        the left edge of the domain
    b: float
        the right edge of the domain
    alpha: float
        the value of u(a)
    beta: float
        the value of u(b)
    N: int
        the number of grid points
    method: string
        the method to solve the linear system: 'root' for SciPy or 'solve' for Numpy
    boundary: string
        the type of boundary condition: 'DD' for Dirichlet, Dirichlet, 'DN' for Dirichlet, Neumann, 'DR' for Dirichlet, Robin
    args: tuple
        the arguments for the function q(x)

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
    
    # modify the matrix A and vector b according to the boundary condition for the first point
    if boundary[0] == 'D':
        b_vec[0] = alpha
    elif boundary[0] == 'N':
        b_vec[0] = 2* alpha * dx
        # modify the matrix A
        A_mat[1, 0] = 2
    elif boundary[0] == 'R':
        b_vec[0] = 2 * alpha * dx
        # modify the matrix A
        A_mat[1, 0] = -2*(1 + alpha*dx)

    # modify the matrix A and vector b according to the boundary condition for the last point
    if boundary[1] == 'D':
        b_vec[-1] = beta
    elif boundary[1] == 'N':
        b_vec[-1] = 2* beta * dx
        # modify the matrix A
        A_mat[-2, -1] = 2
    elif boundary[1] == 'R':
        b_vec[-1] = 2 * beta * dx
        # modify the matrix A
        A_mat[-2, -1] = -2*(1 + beta*dx)


    # define the vector q as a function of u
    def mak_q(q, u, args):
        q_vec = np.zeros(N)
        for i in range(N):
            if len(args) != 0:
                q_vec[i] = q(xi[i], u[i], args)
            else:
                q_vec[i] = q(xi[i], u[i])
        return q_vec

    # solve the linear system
    if method == 'scipy': # use SciPy root
        u = root(lambda u: D*A_mat.dot(u) +  b_vec + mak_q(q,u,args), u).x

    elif method == 'numpy': # use Numpy linalg.solve
        u = solve(D*A_mat, -1*(b_vec + mak_q(q,u,args)))

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
    boundary = 'DR'

    # solve for myu in [0, 4]
    myus = np.linspace(0, 4, 10)
    for myu in myus:

        # solve the problem
        u, xi = bvp_solver(q, a, b, alpha, beta, N, myu, D=D, method=method, boundary=boundary)

        # plot the solution
        plt.plot(xi, u, 'o-', label='myu = %f' % myu)

    
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











