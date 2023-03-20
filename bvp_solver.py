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
        the type of boundary condition: 'Dirichlet' or 'Neumann' or 'Robin'
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
    if boundary == 'Dirichlet':
        b_vec[0] = alpha
        b_vec[-1] = beta

    # define the vector q
    q_vec = np.zeros(N)
    # if there are arguments for the function q(x), then use them
    if len(args) != 0:
        for i in range(N):
            q_vec[i] = q(xi[i], u[i], args)
    else:
        for i in range(N):
            q_vec[i] = q(xi[i], u[i])
            

    # solve the linear system
    if method == 'scipy': # use SciPy root
        u = root(lambda u: D*A_mat.dot(u) +  b_vec + q_vec, u).x

    elif method == 'numpy': # use Numpy linalg.solve
        u = solve(D*A_mat, -1*(b_vec + q_vec))

    return u, xi

if __name__ == '__main__':

    # define the function q(x)
    def q(x, u, args):
        myu = args[0]
        return np.exp(myu*u)

    # define the parameters
    D = 1
    alpha = 0
    beta = 0
    a = 0
    b = 1
    N = 10
    # myu = 4

    # define the method and boundary condition
    method = 'scipy'
    boundary = 'Dirichlet'

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











