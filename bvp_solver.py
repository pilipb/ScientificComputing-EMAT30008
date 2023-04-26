from helpers import *
from scipy.optimize import root
import numpy.linalg
import numpy as np
import matplotlib.pyplot as plt

class ODE():
    '''
    Second order ODE of the form:
    m*u'' + c*u' + k * q(x,u,t,args) = 0

    Parameters
    ----------
    m : float
        2nd order coefficient.
    c : float
        1st order coefficient.
    k : float
        0th order coefficient.
    q : function
        Function of x and u. (source term)
    bound_type : string
        The type of boundary condition: DD, DN, DR, ND, NN, NR, RD, RN, RR (Dirichlet, Neumann, Robin)
    alpha : float
        The left boundary condition.
    beta : float
        The right boundary condition.
    a : float
        The left edge of the domain.
    b : float
        The right edge of the domain.
    args: tuple
        The arguments for the function q(x,u,args)

    Examples
    --------
    >>> ODE(1, 0, 1, lambda x, u: 0, 'DD', 0, 0, 0, 1)


    '''
    def __init__(self, m, c, k, q, bound_type, alpha, beta, a, b, *args):
        
        self.m = m
        self.c = c
        self.k = k
        self.q = q
        self.bound_type = bound_type
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.args = args

class Solver():
    '''
    Class to solve the ODE

    Parameters
    ----------
    ODE : ODE object
        The ODE object to be solved.
    N : int
        The number of interior points.
    method : string
        The method to solve the ODE: .

    Examples
    --------
    >>> ODE(1, 0, 1, lambda x, u: 0, 'DD', 0, 0, 0, 1)
    >>> solver = Solver(ODE, 100, 'scipy')
    >>> u = solver.solve()
    >>> x = solver.x

    ''' 
    def __init__(self, ODE, N, method):

        self.ODE = ODE
        self.N = N
        self.method = method

        # create the grid
        self.x = np.linspace(ODE.a, ODE.b, N+1)
        self.x_int = self.x[1:-1] # interior points
        self.dx = (ODE.b-ODE.a)/N

        # create the matrix A and vector b
        self.A_mat, self.b_vec = boundary(ODE.alpha, ODE.beta, N, self.dx, ODE.bound_type)

        # define an initial guess for the solution
        self.u = np.zeros(N-1)

    def solve(self):
        '''
        Solve the ODE using the method specified in self.method

        Returns
        -------
        u : np.array
            The solution to the ODE

        Examples
        --------
        >>> solver.solve() 


        
        '''

        if self.method == 'scipy':
            u = self.scipy_solve()
        elif self.method == 'numpy':
            u = self.numpy_solve()
        elif self.method == 'tdma':
            u = self.tdma_solve()
        else:
            raise ValueError('method not recognised')


        return u
    
    def q_vector(self, x, u, *args):
        '''
        Makes a q vector for the ODE

        Returns
        -------
        q_vec : np.array
            The q vector for the ODE

        '''
        # define the vector q as a function of u but length N-1
        if callable(self.ODE.q):
            # make a vector of length x_int, and width 1 with q(x,t, u,args) for each x in x_int
            q_vec = self.ODE.q(x, u, *args)
            return q_vec
        
        # if q is a constant
        else:
            return self.ODE.q * np.ones(self.N-1)
    

    def scipy_solve(self):    
        '''
        solves the ODE using scipy root function

        Returns
        -------
        u : np.array
            The solution to the ODE
        
        '''
        # define the function to be solved
        def f(u):
            return self.ODE.m * self.A_mat @ u - self.ODE.k * self.b_vec - self.ODE.c * self.q_vector( self.x_int, u, *self.ODE.args)
        
        # solve the function
        sol = root(f, self.u)

        # check if the solution converged
        if not sol.success:
            raise ValueError(sol.message)

        u = sol.x

        # concatenate the boundary conditions
        u = np.concatenate(([self.ODE.alpha], -u, [self.ODE.beta]))

        return u
    
    def numpy_solve(self):
        '''
        solves the ODE using numpy solve function (if q is a constant)

        Returns
        -------
        u : np.array
            The solution to the ODE
        
        '''
    
        # check if q is a constant
        if callable(self.ODE.q):
            raise ValueError('q is not a constant, use scipy method instead')
        
        # solve the function
        u = numpy.linalg.solve(self.ODE.m * self.A_mat, - self.ODE.k * self.b_vec - self.ODE.c * self.q_vector(self.x_int, self.u, *self.ODE.args))

        # concatenate the boundary conditions
        u = np.concatenate(([self.ODE.alpha], u, [self.ODE.beta]))

        return u
    
    def tdma_solve(self):
        '''
        solves the ODE using the tridiagonal matrix algorithm

        Returns
        -------
        u : np.array
            The solution to the ODE
        
        '''
        from bvp_solver import tdma

        # check if q is a constant
        if callable(self.ODE.q):
            raise ValueError('q is not a constant, use scipy method instead')
        
        # solve the function
        u = tdma(self.ODE.m * self.A_mat, - self.ODE.k * self.b_vec - self.ODE.c * self.q_vector(self.x_int, self.u, *self.ODE.args))

        # concatenate the boundary conditions
        u = np.concatenate(([self.ODE.alpha], u, [self.ODE.beta]))

        return u


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

    
##### test functions #####

if __name__ == '__main__':

    # define the ODE
    m = 1
    c = 1
    k = 1

   
    # def q(x, u, *args):
    #     return 1
    q = 1
    
    bound_type = 'DD'
    alpha = 0
    beta = 0
    a = 0
    b = 1
    args = (3,)

    # create the ODE object
    ODE = ODE(m, c, k, q, bound_type, alpha, beta, a, b, *args)

    # create the solver object
    N = 10
    method = 'tdma'
    solver = Solver(ODE, N, method)

    # solve the ODE
    u = solver.solve()

    # extract the grid
    x = solver.x

    # plot the solution
    plt.plot(x, u, 'o-')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()




