import numpy as np
from math import ceil
from bvp_solver import ODE
from helpers import boundary
from scipy.optimize import root
from scipy.sparse import linalg
import scipy.sparse as sp

class PDE():
    '''
    Second order ODE of the form:
    du/dt = m*u'' + q(x,u,t,args)

    with starting conditions:
    u(a,t) = alpha
    u(b,t) = beta

    u(x,0) = f(x)

    Parameters
    ----------
    f : function
        The initial condition.
    m : float
        2nd order coefficient.
    q : function
        Function of x, t and u. (source term)
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
    >>> PDE(lambda x: 0, 1, lambda x, t, u: 0, 'DD', 0, 0, 0, 1)


    '''
    def __init__(self, f, m,  q, bound_type, alpha, beta, a, b, *args):
        
        self.f = f
        self.m = m
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
    PDE : PDE object
        The PDE object to be solved.
    N : int
        The number of interior points.
    method : string
        The method to solve the ODE: solve_ivp, implicit_euler, crank_nicolson, imex_euler, euler, rk4, heun

    Returns
    -------
    u : np.array
        The solution to the PDE.

    Examples
    --------
    >>> PDE(lambda x: 0, 1, lambda x, t, u: 0, 'DD', 0, 0, 0, 1)
    >>> solver = Solver(PDE, 100, 1, 'solve_ivp')
    >>> u = solver.solve()
    >>> t = solver.t
    >>> x = solver.x

    '''
    def __init__(self, PDE, N, t_final, method, CFL=0.49, sparse=False):
        
        self.PDE = PDE
        self.N = N
        self.t_final = t_final
        self.method = method
        self.CFL = CFL
        self.sparse = sparse


        # create the grid (space discretisation)
        self.x = np.linspace(PDE.a, PDE.b, N+1)
        self.x_int = self.x[1:-1] # interior points
        self.dx = (self.PDE.b-self.PDE.a)/N

        # time discretisation
        self.dt = CFL*self.dx**2/self.PDE.m
        self.N_time = ceil(t_final/self.dt)
        self.t = self.dt * np.arange(self.N_time)

        # preallocate solution and boundary conditions
        self.u = np.zeros((self.N_time+1, N-1))
        self.u[0,:] = self.PDE.f(self.x_int)

        # create the matrix A and vector b
        self.A_mat, self.b_vec = boundary(PDE.alpha, PDE.beta, N, self.dx, PDE.bound_type, sparse=sparse)


    def q_vector(self, x, t, u, *args):
        '''
        Makes a q vector for the ODE

        Returns
        -------
        q_vec : np.array
            The q vector for the ODE
        '''
        # define the vector q as a function of u but length N-1
        if callable(self.PDE.q):
            # make a vector of length x_int, and width 1 with q(x,u,args) for each x in x_int
            q_vec = self.PDE.q(x, t, u, *args)
            return q_vec
        
        # if q is a constant
        else:
            return self.PDE.q * np.ones(self.N-1)



    def solve(self):
        '''
        Solve the PDE using the specified method

        Returns
        -------
        u : np.array
            The solution to the PDE [u(x,t)]

        Examples
        --------
        >>> PDE(lambda x: 0, 1, lambda x, t, u: 0, 'DD', 0, 0, 0, 1)
        >>> solver = Solver(PDE, 100, 1, 'solve_ivp')
        >>> u = solver.solve()



        '''



        if self.method == 'solve_ivp':
            self.u = self.solveivp_solve()
        elif self.method == 'implicit_euler':
            self.u = self.implicit_solve()
        elif self.method == 'crank_nicolson':
            self.u = self.crank_nicolson_solve()
        elif self.method == 'imex_euler':
            self.u = self.imex_euler_solve()
        elif self.method in ['Euler', 'RK4', 'Heun']:
            self.u = self.custom_solve(self.method)
        else:
            raise ValueError('Method not recognised')

        return self.u
    
    def solveivp_solve(self):
        '''
        Solve the PDE using the solve_ivp method

        '''
        from scipy.integrate import solve_ivp
        u = self.u

        if self.sparse:
            raise ValueError('solve_ivp does not support sparse matrices')
        
        # print('Solving using solve_ivp')

        # define the PDE - different form for solve_ivp
        if callable(self.PDE.q):
            def PDE_ivp(t, u, D, A_, b_, q, *args):
                return (D / self.dx**2) * (A_ @ u + b_) + q(self.x_int,t, u, *args)
        else:
            q_vec = self.PDE.q * np.ones(self.N-1)
            def PDE_ivp(t, u, D, A_, b_, q, *args):
                return (D / self.dx**2) * (A_ @ u + b_) + q_vec


        sol = solve_ivp(PDE_ivp, (0, self.t_final), self.PDE.f(self.x_int), args=(self.PDE.m, self.A_mat, self.b_vec, self.PDE.q, self.PDE.args))

        # extract the solution
        u = sol.y
        self.t = sol.t

        # add the boundary conditions at all times by concatenating alpha and beta at all times to the solution
        alphas = np.ones((len(self.t),1)) * self.PDE.alpha
        betas = np.ones((len(self.t),1)) * self.PDE.beta
        self.u = np.concatenate((alphas, u.T, betas), axis=1).T

        return self.u

    
    def implicit_solve(self):
        '''
        Solve the PDE using the implicit Euler method using root
            
        '''
        
        if self.sparse:
            raise ValueError('implicit_solve does not support sparse matrices')
        
        # print('Solving using the implicit Euler method')

        u = self.u
        # function to solve but as a function of u, t, and args
        def F(u, t, *args):
            # unpack the args
            D = args[0][0]
            A_ = args[0][1]
            b_ = args[0][2]
            q = args[0][3]

            # calculate the source term
            if callable(q):
                qval = q(self.x_int, t, u, *args[0][4:])
            else:
                qval = q

            # apply the PDE
            return (D / self.dx**2) * (A_ @ u + b_) + qval - u
        

        # loop over the steps
        for n in range(0, self.N_time):
                
            # define the function to solve (F(u_n+1) = 0 (removing the time dependence)
            def F_solve(u):
                return F(u, self.t[n], (self.PDE.m, self.A_mat, self.b_vec, self.PDE.q, self.PDE.args)) - u

            # solve the function
            sol = root(F_solve, u[n,:], method='hybr')
            # extract the solution
            u[n+1,:] = sol.x

        # concatenate the boundary conditions
        self.u = np.concatenate((self.PDE.alpha*np.ones((self.N_time+1,1)), u, self.PDE.beta*np.ones((self.N_time+1,1))), axis = 1)
        
        return self.u
    
    def crank_nicolson_solve(self):
        '''
        Solve the PDE using the Crank-Nicolson method (linear only)
            
        '''

        # check that q is zero or constant
        if callable(self.PDE.q):
            raise ValueError('q must be zero or constant for the Crank-Nicolson method')
        
        # print('Solving using the Crank-Nicolson method')
        
        u = self.u

        # define the matrices for the implicit method
        C = self.dt * self.PDE.m / self.dx**2
        if self.sparse:
            A = sp.eye(self.N-1) - C/2 * self.A_mat
            b = A.dot(u[-1,:]) + C/2 * (self.b_vec + self.PDE.q * np.ones(self.N-1))

        else:
            A = np.eye(self.N-1) - C/2 * self.A_mat
            b = A * u[-1,:] + C/2 * (self.b_vec + self.PDE.q * np.ones(self.N-1))

        # loop over the steps
        if self.sparse:
            for n in range(0, self.N_time):

                u[n+1,:] = sp.linalg.spsolve(A, b[:])

                # update the boundary conditions
                u[n+1,0] = self.PDE.alpha
                u[n+1,-1] = self.PDE.beta

        else:
            for n in range(0, self.N_time):
        
                u[n+1,:] = np.linalg.solve(A, b[-1,:])

                # update the boundary conditions
                u[n+1,0] = self.PDE.alpha
                u[n+1,-1] = self.PDE.beta

        # concatenate the boundary conditions - for plotting
        self.u = np.concatenate((self.PDE.alpha*np.ones((self.N_time+1,1)), u, self.PDE.beta*np.ones((self.N_time+1,1))), axis = 1).T

        return self.u
    
    def imex_euler_solve(self):
        '''
        Implicit-Explicit Euler method
                nonlinear terms are solved explicitly (q)
                linear terms are solved implicitly (everything else)
            
        '''
        if self.sparse:
            raise ValueError('imex_euler_solve does not currently support sparse matrices')
        # assume q is a function of x, t, u, and args
        # print('Solving using the IMEX Euler method')

        # implicit linear solver
        # define the matrices for the implicit method
        u = self.u
        C = self.dt * self.PDE.m / self.dx**2
        A = np.eye(self.N-1) - C * self.A_mat
        b = u + C * self.b_vec

        # loop over the steps
        for n in range(0, self.N_time):

            if callable(self.PDE.q):
                # calculate the source term
                qval = self.PDE.q(self.x_int, self.t[n], u[n,:], *self.PDE.args)
            else:
                qval = self.PDE.q

            u[n+1,:] = np.linalg.solve(A, b[-1,:]) + self.dt * qval

            # update the boundary conditions
            u[n+1,0] = self.PDE.alpha
            u[n+1,-1] = self.PDE.beta 

        # concatenate the boundary conditions 
        self.u = np.concatenate((self.PDE.alpha*np.ones((self.N_time+1,1)), u, self.PDE.beta*np.ones((self.N_time+1,1))), axis = 1).T

        return self.u
    
    def custom_solve(self, option):
        '''
        solve the PDE using a homemade explicit solver using a method from solvers.py
                Euler, RK4 or Heun's method
    
        '''
        from solvers import euler_step, rk4_step, heun_step

        if self.sparse:
            raise ValueError('custom solvers do not support sparse matrices')
        
        # print('Solving using the ' + option + ' method')

        # function to solve but as a function of u, t, and args
        def PDE_solve(t, u , *args):
            # unpack the args
            D = args[0][0]
            A_ = args[0][1]
            b_ = args[0][2]
            q = args[0][3]

            # calculate the source term
            if callable(q):
                qval = q(self.x_int, t, u, *args[0][4:])
            else:
                qval = q

            # apply the PDE
            return (D / self.dx**2) * (A_ @ u + b_) + qval

        # find method
        methods = {'Euler': euler_step, 'RK4': rk4_step, 'Heun': heun_step}
        method = methods[option]

        u = self.u

        # loop over the time steps
        for n in range(0, self.N_time):

            # update the solution
            u[n+1,:] = method(PDE_solve, u[n,:], self.t[n], self.dt, ( self.PDE.m, self.A_mat, self.b_vec, self.PDE.q , self.PDE.args))[0]

        # concatenate the boundary conditions
        self.u = np.concatenate((self.PDE.alpha*np.ones((self.N_time+1,1)), u, self.PDE.beta*np.ones((self.N_time+1,1))), axis = 1).T

        return self.u
    
def profile(PDE, N, t_final):
    '''
    Returns the profile of the solvers using cProfile

    Parameters
    ----------
    PDE : PDE object
        equation to solve.
    N : int
        number of points in the spatial domain.
    t_final : float
        final time to solve to.

    Returns
    -------
    stats : list
        list of pstats.Stats objects.
    methods : list
        list of methods used. [Euler, RK4, Heun, implicit_euler, crank_nicolson, imex_euler, solve_ivp]

    Example
    -------
    >>> from pde import PDE, Solver
    >>> pde = PDE(---)
    >>> stats, methods = profile(pde, 100, 1)
    >>> stats[0].print_stats()
    
    '''
    import cProfile
    import pstats

    stats = []
    methods = ['Euler', 'RK4', 'Heun','implicit_euler', 'crank_nicolson', 'imex_euler', 'solve_ivp']

    # loop through methods and profile
    for method in methods:

        # profile the function
        pr = cProfile.Profile()
        pr.enable()
        # make solver object
        solver = Solver(PDE, N, t_final, method)
        u = solver.solve()
        pr.disable()
        
        # get the stats
        ps = pstats.Stats(pr).sort_stats('tottime')
        stats.append(ps)

    return stats, methods







        





        

        

        




        


    