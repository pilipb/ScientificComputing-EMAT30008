import numpy as np
from math import ceil
from bvp_solver import ODE
from helpers import boundary
from scipy.optimize import root

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
            The method to solve the ODE:

        '''
    def __init__(self, PDE, N, t_final, method, CFL=0.49):
        
        self.PDE = PDE
        self.N = N
        self.t_final = t_final
        self.method = method
        self.CFL = CFL

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
        self.A_mat, self.b_vec = boundary(PDE.alpha, PDE.beta, N, self.dx, PDE.bound_type)


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

        '''


        if self.method == 'solve_ivp':
            u = self.solveivp_solve()
        elif self.method == 'implicit_euler':
            u = self.implicit_solve()
        elif self.method == 'crank_nicolson':
            u = self.crank_nicolson_solve()
        elif self.method == 'imex_euler':
            u = self.imex_euler_solve()
        elif self.method in ['Euler', 'RK4', 'Heun']:
            u = self.custom_solve(self.method)

        return u
    
    def solveivp_solve(self):
        '''
        Solve the PDE using the solve_ivp method

        '''
        from scipy.integrate import solve_ivp

        # define the PDE - different form for solve_ivp
        def PDE_ivp(t, u, D, A_, b_, q, *args):
            return (D / self.dx**2) * (A_ @ u + b_) + q(self.x_int,t, u, *args)

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
            return (D / self.dx**2) * (A_ @ u + b_) + qval 
        

        # loop over the steps
        for n in range(0, self.N_time):
                
            # define the function to solve (F(u_n+1) = 0 (removing the time dependence)
            def F_solve(u):
                return F(u, self.t[n], (self.PDE.m, self.A_mat, self.b_vec, self.PDE.q, self.PDE.args)) - self.u[n,:]

            # solve the function
            sol = root(F_solve, self.u[n,:], method='hybr')
            # extract the solution
            self.u[n+1,:] = sol.x

        # concatenate the boundary conditions
        self.u = np.concatenate((self.PDE.alpha*np.ones((self.N_time+1,1)), self.u, self.PDE.beta*np.ones((self.N_time+1,1))), axis = 1).T
        
        return self.u
    
    def crank_nicolson_solve(self):
        '''
        Solve the PDE using the Crank-Nicolson method (linear only)
            
        '''
        # check that q is zero or constant
        if callable(self.PDE.q):
            raise ValueError('q must be zero or constant for the Crank-Nicolson method')
        
        u = self.u

        # define the matrices for the implicit method
        C = self.dt * self.PDE.m / self.dx**2
        A = np.eye(self.N-1) - C/2 * self.A_mat
        b = A * u[-1,:] + C/2 * (self.b_vec + self.PDE.q * np.ones(self.N-1))

        # loop over the steps
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
        # assume q is a function of x, t, u, and args

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
    
def profile(PDE, N, t_final, plot = True):
    '''
    return the profile of the solvers using cProfile
    
    '''
    import cProfile
    import pstats

    stats = []

    # loop through methods and profile
    for method in ['Euler', 'RK4', 'Heun','implicit_euler', 'crank_nicolson', 'imex_euler']:

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

    # plot the performance results
    import matplotlib.pyplot as plt
    import numpy as np

    if plot:
        # plot the results as a bar chart
        fig, ax = plt.subplots(1,1, figsize = (10,5))
        ax.set_title('Performance of different solvers')
        ax.set_ylabel('Time (s)')
        ax.set_xlabel('Solver')
        ax.set_xticks(np.arange(0,6))
        ax.set_xticklabels(['Euler', 'RK4', 'Heun', 'Implicit Euler', 'Crank-Nicolson', 'IMEX Euler'])
        ax.bar(np.arange(0,6), [s.total_tt for s in stats])
        plt.show()

    return stats




        
##### TEST #####

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # define the ODE
    m = 0.01

    # q = lambda x, t, u, *args: np.exp(args[0] * u)
    q = 0
    bound_type = 'DD'
    alpha = 0
    beta = 0
    a = 0
    b = 1
    args = (3,)
    f = lambda x: np.sin((np.pi*(x-a)/b-a))

    # create the PDE object
    pde = PDE(f, m, q, bound_type, alpha, beta, a, b, *args)

    # # create the solver object
    # N = 100
    # method = 'explicit_euler'
    # t_final = 0.01
    # solver = Solver(pde, N, t_final, method, CFL=0.6)

    # # solve the ODE
    # u = solver.solve()

    # # extract the grid
    # x = solver.x

    # # plot the solution at 3 different times
    # for n in np.linspace(0, len(solver.t)-1, 10, dtype = int):
    #     plt.plot(x, u[:,n], label = 't = {}'.format(solver.t[n]))

    # plt.legend()
    # plt.show()

    # profile the solvers
    stats = profile(pde, 100, 0.01, plot = False)

    # for each solver, print the top 10 functions
    for s in stats:
        s.print_stats(10)
        
 
    

    






        

        

        




        


    