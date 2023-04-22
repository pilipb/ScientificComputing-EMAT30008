import numpy as np
from math import ceil
from bvp_solver import ODE
from helpers import boundary

class PDE():
    '''
    Class to store the parameters of the ODE
    '''
    def __init__(self, f, m, c, k, q, bound_type, alpha, beta, a, b, *args):
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
    Solver class for the PDE
    
    '''
    def __init__(self, PDE, N, t_final, method, CFL=0.49):
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

        '''
        if self.method == 'solve_ivp':
            u = self.solveivp_solve()
        elif self.method == 'explicit_euler':
            u = self.explicit_solve()

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
    
    def explicit_solve(self):
        '''
        Solve the PDE using the explicit Euler method

        '''
        # define the PDE - for alpha constant time therefore its alpha 1st order ODE
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
        
        u = self.u
        
        # loop over the time steps
        for n in range(0, self.N_time):

            # loop over the interior points
            for i in range(1, self.N):

                u[n+1,i-1] = u[n,i-1] + self.dt * PDE_solve(self.t[n], u[n,:], (self.PDE.m, self.A_mat, self.b_vec, self.PDE.q, self.PDE.args))[i-1]

        # concatenate the boundary conditions 
        u = np.concatenate((alpha*np.ones((self.N_time+1,1)), u, beta*np.ones((self.N_time+1,1))), axis = 1).T

        return u
        
##### TEST #####

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # define the ODE
    m = 0.5
    c = 1
    k = 1

    q = lambda x, t, u, *args: 1
    bound_type = 'DD'
    alpha = 0
    beta = 0
    a = 0
    b = 1
    args = (3,)
    f = lambda x: np.sin((np.pi*(x-a)/b-a))

    # create the PDE object
    pde = PDE(f, m, c, k, q, bound_type, alpha, beta, a, b, *args)

    # create the solver object
    N = 10
    method = 'explicit_euler'
    t_final = 1
    solver = Solver(pde, N, t_final, method)

    # solve the ODE
    u = solver.solve()

    # extract the grid
    x = solver.x

    # plot the solution at 3 different times
    for n in np.linspace(0, len(solver.t)-1, 5, dtype = int):

        plt.plot(x, u[:,n], label = 't = {}'.format(solver.t[n]))

    plt.legend()
    plt.show()


        

        

        




        


    