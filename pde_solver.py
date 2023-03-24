'''
implement a PDE solver using the explicit Euler method to solve the following PDE:

linear diffusion equation:

u_t = D*u_xx

u(a,t) = 0
u(b,t) = 0

u(x,0) = sin((pi*(x-a)/b-a))

the exact solution is:

u(x,t) = sin((pi*(x-a)/b-a)) * exp(-pi**2*D*t/(b-a)**2)

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from solvers import *
from math import ceil

def pde_solver(f, a, b, alpha, beta, D, t_final, N, C= 0.49, method = 'RK4'):

    '''
    A PDE solver that implements different integration methods to solve the PDE

    PDE in the form:

    u_t = D*u_xx

    u(a,t) = alpha
    u(b,t) = beta

    u(x,0) = f(x)

    Parameters:
    ------------------
    f : function
        the initial condition
    a : float
        the left boundary - Dirichlet
    b : float
        the right boundary - Dirichlet
    alpha : float
        the left boundary value - Dirichlet
    beta : float
        the right boundary value - Dirichlet
    D : float
        the diffusion coefficient from form 
    t_final : float
        the final time
    N : int
        the number of grid points
    C : float
        the Courant number - CFL condition default = 0.49 (for homemade methods)
    method : string
        the integration method to be used
        options: 'Euler', 'RK4', 'Heun', 'solve_ivp', 'explicit_euler'

    '''

    # create the grid
    x = np.linspace(a, b, N+1)
    x_int = x[1:-1] # interior points
    dx = (b-a)/N

    # time discretization
    dt = C*dx**2/D
    N_time = ceil(t_final/dt)
    t = dt * np.arange(N_time)

    # preallocate solution and boundary conditions
    u = np.zeros((N_time+1, N-1))
    u[0,:] = f(x_int)

    # define the PDE - for a constant time therefore its a 1st order ODE
    def PDE(t, u , args):
        D, A_DD, b_DD = args
        return (D / dx**2) * (A_DD @ u + b_DD)

    # create the matrix A_DD
    A_DD = np.zeros((N-1, N-1))
    for i in range(1, N-2):
        A_DD[i, i-1] = 1
        A_DD[i, i] = -2
        A_DD[i, i+1] = 1

    A_DD[0, 1] = 1
    A_DD[0, 0] = -2
    A_DD[-1, -2] = 1
    A_DD[-1, -1] = -2

    # create the vector b_DD
    b_DD = np.zeros(N-1)
    b_DD[0] = alpha
    b_DD[-1] = beta


    # identify the method
    if method == 'explicit_euler':

        print('Using the explicit Euler method...\n')

        # print some info about time step
        print('\ndt = %.6f' % dt)
        print('%i time steps will be needed\n' % N_time)

        # loop over the steps
        for n in range(0, N_time):

            for i in range(1, N):

                u[n+1,i-1] = u[n,i-1] + dt * PDE(t[n], u[n,:], args=(D, A_DD, b_DD))[i-1]

        # concatenate the boundary conditions   
        u = np.concatenate((alpha*np.ones((N_time+1,1)), u, beta*np.ones((N_time+1,1))), axis = 1)

        return u, t, x


    elif method == 'solve_ivp':

        print('Using the solve_ivp function...\n')
        
        sol = solve_ivp(PDE, (0, t_final), f(x_int), args=(D, A_DD, b_DD))

        # extract the solution
        u = sol.y
        t = sol.t

        N_time = len(t)

        # add on the u(a,t) and u(b,t) boundary conditions - for plotting
        u = np.concatenate((alpha*np.ones((1,N_time)), u, beta*np.ones((1,N_time))), axis = 0)

        return u.T, t, x

    else: # use the solve_to function

        # find method
        methods = {'Euler': euler_step, 'RK4': rk4_step, 'Heun': heun_step}

        # check if method is valid
        if method not in methods:
            raise ValueError('Invalid method, please enter a valid method: Euler, RK4, Heun or define your own')

        # set method
        method = methods[method]

        # print some info about time step
        print('\ndt = %.6f' % dt)
        print('%i time steps will be needed\n' % N_time)
        
        # loop over the time steps
        for n in range(0, N_time):

            # loop over the steps using solve_to
            for i in range(1, N):

                # update the solution
                u[n+1,:] = method(PDE, u[n,:], t[n], dt,args=( D, A_DD, b_DD))[0]

        # concatenate the boundary conditions
        u = np.concatenate((alpha*np.ones((N_time+1,1)), u, beta*np.ones((N_time+1,1))), axis = 1)

        return u, t, x


def explicit_euler_calc(u, C, alpha, beta, N, n):
    '''
    Explicit Euler method, for an ODE, updates the solution at the next time step.

    Parameters
    ----------
    u : array
        The solution at the current time step.
    C : float
        The time step.
    alpha : float
        The boundary condition at the left end of the domain.
    beta : float
        The boundary condition at the right end of the domain.
    N : int
        The number of grid points.
    n : int
        The current time step.

    Returns
    ----------
    u : array
        The solution at the next time step.

    '''
    
    # loop over the grid
    for i in range(0, N-1):
        if i==0:
            u[n+1,i] = u[n,i] + C*(u[n,i+1] - 2*u[n,i] + alpha)
        elif i < 0 and i < N-2:
            u[n+1,i] = u[n,i] + C*(u[n,i+1] - 2 *u[n,i] + u[n,i-1])
        else:
            u[n+1,N-2] = u[n,N-2] + C*(beta - 2*u[n,N-2]+u[n,N-3])

    return u[n+1,:]


if __name__ == '__main__':

    # test the solver for the linear diffusion equation

    # define the problem
    D = 0.5
    a = 0.0
    b = 1.0
    alpha = 0.0
    beta = 0.0
    f = lambda x: np.sin((np.pi*(x-a)/b-a))
    t_final = 0.5

    # define the exact solution
    u_exact = lambda x, t: np.sin(np.pi*(x-a)/b-a)*np.exp(-np.pi**2*D*t/b**2)


    # solve the problem for RK4, explicit_euler, and solve_ivp
    for method in ['RK4', 'explicit_euler', 'solve_ivp']:

        # solve the problem
        u, t, x = pde_solver(f, a, b, alpha, beta, D, t_final, N = 10, C = 0.49, method = method)

        # plot the solution at 3 different times
        for n in np.linspace(0, len(t)-1, 3, dtype = int):
            plt.plot(x, u[n,:], label = '%s at t = %.2f' % (method, t[n]))

            # plot the exact solution at the same times
            plt.plot(x, u_exact(x, t[n]), '--', label = 'exact at t = %.2f' % t[n])



        plt.legend()
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.show()

    


    
    






