'''
implement alpha PDE solver using the explicit Euler method to solve the following PDE:

linear diffusion equation:

u_t = D*u_xx

u(alpha,t) = 0
u(beta,t) = 0

u(x,0) = sin((pi*(x-alpha)/beta-alpha))

the exact solution is:

u(x,t) = sin((pi*(x-alpha)/beta-alpha)) * exp(-pi**2*D*t/(beta-alpha)**2)

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from solvers import *
from math import ceil

def pde_solver(f, alpha, beta, a, b,bound, D, t_final, N, q = lambda x_int,t,u: 0,  C= 0.49, method = 'RK4'):

    '''
    A PDE solver that implements different integration methods to solve the PDE

    PDE in the form:

    u_t = D*u_xx + q(x,t,u)

    u(alpha,t) = alpha
    u(beta,t) = beta

    u(x,0) = f(x)

    Parameters:
    ------------------
    f : function
        the initial condition
    alpha : float
        the left boundary 
    beta : float
        the right boundary 
    a : float
        the left boundary value 
    b : float
        the right boundary value 
    bound : string
        the type of boundary condition
    D : float
        the diffusion coefficient from form 
    t_final : float
        the final time
    N : int
        the number of grid points
    q : function
        the source term default = 0
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

    # define the PDE - for alpha constant time therefore its alpha 1st order ODE
    def PDE(t, u , *args):
        # unpack the args
        D = args[0][0]
        A_ = args[0][1]
        b_ = args[0][2]
        q = args[0][3]
        return (D / dx**2) * (A_ @ u + b_) + q(x_int,t, u)
        
    
    '''
    annoyingly the solve_ivp function requires the PDE to have args as separate arguments,
    this is not the case for all my own methods so I have to define the PDE again
    '''
    def PDE_ivp(t, u, D, A_, b_, q):
        return (D / dx**2) * (A_ @ u + b_) + q(x_int,t, u)


    '''
    write a function to create the boundary matrices A and beta for different types of boundary conditions
    
    '''
    def boundary(alpha, beta, type):
        '''
        alpha : float
            the left boundary
        beta : float
            the right boundary
        type : string
            the type of boundary condition
            options: 'DD', 'DN', 'DR', 'ND', 'NN', 'NR','RD', 'RN', 'RR'
        '''
        # check the type of boundary condition is valid
        if type not in ['DD', 'DN', 'DR', 'ND', 'NN', 'NR','RD', 'RN', 'RR']:
            raise ValueError('Invalid boundary condition type')
        # check which type of boundary condition for the first point
        if type[0] == 'D':
            A = np.zeros((N-1, N-1))
            b = np.zeros(N-1)
            b[0] = alpha
        elif type[0] == 'N':
            A = np.zeros((N-1, N-1))
            b = np.zeros(N-1)
            b[0] = 2*alpha*dx
            A[0, 1] = 2
        elif type[0] == 'R':
            A = np.zeros((N-1, N-1))
            b = np.zeros(N-1)
            b[0] = 2*alpha*dx
            A[0, 1] = -2*(1+alpha*dx)

        # check which type of boundary condition for the last point
        if type[1] == 'D':
            b[-1] = beta
        elif type[1] == 'N':
            b[-1] = 2*beta*dx
            A[-2, -1] = 2
        elif type[1] == 'R':
            b[-1] = 2*beta*dx
            A[-2, -1] = -2*(1+beta*dx)

        return A, b
    
    # create the boundary matrices
    A_, b_ = boundary(alpha, beta, bound)



    # identify the method
    if method == 'explicit_euler':

        print('Using the explicit Euler method...\n')

        # print some info about time step
        print('\ndt = %.6f' % dt)
        print('%i time steps will be needed\n' % N_time)

        # loop over the steps
        for n in range(0, N_time):

            for i in range(1, N):

                u[n+1,i-1] = u[n,i-1] + dt * PDE(t[n], u[n,:], (D, A_, b_, q))[i-1]

        # concatenate the boundary conditions   
        u = np.concatenate((a*np.ones((N_time+1,1)), u, b*np.ones((N_time+1,1))), axis = 1)

        return u, t, x


    elif method == 'solve_ivp':

        print('Using the solve_ivp function...\n')
        
        sol = solve_ivp(PDE_ivp, (0, t_final), f(x_int), args=(D, A_, b_, q))

        # extract the solution
        u = sol.y
        t = sol.t

        N_time = len(t)

        # add on the u(alpha,t) and u(beta,t) boundary conditions - for plotting
        u = np.concatenate((a*np.ones((1,N_time)), u, b*np.ones((1,N_time))), axis = 0)

        return u.T, t, x

    else: # use the solve_to function

        # find method
        methods = {'Euler': euler_step, 'RK4': rk4_step, 'Heun': heun_step}

        # check if method is valid
        if method not in methods:
            raise ValueError('Invalid method, please enter alpha valid method: Euler, RK4, Heun or define your own in solvers.py')

        # set method
        method = methods[method]

        # print some info about time step
        print('\ndt = %.6f' % dt)
        print('%i time steps will be needed\n' % N_time)
        
        # loop over the time steps
        for n in range(0, N_time):

            # update the solution
            u[n+1,:] = method(PDE, u[n,:], t[n], dt, ( D, A_, b_, q))[0]

        # concatenate the boundary conditions
        u = np.concatenate((a*np.ones((N_time+1,1)), u, b*np.ones((N_time+1,1))), axis = 1)

        return u, t, x









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
    N = 100

    # define the exact solution
    u_exact = lambda x, t: np.sin(np.pi*(x-a)/b-a)*np.exp(-np.pi**2*D*t/b**2)


    # solve the problem for RK4, explicit_euler, and solve_ivp
    for method in ['RK4', 'explicit_euler', 'solve_ivp']:

        # solve the problem
        u, t, x = pde_solver(f, alpha, beta, a, b, 'DD', D, t_final, N, method = method)

        # plot the solution at 3 different times
        for n in np.linspace(0, len(t)-1, 3, dtype = int):
            plt.plot(x, u[n,:], label = '%s at t = %.2f' % (method, t[n]))

            # plot the exact solution at the same times
            plt.plot(x, u_exact(x, t[n]), '--', label = 'exact at t = %.2f' % t[n])



        plt.legend()
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.show()


    ### solve the dynamic Bratu problem

    # define the problem
    D = 1.0
    myu = [2,4]

    # define the function q(x) = exp(myu*u)
    def q(x, u, args):
        myu = args[0]
        return np.exp(myu*u)
    



    


    
    






