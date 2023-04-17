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
from helpers import *

def pde_solver(f, alpha, beta, a, b,bound_type, D, t_final, N, q = lambda x_int,t,u, *args: 0,  C= 0.49, method = 'RK4', args = None):

    '''
    A PDE solver that implements different integration methods to solve the PDE

    PDE in the form:

    u_t = D*u_xx + q(x,t,u)

    left boundary condition:
    D,N,R (a,t) =  alpha (Dirichlet, Neumann, Robin)

    right boundary condition:
    D,N,R (b,t) =  beta (Dirichlet, Neumann, Robin)

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
    args : tuple
        the arguments for the q function

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

        # calculate the source term
        if callable(q):
            qval = q(x_int, t, u, *args[0][4:])
        else:
            qval = q

        # apply the PDE
        return (D / dx**2) * (A_ @ u + b_) + qval

    # define the PDE - different form for solve_ivp
    def PDE_ivp(t, u, D, A_, b_, q, *args):
        return (D / dx**2) * (A_ @ u + b_) + q(x_int,t, u, *args)

    
    # create the boundary matrices
    A_, b_ = boundary(alpha, beta, N, dx, bound_type)

    # identify the method
    if method == 'explicit_euler':

        print('Using the explicit Euler method...\n')

        # print some info about time step
        # print('\ndt = %.6f' % dt)
        # print('%i time steps will be needed\n' % N_time)

        # loop over the steps
        for n in range(0, N_time):

            for i in range(1, N):

                u[n+1,i-1] = u[n,i-1] + dt * PDE(t[n], u[n,:], (D, A_, b_, q, args))[i-1]

        # concatenate the boundary conditions - for plotting
        u = np.concatenate((alpha*np.ones((N_time+1,1)), u, beta*np.ones((N_time+1,1))), axis = 1)
        

        return u, t, x


    elif method == 'solve_ivp':

        print('Using the solve_ivp function...\n')

        sol = solve_ivp(PDE_ivp, (0, t_final), f(x_int), args=(D, A_, b_, q, args))

        # extract the solution
        u = sol.y
        t = sol.t

        N_time = len(t)

        # add on the u(alpha,t) and u(beta,t) boundary conditions - for plotting
        u = np.concatenate((alpha*np.ones((1,N_time)), u, beta*np.ones((1,N_time))), axis = 0)
        

        return u.T, t, x
    
    elif method == 'implicit_euler':

        print('Using the implicit Euler method...\n')

        # print some info about time step
        print('\ndt = %.6f' % dt)
        print('%i time steps will be needed\n' % N_time)

        # define the matrices for the implicit method
        C = dt * D / dx**2
        A = np.eye(N-1) - C * A_
        b = u + C * b_

        # loop over the steps
        for n in range(0, N_time):
                
                u[n+1,:] = np.linalg.solve(A, b[n,:])
    
                # update the boundary conditions
                u[n+1,0] = alpha
                u[n+1,-1] = beta

        # concatenate the boundary conditions - for plotting
        u = np.concatenate((alpha*np.ones((N_time+1,1)), u, beta*np.ones((N_time+1,1))), axis = 1)

        return u, t, x


    else: # use the solve_to function

        # find method
        methods = {'Euler': euler_step, 'RK4': rk4_step, 'Heun': heun_step}

        # check if method is valid
        if method not in methods:
            raise ValueError('Invalid method, please enter alpha valid method: Euler, RK4, Heun or define your own in solvers.py')

        # set method
        method = methods[method]

        print('Using the %s method...\n' % method.__name__)

        # print some info about time step
        # print('\ndt = %.6f' % dt)
        # print('%i time steps will be needed\n' % N_time)
        
        # loop over the time steps
        for n in range(0, N_time):

            # update the solution
            u[n+1,:] = method(PDE, u[n,:], t[n], dt, ( D, A_, b_, q , args))[0]

        # concatenate the boundary conditions
        u = np.concatenate((alpha*np.ones((N_time+1,1)), u, beta*np.ones((N_time+1,1))), axis = 1)

        return u, t, x









if __name__ == '__main__':

    '''
    Testing the solver

    '''

    # test the solver for the linear diffusion equation

    # define the problem
    D = 0.5
    a = 0.0
    b = 1.0
    alpha = 0.0
    beta = 0.0
    f = lambda x: np.sin((np.pi*(x-a)/b-a))
    t_final = 0.5
    N = 10

    # define the exact solution
    u_exact = lambda x, t: np.sin(np.pi*(x-a)/b-a)*np.exp(-np.pi**2*D*t/b**2)


    # solve the problem for RK4, explicit_euler, and solve_ivp
    for method in ['RK4', 'explicit_euler', 'solve_ivp', 'implicit_euler']:

        # solve the problem
        u, t, x = pde_solver(f, alpha, beta, a, b, 'DD', D, t_final, N, method = method)

        # plot the solution at 3 different times
        for n in np.linspace(0, len(t)-1, 3, dtype = int):

            plt.plot(x, u[n,:], label = '%s at t = %.2f' % (method, t[n]))

            # plot the exact solution at the same times
            plt.plot(x, u_exact(x, t[n]), '--', label = 'exact at t = %.2f' % t[n])



        plt.title('Linear diffusion equation - %s' % method)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.show()


    ### solve the dynamic Bratu problem

    # # define the problem
    # D = 1.0
    # myu = [2,4]

    # # u(0,t) = 0
    # alpha = 0.0
    # # u(1,t) = 0
    # beta = 0.0
    # a = 0.0
    # b = 1.0
    # f = lambda x: np.zeros(len(x))
    # t_final = 0.5
    # N = 10


    # # define the function q(x) = exp(myu*u)
    # def q(x,t, u, args):
    #     myu = args[0]
    #     # return np.exp(myu*u)
    #     return np.ones(len(x))
    
    # # define the exact solution
    # # u_exact = lambda x, t, myu: np.exp(-myu**2*t)*np.sin(np.pi*x)

    # # exact solution for source term q(x) = 1
    # def u_exact(x, t, myu):
    #     return (-1/(2*D))*(x - a) * (x - b) + ((beta - alpha)/(b - a))*(x- a) + alpha

    
    # # compute solution for different values of myu
    # for myu in [2]:

    #     # solve the problem
    #     u, t, x = pde_solver(f, alpha, beta, a, b, 'DD', D, t_final, N, method = 'RK4', q = q, args = [myu])

    #     # plot the solution at 3 different times
    #     for n in np.linspace(0, len(t)-1, 10, dtype = int):
                
    #         plt.plot(x, u[n,:], label = 'RK4 at t = %.2f' % t[n])

    #         # plot the exact solution at the same times
    #         plt.plot(x, u_exact(x, t[n], myu), '--', label = 'exact at t = %.2f' % t[n])



    # plt.title('Dynamic Bratu problem - RK4')
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('u(x,t)')
    # plt.show()

    ### solve the heat equation with neumann boundary condition

    # # define the problem
    # D = 1.0
    # a = 0.0
    # b = 1.0

    # # u_x(0,t) = 0
    # alpha = 0.0

    # # u_x(1,t) = 0
    # beta = 0.0

    # # initial condition u(x,0) = sin(pi*x)
    # f = lambda x: np.sin(np.pi*x)

    # # final time
    # t_final = 0.5

    # # number of time steps
    # N = 50

    # # define the exact solution
    # u_exact = lambda x, t: np.sin(np.pi*x)*np.exp(-np.pi**2*D*t)

    # # solve the problem
    # u, t, x = pde_solver(f, alpha, beta, a, b, 'NN', D, t_final, N, method = 'solve_ivp')

    # # plot the solution at 3 different times
    # for n in np.linspace(0, len(t)-1, 10, dtype = int):


    #     plt.plot(x, u[n,:], label = 'RK4 at t = %.2f' % t[n])



    # plt.title('Heat equation with Neumann boundary condition')
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('u(x,t)')
    # plt.show()

    


    



    


    
    






