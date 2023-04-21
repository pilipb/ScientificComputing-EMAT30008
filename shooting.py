from solve_to import *
from solvers import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipy
import math


def shooting_setup(f, y0, T= 0, args = None):

    '''
    Implementing a numerical shooting method to solve an ODE to find a periodic solution
    This method will solve the BVP ODE using root finding method to find the limit cycle, using scipy.optimize.fsolve to
    find the initial conditions and period that satisfy the boundary conditions.

    parameters:
    ----------------------------
    f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
    y0 - array: the initial value of the solution
    T - float: an initial guess for the period of the solution
    args - array: the arguments for the function f

    returns:
    ----------------------------
    sol - array: the initial conditions that cause the solution to be periodic: sol = [x0, y0, ... , T]

    '''

    # define the function that will be solved for the initial conditions and period
    def fun(initial_vals,p):
        '''
        Function F(u) = 0 that will be solved for the initial conditions and period
        Parameters:
        ----------------------------
        initial_vals - array: the initial conditions and period guess: initial_vals = [x0, y0, ... , T]

        Returns:
        ----------------------------
        row - array: the boundary conditions that must be satisfied for the solution to be periodic: 
                    row = [x(T) - x0, y(T) - y0, ... , dx/dt(0) = 0]

        '''

        # unpack the initial conditions and period guess
        T = initial_vals[-1]
        y0 = initial_vals[:-1] 

        Y , _ = solve_to(f, y0, 0, T, 0.01, 'RK4', args=args)

        # make empty array to store the boundary conditions
        num_dim = len(y0)
        row = np.zeros(num_dim)

        # limit cycle condition
        row = Y[-1,:] - y0[:]
  
        # phase condition
        row = np.append(row, f(0, Y[0,:], args)[0]) # dx/dt(0) = 0 

        return row

    # solve the system of equations for the initial conditions [x0, y0, ... ,T] that satisfy the boundary conditions
    u0 = np.append(y0, T)

    return fun, u0


def shooting_solve(fun, u0):
    '''
    Solve the system of equations made in the setup function

    Parameters:
    ----------------------------
    fun - function: such that fun(u) = 0
    u0 - array: the initial guess for the solution

    Returns:
    ----------------------------
    u0 - array: the initial conditions that cause the solution to be periodic: u0 = [x0, y0, ... ]
    T - float: the period of the solution
    
    '''

    sol  = scipy.fsolve(fun, u0, args = (0,))
    
    # return the period and initial conditions that cause the limit cycle: sol = [x0, y0, ... , T]
    u0 = sol[:-1]
    T = sol[-1]

    return u0, T




#### TEST ####

''' 
example code to test the shooting method and period function
the ode is the Lotka-Volterra equation
'''

if __name__ == '__main__':
    
    # define new ode
    a = 1
    d = 0.1
    b = 0.2

    def ode(t, Y, args):

        a, b, d = args
        x,y = Y
        dxdt = x*(1-x) - (a*x*y)/(d+x)
        dydt = b*y*(1- (y/x))

        return np.array([dxdt, dydt])


    '''
    the original guess has to be close to the solution

    '''

    # initial guess
    Y0 = [0.1,0.1]
    
    # solve the ode using the shooting method
    fun, u0 = shooting_setup(ode, Y0, T=20, args=[a,b,d])
    u0, T0 = shooting_solve(fun, u0)

    # solve for one period of the solution
    Y,t = solve_to(ode, u0, 0, T0, 0.01, 'RK4', args=[a,b,d])

    plt.plot(t, Y)
    plt.xlabel('t')
    plt.ylabel('x(t) and y(t)')
    plt.title('Period = %.2f' %T0)
    plt.show()
        
