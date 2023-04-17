from solve_to import *
from solvers import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipy
import math


def shooting(f, y0, T, args = None):

    '''
    Implementing a numerical shooting method to solve an ODE to find a periodic solution
    This method will solve the ODE using root finding method to find the limit cycle, using scipy.optimize.fsolve to
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

    # unpack the initial conditions and period guess
    T_guess = T

    # define the find dx/dt function
    def dxdt(t, Y, f=f):
        return f(t, Y, args=args)[0]

    # define the function that will be solved for the initial conditions and period
    def fun(initial_vals):

        # unpack the initial conditions and period guess
        T = initial_vals[-1]
        y0 = initial_vals[:-1]

        Y , _ = solve_to(f, y0, 0, T, 0.01, 'RK4', args=args)

        num_dim = len(y0)
        row = np.zeros(num_dim)


        for i in range(num_dim):
            row[i] = Y[-1,i] - y0[i]
  
        row = np.append(row, dxdt(T, Y[-1]))

        output = np.array(row)
        return output

    # solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
    y0 = np.append(y0, T_guess)

    sol = scipy.fsolve(fun, y0)

    # return the period and initial conditions that cause the limit cycle: sol = [x0, y0, ... , T]
    return sol


#### TEST ####

''' 
example code to test the shooting method and period function
the ode is the Lotka-Volterra equation
'''

if __name__ == '__main__':
    
    # define new ode
    a = 1
    d = 0.1
    b = 0.1

    def ode(t, Y, args):

        a, b, d = args

        Y = np.array(Y)

        x, y = Y
        return np.array([x*(1-x) - (a*x*y)/(d+x) , b*y*(1- (y/x))])


    '''
    the original guess has to be close to the solution

    '''

    # initial guess
    Y0 = [10,10]
    
    # solve the ode using the shooting method
    sol = shooting(ode, Y0, 60, args=[a,b,d])

#    extract the period and initial conditions
    T = sol[-1]
    Y0 = sol[:-1]

    print('Period = %.2f' %T, '\n')
    print('Y0 = ', Y0, '\n')

    # solve for one period of the solution
    Y,t = solve_to(ode, Y0, 0, T, 0.01, 'RK4', args=[a,b,d])

    plt.plot(t, Y)
    plt.xlabel('t')
    plt.ylabel('x(t) and y(t)')
    plt.title('Period = %.2f' %T)
    plt.show()
        
