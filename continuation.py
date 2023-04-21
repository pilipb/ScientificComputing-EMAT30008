from solve_to import *
from solvers import *
from shooting_2 import shooting
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipy
import math



# natural parameter continuation

'''
increment the parameter by a set amount and attempt to find the solution for
the new parameter value using the previous solution as the initial conditions

'''




# # define a natural parameter continuation function
# def nat_continuation(f, u0, p0, T, *args):

'''
    Function will implement a natural parameter continuation method to find the solution to the ODE f
    and the parameter value p that defines a limit cycle.
    
    Parameters
    ----------------------------
    f : function
            the function to be integrated (with inputs (t, Y, *args)) in first order form of n dimensions
    u0 : array
            the initial conditions guess for the integration
    p0 : float
            the initial guess for the varying parameter value
    T : float
            the initial guess for the period of the solution
    args : tuple
            the constant arguments for the function f


    Returns
    ----------------------------
    
    
    '''

    # # start with f(u0, p0) 
    # u, T = shooting(f, u0, T, p0)

    # dp = 0.1

    # p = p0 + dp

    # while 









### TEST

# define the starting parameter value
b0 = 0
args = b0

# define the function to be integrated
def hopf_bifurcation(t, Y, *args):
    b = args[0][0]
    x, y = Y
    dxdt = b*x - y - x*(x**2 + y**2)
    dydt = x + b*y - y*(x**2 + y**2)
    return np.array([dxdt, dydt])

# store the solutions
T_list = []

# plot the solutions
plt.figure()
while b0 < 2:

    # find the shooting solution for the initial parameter value
    u0 = np.array([4, 1])
    T0 = 1

    Y1, T1 = shooting(hopf_bifurcation, u0, T0, b0)

    # plot the solution
    y,t = solve_to(hopf_bifurcation, u0, 0, T1, 0.01, 'RK4', (b0,))
    plt.plot(t, y[:,1])

    # store the solution
    T_list.append(T1)

    # increment the parameter value
    b0 += 0.1


plt.show()

# plot the period vs the parameter value
plt.figure()
plt.plot(np.arange(0, 2, 0.1), T_list)
plt.show()
