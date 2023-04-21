from solve_to import *
from solvers import *
from shooting_2 import shooting
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipy
import math


# define a natural parameter continuation function
def nat_continuation(f, u0, plim, T, *args, plot = True):
    '''
    Function will implement a natural parameter continuation method to find the solution to the ODE f
    and the parameter value p that defines a limit cycle.
    
    Parameters
    ----------------------------
    f : function
            the function to be integrated (with inputs (t, Y, *args)) in first order form of n dimensions
    u0 : array
            the initial conditions guess for the integration
    plim : array [p0, p1]
            the limit bounds for the varying parameter value (1D)
    T : float
            the initial guess for the period of the solution
    args : tuple
            the constant arguments for the function f


    Returns
    ----------------------------
    T_list : list
            a list of the periods for the solutions found for each parameter value
    p_list : list
            a list of the parameter values used to find the solutions
    
    '''

    # store the solutions
    T_list = []

    if plot == True:
        # plot the solutions on two plots
        fig, ax = plt.subplots(1, 2)

    # plist
    p_list = np.arange(plim[0], plim[1], 0.1)


    for p0 in p_list:

        # find the shooting solution for the initial parameter value
        T0 = T

        Y1, T1 = shooting(f, u0, T0, p0)

        # plot the solution
        y,t = solve_to(hopf_bifurcation, u0, 0, T1, 0.01, 'RK4', (p0,))

        if plot == True:
            # on one plot, plot the y1 vs y2 phase plot
            ax[0].plot(y[:,0], y[:,1])
        
        # store the solution
        T_list.append(T1)

    if plot == True:
        # plot the period vs the parameter value
        ax[1].plot(np.arange(plim[0], plim[1], 0.1), T_list)

        # show and label the plots
        ax[0].set_xlabel('y1')
        ax[0].set_ylabel('y2')
        ax[0].set_title('Phase Plot')
        ax[1].set_xlabel('p')
        ax[1].set_ylabel('T')
        ax[1].set_title('Period vs Parameter')

        plt.show()

    return T_list, p_list


# # define a pseudo arc length continuation function
# def pal_continuation(f, u0, plim, T, *args, plot = True):

#     pass









### TEST

# define the function to be integrated
def hopf_bifurcation(t, Y, *args):
    b = args[0][0]
    x, y = Y
    dxdt = b*x - y - x*(x**2 + y**2)
    dydt = x + b*y - y*(x**2 + y**2)
    return np.array([dxdt, dydt])

# define the initial conditions
u0 = np.array([0.1, 0.1])

# define the parameter limits
plim = [0, 2]

# define the initial guess for the period
T = 2*np.pi

# use the natural parameter continuation method to find the solution
T, p = nat_continuation(hopf_bifurcation, u0, plim, T, plot=True)

