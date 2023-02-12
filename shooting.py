from solve_to import *
from solvers import *
import numpy as np
import matplotlib.pyplot as plt

'''
shooting method will solve the ODE using root finding method to find the limit cycle

param: function - f - the function to be solved
param: float - y0 - the initial guess for y coordinate
param: float - t0 - the initial time

returns:
array - Y - the solution values and the
array - t  - time for these solutions

'''

def shooting(f, y0, method):
    # initialize the solution and constants
    Y = [y0]
    t = [0]

    t0 = t[0]
    t1 = 100

    guess = [1, y0]

    delta_t = 0.01
    step = 0.01

    # find method
    methods = {'Euler': euler_step, 'RK4': rk4_step, 'Lax-Wendroff': lw_step}

    # check if method is valid
    if method not in methods:
        raise ValueError('Invalid method, please enter a valid method: Euler, RK4, Lax-Wendroff or define your own')


    # solve ode at y0 guess
    Y , t = solve_to(f, guess, t0, t1, delta_t, method)
    Y = np.array(Y)

    # find dy/dt 
    dydt = lambda ode: ode[1]

    x,y = Y[:,0], Y[:,1]
    dy = dydt(f([x,y],t))

    print('Initial guess: ', y0, 'dy/dx at t = 100 (arbitrary): ', dy[-1])
    plt.plot(t, y, label='guess = %.2f' %y0)

    # the goal is to make the gradient dy/dx = 0 at t = 100
    # we can do this by shooting the solution until the gradient is zero

    while np.round(dy[-1],3) != 0:
        y0 += step * (0 - dy[-1]) 
        guess = [1, y0]
        Y, t = solve_to(ode, guess, 0, 100, 0.01, 'RK4')
        Y = np.array(Y)
        x , y = Y[:,0], Y[:,1]
        dy = dydt(f(x,y), t)
        print('New guess: ', y0, 'dy/dx at t = 100 (arbitrary): ', dy[-1])
        plt.plot(t, y, label='guess = %.2f' %(y0))

#### TEST ####

a = 1
d = 0.1
b = 0.2

def ode(Y, t, args = (a, b, d)):
    x, y = Y
    return np.array([x*(1-x) - (a*x*y)/(d+x) , b*y*(1- (y/x))])

shooting(ode, 0.1,'RK4')


