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

    guess = 0.01

    delta_t = 0.01
    step = 0.01

    # find method
    methods = {'Euler': euler_step, 'RK4': rk4_step, 'Lax-Wendroff': lw_step}

    # check if method is valid
    if method not in methods:
        raise ValueError('Invalid method, please enter a valid method: Euler, RK4, Lax-Wendroff or define your own')


    # solve ode at y0 guess
    Y , t = solve_to(f, [1,guess], t0, t1, delta_t, method)
    Y = np.array(Y)

    # find dy/dt 
    def dydt(x,y,b):
        return b*y*(1- (y/x))

    x,y = Y[:,0], Y[:,1]
    dy = dydt(x,y,b=0.1)
    # print('Initial guess: ', guess, 'dy/dx at t = 100 (arbitrary): ', dy[-1])
    plt.plot(t, y, label='guess = %.2f' %guess)

    # the goal is to make the gradient dy/dx = 0 at t = 100
    # we can do this by shooting the solution until the gradient is zero

    while np.round(dy[-1],2) != 0:
        guess += step * (0 - dy[-1]) 
        Y, t = solve_to(ode, [1,guess], 0, 100, 0.01, 'RK4')
        Y = np.array(Y)
        x , y = Y[:,0], Y[:,1]
        dy = dydt(x,y,b)
        # print('New guess: ', guess, 'dy/dt at t = 100: ', dy[-1])
        plt.plot(t, y, label='guess = %.2f' %(guess))
        
    return Y, t, guess

'''
the period function will find the period of the limit cycle
param: array - Y - the solution values from the shooting method
param: array - t - the time values from the shooting method

returns:
float - T - the period of the limit cycle
array - Y - the solution values for a single limit cycle
array - t - the time values for a single limit cycle

'''

def period(Y, t):
    # find the index of the max value
    max_index = np.argmax(Y[:,1])
    # reduce the Y and t arrays to only the values after the max value
    Y = Y[max_index:]
    t = t[max_index:]
    # find the index of the min value
    min_index = np.argmin(Y[:,1])
    # reduce the Y and t arrays to only the values after the min value
    Ys = Y[min_index:]
    # find the index of the next max value
    max2_index = np.argmax(Ys[:,1]) + min_index 
    # reduce the Y and t arrays to only the values before the second max value
    Y = Y[:max2_index]
    t = t[:max2_index]
    # find the period
    T = (t[-1] - t[0]) 
    return T, Y, t




#### TEST ####
''' 
example code to test the shooting method and period function
the ode is the Lotka-Volterra equation
'''

if __name__ == '__main__':
    # define the ode
    a = 1
    d = 0.1
    b = 0.1

    def ode(Y, t, args = (a, b, d)):
        x, y = Y
        return np.array([x*(1-x) - (a*x*y)/(d+x) , b*y*(1- (y/x))])

    # solve the ode using the shooting method
    Y,t,guess = shooting(ode, 0.1,'RK4', args = (a, b, d))
    plt.show()

    # plot the period
    T, Y, t = period(Y,t)
    plt.plot(t, Y)
    plt.xlabel('t')
    plt.ylabel('x(t) and y(t)')
    plt.legend('x(t)', 'y(t)')
    plt.title('Period = %.2f, starting condition [1, %f]' %( T, guess))
    plt.show()
        


