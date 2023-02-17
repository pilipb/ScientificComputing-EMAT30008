from solve_to import *
from solvers import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipy
import math


'''
Implementing a numerical shooting method to solve an ODE to find a periodic solution
This method will solve the ODE using root finding method to find the limit cycle, using scipy.optimize.fsolve to
find the initial conditions and period that satisfy the boundary conditions.

Parameters:
f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
y0 - array: the initial value of the solution

Returns:
Y - array: the initial conditions that cause the solution to be periodic
T - float: the period of the solution

'''


def shooting(f, Y0, T):

    # unpack the initial conditions and period guess
    x0,y0  = Y0
    T_guess = T

    # test the initial conditions guess
    Y , _ = solve_to(f, [x0, y0], 0, 300, 0.01, 'RK4')

    # derive better starting guess from the solution
    [x0,y0] = [np.median(Y[:,0]), np.median(Y[:,1])]
    '''
    The initial conditions are not always in the correct range. To fix this
    I have found the phase space trajectory for random guess and given that it is
    going to end up in the periodic solution, I have found the median of the solution
    and used that as the starting guess for the root finding method as this will be
    on the periodic phase space trajectory.
    
    '''

    # define the find dx/dt function
    def dxdt( Y, t, f=f):
        return f(Y, t)[0]


    # define the function that will be solved for the initial conditions and period
    def fun(X):

        x0, y0, T = X

        Y , t = solve_to(f, [x0, y0], 0, T, 0.01, 'RK4')

        row1 = Y[0,0] - Y[-1,0] # x(0) - x(T) = 0
        row2 = Y[0,1] - Y[-1,1] # y(0) - y(T) = 0
        row3 = dxdt([Y[-1,0], Y[-1,1]], T) # dx/dt(T) = 0
        
        return np.array([row1, row2, row3])

    # solve the system of equations for the initial conditions [x0, y0] and period T that satisfy the boundary conditions
    sol = scipy.fsolve(fun, [x0, y0, T_guess])

    '''
    Currently, the initial conditions do not always allow fsolve to find the correct solution. This is because the initial conditions
    are not always in the correct range. To fix this, we can use a root finding method to find the initial conditions that satisfy
    the boundary conditions. 

    The starting guess must fall on the periodic phase space trajectory. 
    
    '''

    x0, y0, period = sol

    # return the period and initial conditions that cause the limit cycle
    return [x0, y0], period





'''
The shooting_dev method will solve the ODE using root finding method to find the limit cycle
of the ODE. The method calls the solve_to method to solve the ODE at each guess.

The function also plots the solution at each guess.

Parameters:
f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
y0 - array: the initial value of the solution 
method - string: the method to be used to solve the ODE (Euler, RK4, Heun)

Returns:
Y - array: the solution at the final guess (limit cycle)
t - array: the time solution at the final guess (limit cycle)
guess - float: the starting condition that gives the limit cycle

'''




def shooting_dev(f, y0, method):

    # initialize the solution and constants
    Y = [y0]  
    t = [0]

    t0 = t[0]
    t1 = 100 # arbitrary end time for search space

    delta_t = 0.01
    step = 0.01

    # solve ode at y0 guess
    Y , t = solve_to(f, y0, t0, t1, delta_t, method)

    # find dx/dt 
    def dxdt(f, Y, t):
        return f(Y, t)[0]

    dx = dxdt(f, Y[-1], t1)
    
    plt.plot(t, Y, label='guess = %.2f, dx/dt at t = %.2f: %.2f' %(y0[0],t1, dx))

    # the goal is to make the gradient dx/dt = 0 at t = 100
    # we can do this by shooting the solution until the gradient is zero
    print('Calculating limit cycle...')
    print('\n')
    while np.round(dx,2) != 0:
        print('-', end='')
        # if the gradient is negative, increase the guess, if positive, decrease the guess
        guess = y0[0]
        guess += step * (0 - dx) 
        y0[0] = guess
        # solve the ODE at the new guess
        Y, t = solve_to(f, y0, 0, t1, delta_t, 'RK4')
        dx = dxdt(f, Y[-1], t1)

        plt.plot(t, Y, label='guess = %.2f, dx/dt at t = %.2f: %.2f' %(t1, y0[0], dx))
        
    print('Limit cycle found at guess = %.2f' %guess)
    print('dx/dt at t = %.2f: %.2f' %(t1, dx))
    return Y, t, guess

'''
The period function will find the period of the limit cycle and plot the solution

Parameters:
Y - array: the solution at the final guess (limit cycle) from the shooting method
t - array: the time solution at the final guess (limit cycle) from the shooting method

Returns:
T - float: the period of the limit cycle
Y - array: the solution at the final guess (limit cycle) from the shooting method for one period oscillation
t - array: the time solution at the final guess (limit cycle) from the shooting method for one period oscillation

'''

def period(Y, t):
    # shorten the search space
    Y, t = Y[1000:], t[1000:]
    # find the max value of the solution
    max_value = np.max(Y[:,0])
    # find the min value of the solution
    min_value = np.min(Y[:,0])
    # find the average value of the solution
    avg_value = (max_value + min_value)/2
    # find the last index where the solution is greater than the average value
    max_index = np.argmax(Y[:,0] - avg_value)
    # reduce the Y and t arrays to only the values before the max value
    Ys = Y[:max_index-50] 
    ts = t[:max_index-50]
    # find the last index where the solution is less than the average value
    min_index = np.argmax(avg_value - Ys[:,0])
    # reduce the Y and t arrays to only the values before the min value
    Ys = Y[:min_index-50]
    ts = t[:min_index-50]
    # find the last index where the solution is greater than the average value
    max2_index = np.argmax(Ys[:,0] - avg_value)
    # find the period
    period_idx = max_index - max2_index
    T = (t[period_idx] - t[0])
    # return one period of the solutions
    Y = Y[max2_index:max_index]
    t = t[max2_index:max_index]

    return T, Y, t


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

    def ode(Y, t, args = (a, b, d)):
        a, b, d = args
        x, y = Y
        return np.array([x*(1-x) - (a*x*y)/(d+x) , b*y*(1- (y/x))])
    
    '''
    the original guess has to be close to the solution

    '''

    # initial guess
    Y0 = [2,3]
    
    # solve the ode using the shooting method
    Y0, T = shooting(ode, Y0,20)

    print('Period = %.2f' %T, '\n')
    print('Y0 = ', Y0, '\n')

    # solve for one period of the solution
    Y,t = solve_to(ode, Y0, 0, T, 0.01, 'RK4')

    # # solve the ode using the shooting method
    # Y,t,guess = shooting_dev(ode3, [0.16,0.1,0.1],'RK4')
    # plt.show()

    # # plot the period
    # T, Y, t = period(Y,t)



    plt.plot(t, Y)
    plt.xlabel('t')
    plt.ylabel('x(t) and y(t)')
    plt.title('Period = %.2f' %T)
    plt.show()
        


