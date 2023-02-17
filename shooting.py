from solve_to import *
from solvers import *
import numpy as np
import matplotlib.pyplot as plt

'''
The shooting method will solve the ODE using root finding method to find the limit cycle
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




def shooting(f, y0, method):

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
    
    # define a simple 3rd order ode
    a = -1
    b = 1

    def ode3(Y, t, args = (a , b)):
        a, b = args
        x, y, z = Y
        return np.array([b*x - y + a*x*(x**2 + y**2), x + b*y + a*y*(x**2 + y**2) , -z]) 

    # solve the ode using the shooting method
    Y,t,guess = shooting(ode3, [0.16,0.1,0.1],'RK4')
    plt.show()

    # plot the period
    T, Y, t = period(Y,t)
    plt.plot(t, Y)
    plt.xlabel('t')
    plt.ylabel('x(t) and y(t)')
    plt.legend('x(t)', 'y(t)')
    plt.title('Period = %.2f, starting condition = [%.4f, 0.1, 0.1] '%( T, guess ))
    plt.show()
        


