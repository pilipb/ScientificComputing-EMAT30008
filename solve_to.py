'''
solve_to method will solve ODE from time t0 to t1 with a given step size
using a given method from the solvers.py file

parameters:
----------------------------
f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
y0 - array: the initial value of the solution
t0 - float: the initial value of time
t1 - float: the end time
delta_t - float: the step size
method - string: the method to be used to solve the ODE (Euler, RK4, Heun)

returns: 
----------------------------
Y - array: the solution at the next step time step
t - float: the next time step

'''

from solvers import *

# solve_to method
def solve_to(f, y0, t0, t1, delta_t, method):
    # initialize the solution
    Y = [y0]
    t = [t0]

    # find method
    methods = {'Euler': euler_step, 'RK4': rk4_step, 'Heun': heun_step}

    # check if method is valid
    if method not in methods:
        raise ValueError('Invalid method, please enter a valid method: Euler, RK4, Heun or define your own')

    # set method
    method = methods[method]

    # loop until we reach the end time
    while t[-1] <= t1-delta_t:
        # take a step
        y1, t0 = method(f, Y[-1], t[-1], delta_t)

        # append the solution
        Y.append(y1)
        t.append(t0)
        # print(t)
    
    # for the last step, we need to take a step of t1 - t[-1]
    last_step = t1 - t[-1]
    y1, t0 = method(f, Y[-1], t[-1], last_step)
    Y.append(y1)
    t.append(t0)



    return np.array(Y), np.array(t)

