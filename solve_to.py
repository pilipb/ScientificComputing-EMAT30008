'''
solve_to method will solve ODE from t0 to t1 with a given step size

param: function - f - the function to be solved
param: float - y0 - the initial condition
param: float - t0 - the initial time
param: float - t1 - the final time
param: float - delta_t - the step size
param: str - method - the method to be used

returns: 
array - Y - the solution values and the 
array - t  - time for these solutions

'''

from solvers import *

# solve_to method
def solve_to(f, y0, t0, t1, delta_t, method):
    # initialize the solution
    Y = [y0]
    t = [t0]

    # find method
    methods = {'Euler': euler_step, 'RK4': rk4_step, 'Lax-Wendroff': lw_step}

    # check if method is valid
    if method not in methods:
        raise ValueError('Invalid method, please enter a valid method: Euler, RK4, Lax-Wendroff or define your own')

    # set method
    method = methods[method]

    # loop until we reach the end time
    while t[-1] < t1:
        # take a step
        y1, t0 = method(f, Y[-1], t[-1], delta_t)

        # append the solution
        Y.append(y1)
        t.append(t0)
        # print(t)


    return Y, t

