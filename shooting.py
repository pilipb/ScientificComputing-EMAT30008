from solve_to import *
from solvers import *

'''
shooting method will solve the ODE using root finding method to find the limit cycle

param: function - f - the function to be solved
param: float - y0 - the initial guess
param: float - t0 - the initial time

returns:
array - Y - the solution values and the
array - t  - time for these solutions

'''

def shooting(f, y0, t0,tol=1e-3):
    # initialize the solution
    t = [t0]

    # set the initial iteration
    iteration = 0

    # set the initial error
    error = 1

    # loop until we reach the tolerance
    while error > tol:
        # solve the ODE with the guess
        Y, t = solve_to(f, [1,y0], t0, 100, tol,  'RK4' )

        # calculate the error
        error = abs(Y[-1][0] - Y[0][0])

        # update the guess
        y0 = [1, y0 - error]

        # update the iteration
        iteration += 1

    return Y, t

