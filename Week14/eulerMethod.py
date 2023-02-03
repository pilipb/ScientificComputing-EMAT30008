# Solving a first order ODE using Euler's method
# equation in the form x' = x

# imports
import numpy as np
import matplotlib.pyplot as plt

# define the function
f = lambda x: x

# define the initial conditions
x0 = 1
t0 = 0

# define euler step
def euler_step(f, x0, t0, delta_t):
    x1 = x0 + delta_t * f(x0)
    t1 = t0 + delta_t
    return x1, t1

# define the exact solution and the error function
x_exact = lambda t: np.exp(t)
def error(x, t):
    return np.abs(x - x_exact(t))

# function solve_to which solves the ODE from x1,t1 to x2,t2 in steps no larger than deltat_max
def solve_to(f, x1, t1, t2, deltat_max):
    # initialize the solution
    x = [x1]
    t = [t1]
    error_values = [error(x1, t1)]
    print( x, t, error_values)
    # loop until we reach the end point
    while t[-1] < t2:
        # take a step
        x1, t1 = euler_step(f, x[-1], t[-1], deltat_max)
        # append the new values to the solution
        x.append(x1)
        t.append(t1)
        # append errors to error list
        error_values.append(error(x1, t1))
    return x, t, error_values



# produce a plot with double logarithmic scale showing how the error depends on the size od the timestep delta t
# define the step size
deltas = np.arange(0.01, 1, 0.01)
for delta_t in deltas:
    # solve the ODE up to t = 1
    x, t, error_vals = solve_to(f, x0, t0, 1, delta_t)
    # plot the error
    print(len(x), len(t), len(error_vals))
    plt.plot(deltas, error_vals, 'o')
    plt.xlabel('delta t')
    plt.ylabel('error')
    plt.show()