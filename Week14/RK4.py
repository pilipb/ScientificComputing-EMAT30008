# Solving a first order ODE using Runge-Kutta Fourth order method
# equation in the form x' = x

# imports
import numpy as np
import matplotlib.pyplot as plt

# define the function
f = lambda x: x

# define the initial conditions
x0 = 1
t0 = 0

# define the exact solution and the error function
x_exact = lambda t: np.exp(t)
def error(x, t):
    return np.abs(x - x_exact(t))

# define the RK4 step
def RK4_step(f, x0, t0, delta_t):
    k1 = delta_t * f(x0)
    k2 = delta_t * f(x0 + k1/2)
    k3 = delta_t * f(x0 + k2/2)
    k4 = delta_t * f(x0 + k3)
    x1 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6
    t1 = t0 + delta_t
    return x1, t1

# function solve_to which solves the ODE from x1,t1 to x2,t2 in steps no larger than deltat_max
def solve_to(f, x1, t1, t2, deltat_max):
    # initialize the solution
    x = [x1]
    t = [t1]
    error_values = [error(x1,t1)]

    # loop until we reach the end point
    while True:
        # take a step
        x1, t1 = RK4_step(f, x[-1], t[-1], deltat_max)
        # append the new values to the solution
        x.append(x1)
        t.append(t1)
        # append errors to error list
        error_values.append(error(x1, t1))
        if t[-1] > t2:
            break
    return x, t, error_values[-1]

# produce a plot with double logarithmic scale showing how the error depends on the size od the timestep delta t
# define the step size
deltas = np.arange(0.001, 1, 0.001)
error_list = []
for delta_t in deltas:
    # solve the ODE up to t = 1
    x, t, error_vals = solve_to(f, x0, t0, 1, delta_t)
    # plot the error
    error_list.append(error_vals)
    # print(len(x), len(t), error_vals)

plt.loglog(deltas, error_list, 'o')
plt.title('Error at x(1) with variation in Delta t (Double Log Scale)')
plt.xlabel('Delta t')
plt.ylabel('Error')
plt.show()