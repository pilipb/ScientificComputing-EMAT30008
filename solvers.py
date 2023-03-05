import numpy as np

'''
The steps will make an integration step of size delta_t 
using the method specified by the function step.

parameters:
---------------------------
f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
y0 - array: the initial value of the solution
t0 - float: the initial value of time
delta_t - float: the step size

returns:
---------------------------
y1 - array: the solution at the next step time step
t1 - float: the next time step

'''

# Euler step - generalised to any number of dimensions
# with added error catching
def euler_step(f, y0, t0, delta_t):
    error_check(f, y0, t0, delta_t)
    y1 = y0 + delta_t * f(y0, t0)
    t1 = t0 + delta_t
    return y1, t1

# RK4 step - generalised to any number of dimensions
def rk4_step(f, y0, t0, delta_t):
    error_check(f, y0, t0, delta_t)
    k1 = delta_t * f(y0, t0)
    k2 = delta_t * f(y0 + delta_t/2 * k1, t0 + delta_t/2)
    k3 = delta_t * f(y0 + delta_t/2 * k2, t0 + delta_t/2)
    k4 = delta_t * f(y0 + delta_t * k3, t0 + delta_t)
    y1 = y0 + (k1)/6 + 2*(k2)/6 + 2*(k3)/6 + (k4)/6
    t1 = t0 + delta_t
    return y1, t1


# Heuns method - generalised to any number of dimensions
def heun_step(f, y0, t0, delta_t):
    error_check(f, y0, t0, delta_t)
    k1 = f(y0, t0)
    k2 = f(y0 + delta_t * k1, t0 + delta_t)
    y1 = y0 + delta_t/2 * (k1 + k2)
    t1 = t0 + delta_t
    return y1, t1

# error checking function for the steps
def error_check(f, y0, t0, delta_t):
    if not callable(f):
        raise ValueError('f must be a function')
    if not isinstance(y0, np.ndarray):
        raise ValueError('y0 must be a numpy array')
    if not isinstance(t0, (int, float)):
        raise ValueError('t0 must be a number')
    if not isinstance(delta_t, (int, float)):
        raise ValueError('delta_t must be a number')
