import numpy as np

'''
The steps will make an integration step of size delta_t 
using the method specified by the function step.

Parameters:
f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
y0 - array: the initial value of the solution
t0 - float: the initial value of time
delta_t - float: the step size

Returns:
y1 - array: the solution at the next step time step
t1 - float: the next time step

'''

# Euler step - generalised to any number of dimensions
def euler_step(f, y0, t0, delta_t):
    y1 = y0 + delta_t * f(y0, t0)
    t1 = t0 + delta_t
    return y1, t1

# RK4 step - generalised to any number of dimensions
def rk4_step(f, y0, t0, delta_t):
    k1 = delta_t * f(y0, t0)
    k2 = delta_t * f(y0 + delta_t/2 * k1, t0 + delta_t/2)
    k3 = delta_t * f(y0 + delta_t/2 * k2, t0 + delta_t/2)
    k4 = delta_t * f(y0 + delta_t * k3, t0 + delta_t)
    y1 = y0 + (k1)/6 + 2*(k2)/6 + 2*(k3)/6 + (k4)/6
    t1 = t0 + delta_t
    return y1, t1


# Heuns method - generalised to any number of dimensions
def heun_step(f, y0, t0, delta_t):
    k1 = f(y0, t0)
    k2 = f(y0 + delta_t * k1, t0 + delta_t)
    y1 = y0 + delta_t/2 * (k1 + k2)
    t1 = t0 + delta_t
    return y1, t1
    