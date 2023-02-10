import numpy as np

'''
The steps will make a step of size delta_t using the method specified by the function step.

param: f - the function to be integrated (with inputs (Y,t))
param: y0 - the initial value of the solution
param: t0 - the initial value of time
param: delta_t - the step size

returns: Y - the solution at the next step and the time t at the next step
'''

# Euler step
def euler_step(f, Y, t0, delta_t):
    x0, y0 = Y
    x1, y1 = [x0, y0] + delta_t * f([x0, y0], t0)
    t1 = t0 + delta_t
    Y1 = [x1, y1]
    return Y1, t1

# RK4 step
def rk4_step(f, Y, t0, delta_t):
    x0, y0 = Y
    k1 = delta_t * f([x0, y0], t0)
    k2 = delta_t * f([x0 + delta_t/2, y0 + delta_t/2], t0 + delta_t/2)
    k3 = delta_t * f([x0 + delta_t/2, y0 + delta_t/2] , t0 + delta_t/2)
    k4 = delta_t * f([x0 + delta_t, y0 + delta_t] , t0 + delta_t)
    x1, y1 = [x0, y0] + (k1)/6 + 2*(k2)/6 + 2*(k3)/6 + (k4)/6
    t1 = t0 + delta_t
    Y1 = [x1, y1]
    return Y1, t1

# Lax-Wendroff step
def lw_step(f, Y, t0, delta_t):
    x0, y0 = Y
    x1 = x0 + delta_t * y0 + (delta_t**2/2) * (-x0)
    y1 = y0 + delta_t * (-x0) + (delta_t**2/2) * (-y0)
    t1 = t0 + delta_t
    Y1 = [x1, y1]
    return Y1, t1

