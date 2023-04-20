'''
we have a pde of form:

u_t = D u_xx; with boundary conditions u(0,t) = A and u(1,t) = B

for the long term solution:

0 = D u_xx ; with boundary conditions u(0) = A and u(1) = B

as a first order system:

u1_t = v
u2_t = 0

with boundary conditions u1(0) = A and u1(1) = B and u2(0) = alpha (unknown)

we can solve this system using the shooting method (and for the limit cycle):

f(u) = [u1(0) - A, u1(1) - B, phase(T) - phase(0)] = 0

where phase(T) is the phase condition of the solution at time T and phase(0) is the phase condition of the solution at time 0
The phase condition can be u_x 

u1(1) is found by numerical integration

start with a guess u(0) = [A, alpha=0]

use newton's method to iterate to a solution:

integrate the system from 0 to 1 with initial conditions u(0) = [A, alpha]

update alpha = u1(1) - B

repeat until f(u) = 0

'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipy
import math

def ode(t, Y, args):
    u1, u2 = Y
    D = args[0]
    return np.array([u2, 0])

# [u1(0) - A, u1(1) - B, phase(T) - phase(0)] = 0

# u(0) = [A, alpha = 0] = [u1(0), u2(0)]
# u(1) = u1(1), u2(1) = [B, _ ]


def shooting(f, y0, T, args = None):






