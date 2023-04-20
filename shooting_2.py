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
from solve_to import solve_to

# consider the ode:
# mx¨ + cx˙ + kx = gamma sin(ωt)

# T = 2π/omega

# in first order form:
# u1' = u2
# u2' = 1/m (-c u2 - k u1 + gamma sin(ωt))

# the time solution is:

# u(t) = [u1(t), u2(t)] = F(t, u0) (where F is the integration of u with the initial conditions u0, up to time t)

# the limit cycle is when u0 - u(T) = 0 for some u0

# u0 - F(T, u0) = 0

# ie: solving the root for G(u0) = 0, where G(u0) = u0 - F(T, u0)

# solve by passing G(u0) to fsolve

# IMPLEMENTATION:

# define the function to be integrated
def f(t, Y, *args):
    # unpack the arguments
    # print(args)
    m, c, k, gamma, omega = args[0]

    # unpack the variables
    u1, u2 = Y

    # define the derivatives
    u1p = u2
    u2p = 1/m*(-c*u2 - k*u1 + gamma*np.sin(omega*t))

    # return the derivatives
    return np.array([u1p, u2p])
 

# define the function to be solved for the initial conditions and period that satisfy the boundary conditions
def G(u0, *args):
    # unpack the arguments
    m, c, k, gamma, omega = args

    # unpack the initial conditions
    u10, u20 = u0

    # define the initial conditions
    u0 = np.array([u10, u20])

    # define the time interval
    t0 = 0
    T = 2*np.pi/omega
    t = np.linspace(t0, T, 1000)

    # solve the ode
    Y, _ = solve_to(f, u0, t0, T, 0.01, 'RK4', args=args)

    # define the boundary conditions
    u1T = Y[-1,0]
    u2T = Y[-1,1]

    # # define the phase condition
    # u1p = f(t, Y, args)[:,0]
    # phase = u1p[-1]

    # return the boundary conditions and phase condition
    return np.array([u1T - u10, u2T - u20])#, phase

# run the code for an example

# define the parameters
m = 1
c = 1
k = 1
gamma = 1
omega = 1

# define the arguments
args = (m, c, k, gamma, omega)

# define the initial conditions
u10 = 0
u20 = 0
u0 = np.array([u10, u20])

# solve for the initial conditions and period that satisfy the boundary conditions
u0 = scipy.fsolve(G, u0, args=args, full_output=True)

print(u0)
# unpack the initial conditions
u10, u20 = u0[0]

# define the initial conditions
u0 = np.array([u10, u20])

# define the time interval
t0 = 0
T = 2*np.pi/omega
# t = np.linspace(t0, T, 1000)

# solve the ode
Y, t = solve_to(f, u0, t0, T, 0.01, 'RK4', args=args)

# unpack the solution
u1 = Y[:,0]
u2 = Y[:,1]

# plot the solution
plt.plot(t, u1, label='u1')
plt.plot(t, u2, label='u2')
plt.legend()
plt.show()

# for autonomous systems, the phase condition is dx/dt(0) = 0










