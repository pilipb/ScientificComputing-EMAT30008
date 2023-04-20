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

# for autonomous systems, G(u0) = [u0 - F(T, u0), phi(0)] = 0 where phi(0) is the phase condition at time 0

# the phase condition is the x derivative of the solution at time t

# phi(t) = u1' = u2

# solve by passing G(u0) to fsolve

# IMPLEMENTATION:

# define the function to be integrated
def f(t, Y, *args):
    # unpack the arguments
    # print(args)
    m, c, k, gamma, omega = args[0]

    # unpack the variables
    u1, u2 = Y

    # define the partial derivatives
    u1p = u2
    u2p = 1/m*(-c*u2 - k*u1 + gamma*np.sin(omega*t))

    # return the derivatives
    return np.array([u1p, u2p])
 
# define the function to be solved for the initial conditions and period
def G(u0, *args):
    # args in the form of (m, c, k, gamma, omega), (A, B)
    # unpack the arguments
    coeff = args[0][0]

    bounds = args[0][1]
    A, B = bounds

    # unpack the initial conditions
    u10, u20, T = u0

    # define the initial conditions
    u0 = np.array([u10, u20])

    # solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
    Y, t = solve_to(f, u0, 0, T, 0.01, 'RK4', args=coeff)

    # phase condition is first row of f(T, Y[-1], *args)
    phi = f(T, Y[-1], coeff)[0]
    
    # define the boundary conditions
    u1A = Y[0,1]
    u1B = Y[-1,1]

    # define the function to be solved
    output = np.array([u1A - A, u1B - B, phi])

    return output

# solve the function using fsolve
def solve_G(u0, *args):

    # solve the function
    u = scipy.fsolve(G, u0, args=args)

    return u

# define the parameters
m = 1
c = 1
k = 1
gamma = 1
omega = 1

# define the boundary conditions
A = 0
B = 0

# define the initial conditions
u10 = 0
u20 = 0
T = 2*np.pi/omega

# define the initial conditions
u0 = np.array([u10, u20, T])

# define the arguments
args = [m, c, k, gamma, omega], [A, B]

# solve the function
u = solve_G(u0, args)

# unpack the solution
u10, u20, T = u

# solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
Y, t = solve_to(f, np.array([u10, u20]), 0, T, 0.01, 'RK4', args=args[0])

# plot the solution
plt.plot(t, Y[:,0])
plt.plot(t, Y[:,1])
plt.title('T = ' + str(T))
plt.show()

# However it is difficult to find correct initial conditions for the shooting method
# so continuation methods can be used

