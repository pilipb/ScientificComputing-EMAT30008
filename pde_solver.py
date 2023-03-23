'''
implement a PDE solver using the explicit Euler method to solve the following PDE:

linear diffusion equation:

u_t = D*u_xx

u(a,t) = 0
u(b,t) = 0

u(x,0) = sin((pi*(x-a)/b-a))

the exact solution is:

u(x,t) = sin((pi*(x-a)/b-a)) * exp(-pi**2*D*t/(b-a)**2)

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from numpy.linalg import solve

from solve_to import *
from solvers import *


### define the problem

# diffusion coefficient
D = 0.5

# domain
a = 0.0
b = 1.0

# boundary conditions
alpha = 0.0
beta = 0.0

# grid for initial condition
f = lambda x: np.sin((np.pi*(x-a)/b-a))

# creating grid
N = 20
x = np.linspace(a, b, N+1)
x_int = x[1:-1]
dx = (b-a)/N

# CFL number
C = 0.49

# time discretization
dt = C*dx**2/D
t_final = 1
N_time = int(t_final/dt)
t = dt * np.arange(N_time)

# print some info about time step
print('dt = %.3f' % dt)
print('%i time steps will be needed' % N_time)

### start time stepping

# preallocate solution
u = np.zeros((N_time+1, N-1))
u[0,:] = f(x_int)

# re make the x grid 
x = np.linspace(a, b, N+1)

# loop over the steps
for n in range(0,N_time):

    # loop over the grid
    for i in range(0, N-2):
        if i==0:
            u[n+1,i] = u[n,i] + C*(u[n,i+1]-2*u[n,i]+alpha)
        elif i < 0 and i < N-2:
            u[n+1,i] = u[n,i] + C*(u[n,i+1]-2*u[n,i]+u[n,i-1])
        else:
            u[n+1,N-2] = u[n,N-2] + C*(beta - 2*u[n,N-2]+u[n,N-3])


### calculate the exact solution
def exact_solution(x, t):
    return np.sin(np.pi*(x-a)/(b-a)) * np.exp(-np.pi**2*D*t/(b-a)**2)

u_exact = np.zeros((N_time+1, N-1))
for n in range(0,N_time):
    u_exact[n,:] = exact_solution(x_int, t[n])


### plot the solution

# display the solution
plt.figure()
plt.title('Solution of the linear diffusion equation')

# for n in range(0,N_time):
plt.plot(x_int, u[0,:], label='numerical solution')

# # plot the exact solution
# for n in range(0,N_time):
#     plt.plot(x_int, u_exact[n,:], label='exact solution')

plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()