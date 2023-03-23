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
N = 50
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


# re attach the boundary conditions and make arrays of alpha and beta for all time steps
u = np.hstack((alpha*np.ones((N_time+1,1)), u, beta*np.ones((N_time+1,1))))
u_exact = np.hstack((alpha*np.ones((N_time+1,1)), u_exact, beta*np.ones((N_time+1,1))))

### plot the solution

# display the solution
plt.figure()
plt.title('Linear diffusion equation: Explicit Euler method')

# plot 3 time steps
for n in range(0,N_time,N_time//3):
    plt.plot(x, u[n,:], label='numerical solution, t=%.2f' % t[n])
    plt.plot(x, u_exact[n,:], label='exact solution, t=%.2f' % t[n], linestyle='--')

plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()


### now solve using solve_ivp

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# define the problem
a = 0.0
b = 1.0
alpha = 0.0
beta = 0.0
D = 0.5
t_final = 1

# grid for initial condition
f = lambda x: np.sin((np.pi*(x-a)/b-a))

# creating grid
N = 40
x = np.linspace(a, b, N+1)
x_int = x[1:-1]
dx = (b-a)/N

def PDE(t, u , D, A_DD, b_DD):
    return (D / dx**2) * (A_DD @ u + b_DD)

# create the matrix A_DD
A_DD = np.zeros((N-1, N-1))
A_DD[0, 0] = -1
A_DD[-1, -1] = -1
for i in range(1, N-2):
    A_DD[i, i-1] = 1
    A_DD[i, i] = -2
    A_DD[i, i+1] = 1

# create the vector b_DD
b_DD = np.zeros(N-1)
b_DD[0] = alpha
b_DD[-1] = beta

sol = solve_ivp(PDE, (0, t_final), f(x_int), args=(D, A_DD, b_DD))

# extract the solution
u = sol.y
t = sol.t

# re attach the boundary conditions and make arrays of alpha and beta for all time steps
u = np.hstack((alpha*np.ones((len(t),1)), u.T, beta*np.ones((len(t),1))))

# display the solution
plt.figure()
plt.title('Linear diffusion equation: solve_ivp')

# plot 3 time steps
for n in range(0,len(t),len(t)//3):
    plt.plot(x, u[n,:], label='numerical solution, t=%.2f' % t[n])
    # plot the exact solution
    plt.plot(x, exact_solution(x, t[n]), label='exact solution, t=%.2f' % t[n], linestyle='--')

plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()
    
### now solve using homemade solvers - RK4
from solve_to import *
from solvers import *

# create a function for the pde
def PDE(u, t , args):
    D, A_DD, b_DD = args
    return D / dx**2 * (A_DD @ u + b_DD)
    
# create the matrix A_DD
A_DD = np.zeros((N-1, N-1))
A_DD[0, 0] = -1
A_DD[-1, -1] = -1
for i in range(1, N-2):
    A_DD[i, i-1] = 1
    A_DD[i, i] = -2
    A_DD[i, i+1] = 1

# create the vector b_DD
b_DD = np.zeros(N-1)
b_DD[0] = alpha
b_DD[-1] = beta

# create the initial condition
u0 = f(x_int)

# for time steps
for n in range(0,N_time):

    # solve the pde
    u[n+1,:] = solve_to(PDE, u0, t[n], t[n+1], dt, 'RK4', args = (D, A_DD, b_DD) )

# re attach the boundary conditions and make arrays of alpha and beta for all time steps
u = np.hstack((alpha*np.ones((N_time+1,1)), u, beta*np.ones((N_time+1,1))))

# display the solution
plt.figure()
plt.title('Linear diffusion equation: RK4')

# plot 3 time steps
for n in range(0,N_time,N_time//3):
    plt.plot(x, u[n,:], label='numerical solution, t=%.2f' % t[n])
    # plot the exact solution
    plt.plot(x, exact_solution(x, t[n]), label='exact solution, t=%.2f' % t[n], linestyle='--')

plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()

