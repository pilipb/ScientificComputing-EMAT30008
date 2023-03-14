from solve_to import *
from solvers import *
from shooting import shooting
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipy
import math


'''
Numerical Continuation Method

The numerical continuation method 


'''

# natural parameter continuation

'''
incrcement the parameter by a set amount and attempt to find the solution for
the new parameter value using the previous solution as the initial conditions

'''
# define the starting parameter value
a = 1
b = 0

# define the function to be integrated
def hopf_bifurcation(Y, t, args = (a, b)):
    a, b = args
    x, y = Y
    dxdt = b*x - y + a*x*(x**2 + y**2)
    dydt = x + b*y + a*y*(x**2 + y**2)
    return np.array([dxdt, dydt])


# define the initial conditions
Y0 = np.array([0.1, 0.1])

# define the parameter increment
db = 0.4

'''
performing pseudo-arclength continuation

the pseudo-arclength equation is added to the ODE system to be integrated
'''

# define the function to be integrated
'''
du . (u - u_guess) + dp .(p - p_guess) = 0

u is state vector
u_guess is predicted state vector
du is the secant of the state vector
p is the parameter vector
p_guess is the predicted parameter vector
dp is the secant of the parameter vector

'''

# unpack the initial conditions and period guess
T_guess = T

# y0 = Y0

# test the initial conditions guess
Y , _ = solve_to(hopf_bifurcation, Y0, 0, 300, 0.01, 'RK4')

# derive better starting guess from the solution
# [x0,y0] = [np.median(Y[:,0]), np.median(Y[:,1])]
y0 = np.median(Y,axis=0)

'''
The initial conditions are not always in the correct range. To fix this
I have found the phase space trajectory for random guess and given that it is
going to end up in the periodic solution, I have found the median of the solution
and used that as the starting guess for the root finding method as this will be
on the periodic phase space trajectory.

'''

# define the find dx/dt function
def dxdt( Y, t, f=f):
    return f(Y, t)[0]


# define the function that will be solved for the initial conditions and period
def fun(initial_vals):
    print(initial_vals)
    # unpack the initial conditions and period guess
    T = initial_vals[-1]
    y0 = initial_vals[:-1]

    Y , _ = solve_to(f, y0, 0, T, 0.01, 'RK4')

    num_dim = len(y0)
    row = np.zeros(num_dim)


    for i in range(num_dim):
        row[i] = Y[-1,i] - y0[i]

    row = np.append(row,dxdt(Y[-1],T))

    # add the pseudo-arclength equation
    psu_eq = 
    row = np.append(row, )

    output = np.array(row)
    return output

# solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
y0 = np.append(y0, T_guess)

sol = scipy.fsolve(fun, y0)

