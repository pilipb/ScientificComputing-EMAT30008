import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

'''
results = continuation(myode,  # the ODE to use
    x0,  # the initial state
    par0,  # the initial parameters
    vary_par=0,  # the parameter to vary
    step_size=0.1,  # the size of the steps to take
    max_steps=100,  # the number of steps to take
    discretisation=shooting,  # the discretisation to use
    solver=scipy.optimize.fsolve)  # the solver to use
    
    '''

# have an algebraic cubic equation

# x^3 - x + c = 0

# natural continuation will find the roots of the equation, then increment c and find the roots again

def cubic(x, c):
    return x**3 - x + c

def nat_continuation(ode, x0, p0 = 0, vary_p =0, step = 0.1, max_steps = 100, discret=lambda x: x, solver=scipy.optimize.fsolve):

    # initialize the solution
    X = []
    C = []

    c0 = c_lim[0]

    # find initial solution F(u) = 0
    sol = scipy.optimize.fsolve(ode, x0, args = (c0))

    # append the solution
    X.append(sol)
    C.append(c0)

    # loop with incrementing c until reaching the limit
    while c0 < c_lim[1]:
        c0 += 0.01
        sol = scipy.optimize.fsolve(ode, sol, args = (c0))
        X.append(sol)
        C.append(c0)

    return X, C

# define the initial conditions
x0 = 1

# define the limit of c
c_lim = [-2, 2]

# solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
X, C = nat_continuation(cubic, x0, c_lim)

# plot the solution
plt.plot(C, X)
plt.grid()
plt.show()



