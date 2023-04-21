import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

# have an algebraic cubic equation

# x^3 - x + c = 0

# natural continuation will find the roots of the equation, then increment c and find the roots again

def cubic(x, c):
    return x**3 - x + c

def natural_continuation(f, x0, c_lim, args = None):

    # initialize the solution
    X = []
    C = []

    c0 = c_lim[0]

    # find initial solution F(u) = 0
    sol = scipy.optimize.fsolve(f, x0, args = (c0))

    # append the solution
    X.append(sol)
    C.append(c0)

    # loop with incrementing c until reaching the limit
    while c0 < c_lim[1]:
        c0 += 0.01
        sol = scipy.optimize.fsolve(f, sol, args = (c0))
        X.append(sol)
        C.append(c0)

    return X, C

# define the initial conditions
x0 = 1

# define the limit of c
c_lim = [-2, 2]

# solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
X, C = natural_continuation(cubic, x0, c_lim)

# plot the solution
plt.plot(C, X)
plt.grid()
plt.show()



