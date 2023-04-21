import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from shooting import shooting_setup

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

def nat_continuation(ode, x0, p0 , vary_p =0, step = 0.1, max_steps = 100, discret=None, solver=scipy.optimize.fsolve):

    # initialize the solution
    X = []
    C = []

    # discretise the ode
    '''
    if the function is an algebraic equation, then the discretisation is just the function itself
    if the function is a differential equation, then the discretisation is calling shooting method
    to make the F(x) = 0 that will be solved by the solver
    
    '''
    # check that the discret either a lambda function or a function
    if not callable(discret):
        raise ValueError("discretisation must be a function or lambda function")
    
    try:
        param = p0[vary_p]
    except:
        param = p0

    fun, u0 = discret(ode, x0, args = param )

    sol = scipy.optimize.fsolve(fun, u0, args = (param))

    # append the solution
    X.append(sol)
    C.append(p0)

    num_steps = 0
    # loop with incrementing c until reaching the limit
    while num_steps < max_steps:
        p0 += step
        sol = scipy.optimize.fsolve(ode, sol, args = (p0))
        X.append(sol)
        C.append(p0)
        num_steps += 1

    return X, C


# define the cubic equation
def cubic(x, *args):
    
    c = args
    return x**3 - x + c

# define the initial conditions
x0 = 1

# define the limit of c
def linear(x, x0, args):
    return x , x0

# solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
X, C = nat_continuation(cubic, x0, -2, vary_p = 0, step = 0.1, max_steps = 100, discret=linear, solver=scipy.optimize.fsolve)

# plot the solution
plt.plot(C, X)
plt.grid()
plt.show()





