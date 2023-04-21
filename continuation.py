import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from shooting import shooting_setup

'''
results = continuation(myode,  # the ODE to use
    x0,  # the initial state
    par0,  # the initial parameters (as a list)
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
    '''
    Natural continuation method to increment a parameter and solve the ODE for the new parameter value
    Parameters:
    ----------------------------
    ode - function: 
            the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
    x0 - array:
            the initial value of the solution
    p0 - list:
            the initial values of the parameters
    vary_p - int:
            the index of the parameter to vary
    step - float:
            the size of the steps to take
    max_steps - int:
            the number of steps to take
    discret - function:
            the discretisation to use (either shooting_setup or linear)
    solver - function:
            the solver to use (either scipy.optimize.fsolve or scipy.optimize.root)

    Returns:
    ----------------------------
    X - array:
            the solution of the equation for each parameter value
    C - array:
            the parameter values that were used to solve the equation

    
    '''
    # initialize the solution
    X = []
    C = []

    # discretise the ode
    '''
    if the function is an algebraic equation, then the discretisation is just the function itself
    if the function is a differential equation, then the discretisation is calling shooting method
    to make the F(x) = 0 that will be solved by the solver
    
    '''
    try:
        param = p0[vary_p]
    except:
        param = p0


    # check that the discret either a lambda function or a function
    if not callable(discret):
        raise ValueError("discretisation must be a function or lambda function")
    
    fun, u0 = discret(ode, x0, T=0, args = param )

    sol = scipy.optimize.fsolve(fun, u0, args = (param,))

    # append the solution
    X.append(sol)
    C.append(param)

    num_steps = 0
    # loop with incrementing c until reaching the limit
    while num_steps < max_steps:
        param += step
        sol = scipy.optimize.fsolve(fun, u0, args = (param))
        X.append(sol)
        C.append(param)
        num_steps += 1

    return X, C


# define the cubic equation
def cubic(x, *args):
    
    c = args
    return x**3 - x + c

# define the initial conditions
x0 = 1

# define the limit of c
def linear(x, x0,T, args):
    return x , x0

# solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
X, C = nat_continuation(cubic, x0, -2, vary_p = 0, step = 0.01, max_steps = 400, discret=linear, solver=scipy.optimize.fsolve)

# plot the solution
plt.figure()
plt.title('Cubic Equation')
plt.plot(C, X)
plt.xlabel('parameter c value')
plt.ylabel('root value')
plt.grid()
plt.show()

# now test natural continuation with a differential equation - Hopf bifurcation
def hopf(t, X, *args):

    b = args[0]

    x = X[0]
    y = X[1]

    dxdt = b*x - y + x*(x**2 + y**2) - x*(x**2 + y**2)**2
    dydt = x + b*y + y*(x**2 + y**2) - y*(x**2 + y**2)**2

    return np.array([dxdt, dydt])

# define the initial conditions
x0 = [-10,10]

# define parameter
p = 2

print('second')

# solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
X, C = nat_continuation(hopf, x0, p, vary_p = 0, step = -0.1, max_steps = 30, discret=shooting_setup, solver=scipy.optimize.fsolve)

# split the X into x, y and period at each parameter value
x = [x[0] for x in X]
y = [x[1] for x in X]
T = [x[2] for x in X]

# plot the solution
plt.figure()
plt.title('Hopf Bifurcation')
plt.plot(C, x )
plt.plot(C, y)
plt.xlabel('parameter b value')
plt.ylabel('solution value')
plt.grid()
plt.show()

# plot the period
plt.figure()
plt.title('Hopf Bifurcation')
plt.plot(C, T)
plt.xlabel('parameter b value')
plt.ylabel('period value')
plt.grid()
plt.show()





