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

while b < 2:
    # run shooting method to find the solution
    sol = shooting(hopf_bifurcation, Y0, 2*np.pi)

    # unpack the solution
    Y = sol[:-1]
    T = sol[-1]

    # get solution
    y, t = solve_to(hopf_bifurcation, Y0, 0, T, 0.01, 'RK4')

    # plot the solution
    plt.plot(t, y, label = 'b = {}'.format(b))

    # increment the parameter
    b += db

    # set the initial conditions to the previous solution
    Y0 = Y

# plot the solution
# put legend outside the plot small
plt.legend()
plt.title('Hopf Bifurcation')
plt.xlabel('t')
plt.ylabel('x(t) and y(t)')
plt.show()

