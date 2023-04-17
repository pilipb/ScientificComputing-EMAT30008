from solve_to import *
from solvers import *
from shooting import shooting
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipy
import math



# natural parameter continuation

'''
increment the parameter by a set amount and attempt to find the solution for
the new parameter value using the previous solution as the initial conditions

'''

# define the starting parameter value
b0 = 0

# define the function to be integrated
def hopf_bifurcation(t, Y, args = (b0)):
    b = args

    x, y = np.array(Y)
    dxdt = b*x - y - x*(x**2 + y**2)
    dydt = x + b*y - y*(x**2 + y**2)
    return np.array([dxdt, dydt])


# define the find dx/dt function
def dxdt( t, Y, f=hopf_bifurcation):
    return f(t, Y)[0]

# by varying b from 0 to 2, perform natural parameter continuation
# define the initial conditions
Y0 = np.array([0.1, 0.1])

# define the initial guess for the period
T_guess = 2*math.pi

# define the initial guess for the parameter
b_guess = 0

# define the step size for the parameter
b_step = 0.2

# repeat the process for the next parameter value
while b_guess < 2:
    
    sol = shooting(hopf_bifurcation, Y0, T_guess, args = b_guess)

    Y0 = sol[:-1]
    T_guess = sol[-1]

    # plot the solution
    Y, t = solve_to(hopf_bifurcation, Y0, 0, T_guess, 0.01, 'RK4', args = b_guess)
    plt.plot(Y[:,0], Y[:,1], label = 'b = ' + str(b_guess))

    b_guess = b_guess + b_step

plt.legend()
plt.show()



# pseudo-arclength continuation
'''
du . (u - u_guess) + dp .(p - p_guess) = 0

u is state vector
u_guess is predicted state vector
du is the secant of the state vector
p is the parameter vector
p_guess is the predicted parameter vector
dp is the secant of the parameter vector

'''


    







