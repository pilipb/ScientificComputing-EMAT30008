from solvers import *
from solve_to import *
from shooting import *

import numpy as np
import matplotlib.pyplot as plt

# define the ODE
def ode(Y, t, args = ()):
    x, y = Y
    return np.array([y, -x])

# run each method
def test_solvers():
    # define the initial conditions
    y0 = [0.3, 0.1]
    t0 = 0
    delta_t = 0.1

    # define the final time
    t1 = 10

    # solve the ODE
    Y_euler, t_e = solve_to(ode, y0, t0, t1, delta_t, 'Euler')
    Y_rk4, t_r = solve_to(ode, y0, t0, t1, delta_t, 'RK4')
    Y_lw, t_l = solve_to(ode, y0, t0, t1, delta_t, 'Lax-Wendroff')

    # plot the solution
    plt.plot(t_e, Y_euler, 'b', label='Euler')
    plt.plot(t_r, Y_rk4, 'r', label='RK4')
    plt.plot(t_l, Y_lw, 'g', label='Lax-Wendroff')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend(loc='best')
    plt.show()

# test the solvers
test_solvers()

###

