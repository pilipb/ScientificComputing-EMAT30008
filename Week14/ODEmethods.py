# imports
import numpy as np
import matplotlib.pyplot as plt

### SETTING UP THE METHODS ###

# define euler step
def euler_step(f, x0, t0, delta_t):
    x1 = x0 + delta_t * f(x0)
    t1 = t0 + delta_t
    return x1, t1

# define the RK4 step
def RK4_step(f, x0, t0, delta_t):
    k1 = delta_t * f(x0)
    k2 = delta_t * f(x0 + k1/2)
    k3 = delta_t * f(x0 + k2/2)
    k4 = delta_t * f(x0 + k3)
    x1 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6
    t1 = t0 + delta_t
    return x1, t1

# define solve_to
def solve_to(f, x1, t1, t2, deltat_max, method):
    # initialize the solution
    x = [x1]
    t = [t1]

    # loop until we reach the end point
    while True:
        # take a step
        if method == 'euler':
            x1, t1 = euler_step(f, x[-1], t[-1], deltat_max)
        elif method == 'RK4':
            x1, t1 = RK4_step(f, x[-1], t[-1], deltat_max)
        else:
            raise ValueError('method must be euler or RK4')
        # append the new values to the solution
        x.append(x1)
        t.append(t1)
        if t[-1] > t2:
            break
    return x, t

### DEFINING THE ODE ###

# define the function
f = lambda x: x

# define the initial conditions
x0 = 1
t0 = 0

# define the exact solution and the error function
x_exact = lambda t: np.exp(t)
def error(x, t):
    return np.abs(x - x_exact(t))

### SOLVING THE ODE ###

# solve for range of step sizes
deltas = np.arange(0.001, 1, 0.001)
euler_errors = []
RK4_errors = []

for delta_t in deltas:
    # solve the ODE up to t = 1
    x, t = solve_to(f, x0, t0, 1, delta_t, 'euler')
    # save the error
    euler_errors.append(error(x[-1], t[-1]))

    x, t = solve_to(f, x0, t0, 1, delta_t, 'RK4')
    # save the error
    RK4_errors.append(error(x[-1], t[-1]))

### PLOTTING THE RESULTS ###
plt.loglog(deltas, euler_errors, 'o', label='Euler')
plt.loglog(deltas, RK4_errors, 'o', label='RK4')
plt.title('Error vs Delta t for Euler and RK4 methods')
plt.xlabel('Delta t')
plt.ylabel('Error')
plt.legend()
plt.show()

### ANALYSIS ###
