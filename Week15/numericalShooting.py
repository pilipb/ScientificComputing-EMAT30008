''' 
Beginning with an example:

Solve the ordinary differential equation

m * x''(t) + c * x'(t) + k * x(t) = gamma * cos(omega * t)

the period of the solution is 2 * pi / omega

'''
### imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

### define the ODE

# rewrite the ODE as a system of first order ODEs
def f(y, t, m, c, k, gamma, omega):
    x, v = y
    return [v, -c/m * v - k/m * x + gamma/m * np.cos(omega * t)]

# define the exact solution
def x_exact(t, m, c, k, gamma, omega):
    return gamma / (m * omega) * np.sin(omega * t) - gamma / (m * omega) * np.cos(omega * t) + gamma / (m * omega) - gamma / (m * omega) * np.exp(-c * t / (2 * m)) * np.cos(np.sqrt(k - c**2 / (4 * m**2)) * t + np.arctan((2 * m * omega) / (c - 2 * m * omega)))

# define the initial conditions
m = 1
c = 1
k = 1
gamma = 1
omega = 1
y0 = [0, 0]

# define the time interval
t = np.linspace(0, 10 * np.pi / omega, 1000)

# solve the ODE
y = odeint(f, y0, t, args=(m, c, k, gamma, omega))

# plot the solution
plt.plot(t, y[:, 0], 'b', label='numerical')
plt.plot(t, x_exact(t, m, c, k, gamma, omega), 'r', label='exact')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend(loc='best')
plt.show()

### Simulate the predator-prey equations below

a = 1
d = 0.1
b = 0.1 # [0.1 0.5]
dxdt = lambda x, y: x*(1-x) - (a*x*y / d+x)
dydt = lambda x, y: b*y*(1 - y/x)

fun = lambda Y, t: [dxdt(Y[0], Y[1]), dydt(Y[0], Y[1])]

# define the initial conditions
x0 = 1
y0 = 1
Y0 = [x0, y0]

# define the time interval
t = np.linspace(0, np.pi / omega, 1000)

# vary b
for b in np.array([0.1, 0.2, 0.25, 0.27, 0.3, 0.4, 0.5]):
    # solve the ODE
    Y = odeint(fun, Y0, t)
    # plot the solution x" against x'
    plt.figure('b = ' + str(b))
    plt.plot(Y[:, 0], Y[:, 1], 'b', label='numerical')
    plt.xlabel('x')
    plt.ylabel('x\'')
    plt.legend(loc='best')
    plt.show()
    




