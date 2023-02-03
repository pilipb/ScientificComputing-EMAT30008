import numpy as np
import matplotlib.pyplot as plt

'''

x''(t) - x(t) = 0

# theta is x, omega is y

x'(t) = y(t)
y'(t) = -x(t)

'''
### USING ODEINT ###

def pend(x0, t):
    x, y = x0
    dxdt = [y, -x]
    return dxdt

x0 = [np.pi - 0.1, 0.0]

t = np.linspace(0, 10, 101)

from scipy.integrate import odeint
sol = odeint(pend, x0, t)

plt.plot(t, sol[:, 0], 'b', label='x(t)')
plt.plot(t, sol[:, 1], 'g', label='y(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
# plt.show()

### USING EULER METHOD ###

from ODEmethods import euler_step
from systemODE import solve_to

### SOLVING THE ODE ###

# solve for range of step sizes
deltas = np.arange(0.001, 1, 0.001)
euler_errors = []
RK4_errors = []

x0 = [np.pi - 0.1, 0.0]
t0 = 0

deltas = np.arange(0.001, 1, 0.001)

# use the timer
for delta_t in deltas:
    # solve the ODE up to t = 1
    x, t = solve_to(pend, x0, t0, 1, delta_t, 'euler')
    # save the error
    euler_errors.append(error(x[-1], t[-1]))

### PLOTTING THE RESULTS ###
plt.loglog(deltas, euler_errors, 'o', label='Euler')
# plt.loglog(deltas, RK4_errors, 'o', label='RK4')
plt.title('Error vs Delta t for Euler and RK4 methods')
plt.xlabel('Delta t')
plt.ylabel('Error')
plt.legend()
plt.show()
