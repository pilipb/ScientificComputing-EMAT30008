import numpy as np
import matplotlib.pyplot as plt

'''

x''(t) - x(t) = 0

# theta is x, omega is y

x'(t) = y(t)
y'(t) = -x(t)

'''
### USING ODEINT ###

# def pend(x0, t):
#     x, y = x0
#     dxdt = [y, -x]
#     return dxdt

# x0 = [np.pi - 0.1, 0.0]

# t = np.linspace(0, 10, 101)

# from scipy.integrate import odeint
# sol = odeint(pend, x0, t)

# plt.figure()

# plt.plot(t, sol[:, 0], 'b', label='x(t)')
# plt.plot(t, sol[:, 1], 'g', label='y(t)')
# plt.legend(loc='best')
# plt.xlabel('t')
# plt.grid()
# plt.show()

### USING EULER METHOD ###

# define function - System ODEs
def fun(x0):
    x, y = x0
    dXdt = [y, -x]
    return dXdt

# define error
x_exact = lambda t: np.sin(t) + np.cos(t)


# define euler step
def euler_step(f, X, t0, delta_t):
    x0,y0 = X
    x1,y1 = [x0,y0] +  np.dot(delta_t, f([x0,y0]))
    t1 = t0 + delta_t
    X1 = [x1,y1]
    return X1, t1

# define the RK4 step
def RK4_step(f, x0, t0, delta_t):
    k1 = delta_t * f(x0, t0)
    k2 = delta_t * f(x0 + k1/2, t0)
    k3 = delta_t * f(x0 + k2/2, t0)
    k4 = delta_t * f(x0 + k3, t0)
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
        x.append([x1[0],x1[1]])
        t.append(t1)
        if t[-1] > t2:
            break
    return x, t


### SOLVING THE ODE ###

# solve for range of step sizes
deltas = np.arange(0.001, 1, 0.001)
euler_errors = []
RK4_errors = []

x0 = [np.pi - 0.1, 0.0]
t0 = 0

deltas = np.arange(0.001, 1, 0.001)


# use the timer
# for delta_t in deltas:
    # solve the ODE up to t = 1
X_euler, t = solve_to(fun, x0, t0, 10, 0.01, 'euler')
X_rk4, t = solve_to(fun, x0, t0, 10, 0.01, 'euler')
    # sol.append(x)
    # solT.append(t)
    # save the error
X_euler = np.array(X_euler)
X_rk4 = np.array(X_rk4)


# errors = []
# for delta_t in deltas:
#     # solve the ODE up to t = 1
#     X, t = solve_to(fun, x0, t0, 10, 0.01, 'euler')
#     # save the error
    

plt.figure()
### PLOTTING THE RESULTS ###
plt.plot(t,X_euler,'-',label='Solutions - Euler')
plt.plot(t,X_rk4,'-',label='Solutions - RK4')
# plt.loglog(deltas, RK4_errors, 'o', label='RK4')
# plt.title('Error vs Delta t for Euler and RK4 methods')
# plt.xlabel('Delta t')
# plt.ylabel('Error')
plt.legend()
plt.show()

