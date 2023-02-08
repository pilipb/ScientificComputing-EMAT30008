import numpy as np
import matplotlib.pyplot as plt

''' This program solves a system of 2 ODEs using the Euler, RK4 and Lax-Wendroff methods. 
The system of ODEs is: 

x''(t) - x(t) = 0

x'(t) = y(t)
y'(t) = -x(t)

The exact solution is:

x(t) = sin(t) + cos(t)

'''

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

### USING RK4 METHOD ###
# define the RK4 step
def RK4_step(f, X, t0, delta_t):
    x0,y0 = X
    k1 = f([x0,y0])
    k2 = f([x0 + delta_t/2, y0 + delta_t/2])
    k3 = f([x0 + delta_t/2, y0 + delta_t/2])
    k4 = f([x0 + delta_t, y0 + delta_t])
    x1,y1 = [x0,y0] + np.dot(delta_t/6, np.array(k1) + 2*np.array(k2) + 2*np.array(k3) + np.array(k4))
    t1 = t0 + delta_t
    X1 = [x1,y1]
    return X1, t1

'''
The Euler method is a first order method, so it is not stable for the system of ODEs.
This is seen in the plots below, where the solutions diverge from the exact solution.
'''

### USING THE A CENTRED METHOD ###
# def centred step
def Centre_step(f, X, t0, delta_t):
    delta_x = 0.0001
    x0,y0 = X
    x1,y1 = f([x0 - delta_x ,y0 - delta_x]) + f([x0 + delta_x,y0 + delta_x]) - (f([x0,y0])*2)
    t1 = t0 + delta_t
    X1 = [x1/(2 * delta_x),y1/(2 * delta_x)]
    return X1, t1

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
        elif method == 'RKF':
            x1, t1 = Centre_step(f, x[-1], t[-1], deltat_max)
        else:
            raise ValueError('method must be defined')
        # append the new values to the solution
        x.append([x1[0],x1[1]])
        t.append(t1)
        if t[-1] > t2:
            break
    return np.array(x), t


### SOLVING THE ODE ###

step_size = 0.1 # with a larger step size, the solutions diverge more from the exact solution
x0 = [np.pi/2, 0] # initial conditions
t0 = 0
t_end = 100 # end time


# solve for all methods
X_euler, t = solve_to(fun, x0, t0, t_end, step_size, 'euler')
X_rk4, t = solve_to(fun, x0, t0, t_end, step_size, 'RK4')
X_rkf, t = solve_to(fun, x0, t0, t_end, step_size, 'RKF')
 

### PLOTTING THE RESULTS ###
# Euler
plt.figure('X vs t - Euler')
plt.plot(t,X_euler,'-',label='Solutions - Euler')
# only plot for euler as it is the same for RK4

# plot the exact solution
plt.plot(t,x_exact(t),'--',label='Exact Solution')

plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()

# RK4
plt.figure('X vs t - RK4')
plt.plot(t,X_rk4,'-',label='Solutions - RK4')

# plot the exact solution
plt.plot(t,x_exact(t),'--',label='Exact Solution')

plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()

# RKF
plt.figure('X vs t - RKF')
plt.plot(t,X_rkf,'-',label='Solutions - RKF')

# plot the exact solution
plt.plot(t,x_exact(t),'--',label='Exact Solution')

plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()

# Plot Euler error
plt.figure('Error - Euler')
plt.plot(t,abs(X_euler[:,0]-x_exact(t)),'-',label='Error - Euler')
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()
plt.show()

# Plot RK4 error
plt.figure('Error - RK4')
plt.plot(t,abs(X_rk4[:,0]-x_exact(t)),'-',label='Error - RK4')
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()
plt.show()

# Plot RKF error
plt.figure('Error - RKF')
plt.plot(t,abs(X_rkf[:,0]-x_exact(t)),'-',label='Error - RKF')
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()
plt.show()

# now plot x(t) vs x'(t)
plt.figure('X vs X\' - Euler')
plt.plot(X_euler[:,0],X_euler[:,1],'-',label='x(t) vs x\'(t) - Euler')
plt.xlabel('x(t)')
plt.ylabel('x\'(t)')
plt.legend()
plt.show()

plt.figure('X vs X\' - RK4')
plt.plot(X_rk4[:,0],X_rk4[:,1],'-',label='x(t) vs x\'(t) - RK4')
plt.xlabel('x(t)')
plt.ylabel('x\'(t)')
plt.legend()
plt.show()

plt.figure('X vs X\' - RKF')
plt.plot(X_rkf[:,0],X_rkf[:,1],'-',label='x(t) vs x\'(t) - RKF')
plt.xlabel('x(t)')
plt.ylabel('x\'(t)')
plt.legend()
plt.show()

'''
The plots show that the Euler and RK4 methods are not stable for the system of ODEs.

The plots of x'(t) vs x(t) show that the solutions diverge from the exact solution over
time. 

'''




