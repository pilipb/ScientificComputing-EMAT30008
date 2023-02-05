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
    k1 = delta_t * f([x0,y0])
    k2 = delta_t * f([x0 + k1/2, y0 + k1/2])
    k3 = delta_t * f([x0 + k2/2, y0 + k2/2])
    k4 = delta_t * f([x0 + k3, y0 + k3])
    x1 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6
    y1 = y0 + (k1 + 2*k2 + 2*k3 + k4)/6
    t1 = t0 + delta_t
    X1 = [x1,y1]
    return X1, t1

'''
Both the Euler and RK4 methods are first order methods, so they are not stable for the system of ODEs.
This is seen in the plots below, where the solutions diverge from the exact solution.

The Lax-Wendroff method is a second order method, so it is stable for the system of ODEs.
This is seen in the plots below, where the solutions converge to the exact solution.


'''

### USING LAX-WENDROFF METHOD ###
# define the Lax-Wendroff step
def Lax_Wendroff_step(f, X, t0, delta_t):
    x0,y0 = X
    x1 = x0 + delta_t*f([x0,y0])[0] + (delta_t**2/2)*f([x0,y0])[1]
    y1 = y0 + delta_t*f([x0,y0])[1] - (delta_t**2/2)*f([x0,y0])[0]
    X1 = [x1,y1]
    t1 = t0 + delta_t
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
        elif method == 'Lax-Wendroff':
            x1, t1 = Lax_Wendroff_step(f, x[-1], t[-1], deltat_max)
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
X_rk4, t = solve_to(fun, x0, t0, t_end, step_size, 'euler')
X_lax, t = solve_to(fun, x0, t0, t_end, step_size, 'Lax-Wendroff')
 

### PLOTTING THE RESULTS ###
plt.figure('X vs t - Euler')
plt.plot(t,X_euler,'-',label='Solutions - Euler')
# only plot for euler as it is the same for RK4

# plot the exact solution
plt.plot(t,x_exact(t),'--',label='Exact Solution')

plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()

plt.figure('X vs t - RK4')
plt.plot(t,X_rk4,'-',label='Solutions - RK4')

# plot the exact solution
plt.plot(t,x_exact(t),'--',label='Exact Solution')

plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()

plt.figure('X vs t - Lax-Wendroff')
plt.plot(t,X_lax,'-',label='Solutions - Lax-Wendroff')

# plot the exact solution
plt.plot(t,x_exact(t),'--',label='Exact Solution')

plt.xlabel('t')
plt.ylabel('x(t)')
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

plt.figure('X vs X\' - Lax-Wendroff')
plt.plot(X_lax[:,0],X_lax[:,1],'-',label='x(t) vs x\'(t) - Lax-Wendroff')
plt.xlabel('x(t)')
plt.ylabel('x\'(t)')
plt.legend()
plt.show()

'''
The plots show that the Euler and RK4 methods are not stable for the system of ODEs.
The Lax-Wendroff method is stable for the system of ODEs.

The plots of x'(t) vs x(t) show that the solutions diverge from the exact solution over
time. 

'''




