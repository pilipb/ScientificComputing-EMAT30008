import numpy as np

# Test euler step for 3 ode's
def ode3(t, Y, *args):
    x, y, z = Y
    return np.array([y, z, -x]) 


# hopf bifurcation for shooting method
def hopf(t, X, *args):

    b = args[0]

    x = X[0]
    y = X[1]

    dxdt = b*x - y + x*(x**2 + y**2) - x*(x**2 + y**2)**2
    dydt = x + b*y + y*(x**2 + y**2) - y*(x**2 + y**2)**2

    return np.array([dxdt, dydt])

# define the cubic equation with parameter c
def cubic(x, *args):
    c = args
    return x**3 - x + c

# hopf polar coor form
def hopf_polar(t, X, *args):

    myu, omega = args[0]

    r = X[0]
    theta = X[1]

    drdt = r*(myu - r**2)
    dthetadt = omega

    return np.array([drdt, dthetadt])

# Lokta-Volterra equations
def lokta_volterra(t, Y, args):

    a, b, d = args
    x,y = Y
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1- (y/x))

    return np.array([dxdt, dydt])

