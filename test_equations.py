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