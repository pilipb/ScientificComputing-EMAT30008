import numpy as np
from scipy.optimize import fsolve

# f1 defines a circle in 3D space (two dimensional output)
# f2 defines a plane in 3D space (one dimensional output)
# g merges f1 and f2 together to find the intersection point of the circle and the plane

def f1(u, p):
    # u is the state vector, p is the parameter vector
    x, y, z = u  # unpack u
    return np.array([
            x**2 + y**2 - p[0]**2,  # circle of radius p[1]
            z - p[1],               # at a height of p[2]
        ])

def f2(u, p):
    x, y, z = u # unpack u
    return np.array([p[0]*x + p[1]*y + p[2]*z])  # a plane passing through (0, 0, 0)

def g(u, p):
    # return the equations for a plane passing through a circle in 3D space
    return np.concatenate((f1(u, p[0]), f2(u, p[1])))

def solve(u0, p):
    # solve the combined problem
    u, info, ier, msg = fsolve(g, u0, args=(p,), full_output=True)
    if ier == 1:
        print("Root finder found the solution u={} after {} function calls; the norm of the final residual is {}".format(u, info["nfev"], np.linalg.norm(info["fvec"])))
        return u
    else:
        print("Root finder failed with error message: {}".format(msg))
        return None

# parameters
p = [[1, 0], [1, 0, 1]]  # p is a list containing two lists; the first list is for f1, the second is for f2

# attempt 1 to find the solution
u0 = np.array([0, 0, 0])
solve(u0, p)

# attempt 2 to find the solution (a better starting guess)
u0 = np.array([0, 0.5, 0])
solve(u0, p)
