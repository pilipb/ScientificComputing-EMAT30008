from helpers import *
from scipy.optimize import root
from numpy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt


# define the parameters
D = 1
alpha = 0
beta = 0
a = 0
b = 1
N = 10
bound_type = 'DD'

# define the method and boundary condition
method = 'scipy'

# define the function q
def q(x_int, u, *args):
    myu = args[0]
    return np.exp(myu*u)

args = (4,)

# create the grid
x = np.linspace(a, b, N+1)
x_int = x[1:-1] # interior points
dx = (b-a)/N

# define an initial guess for the solution
u = np.zeros(N-1)

# use boundary functions to define the boundary conditions (for interior points)
A_mat, b_vec = boundary(alpha, beta, N, dx, bound_type)

# define the vector q as a function of u but length N-1
def mak_q(u):
    if callable(q):
        # make a vector of length x_int, and width 1 with q(x,u,args) for each x in x_int
        q_vec = q(x_int, u, *args)

        return q_vec
    else:
        return q * np.ones(N-1)
    
# solve the linear system
if method == 'scipy': # use SciPy root - use when q is a function of u
    # define the function f(u) = 0
    def f(u):
        return D*A_mat@u + b_vec + mak_q(u)
    # solve the system
    sol = root(f, u)

    if not sol.success:
        raise ValueError('The solution did not converge')
    
    u = sol.x

# add the boundary conditions to the solution
u = np.concatenate(([alpha], u, [beta]))



# plot the solution
plt.plot(x, u, 'o-')
plt.xlabel('x')
plt.ylabel('u')
plt.show()




