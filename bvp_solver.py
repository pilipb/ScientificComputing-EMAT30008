'''
solving poisson equation using finite difference method
Dirichlet boundary condition

in domain:
a <= x <= b
boundary condition:
u(a) = alpha
u(b) = beta

equation:
u''(x) + q(x) = 0

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# step 1: discretise the domain into grid points

def discretise(a, b, N):
    dx = (b - a)/N
    xi = np.linspace(0, N, N+1)

    return dx, xi

# step 2: solve using scipy root

# write system in form of f(u) = 0

def f(u, dx, q, alpha, beta):
    f_1 = (u[2] - 2*u[0] + alpha)/dx**2 + q[1]
    f_i = (u[2:] - 2*u[1:-1] + u[:-2])/dx**2 + q[1:-1]
    f_N1 = (beta - 2*u[-1] + u[-2])/dx**2 + q[-2]

    return np.concatenate(([f_1], f_i, [f_N1]))



# step 3: solve using Numpy - linalg.solve

# A . u = - b - dx**2 * q

def A_matrix(N):
    A = np.zeros((N+1, N+1))
    A[0, 0] = 1
    A[-1, -1] = 1
    A[1:-1, 1:-1] = np.eye(N-1)
    A[1:-1, 2:] -= np.eye(N-1)
    A[1:-1, :-2] -= np.eye(N-1)

    return A

def b_vector(b, dx, q):
    b_vec = - b - dx**2 * q[1:-1]
    return b_vec








