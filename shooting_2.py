'''
we have a pde of form:

u_t = D u_xx; with boundary conditions u(0,t) = A and u(1,t) = B

for the long term solution:

0 = D u_xx ; with boundary conditions u(0) = A and u(1) = B

as a first order system:

u1_t = v
u2_t = 0

with boundary conditions u1(0) = A and u1(1) = B and u2(0) = alpha (unknown)

we can solve this system using the shooting method:

f(u) = [u1(0) - A, u1(1) - B] = 0

u1(1) is found by numerical integration

start with a guess u(0) = [A, alpha=0]

use newton's method to iterate to a solution

first iteration:
u1(1) = u1(0) + u2(0) * dt



u

'''