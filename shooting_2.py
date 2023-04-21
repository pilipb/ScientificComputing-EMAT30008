'''
we have a pde of form:

u_t = D u_xx; with boundary conditions u(0,t) = A and u(1,t) = B

for the long term solution:

0 = D u_xx ; with boundary conditions u(0) = A and u(1) = B

as a first order system:

u1_t = v
u2_t = 0

with boundary conditions u1(0) = A and u1(1) = B and u2(0) = alpha (unknown)

we can solve this system using the shooting method (and for the limit cycle):

f(u) = [u1(0) - A, u1(1) - B, phase(T) - phase(0)] = 0

where phase(T) is the phase condition of the solution at time T and phase(0) is the phase condition of the solution at time 0
The phase condition can be u_x 

u1(1) is found by numerical integration

start with a guess u(0) = [A, alpha=0]

use newton's method to iterate to a solution:

integrate the system from 0 to 1 with initial conditions u(0) = [A, alpha]

update alpha = u1(1) - B

repeat until f(u) = 0

'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipy
import math
from solve_to import solve_to
'''
Explanation of the shooting method:

    consider the ode:
    mx¨ + cx˙ + kx = gamma sin(ωt)

    T = 2π/omega

    in first order form:
    u1' = u2
    u2' = 1/m (-c u2 - k u1 + gamma sin(ωt))

    the time solution is:

    u(t) = [u1(t), u2(t)] = F(t, u0) (where F is the integration of u with the initial conditions u0, up to time t)

    the limit cycle is when u0 - u(T) = 0 for some u0

    u0 - F(T, u0) = 0

    ie: solving the root for G(u0) = 0, where G(u0) = u0 - F(T, u0)

    for autonomous systems, G(u0) = [u0 - F(T, u0), phi(0)] = 0 where phi(0) is the phase condition at time 0

    the phase condition is the x derivative of the solution at time t

    phi(t) = u1' = u2

    solve by passing G(u0) to fsolve

'''

class Shooting():

    def __init__(self, f, u0, T, *args,  u1=None):
        '''
        Initialising the shooting method

        Parameters
        ----------------------------
        f : function
                the function to be integrated (with inputs (t, Y, *args)) in first order form of n dimensions
        u0 : array
                the initial conditions guess for the integration
        T : float
                the initial guess for the period of the solution
        args : tuple
                the arguments for the function f
        u1 : array
                the second initial conditions guess for the integration (for pseudo arclength continuation)
        pal : bool
                whether to use pseudo arclength continuation or not

        '''
        self.f = f
        self.u0 = u0
        self.T = T
        self.args = args[0]

        if u1 is not None:
            self.u1 = u1
            self.pal = True

    
    def shooting_setup(self):
        '''
        Implementing a numerical shooting method to set up the function F(u) to be solved for the limit cycle initial conditions

        Parameters
        ----------------------------
        self

        f : function
                the function to be integrated (with inputs (t, Y, *args)) in first order form of n dimensions
        u0 : array
                the initial conditions guess for the integration
        T : float
                the initial guess for the period of the solution
        args : tuple
                the arguments for the function f
                
        Returns
        ----------------------------
        F(u) : array 
                the array to be solved that finds the initial conditions for the limit cycle

        '''

        # solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
        Y, _ = solve_to(self.f, self.u0, 0, self.T, 0.01, 'RK4', args=self.args)

        # phase condition is first row of f(T, Y[-1], *args)
        phi = self.f(self.T, Y[-1], self.args)[0]

        # define the f(u) = 0 function
        num_dim = len(self.u0)
        row = np.zeros(num_dim)

        for i in range(num_dim):
            row[i] = Y[-1,i] - self.u0[i]

        G = np.append(row, phi)

        if self.pal == True:
            G = self.pal_setup(G, self.u0, self.u1)

        G.self = G

        return G

    def pal_setup(self, G, u0, u1):

        '''
        Adds the pseudo arclength continuation equation to the shooting method setup

        Parameters
        ----------------------------
        G : array
                G(u) = 0 from the shooting method setup
        f : function
                the function to be integrated (with inputs (t, Y, *args)) in first order form of n dimensions
        u0 : array
                the initial conditions for the integration
        u1 : array
                a second initial conditions for the integration
        T : float
                the initial guess for the period of the solution
        args : tuple
                the arguments for the function f


        Returns
        ----------------------------
        G : array 
                the array to be solved that finds the initial conditions for the limit cycle
                in a pseudo arclength continuation method

        '''

        # create secant line
        delta = u1 - u0

        # define a prediction 
        u2_p = u1 +  delta

        # make an estimate of u2 using the solve to method
        Y, _ = solve_to(self.f, self.u0, 0, self.T, 0.01, 'RK4', args=self.args)
        u2 = Y[-1]

        # u2 is what we want to solve for in this equation
        pal_eq = np.dot(u2 -  u2_p , delta) 

        G = np.append(G, pal_eq)

        return G
    
    def G(self, u0):

        '''
        This function will be passed into fsolve to find the initial conditions for the limit cycle

        The initial conditions are:

        u0, T, p
        
        '''

        # unpack the initial conditions
        p0 = u0[-1]
        self.T = u0[-2]
        self.u0 = u0[:-2]









### TEST ###



# # define the function to be integrated
# def f(t, Y, *args):
#     # unpack the arguments
#     m, c, k, gamma, omega = args[0]

#     # unpack the variables
#     u1, u2 = Y

#     # define the partial derivatives
#     u1p = u2
#     u2p = 1/m*(-c*u2 - k*u1 + gamma*np.sin(omega*t))

#     # return the derivatives
#     return np.array([u1p, u2p])
 
# # define the parameters
# m = 1
# c = 1
# k = 1
# gamma = 1
# omega = 1

# # define the initial conditions
# u10 = 0
# u20 = 0
# T = 2*np.pi/omega

# # define the initial conditions
# u0 = np.array([u10, u20])

# # define the arguments
# args = (m, c, k, gamma, omega)

# # solve the function
# y0, T = shooting(f, u0, T, args)

# # solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
# Y, t = solve_to(f, y0, 0, T, 0.01, 'RK4', args=args)

# # plot the solution
# plt.plot(t, Y[:,0])
# plt.plot(t, Y[:,1])
# plt.title('T = ' + str(T))
# plt.show()

# # However it is difficult to find correct initial conditions for the shooting method
# # so continuation methods can be used

