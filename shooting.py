from solve_to import *
from solvers import *
import numpy as np
import matplotlib.pyplot as plt
# import scipy.optimize as scipy
import scipy
import math

class Discretisation():
    def __init__(self):
        '''
        Class to discretise the ODEs

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> discret = Discretisation()
        >>> fun = discret.shooting_setup(f, y0, T= 0, args = None)
        >>> u0, T0 = shooting_solve(fun, u0)

        Finding a limit cycle using a shooting method
        
        >>> Y,t = solve_to(f, u0, 0, T0, 0.01, 'RK4', args=b)
    
        
        '''
        pass

    def shooting_setup(self, f, u, *args):

        '''
        Implementing a numerical shooting method to solve an ODE to find a periodic solution
        This method will solve the BVP ODE using root finding method to find the limit cycle, using scipy.optimize.fsolve to
        find the initial conditions and period that satisfy the boundary conditions.

        parameters:
        ----------------------------
        f - function: the function to be integrated (with inputs (t, Y, *args)) in first order form of n dimensions
        u - array: the initial guess for the solution [x0, y0, ... , T]
        args - array: the arguments for the function f

        returns:
        ----------------------------
        sol - array: the initial conditions that cause the solution to be periodic: sol = [x0, y0, ... , T]

        '''


        # define the function that will be solved for the initial conditions and period
        def fun(u, *args):
            '''
            Function F(u) = 0 that will be solved for the initial conditions and period
            Parameters:
            ----------------------------
            initial_vals - array: the initial conditions and period guess: initial_vals = [x0, y0, ... , T]

            Returns:
            ----------------------------
            row - array: the boundary conditions that must be satisfied for the solution to be periodic: 
                        row = [x(T) - x0, y(T) - y0, ... , dx/dt(0) = 0]

            '''
            # unpack the initial conditions and period guess
            T = u[-1]
            y0 = u[:-1] 

            Y , _ = solve_to(f, y0, 0, T, 0.01, 'RK4', args=args)

            # limit cycle condition (last point - initial point = 0)
            row = Y[-1,:] - y0[:]
    
            # phase condition
            row = np.append(row, f(0, Y[0,:], args)[0]) # dx/dt(0) = 0 

            return row

        return fun
    

    
    def linear(self, f, x, *args):
        '''
        Linear discretisation of the ODE (x => x)

        Parameters:
        ----------------------------
        f - function: the function to be solved (with inputs (x, args))
        args - array: the arguments for the function f

        Returns:
        ----------------------------
        f - function: the function to be solved (with inputs (x, args))
        
        '''
        return f


def shooting_solve(fun, u0, *args):
    '''
    Solve the system of equations made in the setup function

    Parameters:
    ----------------------------
    fun - function: such that fun(u) = 0
    u0 - array: the initial guess for the solution

    Returns:
    ----------------------------
    u0 - array: the initial conditions that cause the solution to be periodic: u0 = [x0, y0, ... ]
    T - float: the period of the solution
    
    '''

    sol  = scipy.optimize.root(fun, u0, args=args)

    if sol.success == False:
        print('Warning: Shooting method did not converge')
        return None
    
    # return the period and initial conditions that cause the limit cycle: sol = [x0, y0, ... , T]
    u0 = sol.x[:-1]
    T = sol.x[-1]

    return u0, T


