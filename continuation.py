import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from shooting import Discretisation
from solve_to import solve_to

'''

FORMAT:

results = continuation(myode,  # the ODE to use
    x0,  # the initial state
    par0,  # the initial parameters (as a list)
    vary_par=0,  # the parameter to vary
    step_size=0.1,  # the size of the steps to take
    max_steps=100,  # the number of steps to take
    discretisation=shooting,  # the discretisation to use
    solver=scipy.optimize.fsolve)  # the solver to use
    
    '''
class Continuation:
    def __init__(self, solver=scipy.optimize.root):
        '''
        Continuation class to solve ODEs and algebraic equations for finding roots in a parameter space

        Parameters:
        ----------------------------
        solver - function:
                the solver to use to solve the discretised equation (either fsolve or root)

        Returns:
        ----------------------------
        None

        Examples:
        ----------------------------
        >>> cont = Continuation( solver=scipy.optimize.fsolve )
        
        >>> X, C = cont.nat_continuation(myode, x0, par0, vary_par=0, step_size=0.1, max_steps=100, discretisation=shooting)
        >>> X, C = cont.nat_continuation(myode, x0, par0, vary_par=0, step_size=0.1, max_steps=100, discretisation=None)


        
        '''
        self.solver = solver

    # natural continuation will find the roots of the equation, then increment c and find the roots again
    def nat_continuation(self, ode, x0, p0 , vary_p =0, step = 0.1, max_steps = 100, discret=None):
        '''
        Natural continuation method to increment a parameter and solve the ODE for the new parameter value
        Parameters:
        ----------------------------
        ode - function: 
                the function to be solved (with inputs (y, *args)) 
        x0 - array:
                the initial value of the solution
        p0 - list:
                the initial values of the parameters
        vary_p - int:
                the index of the parameter to vary
        step - float:
                the size of the steps to take
        max_steps - int:
                the number of steps to take
        discret - function:
                the discretisation to use (either shooting_setup or linear (if None))

        Returns:
        ----------------------------
        X - array:
                the solution of the equation for each parameter value
        C - array:
                the parameter values that were used to solve the equation

        
        '''
        # initialize the solution
        X = []
        C = []

        # discretise the ode
        '''
        if the function is an algebraic equation, then the discretisation is just the function itself
        if the function is a differential equation, then the discretisation is calling shooting method
        to make the F(x) = 0 that will be solved by the solver
        
        '''

        param = p0
        T = 0

        # if no discretisation is given, use the linear discretisation
        if discret is None:
            discret = Discretisation().linear

        # check that the discretisation is a function
        if not callable(discret):
            raise TypeError('discretation must be a function or None')

        # discretise the ode - creating the function that will be solved F(x) = 0 with the parameter
        fun = discret(ode, x0, param)

        if discret == Discretisation().shooting_setup:
            u0 = np.append(x0, T)
        else:
            u0 = x0

        # solve the discretised equation
        sol = self.solver(fun, u0, args=param)

        # append the solution and the parameter value to the solution
        try:
            X.append(sol.x)
        except:
            X.append(sol)

        try:
            C.append(param[vary_p])
        except:
            C.append(param)

        num_steps = 0
        # loop with incrementing c until reaching the limit
        while num_steps < max_steps:
            fun = discret(ode, x0, param)
            try:
                param[vary_p] += step
            except:
                param+=step

            sol = self.solver(fun, u0, args=param)
            try:
                X.append(sol.x)
            except:
                X.append(sol)

            try:
                C.append(param[vary_p])
            except:
                C.append(param)

            num_steps += 1

        return X, C

    def ps_arc_continuation(self, ode, x0, p0 , vary_p =0, step = 0.1, max_steps = 100, discret=None):
        '''
        Pseudo arclength continuation method to increment a parameter and solve the ODE for the new parameter value
        Parameters:
        ----------------------------
        ode - function: 
                the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
        x0 - array:
                the initial value of the solution
        p0 - list:
                the initial values of the parameters
        vary_p - int:
                the index of the parameter to vary
        step - float:
                the size of the steps to take
        max_steps - int:
                the number of steps to take
        discret - function:
                the discretisation to use (either shooting_setup or linear (if None))

        Returns:
        ----------------------------
        X - array:
                the solution of the equation for each parameter value
        C - array:
                the parameter values that were used to solve the equation

        
        '''
        # initialize the solution
        X = []
        C = []

        raise NotImplementedError('This method is not yet implemented')


        return X, C



####################### EXAMPLES ############################
if __name__ == '__main__':
    # define the cubic equation
    def cubic(x, args):
        c = args
        return x**3 - x + c

    # define the initial conditions
    x0 = 1

    # define the linear discretisation
    def linear(x, x0, T, args):
        return x 

    print('\nFirst example: cubic equation with linear discretisation')

    cont = Continuation()
    discret = Discretisation()

    # natural continuation with no discretisation
    X, C = cont.nat_continuation(cubic, x0, -2, vary_p = 0, step = 0.1, max_steps = 40, discret=None)

    # print('X = ', X)
    # print('C = ', C)

    # plot the solution
    plt.figure()
    plt.title('Cubic Equation')
    plt.plot(C, X)
    plt.xlabel('parameter c value')
    plt.ylabel('root value')
    plt.grid()
    plt.show()

    # now test natural continuation with a differential equation - Hopf bifurcation
    def hopf(t, X, *args):

        b = args[0][0]

        x = X[0]
        y = X[1]

        dxdt = b*x - y - x*(x**2 + y**2) 
        dydt = x + b*y - y*(x**2 + y**2)

        return np.array([dxdt, dydt])

    def hopf_polar(t, X, *args):

        myu, omega = args[0]

        r = X[0]
        theta = X[1]

        drdt = r*(myu - r**2)
        dthetadt = omega

        return np.array([drdt, dthetadt])

    # define new ode
    a = 1
    d = 0.1
    b = 1.0

    def ode(t, Y, args):

        a, b, d = args
        x,y = Y
        dxdt = x*(1-x) - (a*x*y)/(d+x)
        dydt = b*y*(1- (y/x))

        return np.array([dxdt, dydt])

    # define the initial conditions
    x0 = [0.1, 0.1, 10]

    # define parameter
    p = -1

    print('\nSecond example: Hopf Bifurcation with shooting discretisation')

    # natural continuation with shooting discretisation
    X, C = cont.nat_continuation(hopf, x0, p, vary_p = 0, step = 0.1, max_steps = 20, discret=discret.shooting_setup)

    # split the X into x, y and period at each parameter value
    # print('\nX = ', X)
    # print('\nC = ', C)

    # extract the period (the last element of the solution)
    T = [x[-1] for x in X] 
    # print('\nT = ', T)


    # plot the period
    plt.figure()
    plt.title('Polar Hopf Bifurcation')
    plt.plot(C, T)
    plt.xlabel('parameter myu value')
    plt.ylabel('period value')
    plt.grid()
    plt.show()








