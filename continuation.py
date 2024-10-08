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
        step_size = step
        # initialize the solution
        X = []
        C = []

        T = 1

        # create step array - len = parameter length
        if not isinstance(p0, list):
            step = step_size
        else:
            step = np.zeros(len(p0))
            step[vary_p] = step_size

        # if no discretisation is given, use the linear discretisation
        if discret is None:
            discret = Discretisation().linear
        else:
            x0 = np.append(x0, T)

        # check that the discretisation is a function
        if not callable(discret):
            raise TypeError('discretation must be a function or None')

        # discretise the ode - creating the function that will be solved F(x) = 0 with the parameter
        fun = discret(ode, x0, (p0,))

        sol = self.solver(fun, x0, args=p0)
        try:
            if sol.success == False:
                raise ValueError('solver failed to find a solution')
        except:
            pass

        # append the solution and the parameter value to the solution
        try:
            X.append(sol.x)
        except:
            X.append(sol)

        try:
            C.append(p0[vary_p])
        except:
            C.append(p0)

        num_steps = 0
        # loop with incrementing c until reaching the limit
        while num_steps < max_steps:

            fun = discret(ode, x0, p0)

            # increment the parameter
            p0 += step

            sol = self.solver(fun, x0, args=p0)
            try:
                if sol.success == False:
                    raise ValueError('solver failed to find a solution')
            except:
                pass
            
            try:
                X.append(sol.x)
            except:
                X.append(sol)

            try:
                C.append(p0[vary_p])
            except:
                C.append(p0)

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
                the solution of the equation for each parameter value [x1, x2, ..., T, [p1, p2, ...]]
        C - array:
                the parameter values that were used to solve the equation

        
        ''' 
        step_size = step
        # initialize the solution
        X = []
        C = []

        # create step array - len = parameter length
        if not isinstance(p0, list):
            step = step_size
        else:
            step = np.zeros(len(p0))
            step[vary_p] = step_size

        if discret is None:
            discret = Discretisation().linear

        # find one solution using natural continuation
        x,c = self.nat_continuation(ode, x0, p0, max_steps = 2, discret = discret)
        # x = np.append(x, step)
        X.append(x[-1])
        C.append(c[-1])
        
        # add step to the parameter
        p0 += step

        # find a second solution using natural continuation
        x,c = self.nat_continuation(ode, x0, p0, max_steps = 2, discret= discret)
        # x = np.append(x, step)
        X.append(x[-1])
        C.append(c[-1])

        # find the change in the solution
        delta_u = X[-1] - X[-2]
        delta_u = np.append(delta_u, step) # change in u and change in p

        # add step to the parameter
        p0 += step

        # predict the next solution as the same x values but with the new parameter
        pred_u = X[-1] #+ delta_u
        pred_u = np.append(pred_u,p0)

        # define the function to be solved (this will be F(x) = 0) > will be located in discretisation in future
        def fun(u):

            # unpack the initial conditions and period guess
            p0 = u[-1]
            T = u[-2]
            y0 = u[:-2] 

            Y , _ = solve_to(ode, y0, 0, T, 0.01, 'RK4', args=p0)

            # limit cycle condition (last point - initial point = 0)
            row = Y[-1,:] - y0[:]
    
            # phase condition
            row = np.append(row, ode(0, Y[0,:], p0)[0]) # dx/dt(0) = 0 

            # arc length condition difference between u1 and u2 dot the difference between the predicted and actual solution
            cond = np.dot(u - pred_u, delta_u)

            row = np.append(row, cond)

            return row
        
        # solve the function
        sol = self.solver(fun, pred_u)
        

        # append the solution and the parameter value to the solution
        try:
            X.append(sol.x)
        except:
            X.append(sol)

        try:
            C.append(p0[vary_p])
        except:
            C.append(p0)

        num_steps = 0
        # loop with incrementing c until reaching the limit
        while num_steps < max_steps:

            # calculate the delta u - accounting for the first where there is no parameter 
            try:
                delta = X[-1] - X[-2]
            except:
                delta = X[-1] - np.append(X[-2], p0 - step)

            # predict the next solution as the same x values but with the new parameter at first and then just change the parameter
            X_ = X[-1] + delta
            pred_u = X_

            # define the function to be solved
            sol = self.solver(fun, pred_u)

            # append the solution and the parameter value to the solution
            try:
                X.append(sol.x)
            except:
                X.append(sol)

            try:
                C.append(p0[vary_p])
            except:
                C.append(p0)

            # add step to the parameter
            p0 += step

            num_steps += 1


        return X[2:], C[2:]












