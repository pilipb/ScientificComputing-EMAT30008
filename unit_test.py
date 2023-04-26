from solvers import *
from solve_to import solve_to
from shooting import *
import unittest 
import numpy as np

class solver_test(unittest.TestCase):

    def test_solve_to(self):
        '''
        Test the solve_to function for incorrect inputs so that
        the user is notified of which inputs are incorrect.

        Inputs:
        ----------------------------
        f - function: the function to be integrated (with inputs (t,Y, args)) in first order form of n dimensions
        y0 - array: the initial value of the solution
        t0 - float: the initial value of time
        t1 - float: the end time
        delta_t - float: the step size
        method - string: the method to be used to solve the ODE (Euler, RK4, Heun)
        args - array: the arguments to be passed to the function f or None if no arguments are to be passed

        Returns:
        ----------------------------
        Y - array: the solution 
        t - float: the next time step
    
        '''
        
        # define a simple test ode
        def test_ode(t, Y , *args):
            a, b = args[0]
            x, y = Y
            return np.array([a*x, b*y])
        
        # initial guess
        Y0 = [1,1]
        t0 = 0
        t1 = 10
        delta_t = 0.1
        method = 'Euler'

        # test it works with correct inputs
        try:
            solve_to(test_ode,Y0, t0, t1, delta_t, method, args = [1,1])
        except:
            self.fail('solve_to function failed with correct inputs')
        
        # test for incorrect f
        try:
            solve_to('test_ode',Y0, t0, t1, delta_t, method, args = [1,1])
        except TypeError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect f')

        # test for incorrect y0
        try:
            solve_to(test_ode,[0,0,0], t0, t1, delta_t, method, args = [1,1])
        except ValueError or TypeError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect y0')

        # test for incorrect t0
        try:
            solve_to(test_ode,Y0, 't0', t1, delta_t, method, args = [1,1])
        except TypeError or ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect t0')

        # test for incorrect t1
        try:
            solve_to(test_ode,Y0, t0, 't1', delta_t, method, args = [1,1])
        except TypeError or ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect t1')

        # test for incorrect delta_t
        try:
            solve_to(test_ode,Y0, t0, t1, 'delta_t', method, args = [1,1])
        except TypeError or ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect delta_t')

        # test for incorrect method
        try:
            solve_to(test_ode,Y0, t0, t1, delta_t, 12, args = [1,1])
        except TypeError or ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect method')

    def test_shooting_setup(self):

        def hopf(t, X, *args):

            b = args[0][0]

            x = X[0]
            y = X[1]

            dxdt = b*x - y - x*(x**2 + y**2)
            dydt = x + b*y - y*(x**2 + y**2)

            return np.array([dxdt, dydt])


        # initial guess
        Y0 = [0.1,0.1, 10]
        b = 1
        
        discret = Discretisation()
        # solve the ode using the shooting method
        fun = discret.shooting_setup(hopf, Y0, (b,))
        assert callable(fun)

        u0, T0 = shooting_solve(fun, Y0, b) 
        assert len(u0) == 2
        assert isinstance(T0, float)
    
    # Define a sample ODE to use for testing
    

def test_continuation(self):
    # define the cubic equation
    def cubic(x, args):
        c = args
        return x**3 - x + c

    from continuation import Continuation
    from discretisation import Discretisation
    cont = Continuation()
    discret = Discretisation()


    # now test natural continuation with a differential equation - Hopf bifurcation
    def hopf(t, X, *args):

        # print(X)
        try:
            b = args[0][0]
        except:
            b = args[0]

        # print('b = ' + str(b) + ' x = ' + str(X))

        x = X[0]
        y = X[1]

        dxdt = b*x - y - x*(x**2 + y**2) 
        dydt = x + b*y - y*(x**2 + y**2)

        return np.array([dxdt, dydt])


    # define the initial conditions
    x0 = [0.1,0.1]

    # define parameter
    p0 = 0

    # natural continuation 
    X, C = cont.nat_continuation(hopf, x0, 0, vary_p = 0, step = 0.1, max_steps = 2, discret=discret.shooting_setup)

    assert len(X) == len(C)
    assert len(X) == 2

    assert isinstance(X[0], np.ndarray)
    
    # pseudo arc length continuation
    X, C = cont.pseudo_arc_length(hopf, x0, 0, vary_p = 0, step = 0.1, max_steps = 4, discret=discret.shooting_setup)

    assert len(X) == len(C)
    assert len(X) == 4

    assert isinstance(X[0], np.ndarray)



        

if __name__ == '__main__':
    unittest.main()
