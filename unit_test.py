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
        f - function: the function to be integrated (with inputs (Y,t, args)) in first order form of n dimensions
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

        print('Testing solve_to function...\n')
        # test it works with correct inputs
        print('---------------------test 1---------------------')
        try:
            solve_to(test_ode,Y0, t0, t1, delta_t, method, args = [1,1])
        except:
            self.fail('solve_to function failed with correct inputs')
        
        # test for incorrect f
        print('---------------------test 2---------------------')
        try:
            solve_to('test_ode',Y0, t0, t1, delta_t, method, args = [1,1])
        except TypeError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect f')

        # test for incorrect y0
        print('---------------------test 3---------------------')
        try:
            solve_to(test_ode,[0,0,0], t0, t1, delta_t, method, args = [1,1])
        except ValueError or TypeError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect y0')

        # test for incorrect t0
        print('---------------------test 4---------------------')
        try:
            solve_to(test_ode,Y0, 't0', t1, delta_t, method, args = [1,1])
        except TypeError or ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect t0')

        # test for incorrect t1
        print('---------------------test 5---------------------')
        try:
            solve_to(test_ode,Y0, t0, 't1', delta_t, method, args = [1,1])
        except TypeError or ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect t1')

        # test for incorrect delta_t
        print('---------------------test 6---------------------')
        try:
            solve_to(test_ode,Y0, t0, t1, 'delta_t', method, args = [1,1])
        except TypeError or ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect delta_t')

        # test for incorrect method
        print('---------------------test 7---------------------')
        try:
            solve_to(test_ode,Y0, t0, t1, delta_t, 12, args = [1,1])
        except TypeError or ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect method')





if __name__ == '__main__':
    unittest.main()
