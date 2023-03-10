from solvers import *
from solve_to import solve_to
from shooting import shooting
import unittest 
import numpy as np

'''
Test the user interface functions, solve_to and shooting, for incorrect inputs so that
the user is notified of which inputs are incorrect.

Test all the possible inputs for the solve_to function:
    - f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
    - y0 - array: the initial value of the solution
    - t0 - float: the initial value of time
    - t1 - float: the end time
    - delta_t - float: the step size
    - method - string: the method to be used to solve the ODE (Euler, RK4, Heun)

and the same for the shooting function


'''

class solver_test(unittest.TestCase):
    # test solve_to
    def test_solve_to(self):
        '''
        Test the solve_to function for incorrect inputs so that
        the user is notified of which inputs are incorrect.

        Inputs:
        ----------------------------
        f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
        y0 - array: the initial value of the solution
        t0 - float: the initial value of time
        t1 - float: the end time
        delta_t - float: the step size
        method - string: the method to be used to solve the ODE (Euler, RK4, Heun)

        Returns:
        ----------------------------
        Y - array: the solution 
        t - float: the next time step
    
        '''

        # test that correct inputs work
        try:
            solve_to(lambda x, y: x, np.ndarray([0]), 0, 10, 0.01, 'Euler')
        except:
            self.fail('solve_to function failed with correct inputs')

        # test f
        with self.assertRaises(ValueError):
            solve_to(1, np.ndarray([0]), 0, 10, 0.01, 'Euler')

        # test y0
        with self.assertRaises(ValueError):
            solve_to(lambda x, y: x, 1, 0, 10, 0.01, 'Euler')

        # test t0
        with self.assertRaises(ValueError):
            solve_to(lambda x, y: x, np.ndarray([0]), '0', 10, 0.01, 'Euler')

        # test t1
        with self.assertRaises(ValueError):
            solve_to(lambda x, y: x, np.ndarray([0]), 0, '10', 0.01, 'Euler')

        # test delta_t
        with self.assertRaises(ValueError):
            solve_to(lambda x, y: x, np.ndarray([0]), 0, 10, '0.01', 'Euler')

        # test method
        with self.assertRaises(ValueError):
            solve_to(lambda x, y: x, np.ndarray([0]), 0, 10, 0.01, 'Eulerian')


        # test solve_to outputs
        def test_ode(y, t):
            return np.array([y[1], -y[0]])
        
        # test that the output is correct
        Y, t = solve_to(test_ode, np.array([1, 0]), 0, 2*np.pi, 0.01, 'Euler')

        # check output type
        self.assertIsInstance(Y, (np.ndarray, list), msg='The solution is not a list or array')
        self.assertIsInstance(t, (list, np.ndarray), msg='The time is not a list or array')

        # check that the final time is correct
        self.assertAlmostEqual(t[-1], 2*np.pi, places=3, msg='The final time is not correct')

        # check that the solution is correct
        self.assertAlmostEqual(Y[-1, 0], 1.0319, places=3, msg='The solution is not correct')




        


    def test_shooting(self):
        '''
        Test the shooting function for incorrect inputs so that
        the user is notified of which inputs are incorrect.
        
        Inputs:
        ----------------------------
        f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
        y0 - array: the initial value of the solution
        T - float: an initial guess for the period of the solution

        Returns:
        ----------------------------
        sol - array: the solution: 
                    where sol[-1] is the period of the solution
                    and sol[:-1] is the initial conditions for a solution with period sol[-1]

        '''

        # define new periodic test ode
        a = 1
        d = 0.1
        b = 0.1

        def test_ode(Y, t, args = (a, b, d)):
            a, b, d = args
            # print('Y = ', Y)
            x, y = Y
            return np.array([x*(1-x) - (a*x*y)/(d+x) , b*y*(1- (y/x))])
        

        # initial guess
        Y0 = [2,3]



        # test it works with correct inputs
        try:
            shooting(test_ode,Y0, 20)
        except:
            self.fail('shooting function failed with correct inputs')

        # test f
        with self.assertRaises(ValueError, msg='f is not a function'):
            shooting(1, Y0, 20)

        # test y0
        with self.assertRaises(ValueError, msg='y0 is not an array'):
            shooting(test_ode, 1, 20)

        # test T
        with self.assertRaises(ValueError, msg='T is not a float'):
            shooting(test_ode, Y0, '20')

        # test that the output is correct
        sol = shooting(test_ode, Y0, 20)

        # check output type
        self.assertIsInstance(sol, (np.ndarray, list), msg='The solution is not a list or array')

        # check that the final time is correct
        self.assertAlmostEqual(sol[-1], 34.118, places=2, msg='The period is not correct')

        # check that the solution is a periodic solution
        Y, t = solve_to(test_ode, sol[:-1], 0, sol[-1], 0.01, 'Euler')
        self.assertAlmostEqual(Y[-1, 0], Y[0, 0], places=2, msg='The solution is not a periodic solution')
        self.assertAlmostEqual(Y[-1, 1], Y[0, 1], places=2, msg='The solution is not a periodic solution')

if __name__ == '__main__':
    unittest.main()
