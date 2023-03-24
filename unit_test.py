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
        def test_ode(Y, t, args):
            a, b = args
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
        except ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect f')

        # test for incorrect y0
        print('---------------------test 3---------------------')
        try:
            solve_to(test_ode,[0,0,0], t0, t1, delta_t, method, args = [1,1])
        except ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect y0')

        # test for incorrect t0
        print('---------------------test 4---------------------')
        try:
            solve_to(test_ode,Y0, 't0', t1, delta_t, method, args = [1,1])
        except ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect t0')

        # test for incorrect t1
        print('---------------------test 5---------------------')
        try:
            solve_to(test_ode,Y0, t0, 't1', delta_t, method, args = [1,1])
        except ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect t1')

        # test for incorrect delta_t
        print('---------------------test 6---------------------')
        try:
            solve_to(test_ode,Y0, t0, t1, 'delta_t', method, args = [1,1])
        except ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect delta_t')

        # test for incorrect method
        print('---------------------test 7---------------------')
        try:
            solve_to(test_ode,Y0, t0, t1, delta_t, 'method', args = [1,1])
        except ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect method')

        # test for incorrect args
        print('---------------------test 8---------------------')
        try:
            solve_to(test_ode,Y0, t0, t1, delta_t, method, args = 'args')
        except ValueError:
            pass
        else:
            self.fail('solve_to function failed to raise ValueError with incorrect args')


    def test_solve_to(self):
        '''
        Test the shooting function for incorrect inputs so that
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


    # def test_shooting(self):
    #     '''
    #     Test the shooting function for incorrect inputs so that
    #     the user is notified of which inputs are incorrect.
        
    #     Inputs:
    #     ----------------------------
    #     f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
    #     y0 - array: the initial value of the solution
    #     T - float: an initial guess for the period of the solution

    #     Returns:
    #     ----------------------------
    #     sol - array: the solution: 
    #                 where sol[-1] is the period of the solution
    #                 and sol[:-1] is the initial conditions for a solution with period sol[-1]

    #     '''

    #     # define new periodic test ode
    #     a = 1
    #     d = 0.1
    #     b = 0.1

    #     def test_ode(Y, t, args = (a, b, d)):
    #         a, b, d = args
    #         # print('Y = ', Y)
    #         x, y = Y
    #         return np.array([x*(1-x) - (a*x*y)/(d+x) , b*y*(1- (y/x))])
        

    #     # initial guess
    #     Y0 = [2,3]


    #     print('Testing shooting function...\n')
    #     # test it works with correct inputs
    #     # test id:
    #     print('---------------------test 1---------------------')
    #     try:
    #         shooting(test_ode,Y0, 20)
    #     except:
    #         self.fail('shooting function failed with correct inputs')

    #     print('---------------------test 2---------------------')
    #     # test f
    #     with self.assertRaises(ValueError, msg='f is not a function'):
    #         shooting(1, Y0, 20)

    #     print('---------------------test 3---------------------')
    #     # test y0
    #     with self.assertRaises(ValueError, msg='y0 is not an array'):
    #         shooting(test_ode, 1, 20)

    #     print('---------------------test 4---------------------')
    #     # test T
    #     with self.assertRaises(ValueError, msg='T is not a float'):
    #         shooting(test_ode, Y0, '20')

    #     print('---------------------test 5---------------------')
    #     # test that the output is correct
    #     sol = shooting(test_ode, Y0, 20)

    #     # check output type
    #     self.assertIsInstance(sol, (np.ndarray, list), msg='The solution is not a list or array')

    #     # check that the final time is correct
    #     self.assertAlmostEqual(sol[-1], 34.118, places=2, msg='The period is not correct')

    #     print('---------------------test 6---------------------')
    #     # check that the solution is a periodic solution
    #     Y, t = solve_to(test_ode, sol[:-1], 0, sol[-1], 0.01, 'Euler')
    #     self.assertAlmostEqual(Y[-1, 0], Y[0, 0], places=2, msg='The solution is not a periodic solution')
    #     self.assertAlmostEqual(Y[-1, 1], Y[0, 1], places=2, msg='The solution is not a periodic solution')

    #     print('---------------------test 7---------------------')
    #     # test shooting with Hopf bifurcation a = -1
    #     # params:
    #     a = -1
    #     b = 1
    #     def test_ode(Y, t, args = (a,b)):
    #         a, b = args
    #         x, y = Y
    #         dxdt = b*x - y + a*x*(x**2 + y**2)
    #         dydt = x + b*y + a*y*(x**2 + y**2)
    #         return np.array([dxdt, dydt])
        
    #     # define true solution
    #     def true_sol(t, args = (a,b)):
    #         a, b = args
    #         x_t = np.sqrt(b) * np.cos(t)
    #         y_t = np.sqrt(b) * np.sin(t)

    #         return np.array([x_t, y_t])
        
    #     # initial guess
    #     Y0 = [1,1]
    #     T = 20

    #     # test it works with correct inputs
    #     try:
    #         sol = shooting(test_ode,Y0, T)
    #         print('Solution to Bifurcation: ' , sol)
    #         # test that the solution is correct
    #         Y, t = solve_to(test_ode, sol[:-1], 0, sol[-1], 0.01, 'RK4')

    #         Y_true = true_sol(t[-1])

    #         self.assertAlmostEqual(abs(Y[-1, 0]), abs(Y_true[0]), places=2, msg='The solution is not correct')
    #         self.assertAlmostEqual(abs(Y[-1, 1]), abs(Y_true[1]), places=2, msg='The solution is not correct')

    #     except:
    #         self.fail('shooting function failed with correct inputs')

    #     print('---------------------test 8---------------------')

    #     # test incorrect dimension of y0
    #     with self.assertRaises(ValueError, msg='y0 is not the correct dimension'):

    #         shooting(test_ode, np.array([1,1,1]), 20)

        


if __name__ == '__main__':
    unittest.main()
