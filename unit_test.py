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
        # test f
        with self.assertRaises(ValueError):
            solve_to(1, [0], 0, 10, 0.01, 'Euler')

        # test y0
        with self.assertRaises(ValueError):
            solve_to(lambda x, y: x, 1, 0, 10, 0.01, 'Euler')

        # test t0
        with self.assertRaises(ValueError):
            solve_to(lambda x, y: x, [0], '0', 10, 0.01, 'Euler')

        # test t1
        with self.assertRaises(ValueError):
            solve_to(lambda x, y: x, [0], 0, '10', 0.01, 'Euler')

        # test delta_t
        with self.assertRaises(ValueError):
            solve_to(lambda x, y: x, [0], 0, 10, '0.01', 'Euler')

        # test method
        with self.assertRaises(ValueError):
            solve_to(lambda x, y: x, [0], 0, 10, 0.01, 'Eulerian')

    # def test_shooting(self):





if __name__ == '__main__':
    unittest.main()
