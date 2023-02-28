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
    def test_euler(self):
        step = 3
        def func(x, t):
            return x
        x1, t1 = euler_step(func, 0,0,step)
        self.assertEqual(t1, step)

    def test_rk4(self):
        step = 3
        def func(x, t):
            return x
        x1, t1 = rk4_step(func, 0,0,step)
        self.assertEqual(t1, step)

    def test_heun(self):
        step = 3
        def func(x, t):
            return x
        x1, t1 = heun_step(func, 0,0,step)
        self.assertEqual(t1, step)

if __name__ == '__main__':
    unittest.main()
