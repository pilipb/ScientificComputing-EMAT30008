from solvers import *
from solve_to import solve_to
from shooting import shooting
import unittest 
import numpy as np

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
