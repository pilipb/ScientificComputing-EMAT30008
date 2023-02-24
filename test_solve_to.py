from solve_to import solve_to
import unittest as ut
import numpy as np

step = 3
delta_t = 0.1
def func(x, t):
    return x

def test_solve_to_step():
    x1, t1 = solve_to(func, 0,0,step,delta_t, method='Euler')
    assert round(t1[-1] -  step, 1) ==0

def test_solve_to_output():
    x1, t1 = solve_to(func, 0,0,step,delta_t, method='Euler')
    assert type(x1) == np.ndarray
    assert type(t1) == np.ndarray

    