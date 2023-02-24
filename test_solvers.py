from solvers import *

def test_euler():
    step = 3
    def func(x, t):
        return x
    x1, t1 = euler_step(func, 0,0,step)
    assert t1 == step

def test_rk4():
    step = 3
    def func(x, t):
        return x
    x1, t1 = rk4_step(func, 0,0,step)
    assert t1 == step

def test_heun():
    step = 3
    def func(x, t):
        return x
    x1, t1 = heun_step(func, 0,0,step)
    assert t1 == step