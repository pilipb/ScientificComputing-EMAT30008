import numpy as np

def euler_step(f, y0, t0, delta_t, args):

    '''
    The steps will make an integration step of size delta_t 
    using the method specified by the function step.

    parameters:
    ---------------------------
    f - function: the function to be integrated (with inputs (Y,t, args)) in first order form of n dimensions
    y0 - array: the initial value of the solution
    t0 - float: the initial value of time
    delta_t - float: the step size
    args - array: the arguments to be passed to the function f
            or None if no arguments are to be passed

    returns:
    ---------------------------
    y1 - array: the solution at the next step time step
    t1 - float: the next time step

    '''
    # run error check
    error_check(f, y0, t0, delta_t, args = args)

    y1 = y0 + delta_t * f(y0, t0, args)
    t1 = t0 + delta_t
    return y1, t1

# RK4 step - generalised to any number of dimensions
def rk4_step(f, y0, t0, delta_t, args):

    '''
    The steps will make an integration step of size delta_t 
    using the method specified by the function step.

    parameters:
    ---------------------------
    f - function: the function to be integrated (with inputs (Y,t, args)) in first order form of n dimensions
    t0 - float: the initial value of time    
    y0 - array: the initial value of the solution
    delta_t - float: the step size
    args - array: the arguments to be passed to the function f
            or None if no arguments are to be passed

    returns:
    ---------------------------
    y1 - array: the solution at the next step time step
    t1 - float: the next time step

    '''
    # run error check
    # error_check(f, y0, t0, delta_t, args = args)


    k1 = delta_t * f( t0, y0, args)
    k2 = delta_t * f( t0 + delta_t/2, y0 + k1/2, args)  
    k3 = delta_t * f( t0 + delta_t/2, y0 + k2/2, args)
    k4 = delta_t * f( t0 + delta_t, y0 + k3, args)
    y1 = y0 + (k1 + 2*k2 + 2*k3 + k4)/6
    t1 = t0 + delta_t

    return np.array(y1), t1


# Heuns method - generalised to any number of dimensions
def heun_step(f, y0, t0, delta_t, args):

    '''
    The steps will make an integration step of size delta_t 
    using the method specified by the function step.

    parameters:
    ---------------------------
    f - function: the function to be integrated (with inputs (Y,t, args)) in first order form of n dimensions
    y0 - array: the initial value of the solution
    t0 - float: the initial value of time
    delta_t - float: the step size
    args - array: the arguments to be passed to the function f
            or None if no arguments are to be passed

    returns:
    ---------------------------
    y1 - array: the solution at the next step time step
    t1 - float: the next time step

    '''
    # run error check
    error_check(f, y0, t0, delta_t, args = args)

    k1 = f(y0, t0, args)
    k2 = f(y0 + delta_t * k1, t0 + delta_t, args)
    y1 = y0 + delta_t/2 * (k1 + k2)
    t1 = t0 + delta_t
    return y1, t1

# error checking function for the steps
def error_check(f, y0, t0, delta_t, t1=None, method=None, args=None):
    if not callable(f):
        raise TypeError('f must be a function')
    # if not isinstance(y0, (np.ndarray, list)):
    #     raise ValueError('y0 must be a numpy array or list')
    # if not isinstance(t0, (int, float)):
    #     raise ValueError('t0 must be a number')
    if not isinstance(delta_t, (int, float)):
        raise TypeError('delta_t must be a number')
    if t1 is not None:
        if not isinstance(t1, (int, float)):
            raise ValueError('t1 must be a number')
        elif t1 < t0:
            raise ValueError('t1 must be greater than t0')
    if method is not None:
        if not isinstance(method, str):
            raise TypeError('method must be a string')
    if args is not None:
        if not isinstance(args, (np.ndarray, list, tuple)):
            raise TypeError('args must be a numpy array or list')
        
        

