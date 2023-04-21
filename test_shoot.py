# pal shooting:
import numpy as np
from solve_to import solve_to
import scipy.optimize as scipy
import matplotlib.pyplot as plt


def test_shooting(f, y0, T0, args = None):

    '''
    Implementing a numerical shooting method to solve an ODE to find a periodic solution
    This method will solve the BVP ODE using root finding method to find the limit cycle, using scipy.optimize.fsolve to
    find the initial conditions and period that satisfy the boundary conditions.

    parameters:
    ----------------------------
    f - function: the function to be integrated (with inputs (Y,t)) in first order form of n dimensions
    y0 - array: the initial value of the solution
    T - float: an initial guess for the period of the solution
    args - array: the arguments for the function f

    returns:
    ----------------------------
    sol - array: the initial conditions that cause the solution to be periodic: sol = [x0, y0, ... , T]

    '''

    # unpack the initial conditions and period guess
    T_guess = T0
    p_guess = args

    print('\nPseudo-arclength continuation method\n')

    # define the function that will be solved for the initial conditions and period
    def fun(initial_vals):
        print(initial_vals)

        # unpack the initial conditions and period guess
        p0 = initial_vals[-1]
        T = initial_vals[-2]
        y0 = initial_vals[:-2] 

        Y , _ = solve_to(f, y0, 0, T, 0.01, 'RK4', args=p0)

        num_dim = len(y0)
        row = np.zeros(num_dim)

        row = Y[-1,:] - y0
  
        row = np.append(row, f(0, Y[0,:], args)[0])


        output = np.array(row)

        # making the pseudo-arclength condition
        # (v_i+1 - v_pred_i+1) dot sec = 0

        # u1 - u0
        secant = Y[-1] - y0

        y,t = solve_to(f, Y[-1], 0, T, 0.01, 'RK4', args=p0) 
        vi1 = y[-1]

        vi1_pred = Y[-1] + secant

        # (v_i+1 - v_pred_i+1) dot sec = 0
        output = np.append(output, np.dot(vi1 - vi1_pred, secant))

        return output

    # solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
    y0 = np.append(y0, T_guess)
    y0 = np.append(y0, p_guess)

    sol = scipy.fsolve(fun, y0)

    # return the period and initial conditions that cause the limit cycle: sol = [x0, y0, ... , T, p]
    return sol[:-2], sol[-2], sol[-1]



#### TEST ####

# define the function to be integrated
def f(t, Y, *args):
    # unpack the arguments
    try:
        b = float(args)
    except:
        b = float(args[0])

    # unpack the variables
    u1, u2 = Y

    # define the partial derivatives
    u1p = u1*(1 - u1) - ((1*u1*u2)/(0.1 + u1))
    u2p = b*u2 *(1 - (u2/u1))

    # return the derivatives
    return np.array([u1p, u2p])
 
# define the parameters
b = 0.1 # the parameter that determines the stability of the limit cycle

# define the initial conditions
u10 = 1
u20 = 1
T = 20

# define the initial conditions
u0 = np.array([u10, u20])

# define the arguments
args = b

# solve the function
y0 = test_shooting(f, u0, T, args)

T = y0[-1]
y0 = y0[:-2]

# solve the system of equations for the initial conditions [x0, y0, ... ] and period T that satisfy the boundary conditions
Y, t = solve_to(f, y0, 0, T, 0.01, 'RK4', args=args)

# plot the solution
plt.plot(t, Y[:,0])
plt.plot(t, Y[:,1])
plt.title('T = ' + str(T))
plt.show()

# However it is difficult to find correct initial conditions for the shooting method
# so continuation methods can be used
