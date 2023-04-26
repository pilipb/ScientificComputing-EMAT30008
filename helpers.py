import numpy as np

def boundary(alpha, beta, N, dx, type, sparse=False):
    '''
    This function returns the boundary condition matrix and vector for a given boundary condition type

    Parameters
    ----------
    alpha : float
        the left boundary
    beta : float
        the right boundary
    N : int
        the number of grid points
    dx : float
        the step size
    type : string
        the type of boundary condition
        options: 'DD', 'DN', 'DR', 'ND', 'NN', 'NR','RD', 'RN', 'RR'

    Returns
    -------
    A : array
        the matrix size (N-1, N-1)
    b : array
        the boundary condition vector, len = N-1

    '''
    # check the type of boundary condition is valid
    if type not in ['DD', 'DN', 'DR', 'ND', 'NN', 'NR','RD', 'RN', 'RR']:
        raise ValueError('Invalid boundary condition type')
    
    # make basic A tri-diagonal matrix
    A = np.zeros((N-1, N-1))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:,1:], 1)
    np.fill_diagonal(A, -2)

    # make the b vector
    b = np.zeros(N-1)

    # check which type of boundary condition for the first point
    if type[0] == 'D':
        b[0] = alpha
    elif type[0] == 'N':
        b[0] = 2*alpha*dx
        A[0, 1] = 2
    elif type[0] == 'R':
        b[0] = 2*alpha*dx
        A[0, 1] = -2*(1+alpha*dx)/dx

    # check which type of boundary condition for the last point
    if type[1] == 'D':
        b[-1] = beta
    elif type[1] == 'N':
        b[-1] = 2*beta*dx
        A[-2, -1] = 2
    elif type[1] == 'R':
        b[-1] = 2*beta*dx
        A[-2, -1] = -2*(1+beta*dx)/dx

    if sparse:
        import scipy.sparse as sparse
        A = sparse.csr_matrix(A)

    return A, b


def plot_help(plt, xlabel, ylabel, title=None, legend=False):
    '''
    This function is a helper function for plotting
    '''
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)
    
    if legend:
        plt.legend()

    plt.grid()
    plt.show()

# compare the implementation of crank-nicolson with sparse and dense matrices
import cProfile, pstats
def help_prof(N, sparse, Solver, pde):
    '''
    Profiling all solvers in the pde_solver.py file
    
    '''
    pr = cProfile.Profile()
    pr.enable()
    solver = Solver(pde, N=N, t_final=1, method='crank_nicolson', sparse=sparse)
    u = solver.solve()
    try:
        nbytes = solver.A_mat.nbytes
    except:
        nbytes = solver.A_mat.data.nbytes
    pr.disable()
    stats = pstats.Stats(pr)
    return stats, nbytes