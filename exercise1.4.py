
import numpy as np
from scipy.special import erfc
from numba import njit

@njit  # Accelerate computation with Numba
def jacobi_iteration(c_old, max_iter, tol):
    """
    Solves the Laplace equation using the Jacobi iterative method.
    
    Parameters:
    c_old (numpy.ndarray): Initial grid values, including boundary conditions.
    max_iter (int): Maximum number of iterations.
    tol (float): Convergence tolerance.
    
    Returns:
    numpy.ndarray: The steady-state solution.
    """
    c_new = c_old.copy()  # Copy the initial grid to store updated values
    m, n = c_old.shape  # Get the number of rows and columns

    for iteration in range(max_iter):
        # Update the grid using the Jacobi formula: 
        # c_new(i,j) = 1/4 * (c_old(i+1,j) + c_old(i-1,j) + c_old(i,j+1) + c_old(i,j-1))
        for i in range(1, m-1):
            for j in range(1, n-1):
                c_new[i, j] = 0.25 * (c_old[i-1, j] + c_old[i+1, j] + c_old[i, j-1] + c_old[i, j+1])

        # Check for convergence
        diff = np.max(np.abs(c_new - c_old))
        if diff < tol:
            print(f'Converged in {iteration + 1} iterations.')
            break
        
        c_old[:] = c_new  # Update c_old with new values for the next iteration
    
    return c_new





##1.4
N = 150
# eps = 1e-5
# max_iter = int(2e4)
# omegas = np.linspace(1.7, 2, 100)
# Ns = np.linspace(10, 200, 20)

# per row: i_min, i_max, j_min, j_max # c_old= np.array([[18, 24, 47, 53]])*N/100 # # two_square = np.array([[18, 24, 30, 36],
#                        [18, 24, 63, 69]])*N/100

# three_square = np.array([[18, 24, 22, 28],
#                          [18, 24, 47, 53],
#                          [18, 24, 72, 78]])*N/100


j=jacobi_iteration(c_old, max_iter=1000, tol=1e-5)
print(j)···