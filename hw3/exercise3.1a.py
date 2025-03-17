import numpy as np
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg

# Define matrix M: n^2 * n^2
# 1. Square
def create_square_matrix(N, L):
    h = L / (N + 1)
    size = N * N
    M = sp.lil_matrix((size, size))
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            if i > 0:
                M[idx, idx - N] = 1 / h**2
            if i < N - 1:
                M[idx, idx + N] = 1 / h**2
            if j > 0:
                M[idx, idx - 1] = 1 / h**2
            if j < N - 1:
                M[idx, idx + 1] = 1 / h**2
            M[idx, idx] = -4 / h**2
    return M.tocsr()

# 2. Rectangle boundary
def create_rectangle_matrix(N, L):
    h = L / (N + 1)
    size = N * (2 * N)
    M = sp.lil_matrix((size, size))
    for i in range(N):
        for j in range(2 * N):
            idx = i * (2 * N) + j
            if i > 0:
                M[idx, idx - (2 * N)] = 1 / h**2
            if i < N - 1:
                M[idx, idx + (2 * N)] = 1 / h**2
            if j > 0:
                M[idx, idx - 1] = 1 / h**2
            if j < 2 * N - 1:
                M[idx, idx + 1] = 1 / h**2
            M[idx, idx] = -4 / h**2
    return M.tocsr()
# Define matrix for rectangle and circle

# 3. Circle boundary
def create_circle_matrix(N, L):
    # h = L / (N + 1)
    size = N * N
    M = np.zeros((size, size))
    xc, yc = L / 2, L / 2  # Center coordinates of the circle
    valid_indices = []  # Store indices of points inside the circle

    # Traverse all points and remove points outside the circle
    for i in range(N):
        for j in range(N):
            x = (i + 1) * h  # Grid point coordinates
            y = (j + 1) * h
            r = np.sqrt((x - xc)**2 + (y - yc)**2)
            if r <= L / 2:  # Point is inside the circle
                valid_indices.append(i * N + j)

    # Build the matrix M for the circle region
    M_circle = np.zeros((len(valid_indices), len(valid_indices)))
    for idx1, i in enumerate(valid_indices):
        for idx2, j in enumerate(valid_indices):
            if i == j:
                M_circle[idx1, idx2] = -4 / h**2  # Current point
            elif abs(i - j) == 1 or abs(i - j) == N:
                M_circle[idx1, idx2] = 1 / h**2  # Neighboring point
    return M_circle, valid_indices

# Restore frequencies
def restore_frequencies(N, valid_indices, eigenvectors):
    num_modes = eigenvectors.shape[1]
    restored_modes = np.zeros((N, N, num_modes))  # Create a 3D matrix of size `N × N × k`

    for i in range(num_modes):
        restored_matrix = np.zeros((N, N))  # Restore a single mode
        for idx, valid_idx in enumerate(valid_indices):
            row, col = divmod(valid_idx, N)  # Calculate original (row, col)
            restored_matrix[row, col] = np.real(eigenvectors[idx, i])  # Fill the eigenmode
        restored_modes[:, :, i] = restored_matrix
    return restored_modes

# Solve the eigenvalue problem
def solve_eigenproblem(M, k=4):
    # eigenvalues, eigenvectors = eigh(M)
    # eigenvalues, eigenvectors = eig(M)
    eigenvalues, eigenvectors = eigs(M, k=k, which='SM')  # Return the smallest k eigenvalues
    return eigenvalues, eigenvectors

# Define eigenvalues and draw the heat map
# Plot eigenmodes
def plot_eigenmode(ax, reshaped_eigenvector, title):
    '''
    Input: eigenvector of length A*B reshaped to matrix A*B
    Length N^2

    Plot the eigenmode on the specified subplot ax
    '''
    im = ax.imshow(reshaped_eigenvector, cmap='viridis', extent=[0, reshaped_eigenvector.shape[1], 0, reshaped_eigenvector.shape[0]])
    ax.set_title(title)
    return im  # Return the imshow object for adding colorbar later

# Main program
if __name__ == "__main__":
    N = 50  # Number of grid points (internal points)
    L = 1.0  # Side length of the square
    h = L / (N + 1)  # Grid spacing
    M_square = create_square_matrix(N, L)
    # print(M_square)

    # 1. Square boundary
    M_square = create_square_matrix(N, L)
    eigenvalues_square, eigenvectors_square = solve_eigenproblem(M_square)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Create 3x1 subplots
    for i in range(3):
        #eigenvector = eigenvectors_square[:, i]  # The i-th eigenvector
        #print("vector", eigenvectors_square[i, :])
        #reshaped_eigenvector = eigenvectors_square[:, i].reshape((N, -1))
        reshaped_eigenvector = np.real(eigenvectors_square[:, i].reshape(N, N))
        lambda_value = np.sqrt(abs(eigenvalues_square[i])).real
        title = f"Square, λ = {lambda_value:.2f}"

        im = plot_eigenmode(axes[i], reshaped_eigenvector, title)  # Pass the subplot object
    fig.colorbar(im, ax=axes.ravel(), orientation='horizontal', shrink=0.5)

    ######

    plt.show()

    # 2. Rectangle boundary
    M_rectangle = create_rectangle_matrix(N, L)

    eigenvalues_rectangle, eigenvectors_rectangle = solve_eigenproblem(M_rectangle)
    print("Rectangle Eigenvalues:", eigenvalues_rectangle[:6])  # Output the first 6 eigenvalues
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Create 3x1 subplots
    for i in range(3):
        eigenvector = eigenvectors_rectangle[:, i]  # The i-th eigenvector
        print("vector", eigenvectors_rectangle[i, :])
        reshaped_eigenvector = np.real(eigenvectors_rectangle[:, i].reshape(N, 2 * N))
        lambda_value = np.sqrt(-eigenvalues_rectangle[i]).real
        title = f"Square, λ = {lambda_value:.2f}"
        im = plot_eigenmode(axes[i], reshaped_eigenvector, title)  # Pass the subplot object
    fig.colorbar(im, ax=axes.ravel(), orientation='horizontal', shrink=0.5)
    plt.show()

    # 3. Circle boundary
    M_circle, valid_indices = create_circle_matrix(N, L)
    # print(M_circle)

    eigenvalues_circle, eigenvectors_circle = solve_eigenproblem(M_circle, k=3)
    # Restore the eigenmodes in `N × N` format
    restored_modes = restore_frequencies(N, valid_indices, eigenvectors_circle)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        lambda_value = np.sqrt(-eigenvalues_circle[i]).real
        title = f"Circle, λ = {lambda_value:.2f}"
        im = plot_eigenmode(axes[i], restored_modes[:, :, i], title)
    fig.colorbar(im, ax=axes.ravel(), orientation='horizontal', shrink=0.5)
    plt.show()