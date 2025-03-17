import numpy as np
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Define matrix M: n^2 * n^2
# 1. Square domain
def create_square_matrix(N, L):
    h = L / (N + 1)
    size = N * N
    M = np.zeros((size, size))
    for i in range(N):
        for j in range(N):
            idx = i * N + j  # Index of the current point
            if i > 0:
                M[idx, idx - N] = 1 / h**2  # Upper neighbor
            if i < N - 1:
                M[idx, idx + N] = 1 / h**2  # Lower neighbor
            if j > 0:
                M[idx, idx - 1] = 1 / h**2  # Left neighbor
            if j < N - 1:
                M[idx, idx + 1] = 1 / h**2  # Right neighbor
            M[idx, idx] = -4 / h**2  # Current point
    return M

# 2. Rectangular domain
def create_rectangle_matrix(N, L):
    h = L / (N + 1)
    size = N * (2 * N)
    M = np.zeros((size, size))
    for i in range(N):
        for j in range(2 * N):
            idx = i * (2 * N) + j  # Index of the current point
            if i > 0:
                M[idx, idx - (2 * N)] = 1 / h**2  # Upper neighbor
            if i < N - 1:
                M[idx, idx + (2 * N)] = 1 / h**2  # Lower neighbor
            if j > 0:
                M[idx, idx - 1] = 1 / h**2  # Left neighbor
            if j < 2 * N - 1:
                M[idx, idx + 1] = 1 / h**2  # Right neighbor
            M[idx, idx] = -4 / h**2  # Current point
    return M

# 3. Circular domain
def create_circle_matrix(N, L):
    h = L / (N + 1)
    size = N * N
    M = np.zeros((size, size))
    xc, yc = L / 2, L / 2  # Center of the circle
    valid_indices = []  # Store the indices of points inside the circular region

    # Iterate through all points and remove those outside the circular region
    for i in range(N):
        for j in range(N):
            x = (i + 1) * h  # Grid point coordinate
            y = (j + 1) * h
            r = np.sqrt((x - xc)**2 + (y - yc)**2)
            if r <= L / 2:  # Point is inside the circular region
                valid_indices.append(i * N + j)

    # Construct the matrix M for the circular region
    M_circle = np.zeros((len(valid_indices), len(valid_indices)))
    for idx1, i in enumerate(valid_indices):
        for idx2, j in enumerate(valid_indices):
            if i == j:
                M_circle[idx1, idx2] = -4 / h**2  # Current point
            elif abs(i - j) == 1 or abs(i - j) == N:
                M_circle[idx1, idx2] = 1 / h**2  # Neighboring point
    return M_circle, valid_indices

# Restore frequencies for the circular domain
def restore_frequencies(N, valid_indices, eigenvectors):
    num_modes = eigenvectors.shape[1]
    restored_modes = np.zeros((N, N, num_modes))  # Create a 3D matrix of size `N × N × k`

    for i in range(num_modes):
        restored_matrix = np.zeros((N, N))  # Restored individual mode
        for idx, valid_idx in enumerate(valid_indices):
            row, col = divmod(valid_idx, N)  # Compute original (row, col)
            restored_matrix[row, col] = np.real(eigenvectors[idx, i])  # Fill in the eigenmode
        restored_modes[:, :, i] = restored_matrix
    return restored_modes

# Solve eigenvalue problem
def solve_eigenproblem(M, k=4):    
    #eigenvalues, eigenvectors = eigh(M) 
    #eigenvalues, eigenvectors = eig(M) 
    eigenvalues, eigenvectors = eigs(M, k=k, which='SM')  # Return the smallest k eigenvalues
    return np.real(eigenvalues), np.real(eigenvectors)

# Animate eigenmodes
def animate_eigenmode(shape, eigenvector, lambda_val, N, index, save=True):
    fig, ax = plt.subplots()
    time = np.linspace(0, 2 * np.pi, 200)
    ims = []

    for t in time:
        frame = np.sin(lambda_val * t) * eigenvector
        im = ax.imshow(frame, animated=True, cmap='viridis')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
    ax.set_title(f"{shape}, λ = {lambda_val:.4f}")

    # Save animation
    if save:
        gif_path = os.path.join(figures_dir, f"{shape.lower()}{index+1}.gif")
        ani.save(gif_path, writer="pillow", fps=15)
        print(f"Animation saved to {gif_path}")

    plt.close(fig)  # Close figure to avoid opening too many windows

# Main program
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Script directory:", script_dir)  # Print to check if correct
    figures_dir = os.path.join(script_dir, "fig3.1")
    os.makedirs(figures_dir, exist_ok=True)  # Ensure the directory exists
    print("Figures will be saved in:", figures_dir)

    N = 50  # Number of grid points (internal points)
    L = 1.0  # Side length of the square
    h = L / (N + 1)  # Grid spacing
    shapes = ["Square", "Rectangle"]#, "Circle"]
    matrix_creators = [create_square_matrix, create_rectangle_matrix, create_circle_matrix]

    for shape, create_matrix in zip(shapes, matrix_creators):   
        M = create_matrix(N, L)
        eigenvalues, eigenvectors = solve_eigenproblem(M, k=3)
        for i in range(3):
            lambda_val = np.sqrt(abs(eigenvalues[i]))
            eigenvector = eigenvectors[:, i].reshape((N, -1))
            animate_eigenmode(shape, eigenvector, lambda_val, N, i)

    # Animation for the circular domain
    M_circle, valid_indices = create_circle_matrix(N, L)
    eigenvalues_circle, eigenvectors_circle = solve_eigenproblem(M_circle, k=3)
    restored_modes = restore_frequencies(N, valid_indices, eigenvectors_circle)

    for i in range(3):
        lambda_value = np.sqrt(-eigenvalues_circle[i]).real
        animate_eigenmode("Circle", restored_modes[:, :, i], lambda_value, N, i)
