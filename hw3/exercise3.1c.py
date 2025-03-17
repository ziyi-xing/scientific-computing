import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import time

# Define the finite difference matrix for a square region

# Define matrix M: n^2 * n^2
# 1. Square
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

# 2. Rectangle boundary
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

# Define matrix for rectangle and circle

# 3. Circle boundary
def create_circle_matrix(N, L):
    h = L / (N + 1)
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
    return M_circle

# Solve the eigenvalue problem
def solve_eigenproblem(M, k=4):
    eigenvalues, eigenvectors = eigs(M, k=k, which='SM')  # Return the smallest k eigenvalues
    frequencies = [np.sqrt(-ev) if ev < 0 else 0 for ev in eigenvalues]
    return frequencies

def compute_frequencies(N_values, L, shape_func):
    all_frequencies = {i: [] for i in range(1, 11)}
    for N in N_values:
        M = shape_func(int(N), L)
        frequencies = solve_eigenproblem(M, k=10)
        for i in range(10):
            all_frequencies[i + 1].append(frequencies[i])
    return all_frequencies

def compute_frequencies_L(N, L_values, shape_func):
    all_frequencies = {i: [] for i in range(1, 11)}
    for L in L_values:
        M = shape_func(int(N), L)
        frequencies = solve_eigenproblem(M, k=10)
        for i in range(10):
            all_frequencies[i + 1].append(frequencies[i])
    return all_frequencies

def plot_frequencies(ax, x_values, frequencies, title, xlabel):
    for i in range(10):
        ax.plot(x_values, frequencies[i + 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Eigenfrequency")
    ax.set_title(title, fontsize=10)

if __name__ == "__main__":
    N_values = np.arange(5, 55, 5)
    fixed_N = 50  # Fixed discretization steps for L-variation plot
    fixed_L = 1.0  # Fixed size for N-variation plot
    L_values = np.linspace(0.5, 1.0, 20)  # Range of L values

    freq_square = compute_frequencies(N_values, fixed_L, create_square_matrix)
    freq_rectangle = compute_frequencies(N_values, fixed_L, create_rectangle_matrix)
    freq_circle = compute_frequencies(N_values, fixed_L, create_circle_matrix)

    freq_square_L = compute_frequencies_L(fixed_N, L_values, create_square_matrix)
    freq_rectangle_L = compute_frequencies_L(fixed_N, L_values, create_rectangle_matrix)
    freq_circle_L = compute_frequencies_L(fixed_N, L_values, create_circle_matrix)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    plot_frequencies(axes[0, 0], N_values, freq_square, "Square (N)", "N")
    plot_frequencies(axes[1, 0], N_values, freq_rectangle, "Rectangle (N)", "N")
    plot_frequencies(axes[2, 0], N_values, freq_circle, "Circle (N)", "N")

    plot_frequencies(axes[0, 1], L_values, freq_square_L, "Square (L)", "L")
    plot_frequencies(axes[1, 1], L_values, freq_rectangle_L, "Rectangle (L)", "L")
    plot_frequencies(axes[2, 1], L_values, freq_circle_L, "Circle (L)", "L")

    plt.tight_layout()
    plt.show()