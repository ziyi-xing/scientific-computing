import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time

# Create a sparse matrix for a square region
def create_square_sparse(N, L):
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

# Create a sparse matrix for a rectangular region
def create_rectangle_sparse(N, L):
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

# Create a sparse matrix for a circular region
def create_circle_sparse(N, L):
    h = L / (N + 1)
    size = N * N
    xc, yc = L / 2, L / 2
    valid_indices = []
    
    for i in range(N):
        for j in range(N):
            x = (i + 1) * h
            y = (j + 1) * h
            r = np.sqrt((x - xc)**2 + (y - yc)**2)
            if r <= L / 2:
                valid_indices.append(i * N + j)
    
    M_circle = sp.lil_matrix((len(valid_indices), len(valid_indices)))
    for idx1, i in enumerate(valid_indices):
        for idx2, j in enumerate(valid_indices):
            if i == j:
                M_circle[idx1, idx2] = -4 / h**2
            elif abs(i - j) == 1 or abs(i - j) == N:
                M_circle[idx1, idx2] = 1 / h**2
    return M_circle.tocsr()

# Compute eigenvalues for a dense matrix
def solve_eigen_dense(M, k=4):
    start = time.perf_counter()
    eigenvalues, _ = np.linalg.eigh(M.toarray())
    end = time.perf_counter()
    return eigenvalues[:k], end - start

# Compute eigenvalues for a sparse matrix
def solve_eigen_sparse(M, k=4):
    start = time.perf_counter()
    eigenvalues, _ = scipy.sparse.linalg.eigs(M, k=k, which='SR')
    end = time.perf_counter()
    return eigenvalues.real, end - start

N_values = np.arange(10, 55, 5)

dense_times_square, sparse_times_square = [], []
dense_times_rectangle, sparse_times_rectangle = [], []
dense_times_circle, sparse_times_circle = [], []

for N in N_values:
    N = int(N)
    M_square = create_square_sparse(N, 1.0)
    M_rectangle = create_rectangle_sparse(N, 1.0)
    M_circle = create_circle_sparse(N, 1.0)
    
    _, t_dense_square = solve_eigen_dense(M_square)
    _, t_sparse_square = solve_eigen_sparse(M_square)
    _, t_dense_rectangle = solve_eigen_dense(M_rectangle)
    _, t_sparse_rectangle = solve_eigen_sparse(M_rectangle)
    _, t_dense_circle = solve_eigen_dense(M_circle)
    _, t_sparse_circle = solve_eigen_sparse(M_circle)
    
    dense_times_square.append(t_dense_square)
    sparse_times_square.append(t_sparse_square)
    dense_times_rectangle.append(t_dense_rectangle)
    sparse_times_rectangle.append(t_sparse_rectangle)
    dense_times_circle.append(t_dense_circle)
    sparse_times_circle.append(t_sparse_circle)

plt.figure(figsize=(8, 6))
plt.plot(N_values, dense_times_square, color='pink', linestyle='--', label='Square without sparse')
plt.plot(N_values, sparse_times_square, color='pink', linestyle='-', label='Square Sparse')
plt.plot(N_values, dense_times_rectangle, color='green', linestyle='--', label='Rectangle without sparse')
plt.plot(N_values, sparse_times_rectangle, color='green', linestyle='-', label='Rectangle Sparse')
plt.plot(N_values, dense_times_circle, color='blue', linestyle='--', label='Circle without sparse')
plt.plot(N_values, sparse_times_circle, color='blue', linestyle='-', label='Circle Sparse')
plt.xlabel('N')
plt.ylabel('Time (seconds)')
plt.xlim(10, 50)
plt.xticks(np.arange(10, 55, 5))
plt.legend()
plt.grid()
plt.show()
