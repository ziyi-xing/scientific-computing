import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Parameters
R = 2           # Radius of the circle
h = 0.05        # Finer grid spacing
x = np.arange(-R, R+h, h)
y = np.arange(-R, R+h, h)
X, Y = np.meshgrid(x, y)
Nx, Ny = len(x), len(y)

# Mask for points inside the circle (domain restriction)
mask = X**2 + Y**2 <= R**2
idx_map = -np.ones((Ny, Nx), dtype=int)
idx_map[mask] = np.arange(np.sum(mask))  # assign indices only inside the circle

# Assemble sparse matrix M and vector b
N = np.sum(mask)
M = lil_matrix((N, N))
b = np.zeros(N)

for j in range(Ny):
    for i in range(Nx):
        if not mask[j, i]:
            continue
        p = idx_map[j, i]

        # Boundary condition: c = 0 on boundary
        if (np.abs(np.sqrt(X[j, i]**2 + Y[j, i]**2) - R) < h):
            M[p, p] = 1
            b[p] = 0
            continue

        # Interior points: apply 5-point stencil
        M[p, p] = -4
        for (jj, ii) in [(j+1, i), (j-1, i), (j, i+1), (j, i-1)]:
            if mask[jj, ii]:
                q = idx_map[jj, ii]
                M[p, q] = 1

# Source point at (0.6, 1.2)
source_x, source_y = 0.6, 1.2
i_src = np.argmin(np.abs(x - source_x))
j_src = np.argmin(np.abs(y - source_y))
p_src = idx_map[j_src, i_src]
b[p_src] = 1

# Solve Mc = b
c = spsolve(M.tocsr(), b)

# Plot the solution
C = np.full_like(X, np.nan)
C[mask] = c

plt.figure(figsize=(7,6))
plt.contourf(X, Y, C, levels=100, cmap='viridis')
plt.colorbar(label='c(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Steady-state concentration c(x,y)')
plt.axis('equal')
plt.savefig('homework3/figure3.2/poisson_solution.png')
plt.show()
