import numpy as np
from numba import njit


# Jacobi求解器
@njit
def jacobi_solve(grid_size, tolerance, max_steps):
    grid = np.zeros((grid_size, grid_size))
    grid[0, :] = 1
    new_grid = grid.copy()
    error_list = np.zeros(max_steps)
    for step in range(max_steps):
        for i in range(1, grid_size - 1):
            for j in range(grid_size):
                new_grid[i, j] = 0.25 * (
                    grid[i + 1, j] + grid[i - 1, j] + grid[i, (j + 1) % grid_size] + grid[i, (j - 1) % grid_size]
                )
        diff = np.max(np.abs(new_grid - grid))
        error_list[step] = diff
        if diff < tolerance:
            break
        grid = new_grid.copy()
    return new_grid, error_list[:step + 1], np.arange(1, step + 2)


# Gauss - Seidel求解器
@njit
def gauss_seidel_solve(grid_size, tolerance, max_steps):
    grid = np.zeros((grid_size, grid_size))
    grid[0, :] = 1
    error_list = np.zeros(max_steps)
    for step in range(max_steps):
        max_error = 0.0
        for i in range(1, grid_size - 1):
            for j in range(grid_size):
                old_value = grid[i, j]
                grid[i, j] = 0.25 * (
                    grid[i + 1, j] + grid[i - 1, j] + grid[i, (j + 1) % grid_size] + grid[i, (j - 1) % grid_size]
                )
                max_error = max(max_error, abs(grid[i, j] - old_value))
        error_list[step] = max_error
        if max_error < tolerance:
            break
    return grid, error_list[:step + 1], np.arange(1, step + 2)


# SOR求解器
@njit
def sor_solve(grid_size, relaxation_factor, tolerance, max_steps):
    grid = np.zeros((grid_size, grid_size))
    grid[0, :] = 1
    error_list = np.zeros(max_steps)
    for step in range(max_steps):
        max_error = 0.0
        for i in range(1, grid_size - 1):
            for j in range(grid_size):
                prev_value = grid[i, j]
                new_value = relaxation_factor * 0.25 * (
                    grid[i + 1, j] + grid[i - 1, j] + grid[i, (j + 1) % grid_size] + grid[i, (j - 1) % grid_size]
                ) + (1 - relaxation_factor) * prev_value
                grid[i, j] = new_value
                max_error = max(max_error, abs(new_value - prev_value))
        error_list[step] = max_error
        if max_error < tolerance:
            break
    return grid, error_list[:step + 1], np.arange(1, step + 2)