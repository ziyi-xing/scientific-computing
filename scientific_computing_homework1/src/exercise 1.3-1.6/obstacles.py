import numpy as np


# 初始化障碍物
def setup_obstacles(obstacle_list, grid_size):
    grid = np.zeros((grid_size, grid_size))
    grid[0, :] = 1
    for obj in obstacle_list:
        i_min, i_max, j_min, j_max = obj
        if i_min > i_max or j_min > j_max:
            raise ValueError('Invalid obstacle bounds')
        grid[i_min:i_max + 1, j_min:j_max + 1] = 1
    grid[-1, :] = 0
    return grid


# 带障碍物的SOR求解器
def sor_with_obstacle_solve(obstacle_grid, grid_size, relaxation_factor, tolerance, max_steps):
    from solvers import sor_solve
    grid = np.zeros((grid_size, grid_size))
    grid[0, :] = 1
    error_list = np.zeros(max_steps)
    for step in range(max_steps):
        max_error = 0.0
        for i in range(1, grid_size - 1):
            for j in range(grid_size):
                if obstacle_grid[i, j] == 1:
                    grid[i, j] = 0
                else:
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