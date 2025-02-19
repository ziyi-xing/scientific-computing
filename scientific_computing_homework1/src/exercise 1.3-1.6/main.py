import numpy as np

# 参数设置
grid_size = 100
tolerance = 1e-5
max_steps = int(2e4)
relaxation_factors = np.linspace(1.7, 2, 100)
grid_sizes = np.linspace(10, 100, 10)
obstacle_1 = np.array([[15, 25, 45, 55]])
obstacle_2 = np.array([[15, 25, 30, 40], [15, 25, 60, 70]])

from visualization import draw_concentration, draw_convergence, draw_obstacle_convergence, draw_heatmaps
from optimization import search_optimal_omega, search_omega_with_obstacles, search_omega_no_obstacles

# 运行模拟
draw_concentration(grid_size, 1.95, tolerance, max_steps)
draw_convergence(grid_size, [1.75, 1.85, 1.95], tolerance, max_steps)
search_optimal_omega(grid_sizes, relaxation_factors, tolerance, max_steps)
draw_obstacle_convergence([obstacle_1, obstacle_2], grid_size, 1.95, tolerance, max_steps)
search_omega_no_obstacles(grid_size, relaxation_factors, tolerance, max_steps)
search_omega_with_obstacles([obstacle_1, obstacle_2], grid_size, relaxation_factors, tolerance, max_steps)
draw_heatmaps(grid_size, 1.95, tolerance, max_steps, [obstacle_1, obstacle_2])