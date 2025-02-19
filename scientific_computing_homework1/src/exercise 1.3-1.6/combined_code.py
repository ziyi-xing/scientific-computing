import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os


# 确保figures文件夹存在
figures_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../figures'))
if not os.path.exists(figures_path):
    os.makedirs(figures_path)


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


# 绘制沿y轴的浓度图
def draw_concentration(grid_size, relaxation_factor, tolerance, max_steps):
    grid_jacobi, _, _ = jacobi_solve(grid_size, tolerance, max_steps)
    grid_gauss, _, _ = gauss_seidel_solve(grid_size, tolerance, max_steps)
    grid_sor, _, _ = sor_solve(grid_size, relaxation_factor, tolerance, max_steps)
    y = np.linspace(0, 1, grid_size)
    c_analytical = y
    c_jacobi = np.flip(grid_jacobi[:, 0])
    c_gauss = np.flip(grid_gauss[:, 0])
    c_sor = np.flip(grid_sor[:, 0])
    plt.figure(figsize=(8, 6))
    plt.title('Late - Time Concentration Profile', fontsize=16)
    plt.plot(y, c_analytical, color='gray', ls='dashed', label='Analytical', zorder=2.5)
    plt.plot(y, c_jacobi, color='purple', label='Jacobi')
    plt.plot(y, c_gauss, color='brown', label='Gauss - Seidel')
    plt.plot(y, c_sor, color='teal', label='SOR')
    plt.xlabel('y - coordinate', fontsize=15)
    plt.ylabel('Concentration Value', fontsize=15)
    plt.legend()
    plt.tight_layout()
    # 保存图片到figures文件夹
    plt.savefig(os.path.join(figures_path, 'Late_Time_Concentration_Profile.png'))
    plt.show()


# 绘制不同求解器的收敛图
def draw_convergence(grid_size, relaxation_factors, tolerance, max_steps):
    _, errors_jacobi, iter_jacobi = jacobi_solve(grid_size, tolerance, max_steps)
    _, errors_gauss, iter_gauss = gauss_seidel_solve(grid_size, tolerance, max_steps)
    _, errors_sor_75, iter_sor_75 = sor_solve(grid_size, relaxation_factors[0], tolerance, max_steps)
    _, errors_sor_85, iter_sor_85 = sor_solve(grid_size, relaxation_factors[1], tolerance, max_steps)
    _, errors_sor_95, iter_sor_95 = sor_solve(grid_size, relaxation_factors[2], tolerance, max_steps)
    plt.figure(figsize=(8, 6))
    plt.title('Convergence Analysis of Solvers', fontsize=16)
    plt.loglog(iter_sor_75, errors_sor_75, color='magenta', label=fr'SOR, $\omega = {{{relaxation_factors[0]}}}$')
    plt.loglog(iter_sor_85, errors_sor_85, color='cyan', label=fr'SOR, $\omega = {{{relaxation_factors[1]}}}$')
    plt.loglog(iter_sor_95, errors_sor_95, color='lime', label=fr'SOR, $\omega = {{{relaxation_factors[2]}}}$')
    plt.loglog(iter_gauss, errors_gauss, color='olive', label='Gauss - Seidel')
    plt.loglog(iter_jacobi, errors_jacobi, color='navy', label='Jacobi')
    plt.xlabel('Iteration Count', fontsize=15)
    plt.ylabel(r'$\delta$ (Error)', fontsize=15)
    plt.legend()
    plt.tight_layout()
    # 保存图片到figures文件夹
    plt.savefig(os.path.join(figures_path, 'Convergence_Analysis_of_Solvers.png'))
    plt.show()


# 寻找SOR的最优松弛因子
def search_optimal_omega(grid_sizes, relaxation_factors, tolerance, max_steps):
    optimal_factors = []
    for N in grid_sizes:
        iteration_counts = np.zeros(len(relaxation_factors))
        for i, omega in enumerate(relaxation_factors):
            _, _, iter_sor = sor_solve(int(N), omega, tolerance, max_steps)
            iteration_counts[i] = iter_sor[-1]
        optimal_factors.append(relaxation_factors[np.argmin(iteration_counts)])
    plt.figure(figsize=(8, 6))
    plt.title('Optimal Relaxation Factor vs Grid Size', fontsize=16)
    plt.scatter(grid_sizes, optimal_factors, color='green')
    plt.xlabel('Grid Size (N)', fontsize=15)
    plt.ylabel(r'Optimal $\omega$', fontsize=15)
    plt.tight_layout()
    # 保存图片到figures文件夹
    plt.savefig(os.path.join(figures_path, 'Optimal_Relaxation_Factor_vs_Grid_Size.png'))
    plt.show()
    return optimal_factors


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
@njit
def sor_with_obstacle_solve(obstacle_grid, grid_size, relaxation_factor, tolerance, max_steps):
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


# 绘制带障碍物的收敛图
def draw_obstacle_convergence(obstacle_list, grid_size, relaxation_factor, tolerance, max_steps):
    grid_obstacle_1 = setup_obstacles(obstacle_list[0], grid_size)
    _, errors_1, iter_1 = sor_with_obstacle_solve(grid_obstacle_1, grid_size, relaxation_factor, tolerance, max_steps)
    grid_obstacle_2 = setup_obstacles(obstacle_list[1], grid_size)
    _, errors_2, iter_2 = sor_with_obstacle_solve(grid_obstacle_2, grid_size, relaxation_factor, tolerance, max_steps)
    _, errors, iter = sor_solve(grid_size, relaxation_factor, tolerance, max_steps)
    plt.figure(figsize=(8, 6))
    plt.title('Convergence with Obstacles', fontsize=16)
    plt.loglog(iter_1, errors_1, color='coral', label='SOR with 1 obstacle')
    plt.loglog(iter_2, errors_2, color='indigo', label='SOR with 2 obstacles')
    plt.loglog(iter, errors, color='black', label='SOR without obstacles')
    plt.xlabel('Iteration Number', fontsize=15)
    plt.ylabel(r'$\delta$ (Convergence Error)', fontsize=15)
    plt.legend()
    plt.tight_layout()
    # 保存图片到figures文件夹
    plt.savefig(os.path.join(figures_path, 'Convergence_with_Obstacles.png'))
    plt.show()


# 寻找带障碍物的最优松弛因子
def search_omega_with_obstacles(obstacle_list, grid_size, relaxation_factors, tolerance, max_steps):
    optimal_factors = []
    for obj in obstacle_list:
        iteration_counts = np.zeros(len(relaxation_factors))
        for i, omega in enumerate(relaxation_factors):
            grid_with_obj = setup_obstacles(obj, grid_size)
            _, _, iter_sor = sor_with_obstacle_solve(grid_with_obj, grid_size, omega, tolerance, max_steps)
            iteration_counts[i] = iter_sor[-1]
        optimal_factors.append(relaxation_factors[np.argmin(iteration_counts)])
        print(f'Optimal omega with {len(obj)} obstacle(s): {optimal_factors[-1]}')


# 计算不带障碍物的最优松弛因子
def search_omega_no_obstacles(grid_size, relaxation_factors, tolerance, max_steps):
    iteration_counts = np.zeros(len(relaxation_factors))
    for i, omega in enumerate(relaxation_factors):
        _, _, iter_sor = sor_solve(grid_size, omega, tolerance, max_steps)
        iteration_counts[i] = iter_sor[-1]
    optimal_omega = relaxation_factors[np.argmin(iteration_counts)]
    print(f'Optimal omega without obstacles: {optimal_omega}')
    return optimal_omega


# 绘制不同障碍物配置的热图
def draw_heatmaps(grid_size, relaxation_factor, tolerance, max_steps, obstacle_list):
    # 无障碍物
    grid_no_obstacle, _, _ = sor_solve(grid_size, relaxation_factor, tolerance, max_steps)
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_no_obstacle, extent=[0, 1, 0, 1], cmap='cividis')
    plt.xlabel('x - coordinate', fontsize=13)
    plt.ylabel('y - coordinate', fontsize=13)
    plt.title('2D Diffusion - No Obstacles', fontsize=16)
    plt.colorbar(label='Concentration Level')
    plt.tight_layout()
    # 保存图片到figures文件夹
    plt.savefig(os.path.join(figures_path, '2D_Diffusion_No_Obstacles.png'))
    plt.show()
    # 一个障碍物
    grid_one_obstacle, _, _ = sor_with_obstacle_solve(setup_obstacles(obstacle_list[0], grid_size), grid_size, relaxation_factor, tolerance, max_steps)
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_one_obstacle, extent=[0, 1, 0, 1], cmap='plasma')
    plt.xlabel('x - coordinate', fontsize=13)
    plt.ylabel('y - coordinate', fontsize=13)
    plt.title('2D Diffusion - One Obstacle', fontsize=16)
    plt.colorbar(label='Concentration Level')
    plt.tight_layout()
    # 保存图片到figures文件夹
    plt.savefig(os.path.join(figures_path, '2D_Diffusion_One_Obstacle.png'))
    plt.show()
    # 两个障碍物
    grid_two_obstacles, _, _ = sor_with_obstacle_solve(setup_obstacles(obstacle_list[1], grid_size), grid_size, relaxation_factor, tolerance, max_steps)
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_two_obstacles, extent=[0, 1, 0, 1], cmap='plasma')
    plt.xlabel('x - coordinate', fontsize=13)
    plt.ylabel('y - coordinate', fontsize=13)
    plt.title('2D Diffusion - Two Obstacles', fontsize=16)
    plt.colorbar(label='Concentration Level')
    plt.tight_layout()
    # 保存图片到figures文件夹
    plt.savefig(os.path.join(figures_path, '2D_Diffusion_Two_Obstacles.png'))
    plt.show()


# 参数设置
grid_size = 100
tolerance = 1e-5
max_steps = int(2e4)
relaxation_factors = np.linspace(1.7, 2, 100)
grid_sizes = np.linspace(10, 100, 10)
obstacle_1 = np.array([[15, 25, 45, 55]])
obstacle_2 = np.array([[15, 25, 30, 40], [15, 25, 60, 70]])

# 运行模拟
draw_concentration(grid_size, 1.95, tolerance, max_steps)
draw_convergence(grid_size, [1.75, 1.85, 1.95], tolerance, max_steps)
search_optimal_omega(grid_sizes, relaxation_factors, tolerance, max_steps)
draw_obstacle_convergence([obstacle_1, obstacle_2], grid_size, 1.95, tolerance, max_steps)
search_omega_no_obstacles(grid_size, relaxation_factors, tolerance, max_steps)
search_omega_with_obstacles([obstacle_1, obstacle_2], grid_size, relaxation_factors, tolerance, max_steps)
draw_heatmaps(grid_size, 1.95, tolerance, max_steps, [obstacle_1, obstacle_2])