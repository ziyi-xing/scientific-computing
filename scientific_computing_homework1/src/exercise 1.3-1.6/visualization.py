import numpy as np
import matplotlib.pyplot as plt
from solvers import jacobi_solve, gauss_seidel_solve, sor_solve
from obstacles import setup_obstacles, sor_with_obstacle_solve


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
    plt.show()


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
    plt.show()


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
    plt.show()