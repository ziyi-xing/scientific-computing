import numpy as np
from solvers import sor_solve
from obstacles import setup_obstacles, sor_with_obstacle_solve


# 寻找SOR的最优松弛因子
def search_optimal_omega(grid_sizes, relaxation_factors, tolerance, max_steps):
    optimal_factors = []
    for N in grid_sizes:
        iteration_counts = np.zeros(len(relaxation_factors))
        for i, omega in enumerate(relaxation_factors):
            _, _, iter_sor = sor_solve(int(N), omega, tolerance, max_steps)
            iteration_counts[i] = iter_sor[-1]
        optimal_factors.append(relaxation_factors[np.argmin(iteration_counts)])
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.title('Optimal Relaxation Factor vs Grid Size', fontsize=16)
    plt.scatter(grid_sizes, optimal_factors, color='green')
    plt.xlabel('Grid Size (N)', fontsize=15)
    plt.ylabel(r'Optimal $\omega$', fontsize=15)
    plt.tight_layout()
    plt.show()
    return optimal_factors


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