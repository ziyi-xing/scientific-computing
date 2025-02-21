import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os

# Ensure the figures directory exists
figures_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../figures'))
if not os.path.exists(figures_path):
    os.makedirs(figures_path)

# Jacobi solver: Solves the Laplace equation using the Jacobi iteration method
@njit
def jacobi_solve(N, tolerance, max_steps):
    # Initialize the grid with zeros and set the top boundary to 1.0
    grid = np.zeros((N, N))
    grid[0, :] = 1.0
    new_grid = np.copy(grid)  # Create a copy of the grid for updates
    error_list = []  # Store the maximum error at each step
    steps = []  # Store the step count

    # Iterate until convergence or max_steps is reached
    for step in range(1, max_steps + 1):
        # Update the grid using the Jacobi iteration formula
        for i in range(1, N - 1):
            for j in range(N):
                new_grid[i, j] = 0.25 * (
                    grid[i + 1, j] + grid[i - 1, j] + grid[i, (j + 1) % N] + grid[i, (j - 1) % N]
                )

        # Calculate the maximum error between the old and new grid
        max_error = np.max(np.abs(new_grid - grid))
        error_list.append(max_error)
        steps.append(step)

        # Check for convergence
        if max_error < tolerance:
            break

        # Update the grid for the next iteration
        grid[:, :] = new_grid

    return grid, np.array(error_list), np.array(steps)

# Gauss-Seidel solver: Solves the Laplace equation using the Gauss-Seidel iteration method
@njit
def gauss_seidel_solve(N, tolerance, max_steps):
    # Initialize the grid with zeros and set the top boundary to 1.0
    grid = np.zeros((N, N))
    grid[0, :] = 1.0
    error_list = []  # Store the maximum error at each step
    steps = []  # Store the step count

    # Iterate until convergence or max_steps is reached
    for step in range(1, max_steps + 1):
        max_error = 0.0  # Reset the maximum error for this step
        # Update the grid using the Gauss-Seidel iteration formula
        for i in range(1, N - 1):
            for j in range(N):
                old_value = grid[i, j]
                grid[i, j] = 0.25 * (
                    grid[i + 1, j] + grid[i - 1, j] + grid[i, (j + 1) % N] + grid[i, (j - 1) % N]
                )
                # Calculate the error for this cell
                max_error = max(max_error, abs(grid[i, j] - old_value))

        # Store the maximum error and step count
        error_list.append(max_error)
        steps.append(step)

        # Check for convergence
        if max_error < tolerance:
            break

    return grid, np.array(error_list), np.array(steps)

# SOR solver: Solves the Laplace equation using the Successive Over-Relaxation (SOR) method
@njit
def sor_solve(N, relaxation_factor, tolerance, max_steps):
    # Initialize the grid with zeros and set the top boundary to 1.0
    grid = np.zeros((N, N))
    grid[0, :] = 1.0
    error_list = []  # Store the maximum error at each step
    steps = []  # Store the step count

    # Iterate until convergence or max_steps is reached
    for step in range(1, max_steps + 1):
        max_error = 0.0  # Reset the maximum error for this step
        # Update the grid using the SOR iteration formula
        for i in range(1, N - 1):
            for j in range(N):
                old_value = grid[i, j]
                updated_value = relaxation_factor * 0.25 * (
                    grid[i - 1, j] + grid[i + 1, j] + grid[i, (j - 1) % N] + grid[i, (j + 1) % N]
                ) + (1 - relaxation_factor) * old_value
                grid[i, j] = updated_value
                # Calculate the error for this cell
                max_error = max(max_error, abs(updated_value - old_value))

        # Store the maximum error and step count
        error_list.append(max_error)
        steps.append(step)

        # Check for convergence
        if max_error < tolerance:
            break

    return grid, np.array(error_list), np.array(steps)

# Draw concentration profile along the y-axis for Jacobi, Gauss-Seidel, and SOR methods
def draw_concentration(N, relaxation_factor, tolerance, max_steps):
    # Solve the Laplace equation using all three methods
    grid_jacobi, _, _ = jacobi_solve(N, tolerance, max_steps)
    grid_gauss, _, _ = gauss_seidel_solve(N, tolerance, max_steps)
    grid_sor, _, _ = sor_solve(N, relaxation_factor, tolerance, max_steps)

    # Extract the concentration profile along the y-axis
    y = np.linspace(0, 1, N)
    c_analytical = y  # Analytical solution for comparison
    c_jacobi = np.flip(grid_jacobi[:, 0])  # Jacobi solution
    c_gauss = np.flip(grid_gauss[:, 0])  # Gauss-Seidel solution
    c_sor = np.flip(grid_sor[:, 0])  # SOR solution

    # Plot the concentration profiles
    plt.figure(figsize=(8, 6))
    plt.title('Steady-State Concentration Profile Along y-axis', fontsize=16)
    plt.plot(y, c_analytical, color='gray', ls='dashed', label='Analytical', zorder=2.5)
    plt.plot(y, c_jacobi, color='purple', label='Jacobi')
    plt.plot(y, c_gauss, color='brown', label='Gauss - Seidel')
    plt.plot(y, c_sor, color='teal', label='SOR')
    plt.xlabel('y - coordinate', fontsize=14)
    plt.ylabel('Concentration Value', fontsize=14)
    plt.legend()
    plt.tight_layout()
    # Save the plot to the figures directory
    plt.savefig(os.path.join(figures_path, 'Steady-State Concentration Profile Along y-axis.png'))
    plt.show()

# Draw convergence behavior for Jacobi, Gauss-Seidel, and SOR methods
def draw_convergence(N, relaxation_factors, tolerance, max_steps):
    # Solve the Laplace equation using all three methods
    _, errors_jacobi, iter_jacobi = jacobi_solve(N, tolerance, max_steps)
    _, errors_gauss, iter_gauss = gauss_seidel_solve(N, tolerance, max_steps)
    _, errors_sor_75, iter_sor_75 = sor_solve(N, relaxation_factors[0], tolerance, max_steps)
    _, errors_sor_85, iter_sor_85 = sor_solve(N, relaxation_factors[1], tolerance, max_steps)
    _, errors_sor_95, iter_sor_95 = sor_solve(N, relaxation_factors[2], tolerance, max_steps)

    # Plot the convergence behavior
    plt.figure(figsize=(8, 6))
    plt.title('Convergence Behavior of Iterative Solvers', fontsize=16)
    plt.semilogy(iter_sor_75, errors_sor_75, color='magenta', label=fr'SOR, $\omega = {{{relaxation_factors[0]}}}$')
    plt.semilogy(iter_sor_85, errors_sor_85, color='cyan', label=fr'SOR, $\omega = {{{relaxation_factors[1]}}}$')
    plt.semilogy(iter_sor_95, errors_sor_95, color='lime', label=fr'SOR, $\omega = {{{relaxation_factors[2]}}}$')
    plt.semilogy(iter_gauss, errors_gauss, color='olive', label='Gauss - Seidel')
    plt.semilogy(iter_jacobi, errors_jacobi, color='navy', label='Jacobi')
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel(r'$\delta$ (Convergence Measure)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    # Save the plot to the figures directory
    plt.savefig(os.path.join(figures_path, 'Convergence Behavior of Iterative Solvers.png'))
    plt.show()

# Search for the optimal relaxation factor (omega) for the SOR method
def search_optimal_omega(grid_sizes, relaxation_factors, tolerance, max_steps):
    optimal_factors = []  # Store the optimal omega for each grid size

    # Iterate over different grid sizes
    for N in grid_sizes:
        iteration_counts = []  # Store the number of iterations for each omega
        # Iterate over different relaxation factors
        for omega in relaxation_factors:
            _, _, iter_sor = sor_solve(int(N), omega, tolerance, max_steps)
            iteration_counts.append(iter_sor[-1])  # Store the number of iterations

        # Find the omega that results in the fewest iterations
        optimal_factors.append(relaxation_factors[np.argmin(iteration_counts)])

    # Plot the optimal omega vs. grid size
    plt.figure(figsize=(8, 6))
    plt.title(' Optimal Omega vs. Grid Size (SOR Method)', fontsize=16)
    plt.scatter(grid_sizes, optimal_factors, color='green')
    plt.xlabel('Grid Size (N)', fontsize=14)
    plt.ylabel(r'Optimal Relaxation Factor $\omega$', fontsize=14)
    plt.tight_layout()
    # Save the plot to the figures directory
    plt.savefig(os.path.join(figures_path, 'Optimal Omega vs. Grid Size (SOR Method).png'))
    plt.show()
    return optimal_factors

# Initialize sinks in the grid
def setup_sinks(sink_list, N):
    grid = np.zeros((N, N))
    grid[0, :] = 1.0  # Set the top boundary to 1.0
    # Set the sink regions to 1.0
    for obj in sink_list:
        i_min, i_max, j_min, j_max = obj
        if i_min > i_max or j_min > j_max:
            raise ValueError('Invalid sink bounds')
        grid[i_min:i_max + 1, j_min:j_max + 1] = 1
    grid[-1, :] = 0  # Set the bottom boundary to 0
    return grid

# SOR solver with sinks: Solves the Laplace equation with sink regions
@njit
def sor_with_sink_solve(sink_grid, N, relaxation_factor, tolerance, max_steps):
    grid = np.zeros((N, N))
    grid[0, :] = 1.0  # Set the top boundary to 1.0
    error_list = []  # Store the maximum error at each step
    steps = []  # Store the step count

    # Iterate until convergence or max_steps is reached
    for step in range(1, max_steps + 1):
        max_error = 0.0  # Reset the maximum error for this step
        # Update the grid using the SOR iteration formula
        for i in range(1, N - 1):
            for j in range(N):
                if sink_grid[i, j] == 1:  # If the cell is a sink, set it to 0
                    grid[i, j] = 0
                else:
                    prev_value = grid[i, j]
                    new_value = relaxation_factor * 0.25 * (
                        grid[i + 1, j] + grid[i - 1, j] + grid[i, (j + 1) % N] + grid[i, (j - 1) % N]
                    ) + (1 - relaxation_factor) * prev_value
                    grid[i, j] = new_value
                    # Calculate the error for this cell
                    max_error = max(max_error, abs(new_value - prev_value))

        # Store the maximum error and step count
        error_list.append(max_error)
        steps.append(step)

        # Check for convergence
        if max_error < tolerance:
            break

    return grid, np.array(error_list), np.array(steps)

# Draw convergence behavior with and without sinks
def draw_sink_convergence(sink_list, N, relaxation_factor, tolerance, max_steps):
    # Set up the grid with sinks
    grid_sink_1 = setup_sinks(sink_list[0], N)
    _, errors_1, iter_1 = sor_with_sink_solve(grid_sink_1, N, relaxation_factor, tolerance, max_steps)

    grid_sink_2 = setup_sinks(sink_list[1], N)
    _, errors_2, iter_2 = sor_with_sink_solve(grid_sink_2, N, relaxation_factor, tolerance, max_steps)

    # Solve without sinks for comparison
    _, errors, iter = sor_solve(N, relaxation_factor, tolerance, max_steps)

    # Plot the convergence behavior
    plt.figure(figsize=(8, 6))
    plt.title('Convergence Behavior with and without Sinks', fontsize=16)
    plt.semilogy(iter_1, errors_1, color='coral', label='SOR with 1 sink')
    plt.semilogy(iter_2, errors_2, color='indigo', label='SOR with 2 sinks')
    plt.semilogy(iter, errors, color='black', label='SOR without sinks')
    plt.xlabel('Iteration Number', fontsize=15)
    plt.ylabel(r'$\delta$ (Convergence Measure)', fontsize=15)
    plt.legend()
    plt.tight_layout()
    # Save the plot to the figures directory
    plt.savefig(os.path.join(figures_path, 'Convergence Behavior with and without Sinks.png'))
    plt.show()

# Search for the optimal omega with sinks
def search_omega_with_sinks(sink_list, N, relaxation_factors, tolerance, max_steps):
    optimal_factors = []  # Store the optimal omega for each sink configuration

    # Iterate over different sink configurations
    for obj in sink_list:
        iteration_counts = []  # Store the number of iterations for each omega
        # Iterate over different relaxation factors
        for omega in relaxation_factors:
            grid_with_sink = setup_sinks(obj, N)
            _, _, iter_sor = sor_with_sink_solve(grid_with_sink, N, omega, tolerance, max_steps)
            iteration_counts.append(iter_sor[-1])  # Store the number of iterations

        # Find the omega that results in the fewest iterations
        optimal_factors.append(relaxation_factors[np.argmin(iteration_counts)])
        print(f'Optimal omega with {len(obj)} sink(s): {optimal_factors[-1]}')

# Search for the optimal omega without sinks
def search_omega_no_sinks(N, relaxation_factors, tolerance, max_steps):
    iteration_counts = []  # Store the number of iterations for each omega

    # Iterate over different relaxation factors
    for omega in relaxation_factors:
        _, _, iter_sor = sor_solve(N, omega, tolerance, max_steps)
        iteration_counts.append(iter_sor[-1])  # Store the number of iterations

    # Find the omega that results in the fewest iterations
    optimal_omega = relaxation_factors[np.argmin(iteration_counts)]
    print(f'Optimal omega without sinks: {optimal_omega}')
    return optimal_omega

# Draw heatmaps for different sink configurations
def draw_heatmaps(N, relaxation_factor, tolerance, max_steps, sink_list):
    # Solve without sinks
    grid_no_sink, _, _ = sor_solve(N, relaxation_factor, tolerance, max_steps)
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_no_sink, extent=[0, 1, 0, 1], cmap='viridis')
    plt.title('Concentration Field without Sinks', fontsize=16)
    plt.xlabel('x - coordinate', fontsize=14)
    plt.ylabel('y - coordinate', fontsize=14)
    plt.colorbar(label='Concentration Level')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'Concentration Field without Sinks.png'))
    plt.show()

    # Solve with one sink
    grid_one_sink, _, _ = sor_with_sink_solve(setup_sinks(sink_list[0], N), N, relaxation_factor, tolerance, max_steps)
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_one_sink, extent=[0, 1, 0, 1], cmap='viridis')
    plt.title('Concentration Field with One Sink', fontsize=16)
    plt.xlabel('x - coordinate', fontsize=14)
    plt.ylabel('y - coordinate', fontsize=14)
    plt.colorbar(label='Concentration Level')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'Concentration Field with One Sink.png'))
    plt.show()

    # Solve with two sinks
    grid_two_sinks, _, _ = sor_with_sink_solve(setup_sinks(sink_list[1], N), N, relaxation_factor, tolerance, max_steps)
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_two_sinks, extent=[0, 1, 0, 1], cmap='viridis')
    plt.title('Concentration Field with Two Sinks', fontsize=16)
    plt.xlabel('x - coordinate', fontsize=14)
    plt.ylabel('y - coordinate', fontsize=14)
    plt.colorbar(label='Concentration Level')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'Concentration Field with Two Sinks.png'))
    plt.show()

# Parameter settings
N = 100
tolerance = 1e-5
max_steps = int(2e4)
relaxation_factors = np.linspace(1.7, 2, 100)
grid_sizes = np.linspace(10, 100, 10)
sink_1 = np.array([[15, 25, 45, 55]])
sink_2 = np.array([[15, 25, 30, 40], [15, 25, 60, 70]])

# Run simulations
draw_concentration(N, 1.95, tolerance, max_steps)
draw_convergence(N, [1.75, 1.85, 1.95], tolerance, max_steps)
search_optimal_omega(grid_sizes, relaxation_factors, tolerance, max_steps)
draw_sink_convergence([sink_1, sink_2], N, 1.95, tolerance, max_steps)
search_omega_no_sinks(N, relaxation_factors, tolerance, max_steps)
search_omega_with_sinks([sink_1, sink_2], N, relaxation_factors, tolerance, max_steps)
draw_heatmaps(N, 1.95, tolerance, max_steps, [sink_1, sink_2])
