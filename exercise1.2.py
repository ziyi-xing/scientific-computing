import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import matplotlib.animation as animation
from numba import njit
from matplotlib.animation import FuncAnimation
from matplotlib.animation import HTMLWriter

# Define the analytical solution function
def analytical_solution(y, t, D=1.0, sum_terms=50):
    """
    Compute the analytical solution of the diffusion equation.
    
    Parameters:
        y (array): Spatial coordinate array.
        t (float): Time value.
        D (float): Diffusion constant (default is 1.0).
        sum_terms (int): Number of terms to sum in the series expansion.
    
    Returns:
        sol (array): Concentration values at each y for given time t.
    """
    if t == 0:
        sol = np.zeros(len(y))
        sol[-1] = 1  # Upper boundary condition
        return sol
    
    sol = np.zeros(len(y))
    for i in range(sum_terms):
        sol += erfc((1 - y + 2 * i) / (2 * np.sqrt(D * t))) - erfc((1 + y + 2 * i) / (2 * np.sqrt(D * t)))
    return sol

# Define function to plot analytical solution
def plot_results(times):
    """
    Plot concentration C(y) over y at different time steps using the analytical solution.
    
    Parameters:
        times (list): List of time values to plot.
    """
    y = np.linspace(0, 1, 100)  # Define y-axis range
    plt.figure(figsize=(8, 6))
    for t in times:
        plt.plot(y, analytical_solution(y, t), label=f't={t}')
    plt.xlabel('y')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title('C(y) by analytical method')
    plt.show()

@njit  # Accelerate computation with Numba
def explicit_finite_difference_result(t, dx, dt, N, D=1.0, num_frames=100):
    """
    Compute the numerical solution of the diffusion equation using the explicit finite difference method.
    
    Parameters:
        t (float): Total simulation time.
        dx (float): Grid spacing.
        dt (float): Time step size.
        N (int): Grid size.
        D (float): Diffusion constant.
        num_frames (int): Number of frames to store for visualization.
    
    Returns:
        all_frames (array): 3D array storing grid states at different time steps.
    """
    grid = np.zeros((N, N))  # Initialize grid
    grid[0, :] = 1  # Upper boundary condition
    grid[-1, :] = 0  # Lower boundary condition
    new_grid = grid.copy()
    num_time_steps = int(t / dt)
    frame_times = np.round(np.linspace(0, num_time_steps - 1, num_frames)).astype(np.int32)
    all_frames = np.zeros((num_frames, N, N))
    index = 0
    
    for t in range(num_time_steps):
        for i in range(1, N - 1):
            for j in range(N):
                new_grid[i, j] = grid[i, j] + dt * D / (dx**2) * (grid[i + 1, j] + grid[i - 1, j] + grid[i, (j + 1) % N] + grid[i, (j - 1) % N] - 4 * grid[i, j])
        
        grid[:] = new_grid
        if t in frame_times:
            all_frames[index] = np.copy(grid)
            index += 1
    
    return all_frames

# Plot concentration profile using explicit finite difference method
def ex_plot_result(t_max, dx, dt, N, times, num_frames=100):
    """
    Plot C(y) using the explicit finite difference formulation.
    
    Parameters:
        t_max (float): Total simulation time.
        dx (float): Grid spacing.
        dt (float): Time step size.
        N (int): Grid size.
        times (list): List of time points to plot.
        num_frames (int): Number of frames to store.
    """
    y = np.linspace(0, 1, N)
    all_frames = explicit_finite_difference_result(t_max, dx, dt, N, num_frames=100)
    frame_index = [round(t / t_max * (num_frames - 1)) for t in times]
    print(frame_index)
    plt.figure(figsize=(8, 6))
    for t_idx in frame_index:
        t_value = times[frame_index.index(t_idx)]
        reverse_cy = all_frames[t_idx, :, N // 2]
        cy = reverse_cy[::-1]
        plt.plot(y, cy, label=f't={t_value:.2f}')
    
    plt.xlabel('y')
    plt.ylabel('Concentration c(y)')
    plt.xticks([-0.1, 1, 1])  # Set x-axis ticks
    plt.title('Explicit Finite Difference Formulation C(y)')
    plt.legend()
    plt.show()

# Function to visualize 2D concentration distribution
def plot_2d_concentration(times, t_max, dx, dt, N, D=1.0, num_frames=100):
    """
    Plot 2D heatmap of concentration over time using explicit finite difference method.
    """
    all_frames = explicit_finite_difference_result(t_max, dx, dt, N, D, num_frames)
    frame_indices = [round(t / t_max * (num_frames - 1)) for t in times]
    frames = [all_frames[idx] for idx in frame_indices]
    
    plt.figure(figsize=(15, 10))
    for i, (t, frame) in enumerate(zip(times, frames)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(frame, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Concentration')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f't = {t:.3f}')
    
    plt.tight_layout()
    plt.show()

# Function to create an animation
def create_animation(t_max, dx, dt, N, D=1.0, num_frames=100):
    """
    Generate an animation of the time-dependent diffusion equation.
    """
    all_frames = explicit_finite_difference_result(t_max, dx, dt, N, D, num_frames)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(all_frames[0], extent=[0, 1, 0, 1], origin='lower', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, label='Concentration')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Time Dependent Diffusion Equation')
    
    def update(frame):
        im.set_data(all_frames[frame])
        ax.set_title(f'Time = {frame * t_max / (num_frames - 1):.3f}')
        return im,
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
    plt.show()
    return ani

# show the result
D=1
t_max = 1
dx = 1 / 100
dt = 0.25 * dx**2 / (4 * D)
N = 100
times = [0.001,0.01,0.1, 1]
##
## show the analytical C(y)
plot_results(times)
##explicit finite difference C(y)
ex_plot_result(t_max, dx, dt, N, times, num_frames=100)
plot_2d_concentration(times, t_max, dx, dt, N, D=1.0, num_frames=100)
ani=create_animation(t_max, dx, dt, N, D=1.0, num_frames=100)
ani.save(filename="/Users/yuzongyao/Downloads/出国/SFM-STUDY/scinectific computing/hm/diffusion.gif", writer="imagemagick")
