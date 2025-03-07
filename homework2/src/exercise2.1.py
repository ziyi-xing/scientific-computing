import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from numba import jit

# SOR Solver
@jit(nopython=True)
def solve_diffusion(grid, sinks, omega=1.7, tol=1e-5, max_iter=10000):
    """
    Solve the diffusion equation using the Successive Over-Relaxation (SOR) method.
    :param grid: The grid field representing the nutrient concentration.
    :param sinks: Fixed points (particle aggregation points).
    :param omega: SOR relaxation factor.
    :param tol: Convergence tolerance.
    :param max_iter: Maximum number of iterations.
    :return: Updated grid field and the number of iterations.
    """
    ny, nx = grid.shape
    residual = tol + 1  # Ensure at least one iteration

    for iteration in range(max_iter):
        residual = 0.0

        for i in range(1, ny - 1):
            for j in range(nx):
                if sinks[i, j]:
                    continue  # Skip fixed points

                # Periodic boundary conditions
                left = grid[i, (j - 1) % nx]
                right = grid[i, (j + 1) % nx]
                up = grid[i - 1, j]
                down = grid[i + 1, j]

                # SOR update
                old_value = grid[i, j]
                new_value = (1 - omega) * old_value + omega * (up + down + left + right) / 4.0
                new_value = max(0.0, new_value)
                residual = max(residual, abs(new_value - old_value))
                grid[i, j] = new_value

        if residual < tol:
            return grid, iteration + 1

    return grid, max_iter

def get_growth_candidates(cluster_field):
    """
    Identify potential growth points.
    :param cluster_field: The field representing particle aggregation.
    :return: List of candidate coordinates.
    """
    east = np.roll(cluster_field, shift=-1, axis=1)
    west = np.roll(cluster_field, shift=1, axis=1)
    north = np.roll(cluster_field, shift=1, axis=0)
    south = np.roll(cluster_field, shift=-1, axis=0)

    # Exclude invalid boundaries
    north[0, :] = 0  # Top boundary invalid
    south[-1, :] = 0  # Bottom boundary invalid

    # Candidate condition: at least one neighbor is occupied and the point itself is not occupied
    neighbor_occupied = (north + south + east + west) > 0
    candidates = (cluster_field == 0) & neighbor_occupied

    return np.argwhere(candidates)

def choose_growth_candidate(candidates, nutrient_field, eta):
    """
    Select a growth point based on nutrient concentration.
    :param candidates: List of candidate coordinates.
    :param nutrient_field: Nutrient concentration field.
    :param eta: Parameter controlling growth probability.
    :return: Selected growth point coordinates.
    """
    if len(candidates) == 0:
        return None

    # Calculate growth probabilities based on nutrient concentration
    nutrient_values = nutrient_field[candidates[:, 0], candidates[:, 1]] ** eta
    probabilities = nutrient_values / nutrient_values.sum()

    # Randomly select a growth point
    chosen_index = candidates[np.random.choice(len(candidates), p=probabilities)]
    return chosen_index

class DiffusionLimitedAggregation:
    """
    Class for simulating Diffusion-Limited Aggregation (DLA) process.
    """
    def __init__(self, grid_size: tuple, eta: float):
        """
        Initialize the DLA model.
        :param grid_size: Grid size (height, width).
        :param eta: Parameter controlling the relationship between growth probability and nutrient concentration.
        """
        self.grid_size = grid_size
        self.eta = eta
        self.nutrient_field = np.zeros(grid_size)  # Nutrient concentration field
        self.cluster_field = np.zeros_like(self.nutrient_field)  # Particle aggregation field
        self.nutrient_field[0, :] = 1.0  # Set top boundary as nutrient source

        # Set initial seed at the bottom center of the grid
        seed_position = grid_size[1] // 2
        self.cluster_field[-1, seed_position] = 1
        self.nutrient_field[-1, seed_position] = 0.0  # Set seed as a fixed point

        self.termination_flag = False  # Termination flag
        self.termination_step = -1  # Termination step
        self.history = []  # Record grid states at each iteration

    def update_nutrient_field(self, omega=1.7, tol=1e-5, max_iter=10000):
        """
        Update the nutrient concentration field using SOR.
        :param omega: SOR relaxation factor.
        :param tol: Convergence tolerance.
        :param max_iter: Maximum number of iterations.
        :return: Number of iterations.
        """
        sinks = self.cluster_field == 1  # Fixed points
        self.nutrient_field, iter_count = solve_diffusion(self.nutrient_field, sinks, omega=omega, tol=tol, max_iter=max_iter)
        return iter_count

    def grow(self, growth_steps, plot_interval=100, omega=1.7, tol=1e-5, max_iter=10000):
        """
        Run the DLA growth process.
        :param growth_steps: Maximum number of growth steps.
        :param plot_interval: Interval for plotting (if 0, no plotting).
        :param omega: SOR relaxation factor.
        :param tol: Convergence tolerance.
        :param max_iter: Maximum number of iterations.
        """
        for step in range(growth_steps):
            if self.termination_flag:
                self.termination_step = step
                print(f"Termination at step {step} with η = {self.eta}")
                break

            self.update_nutrient_field(omega=omega, tol=tol, max_iter=max_iter)
            candidates = get_growth_candidates(self.cluster_field)
            chosen_index = choose_growth_candidate(candidates, self.nutrient_field, self.eta)

            if chosen_index is not None:
                self.cluster_field[chosen_index[0], chosen_index[1]] = 1
                self.nutrient_field[chosen_index[0], chosen_index[1]] = 0  # Set as fixed point

                # Check if the top boundary is reached
                if chosen_index[0] == 0:
                    self.termination_flag = True

            # Record current state
            self.history.append((self.nutrient_field.copy(), self.cluster_field.copy()))

            if plot_interval > 0 and (step + 1) % plot_interval == 0:
                print(f"Step {step + 1} with η = {self.eta}")

        if self.termination_step < 0:
            self.termination_step = growth_steps

def update_animation(frame, img, history, dla):
    """
    Update the animation frame.
    :param frame: Current frame number.
    :param img: Image object.
    :param history: History of grid states.
    :param dla: DLA model instance.
    :return: Updated image object.
    """
    nutrient_field, cluster_field = history[frame]
    
    # Create a copy of the nutrient field
    image = nutrient_field.copy()
    
    # Set cluster points to NaN (to ignore them in the nutrient field plot)
    image[cluster_field == 1] = np.nan
    
    # Plot the nutrient field with viridis colormap
    img.set_data(image)
    img.set_cmap("viridis")  # Use viridis for nutrient field
    img.set_clim(0, 1)  # Set color limits

    # Overlay cluster points in white (fully opaque)
    cluster_mask = np.ma.masked_where(cluster_field == 0, cluster_field)  # Mask non-cluster points
    plt.gca().images[-1].set_data(cluster_mask)  # Update the overlay
    plt.gca().images[-1].set_cmap("binary")  # Use binary colormap (white for 1, black for 0)
    plt.gca().images[-1].set_alpha(1.0)  # Set alpha to 1.0 for fully opaque

    # Update title with eta value
    ax.set_title(f"Step {frame} with η = {dla.eta}", fontsize=12)  # Add eta value
    fig.canvas.draw_idle()  # Force refresh to ensure title update

    # Stop animation when top boundary is reached
    if frame >= dla.termination_step - 1:
        print("Animation stopped at step:", frame)
        ani.event_source.stop()  # Stop animation without closing the window

    return img,

def calculate_fractal_dimension(cluster_field):
    """
    Calculate the fractal dimension using the box-counting method.
    :param cluster_field: Particle aggregation field.
    :return: Fractal dimension.
    """
    cluster_points = np.argwhere(cluster_field == 1)
    if len(cluster_points) == 0:
        return 0.0

    # Define range of box sizes
    box_sizes = np.logspace(0, np.log10(min(cluster_field.shape)), base=10, num=20)
    box_sizes = np.unique(np.floor(box_sizes)).astype(int)
    box_sizes = box_sizes[box_sizes > 0]

    # Calculate number of boxes for each box size
    box_counts = []
    for box_size in box_sizes:
        bins = [np.arange(0, cluster_field.shape[i] + box_size, box_size) for i in range(2)]
        hist, _, _ = np.histogram2d(cluster_points[:, 0], cluster_points[:, 1], bins=bins)
        box_counts.append(np.sum(hist > 0))

    # Fit fractal dimension
    if len(box_sizes) < 2:
        return 0.0

    coeffs = np.polyfit(np.log(1 / box_sizes), np.log(box_counts), 1)
    return coeffs[0]

def study_eta_impact(eta_values, grid_size=(100, 100), growth_steps=5000, omega=1.7):
    """
    Study the impact of eta on the DLA model.
    :param eta_values: List of eta values.
    :param grid_size: Grid size.
    :param growth_steps: Maximum number of growth steps.
    :param omega: SOR relaxation factor.
    """
    results = []
    fractal_dimensions = []
    for eta in eta_values:
        dla = DiffusionLimitedAggregation(grid_size, eta)
        dla.grow(growth_steps, plot_interval=0, omega=omega)
        results.append((eta, dla.termination_step, dla.cluster_field, dla.nutrient_field))

        # Calculate fractal dimension
        fractal_dim = calculate_fractal_dimension(dla.cluster_field)
        fractal_dimensions.append(fractal_dim)
        print(f"η = {eta}, Fractal Dimension = {fractal_dim:.4f}")

    # Plot cluster morphology for different eta values
    plt.figure(figsize=(10, 10))  # Adjust canvas size for 2x2 layout

    for i, (eta, step, cluster_field, nutrient_field) in enumerate(results):
        plt.subplot(2, 2, i + 1)  # 2 rows, 2 columns, i+1 subplot
        
        # Combine nutrient_field and cluster_field for gradient heatmap
        image = nutrient_field.copy()
        image[cluster_field == 1] = np.nan  # Set cluster points to NaN
        
        # Plot nutrient field with viridis colormap
        plt.imshow(image, cmap="viridis", vmin=0, vmax=1, extent=[0, grid_size[1], 0, grid_size[0]], origin='upper')
        
        # Overlay cluster points in white (fully opaque)
        cluster_mask = np.ma.masked_where(cluster_field == 0, cluster_field)  # Mask non-cluster points
        plt.imshow(cluster_mask, cmap="binary", vmin=0, vmax=1, alpha=1.0)  # Use binary colormap (white for 1, black for 0)
        
        # Set title with eta value and step count
        plt.title(f"η = {eta}, Steps = {step}", fontsize=22)

    # Adjust subplot spacing
    plt.tight_layout()

    # Save figure to 'figures' directory
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")  # Path to 'figures' directory
    os.makedirs(figures_dir, exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(os.path.join(figures_dir, "eta_impact.png"))  # Save figure
    plt.show()

    # Plot eta vs fractal dimension
    plt.figure()
    plt.plot(eta_values, fractal_dimensions, marker='o')
    plt.xlabel('η', fontsize=18)
    plt.ylabel('Fractal Dimension', fontsize=18)
    plt.title('Impact of Eta on Fractal Dimension', fontsize=18)
    
    # Save figure to 'figures' directory
    plt.savefig(os.path.join(figures_dir, "eta_fractal_dimension.png"))  # Save figure
    plt.show()

def study_eta_omega_impact(eta_values, omega_values, grid_size=(100, 100), growth_steps=5000):
    """
    Study the impact of eta and omega on SOR iterations.
    :param eta_values: List of eta values to test.
    :param omega_values: List of omega values to test.
    :param grid_size: Grid size.
    :param growth_steps: Maximum number of growth steps.
    """
    results = []
    for eta in eta_values:
        for omega in omega_values:
            dla = DiffusionLimitedAggregation(grid_size, eta)
            total_iterations = 0  # Track total SOR iterations
            for step in range(growth_steps):
                if dla.termination_flag:
                    break

                # Update nutrient field and record iterations
                iter_count = dla.update_nutrient_field(omega=omega)
                total_iterations += iter_count

                # Growth process
                candidates = get_growth_candidates(dla.cluster_field)
                chosen_index = choose_growth_candidate(candidates, dla.nutrient_field, eta)
                if chosen_index is not None:
                    dla.cluster_field[chosen_index[0], chosen_index[1]] = 1
                    dla.nutrient_field[chosen_index[0], chosen_index[1]] = 0  # Set as fixed point

                    # Check if top boundary is reached
                    if chosen_index[0] == 0:
                        dla.termination_flag = True

            # Record average iterations for current eta and omega
            average_iterations = total_iterations / (step + 1)  # Average iterations per nutrient update
            results.append((eta, omega, average_iterations))

    # Convert results to 2D arrays for plotting
    eta_grid, omega_grid = np.meshgrid(eta_values, omega_values)  # Swap eta and omega order
    iteration_grid = np.zeros_like(eta_grid, dtype=float)

    for result in results:
        eta, omega, avg_iter = result
        eta_index = eta_values.index(eta)
        omega_index = omega_values.index(omega)
        iteration_grid[omega_index, eta_index] = avg_iter  # Note index order

    # Plot contour map
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(eta_grid, omega_grid, iteration_grid, levels=50, cmap='viridis')
    
    # Add colorbar and adjust font size
    cbar = plt.colorbar(contour)
    cbar.set_label("Average SOR Iterations", fontsize=22)  
    cbar.ax.tick_params(labelsize=12)  

    # Adjust axis labels and title
    plt.xlabel('η', fontsize=22)  
    plt.ylabel('ω', fontsize=22)  
    plt.title('Contour Plot', fontsize=22)  

    # Save figure to 'figures' directory
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")  # Path to 'figures' directory
    os.makedirs(figures_dir, exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(os.path.join(figures_dir, "eta_omega_impact.png"))  # Save figure
    plt.show()

if __name__ == "__main__":
    # Parameter settings
    grid_size = (100, 100)  # Grid size
    eta = 1.0  # Parameter controlling growth probability
    growth_steps = 5000  # Maximum growth steps
    plot_interval = 100  # Plot interval

    # Initialize DLA model
    dla = DiffusionLimitedAggregation(grid_size, eta)
    dla.grow(growth_steps, plot_interval)

    # Create animation
    fig, ax = plt.subplots()
    nutrient_field, cluster_field = dla.history[0]
    image = nutrient_field.copy()
    image[cluster_field == 1] = 1  # Highlight cluster points
    img = ax.imshow(image, cmap="viridis", vmin=0, vmax=1, extent=[0, grid_size[1], 0, grid_size[0]], origin='upper')
    # Use FuncAnimation
    ani = FuncAnimation(fig, update_animation, fargs=(img, dla.history, dla),
                        frames=len(dla.history), interval=50, blit=False)

    plt.colorbar(img, label="Nutrient Concentration / Cluster")

    # Save animation as GIF
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")  # Path to 'figures' directory
    os.makedirs(figures_dir, exist_ok=True)  # Create directory if it doesn't exist
    gif_path = os.path.join(figures_dir, "dla_animation.gif")  # GIF file path
    ani.save(gif_path, writer="pillow", fps=15)  # Save as GIF, fps controls frame rate
    print(f"Animation saved to {gif_path}")

    plt.show()

    # Study the impact of eta
    eta_values = [0.5, 1.0, 1.5, 2.0]
    study_eta_impact(eta_values)

    # Study the impact of eta and omega
    omega_values = [1.0, 1.2, 1.5, 1.7, 1.9]
    study_eta_omega_impact(eta_values, omega_values, grid_size, growth_steps)