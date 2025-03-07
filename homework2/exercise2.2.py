import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import os  # Save gif


class MonteCarloDLA:
    def __init__(self, grid_size, ps, max_iter):
        """
        Initialize the Monte Carlo DLA model.
        :param grid_size: Grid size (height, width).
        :param ps: Sticking probability.
        :param max_iter: Maximum number of iterations.
        """
        self.grid_size = grid_size
        self.ps = ps
        self.max_iter = max_iter

        # Initialize the grid
        self.grid = np.zeros(grid_size, dtype=int)  # 0 represents an empty point, 1 represents a cluster point
        self.history = []  # Record the grid state after each cluster change

        # Set the initial cluster point (middle of the bottom)
        self.seed_position = (grid_size[0] - 1, grid_size[1] // 2)
        self.grid[self.seed_position] = 1
        self.history.append(self.grid.copy())  # Record the initial state

    def MonteCarlo(self):
        """
        Run the Monte Carlo simulation.
        """
        self.grid = run_montecarlo(self.grid, self.grid_size, self.ps, self.max_iter, self.history)

    def animate(self):
        """
        Generate an animation.
        """
        fig, ax = plt.subplots()
        flipped_data = self.history[0][::-1, :]  # Flip the y-axis
        img = ax.imshow(flipped_data, cmap='viridis', vmin=0, vmax=1)

        ax.set_ylim([0, 100])

        def update(frame):
            img.set_data(self.history[frame][::-1, :])
            ax.set_title(f"ps={ps}, Step {frame}")
            return img,

        ani = FuncAnimation(fig, update, frames=len(self.history), interval=50, blit=True)
        plt.show()
        return ani


@njit
def is_neighbor_of_cluster(grid, x, y):
    """
    Check if the point (x, y) is a neighbor of the cluster.
    :param grid: The grid.
    :param x: Row coordinate.
    :param y: Column coordinate.
    :return: True if it is a neighbor of the cluster, otherwise False.
    """
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx, ny] == 1:
                return True
    return False


@njit
def run_montecarlo(grid, grid_size, ps, max_iter, history):
    """
    Run the Monte Carlo simulation, accelerated by Numba.
    """
    for _ in range(max_iter):
        x, y = 0, np.random.randint(0, grid_size[1])  # Release a particle randomly at the top

        while True:
            # Random walk
            direction = np.random.randint(4)
            if direction == 0:  # Up
                x_new, y_new = x - 1, y
            elif direction == 1:  # Down
                x_new, y_new = x + 1, y
            elif direction == 2:  # Left
                x_new, y_new = x, y - 1
            else:  # Right
                x_new, y_new = x, y + 1

            # Check boundary conditions
            if x_new < 0:  # Exceed the top boundary
                break  # Stop the simulation if the cluster touches the top boundary
            if x_new >= grid_size[0]:  # Exceed the bottom boundary
                break  # Remove the particle and release a new one
            if y_new < 0:  # Exceed the left boundary
                y_new = grid_size[1] - 1
            elif y_new >= grid_size[1]:  # Exceed the right boundary
                y_new = 0
            # Check if the particle moves inside the cluster
            if grid[x_new, y_new] == 1:
                continue  # Cannot move inside the cluster, choose a new direction

            # Check if the particle reaches the neighborhood of the cluster
            if is_neighbor_of_cluster(grid, x_new, y_new) and (grid[x_new, y_new] != 1):
                if np.random.rand() < ps:  # Stick with probability ps
                    grid[x_new, y_new] = 1  # Join the cluster
                    history.append(grid.copy())  # Record the current state
                    if x_new == 0:  # If the cluster touches the top boundary
                        return grid  # Stop the simulation
                    break
                else:
                    x, y = x_new, y_new  # Continue walking
            else:
                x, y = x_new, y_new  # Continue walking
    return grid


# Test code
if __name__ == "__main__":
    # Define different ps values
    ps_values = [0.1, 0.5, 0.7, 1]

    # Create a directory to save results
    figures_dir = os.path.join(os.path.dirname(__file__), "figures2.2")
    os.makedirs(figures_dir, exist_ok=True)  # Create the directory if it does not exist

    for ps in ps_values:
        # Initialize the Monte Carlo DLA model
        dla = MonteCarloDLA(grid_size=(100, 100), ps=ps, max_iter=70500)

        # Run the Monte Carlo simulation
        dla.MonteCarlo()

        # Generate animation and get the ani object
        ani = dla.animate()

        # Save the animation as a GIF
        gif_path = os.path.join(figures_dir, f"dla_MCanimation_ps_{ps}.gif")  # Dynamically generate the filename
        ani.save(gif_path, writer="pillow", fps=15)  # Save as GIF, fps controls the frame rate
        print(f"Animation saved to {gif_path}")  # Print the path where the GIF is saved

        # Save the final frame as an image
        final_frame = dla.history[-1][::-1, :]  # Get the final frame
        final_image_path = os.path.join(figures_dir, f"MC_final_frame_ps_{ps}.png")  # Dynamically generate the filename
        plt.imshow(final_frame, cmap='viridis', vmin=0, vmax=1, origin='lower')
        step_when_top_reached = len(dla.history)
        plt.title(f"ps={ps}, Step={step_when_top_reached}")  # Set the title with ps and step number
        plt.savefig(final_image_path)  # Save the final frame image
        plt.close()  # Close the image window
        print(f"Final frame saved to {final_image_path}")  # Print the path where the final frame is saved