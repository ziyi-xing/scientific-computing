import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Ensure the figures directory exists
os.makedirs("figures", exist_ok=True)

# Define parameters
L = 1       # String length
N = 100     # Number of spatial grid points
dx = L / N  # Spatial step size
dt = 0.001  # Time step size
c = 1       # Wave propagation speed
T = 2       # Total simulation time
steps = int(T / dt)  # Number of time steps

# Initialize spatial grid
x = np.linspace(0, L, N)

# Different initial conditions
initial_conditions = {
    1: lambda x: np.sin(2 * np.pi * x),
    2: lambda x: np.sin(5 * np.pi * x),
    3: lambda x: np.where((1/5 < x) & (x < 2/5), np.sin(5 * np.pi * x), 0)
}

# Time indices for plotting static images
time_indices = [0, int(steps/4), int(steps/2), int(3*steps/4), steps-1]

# Loop over three initial conditions to create both static and animated plots
for ic in initial_conditions:
    u = np.zeros((N, steps))  # Store wave amplitude at each time step
    u[:, 0] = initial_conditions[ic](x)  # Apply initial condition

    # Compute second time step using finite difference method
    for i in range(1, N - 1):
        u[i, 1] = u[i, 0] + 0.5 * (c * dt / dx) ** 2 * (u[i + 1, 0] - 2 * u[i, 0] + u[i - 1, 0])

    # Time-stepping loop using finite difference method
    for t in range(1, steps - 1):
        for i in range(1, N - 1):
            u[i, t + 1] = 2 * u[i, t] - u[i, t - 1] + (c * dt / dx) ** 2 * (u[i + 1, t] - 2 * u[i, t] + u[i - 1, t])

    # **Create and save static plot**
    plt.figure(figsize=(8, 5))
    for t in time_indices:
        plt.plot(x, u[:, t], label=f"t = {t*dt:.3f}s")
    
    plt.xlabel("Position (x)")
    plt.ylabel("Amplitude Ψ(x, t)")
    plt.title(f"Wave evolution for Initial Condition {ic}")
    plt.legend()

    # Save figure before showing
    plt.savefig(f"figures/wave_evolution_{ic}.png", dpi=300)
    print(f"Saved: figures/wave_evolution_{ic}.png")  # Debugging output
    plt.show()

    # **Create animated plot**
    fig, ax = plt.subplots()
    line, = ax.plot(x, u[:, 0])
    ax.set_ylim(-1, 1)  # Set y-axis limits for better visualization
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Amplitude Ψ(x, t)")
    ax.set_title(f"Wave Animation for Initial Condition {ic}")

    def update(frame):
        line.set_ydata(u[:, frame])  # Update wave amplitude for each frame
        return line,

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=10)

    # Define FFMpegWriter
    writer = animation.FFMpegWriter(fps=30, codec="libx264")

    # Save animation file
    ani.save(f"figures/wave_animation_{ic}.mp4", writer=writer)
    print(f"Saved: figures/wave_animation_{ic}.mp4")  # Debugging output

    plt.show()
