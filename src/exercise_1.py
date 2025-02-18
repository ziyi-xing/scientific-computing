import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
u = np.zeros((N, steps))  # Store wave amplitude at each time step

# Set initial conditions
initial_condition = 3  # Choose initial condition: 1, 2, or 3

if initial_condition == 1:
    u[:, 0] = np.sin(2 * np.pi * x)  # Ψ(x,0) = sin(2πx)
elif initial_condition == 2:
    u[:, 0] = np.sin(5 * np.pi * x)  # Ψ(x,0) = sin(5πx)
elif initial_condition == 3:
    for i in range(N):
        if 1/5 < x[i] < 2/5:
            u[i, 0] = np.sin(5 * np.pi * x[i])  # Apply condition in the interval (1/5, 2/5)

# Compute second time step using finite difference method
for i in range(1, N - 1):
    u[i, 1] = u[i, 0] + 0.5 * (c * dt / dx) ** 2 * (u[i + 1, 0] - 2 * u[i, 0] + u[i - 1, 0])

# Time-stepping loop using finite difference method
for t in range(1, steps - 1):
    for i in range(1, N - 1):
        u[i, t + 1] = 2 * u[i, t] - u[i, t - 1] + (c * dt / dx) ** 2 * (u[i + 1, t] - 2 * u[i, t] + u[i - 1, t])

# Plot wave evolution at different time steps
time_indices = [0, int(steps/4), int(steps/2), int(3*steps/4), steps-1]
plt.figure(figsize=(8, 5))
for t in time_indices:
    plt.plot(x, u[:, t], label=f"t = {t*dt:.3f}s")
plt.xlabel("Position (x)")
plt.ylabel("Amplitude Ψ(x, t)")
plt.title("Wave evolution at different time steps")
plt.legend()
plt.savefig("wave_evolution.png", dpi=300)  # Save the figure
plt.show()

# Create animated plot
fig, ax = plt.subplots()
line, = ax.plot(x, u[:, 0])
ax.set_ylim(-1, 1)  # Set y-axis limits for better visualization

def update(frame):
    line.set_ydata(u[:, frame])  # Update wave amplitude for each frame
    return line,

ani = animation.FuncAnimation(fig, update, frames=steps, interval=10)
# Define FFMpegWriter
writer = animation.FFMpegWriter(fps=30, codec="libx264")

# Save animation using the updated method
ani.save("wave_animation.mp4", writer=writer)

plt.show()



