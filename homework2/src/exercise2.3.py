import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Nx, Ny = 128, 128   # Grid size
dx = 1.0            # Space step
dt = 1.0            # Time step
D_u, D_v = 0.16, 0.08  # Diffusion coefficients
T = 5000  # Total time steps
save_interval = 500  # Save interval for visualization


param_sets = [
    (0.035, 0.060),  # Default values
    (0.025, 0.060),  # Lower F
    (0.045, 0.060),  # Higher F
    (0.035, 0.070),  # Higher K
    (0.035, 0.050)   # Lower K
]


def laplacian(Z):
    """Computes the discrete Laplacian using a 5-point stencil."""
    Z_top    = np.roll(Z, shift=-1, axis=0)
    Z_bottom = np.roll(Z, shift=1, axis=0)
    Z_left   = np.roll(Z, shift=-1, axis=1)
    Z_right  = np.roll(Z, shift=1, axis=1)
    return (Z_top + Z_bottom + Z_left + Z_right - 4*Z) / dx**2


def evolve(U, V, D_u, D_v, F, K, dt, steps, save_interval):
    """Runs the Gray-Scott reaction-diffusion model."""
    snapshots = []  # Store snapshots for visualization

    for t in range(steps):
        Lu = laplacian(U)
        Lv = laplacian(V)
        
        UVV = U * V**2
        dU = D_u * Lu - UVV + F * (1 - U)
        dV = D_v * Lv + UVV - (F + K) * V
        
        U += dt * dU
        V += dt * dV
        
        if t % save_interval == 0:
            snapshots.append(U.copy())  # Store U state
    
    return snapshots


for F, K in param_sets:
    print(f"Running simulation for F={F}, K={K}")

    # Initialize U, V with correct initial conditions
    U = np.full((Nx, Ny), 0.5)  # Set U = 0.5 everywhere
    V = np.zeros((Nx, Ny))      # Set V = 0 everywhere

    # Define the central perturbation region
    mid = Nx // 2
    r = int(Nx * 0.05)  # Perturbation size
    noise = np.random.normal(0, 0.02, (2*r, 2*r))  # Add noise

    # Apply perturbation to the center
    V[mid-r:mid+r, mid-r:mid+r] = 0.25 + noise  # Set V = 0.25 + noise in the center

    # Run simulation
    snapshots = evolve(U, V, D_u, D_v, F, K, dt, T, save_interval)

    
    fig, axes = plt.subplots(1, len(snapshots), figsize=(15, 5))

    for ax, snapshot, step in zip(axes, snapshots, range(0, T, save_interval)):
        im = ax.imshow(snapshot, cmap='inferno', extent=[0, Nx, 0, Ny])
        ax.set_title(f"t = {step}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(f"homework2/figures/gray_scott_F{F}_K{K}.png", dpi=300)  # Save figure
    plt.show()

   
    fig, ax = plt.subplots()
    im = ax.imshow(snapshots[0], cmap="inferno", animated=True)
    ax.set_title(f"F={F}, K={K}")

    def update(frame):
        im.set_array(snapshots[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=50)
    ani.save(f"homework2/figures/gray_scott_F{F}_K{K}.gif", fps=10, dpi=300)
    plt.close(fig)  # Close figure to prevent duplicate display
