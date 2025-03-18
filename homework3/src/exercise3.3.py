import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ----------------- Utility Function: Create Directory ----------------- #
def create_directory(path):
    """
    Create a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

# ----------------- Task I: Leapfrog method for a simple harmonic oscillator ----------------- #
def leapfrog_harmonic_oscillator(k, x0, v0, dt, T):
    """
    Simulates a simple harmonic oscillator using the Leapfrog method.

    Parameters:
    k   - Spring constant
    x0  - Initial position
    v0  - Initial velocity
    dt  - Time step
    T   - Total simulation time

    Returns:
    t, x, v - Time, position, and velocity arrays
    """
    m = 1  # Given that mass is 1
    N = int(T / dt)  # Number of timesteps
    t = np.linspace(0, T, N)  # Time array

    x = np.zeros(N)
    v_half = np.zeros(N)  # Velocity at half-steps

    # Initial conditions
    x[0] = x0
    v_half[0] = v0 - 0.5 * dt * (-k * x0 / m)  # Initialize v at t = dt/2

    # Leapfrog integration loop
    for n in range(N - 1):
        x[n + 1] = x[n] + dt * v_half[n]  # Position update at full step
        a_new = -k * x[n + 1] / m  # Compute new acceleration
        v_half[n + 1] = v_half[n] + dt * a_new  # Velocity update at half-step

    # Approximate velocity at integer time steps (optional, if needed)
    v = v_half + 0.5 * dt * (-k * x / m)  # Optional: approximate v at integer steps

    return t, x, v_half  # Return v_half instead of v

# ----------------- Task J: Adding external driving force ----------------- #
def leapfrog_driven_oscillator(k, A, omega, x0, v0, dt, T):
    """
    Simulates a driven harmonic oscillator with an external periodic force using the Leapfrog method.

    Parameters:
    k     - Spring constant
    A     - Amplitude of driving force
    omega - Frequency of driving force
    x0    - Initial position
    v0    - Initial velocity
    dt    - Time step
    T     - Total simulation time

    Returns:
    t, x, v_half - Time, position, and velocity (at half-steps) arrays
    """
    m = 1  # Given that mass is 1
    N = int(T / dt)
    t = np.linspace(0, T, N)

    x = np.zeros(N)
    v_half = np.zeros(N)  # Velocity at half-steps

    # Initial conditions
    x[0] = x0
    v_half[0] = v0 - 0.5 * dt * (-k * x0 / m + A * np.cos(0) / m)  # Initialize v at t = dt/2

    # Leapfrog integration loop
    for n in range(N - 1):
        x[n + 1] = x[n] + dt * v_half[n]  # Position update
        a_new = (-k * x[n + 1] + A * np.cos(omega * t[n + 1])) / m  # Acceleration with external force
        v_half[n + 1] = v_half[n] + dt * a_new  # Velocity update at half-step

    # Approximate velocity at integer time steps (optional, if needed)
    v = v_half + 0.5 * dt * (-k * x / m + A * np.cos(omega * t) / m)  # Optional: approximate v at integer steps

    return t, x, v_half  # Return v_half instead of v

# ----------------- Extra Bonus: Compare Leapfrog with RK45 ----------------- #
def harmonic_oscillator_ode(t, y, k):
    """
    ODE system for the harmonic oscillator (used by RK45).
    """
    x, v = y
    return [v, -k * x]

def compare_leapfrog_rk45(k, x0, v0, dt, T):
    """
    Compare Leapfrog method with RK45 for a harmonic oscillator.
    """
    # Leapfrog method
    t_leapfrog, x_leapfrog, v_half_leapfrog = leapfrog_harmonic_oscillator(k, x0, v0, dt, T)

    # RK45 method
    sol = solve_ivp(harmonic_oscillator_ode, (0, T), [x0, v0], args=(k,), method='RK45', t_eval=t_leapfrog)
    t_rk45, x_rk45, v_rk45 = sol.t, sol.y[0], sol.y[1]

    # Plot phase space (position vs. velocity) for first 10s and last 10s
    plt.figure(figsize=(12, 6))

    # Leapfrog Phase Space Plot
    plt.subplot(1, 2, 1)
    plt.plot(x_leapfrog[:1000], v_half_leapfrog[:1000], label='Leapfrog (0-10s)')
    plt.plot(x_leapfrog[-1000:], v_half_leapfrog[-1000:], label='Leapfrog (90-100s)')
    plt.xlabel("Position (x)", fontsize=20)
    plt.ylabel("Velocity (v)", fontsize=20)
    plt.title("Leapfrog Method", fontsize=22)
    plt.legend(fontsize=18)
    plt.grid()

    # RK45 Phase Space Plot
    plt.subplot(1, 2, 2)
    plt.plot(x_rk45[:1000], v_rk45[:1000], label='RK45 (0-10s)')
    plt.plot(x_rk45[-1000:], v_rk45[-1000:], label='RK45 (90-100s)')
    plt.xlabel("Position (x)", fontsize=20)
    plt.ylabel("Velocity (v)", fontsize=20)
    plt.title("RK45 Method", fontsize=22)
    plt.legend(fontsize=18)
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join("homework3/figure/figure3.3", "extra_bonus.png"), dpi=300)  # Save the figure
    plt.show()

# ----------------- Plotting Results ----------------- #
def plot_results():
    # Task I: Initial conditions
    x0, v0, dt, T_taskI = 1.0, 0.0, 0.01, 20.0  # Task I: T = 20

    # Different k values for Task I
    k_values = [0.5, 1.0, 2.0]

    # Task I: Position vs. Time
    plt.figure(figsize=(12, 4))
    for k in k_values:
        t, x, _ = leapfrog_harmonic_oscillator(k, x0, v0, dt, T_taskI)
        plt.plot(t, x, label=f"k={k}")
    plt.xlabel("Time (t)", fontsize=13)
    plt.ylabel("Position (x)", fontsize=13)
    plt.title("Position vs. Time for Different k Values", fontsize=15)
    plt.legend(fontsize=13)
    plt.grid()
    plt.savefig(os.path.join("homework3/figure/figure3.3", "taskI_position_vs_time.png"), dpi=300)  # Save the figure
    plt.show()

    # Task I: Velocity vs. Time
    plt.figure(figsize=(12, 4))
    for k in k_values:
        t, _, v_half = leapfrog_harmonic_oscillator(k, x0, v0, dt, T_taskI)
        plt.plot(t, v_half, label=f"k={k}")
    plt.xlabel("Time (t)", fontsize=13)
    plt.ylabel("Velocity (v)", fontsize=13)
    plt.title("Velocity vs. Time for Different k Values", fontsize=15)
    plt.legend(fontsize=13)
    plt.grid()
    plt.savefig(os.path.join("homework3/figure/figure3.3", "taskI_velocity_vs_time.png"), dpi=300)  # Save the figure
    plt.show()

    # Task I: Phase Space (v vs. x)
    plt.figure(figsize=(12, 4))
    for k in k_values:
        t, x, v_half = leapfrog_harmonic_oscillator(k, x0, v0, dt, T_taskI)
        plt.plot(x, v_half, label=f"k={k}")
    plt.xlabel("Position (x)", fontsize=13)
    plt.ylabel("Velocity (v)", fontsize=13)
    plt.title("Phase Space (v vs. x) for Different k Values", fontsize=15)
    plt.legend(fontsize=13)
    plt.grid()
    plt.savefig(os.path.join("homework3/figure/figure3.3", "taskI_phase_space.png"), dpi=300)  # Save the figure
    plt.show()

    # Task J: Initial conditions
    T_taskJ = 50.0  # Task J: T = 50
    A = 0.5
    omega_values = [0.8, 1.0, 1.2]  # Near resonance
    colors = ['orange', 'blue', 'red']

    # Task J: Combined Phase Space
    plt.figure(figsize=(8, 6))
    for i, omega in enumerate(omega_values):
        t, x, v_half = leapfrog_driven_oscillator(1.0, A, omega, x0, v0, dt, T_taskJ)
        plt.plot(x, v_half, color=colors[i], label=f"ω={omega}")
    plt.xlabel("Position (x)", fontsize=20)
    plt.ylabel("Velocity (v)", fontsize=20)
    plt.title("Combined Phase Space (v vs. x)", fontsize=22)
    plt.legend(fontsize=16)
    plt.grid()
    plt.savefig(os.path.join("homework3/figure/figure3.3", "taskJ_combined_phase_space.png"), dpi=300)  # Save the figure
    plt.show()

    # Task J: Individual Phase Space for each omega
    for i, omega in enumerate(omega_values):
        t, x, v_half = leapfrog_driven_oscillator(1.0, A, omega, x0, v0, dt, T_taskJ)
        plt.figure(figsize=(8, 6))
        plt.plot(x, v_half, color=colors[i], label=f"ω={omega}")
        plt.xlabel("Position (x)", fontsize=20)
        plt.ylabel("Velocity (v)", fontsize=20)
        plt.title(f"Phase Space for ω={omega}", fontsize=22)
        plt.legend(fontsize=16)
        plt.grid()
        plt.savefig(os.path.join("homework3/figure/figure3.3", f"taskJ_phase_space_omega_{omega}.png"), dpi=300)  # Save the figure
        plt.show()

    
# ----------------- Main Execution ----------------- #
if __name__ == "__main__":
    # Create the figure/figure3.3 directory if it does not exist
    create_directory("homework3/figure/figure3.3")

    # Run the simulation and plot results
    plot_results()

    # Compare Leapfrog with RK45 (Extra Bonus)
    compare_leapfrog_rk45(k=1.0, x0=1.0, v0=0.0, dt=0.01, T=10000.0)