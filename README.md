# Scientific Computing Coursework

This repository contains solutions for assignments in the Scientific Computing course. Each assignment is structured into separate folders with relevant code and figures.

## Repository Structure

- `scientific_computing_homework1/`

  - `src/`: Contains Python scripts for the first assignment.
    - `exercise_1.py`: Solution for Set 1, Question 1.
    - `exercise_2.py`: Solution for Set 1, Question 2.
    - `exercise_3-6.py`: Solutions for Set 1, Questions 3-6.
  - `figures/`: Stores generated figures from simulations.

- `scientific_computing_homework2/`

  - `src/`: Contains Python scripts for the second assignment.
    - `exercise_2.1.py`: Solution for Set 2, Question 1 (Diffusion-Limited Aggregation using SOR method).
    - `exercise_2.2.py`: Solution for Set 2, Question 2 (Monte Carlo simulation of DLA).
    - `exercise_2.3.py`: Solution for Set 2, Question 3 (Gray-Scott reaction-diffusion model).
  - `figures/`: Stores images and animations generated from simulations.

## Usage

Clone the repository and navigate to the relevant assignment directory. Run the scripts in the `src/` folder to execute simulations and generate results.

```bash
git clone https://github.com/ziyi-xing/scientific-computing.git
cd homework2/src
python exercise_2.3.py
```

## Running Individual Scripts

#### **1. Diffusion-Limited Aggregation with SOR (exercise\_2.1.py)**

```bash
python exercise_2.1.py
```

- Runs a diffusion-limited aggregation simulation using the Successive Over-Relaxation (SOR) method.
- Generates animations and figures saved in `figures/`.
- Key parameters: `grid_size`, `eta`, `growth_steps`, `omega`.

#### **2. Monte Carlo Diffusion-Limited Aggregation (exercise\_2.2.py)**

```bash
python exercise_2.2.py
```

- Implements Monte Carlo simulation of DLA, where particles undergo random walks and stick with probability `ps`.
- Generates GIF animations and a combined final-state image in `figures2.2/`.
- Key parameters: `grid_size`, `ps`, `max_iter`.

#### **3. Gray-Scott Reaction-Diffusion Model (exercise\_2.3.py)**

```bash
python exercise_2.3.py
```

- Simulates the Gray-Scott reaction-diffusion process for different `F, K` values.
- The script **does not contain ************************`if __name__ == "__main__":`************************, but automatically runs all predefined simulations**.
- If you want to customize parameters or run specific cases, modify the `param_sets` list or directly call the `evolve()` function.
- Saves visualizations as images and GIFs in `homework2/figures/`.
- Key parameters: `Nx`, `Ny`, `D_u`, `D_v`, `T`, `save_interval`, `F`, `K`.

## Understanding the Code Structure

Some scripts use `if __name__ == "__main__":` to define the execution entry point, while others automatically execute upon running.

#### **Code Organization in Each Script**

- **exercise\_2.1.py** (DLA with SOR)

  - Contains functions for diffusion simulation.
  - Uses `if __name__ == "__main__":` to execute the main simulation.

- **exercise\_2.2.py** (Monte Carlo DLA)

  - Defines a `MonteCarloDLA` class.
  - Uses `if __name__ == "__main__":` to execute the full Monte Carlo simulation.

- **exercise\_2.3.py** (Gray-Scott Reaction-Diffusion)

  - Implements reaction-diffusion equations.
  - **Does not contain ************************`if __name__ == "__main__":`************************, but runs simulations automatically**.
  - To customize behavior, modify the `param_sets` list or manually call `evolve()`.

You can modify parameters within the script before running them.

## Dependencies

The scripts require the following Python packages:

- `numpy`
- `matplotlib`
- `numba`
- `pillow`

To install the required dependencies, use the following command:

```bash
pip install numpy matplotlib numba pillow
```

## Notes

- Ensure output directories (`figures/`, `figures2.2/`, `homework2/figures/`) exist before running the scripts.
- Some functions are accelerated using `numba` for performance improvements.
- Modify parameter values inside the scripts to observe different simulation results.

## License

This project is open-source. Feel free to modify and use it for research and learning purposes.

