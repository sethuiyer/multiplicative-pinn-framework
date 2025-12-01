# Multiplicative PINN Framework

**A Research Framework for Solving PDEs with Multiplicative Constraints**

This package implements the Multiplicative Axis Framework for physics-informed neural networks (PINNs). It provides a method for enforcing physical constraints in deep learning models using prime-based Euler gates and exponential barriers, rather than traditional additive penalties.

## Core Concepts

### 1. Multiplicative Axis
Instead of `Loss = DataLoss + λ * PhysicsLoss`, we use:
`Loss = DataLoss * ConstraintFactor(PhysicsLoss)`

This allows the physics constraints to dynamically modulate the gradient flow, preventing the "fighting" often seen between data and physics terms.

### 2. Euler Product Gate (Attenuation)
We use a truncated Euler product over small primes to create a "gate" that scales down the loss when constraints are satisfied:
$$ G(v) = \prod_{p \in \mathcal{P}} (1 - p^{-\tau v}) $$
This prevents gradient vanishing in valid regions while maintaining a smooth geometric structure.

### 3. Exponential Barrier (Amplification)
We use an exponential function to massively amplify gradients when violations occur, acting as a soft "hard constraint":
$$ B(v) = e^{\gamma v} $$

## Directory Structure

- **`core/`**: The heart of the framework.
    - `pinn_multiplicative_constraints.py`: Main implementation of the PINN logic with multiplicative layers.
    - `multi_constraint_graph.py`: Graph-based approach for handling multiple conflicting constraints.
- **`examples/`**: Runnable demonstrations.
    - `navier_stokes_test.py`: **Key Demo**. Solves 2D Navier-Stokes equations using the framework.
    - `fluid_simulation_demo.py`: Visual demo of fluid dynamics.
- **`analysis/`**: Validation and benchmarking scripts.
- **`docs/`**: Detailed research summaries and validation reports.

## Getting Started

To run the Navier-Stokes demonstration:

```bash
# From the parent directory
python3 -m multiplicative_pinn_framework.examples.navier_stokes_test
```

*Note: You may need to adjust python path or run as a module to handle imports correctly.*
