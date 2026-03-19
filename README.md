# ML4Sci GSoC Submission

This submission is a test for ML4Sci's GSOC problem statement.

**Author:** Basit Warsi

## Overview

This project applies a Physics-Informed Neural Network (PINN) to a 2D incompressible Navier-Stokes problem.  
The main notebook trains a neural network to learn the velocity and pressure fields directly from:

- the Navier-Stokes momentum equations,
- the incompressibility constraint,
- no-slip boundary conditions on the walls,
- and an imposed initial velocity field.

The result is a PINN-based solver that predicts the time-decay of a viscous flow inside a square domain and generates visual outputs showing how the flow weakens over time.

## Problem

The governing equations used in the notebook are the incompressible Navier-Stokes equations:

$$ \rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f} $$

$$ \nabla \cdot \mathbf{u} = 0 $$

where the network predicts:

- `u(x, y, t)` = horizontal velocity,
- `v(x, y, t)` = vertical velocity,
- `p(x, y, t)` = pressure.

The notebooks use:

- density `rho = 1`
- kinematic viscosity `nu = 0.01`
- square spatial domain `x, y in [-1, 1]`
- time domain `t in [0, 10]`

Using a characteristic velocity of about 1 and length scale of 1, the notebook notes an approximate Reynolds number of:

Re = \frac{UL}{\nu} \approx 100

This corresponds to a laminar-to-transitional regime where viscous decay is expected and is practical for a PINN formulation.

## Boundary and Initial Conditions

The training notebook enforces the following conditions:

### Boundary condition

All four walls satisfy no-slip conditions:

\[
u = 0, \quad v = 0
\]

### Initial condition

At `t = 0`, the initial velocity field is:

\[
u(x,y,0) = \sin(\pi x)\sin(\pi y), \quad v(x,y,0)=0
\]

This gives a smooth starting flow that decays with time due to viscosity.

## PINN Architecture

The network used in both notebooks is a fully connected neural network:

- input: `(x, y, t)`
- output: `(u, v, p)`
- hidden layers: `128 -> 64 -> 32`
- activation: `Tanh`

`Tanh` is a suitable choice here because the physics loss requires first- and second-order derivatives from autograd, and smooth activations help with that.

## Training Strategy

The main notebook is [`navierStrokesPINN.ipynb`](/z:/CODE%20OFFICIAL/ML4S%20submission/navierStrokesPINN.ipynb). It samples:

- interior collocation points for the PDE residual,
- boundary points on all four walls,
- and initial-condition points on a grid at `t = 0`.

The loss combines:

1. physics loss from the Navier-Stokes momentum equations,
2. continuity loss from `du/dx + dv/dy = 0`,
3. boundary-condition loss,
4. initial-condition loss,
5. and a pressure-mean penalty to keep pressure anchored near zero.

In the implemented loop, the model is trained for `5000` epochs with stronger weights on the boundary and initial losses:

\[
\text{loss} = \text{physics} + 15 \cdot \text{boundary} + 15 \cdot \text{initial}
\]

The trained weights are saved as [`navier_stokes_pinn.pth`](/z:/CODE%20OFFICIAL/ML4S%20submission/navier_stokes_pinn.pth).

## Evaluation Summary

The notebook reports the following post-training metrics:

- mean continuity residual: `0.013283039443194866`
- max continuity residual: `1.2919423580169678`
- mean boundary velocity error: `0.008336435072124004`
- mean initial error in `u`: `0.005493460688740015`

These values indicate that:

- incompressibility is reasonably enforced on average,
- wall velocities are kept close to zero,
- the initial condition is reproduced with low average error,
- and the largest residuals are likely concentrated near hard regions such as wall corners.

## Visualisation Notebook

The second notebook, [`visualise.ipynb`](/z:/CODE%20OFFICIAL/ML4S%20submission/visualise.ipynb), loads the trained model and generates flow visualisations over time.

It:

- evaluates the model on a regular `x-y` grid,
- advances time in steps of `0.1`,
- computes the velocity magnitude,
- creates streamplots for each frame,
- and combines the saved frames into an animated GIF.

The animation stops once the maximum velocity drops below a fixed threshold, showing the expected decay of the flow.

## Repository Contents

- [`navierStrokesPINN.ipynb`](/z:/CODE%20OFFICIAL/ML4S%20submission/navierStrokesPINN.ipynb): main training and evaluation notebook
- [`visualise.ipynb`](/z:/CODE%20OFFICIAL/ML4S%20submission/visualise.ipynb): visualization notebook for generating the flow animation
- [`navier_stokes_pinn.pth`](/z:/CODE%20OFFICIAL/ML4S%20submission/navier_stokes_pinn.pth): saved trained model weights
- [`loss_curve.png`](/z:/CODE%20OFFICIAL/ML4S%20submission/loss_curve.png): training loss plot
- [`analytical_vs_pinn.png`](/z:/CODE%20OFFICIAL/ML4S%20submission/analytical_vs_pinn.png): comparison figure included with the submission
- [`navier_stokes_decay.gif`](/z:/CODE%20OFFICIAL/ML4S%20submission/navier_stokes_decay.gif): animated flow decay visualisation

## How to Run

1. Open [`navierStrokesPINN.ipynb`](/z:/CODE%20OFFICIAL/ML4S%20submission/navierStrokesPINN.ipynb) to inspect training, loss construction, and evaluation.
2. Run the notebook to retrain the model if needed.
3. Open [`visualise.ipynb`](/z:/CODE%20OFFICIAL/ML4S%20submission/visualise.ipynb) to load the saved weights and regenerate the animation.

## Notes

- The implementation is notebook-based and intended as a clear submission rather than a packaged library.
- The training notebook includes both the physics formulation and direct numerical checks of the learned solution.
- The visualization notebook focuses on interpreting the learned dynamics through streamplots and a GIF.

## Submission Statement

This repository is submitted by **Basit Warsi** as a test submission for **ML4Sci's GSoC problem statement**.
