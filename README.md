# Dynamic LMI-Based NMPC for Dubins Path Tracking

This repository contains simulations of a Nonlinear Model Predictive Controller (NMPC) for a nonholonomic Dubins vehicle, provided in both **MATLAB** and **Python**.

The control architecture is specifically designed to bridge finite-horizon NMPC with infinite-horizon Lyapunov stability.

## Core Mathematical Framework
This controller departs from standard "short-sighted" NMPC by guaranteeing Asymptotic Stability via the following mechanisms:

1. **Solving the Singular Problem:** A running stage cost is implemented to convexify the Hamiltonian, preventing the erratic "bang-bang" steering associated with pure terminal-cost affine systems.
2. **Constant Upper-Bound Bellman Matrix:** Instead of computing a real-time state-dependent Riccati equation, a constant upper-bound matrix `P` is synthesized offline to act as the terminal penalty.
3. **LTV Embedding & LMI Synthesis:** The physical steering saturator is modeled as an uncertainty parameter (`rho`), embedding the closed-loop system into a Linear Time-Varying (LTV) differential inclusion. Linear Matrix Inequalities (LMIs) are solved dynamically to find the `P` matrix, guaranteeing absolute stability across the entire saturation polytope.
4. **Curved Path Tracking:** The exact feedback linearization states (`z`) dynamically incorporate path curvature (`kappa`), allowing the controller to track rotating geometric frames rather than just straight lines.

## Included Implementations

This repository includes two identical implementations of the control framework to accommodate different software environments.

### 1. Python Implementation (`nmpc_dubins.py`)
This is the recommended version for users without a MATLAB license. It uses `cvxpy` for the LMI synthesis and `scipy.optimize` for the nonlinear MPC loop.
* **Prerequisites:** Python 3.x, `numpy`, `scipy`, `cvxpy`, `matplotlib`
* **How to run:** Simply execute the script via terminal or your preferred IDE: `python nmpc_dubins.py`

### 2. MATLAB Implementation (`nmpc_dubins.m`)
* **Prerequisites:** MATLAB, Optimization Toolbox (for `fmincon`), YALMIP, and a semidefinite solver (e.g., `lmilab`, `sedumi`, or `mosek`).
* **How to run:** Open the script in MATLAB and run it. 

## Simulation Flow
Both scripts run in two automated phases:
* **Phase 1:** Dynamically defines the LMI variables, applies the Schur complement for control bounds, and solves the Semi-Definite Program to compute the `P` matrix (Quadratic Attraction Domain).
* **Phase 2:** Executes the NMPC loop over an 8-second simulation to track a curved path. It outputs a live console feed of the metrics and generates two verification plots.
