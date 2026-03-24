# Dynamic LMI-Based NMPC for Dubins Path Tracking

This repository contains a MATLAB simulation of a Nonlinear Model Predictive Controller (NMPC) for a nonholonomic Dubins vehicle. 

The control architecture is specifically designed to bridge finite-horizon NMPC with infinite-horizon Lyapunov stability.

## Core Mathematical Framework
This controller departs from standard "short-sighted" NMPC by guaranteeing Asymptotic Stability via the following mechanisms:

1. **Solving the Singular Problem:** A running stage cost is implemented to convexify the Hamiltonian, preventing the erratic "bang-bang" steering associated with pure terminal-cost affine systems.
2. **Constant Upper-Bound Bellman Matrix:** Instead of computing a real-time state-dependent Riccati equation, a constant upper-bound matrix `P` is synthesized offline to act as the terminal penalty.
3. **LTV Embedding & LMI Synthesis:** The physical steering saturator is modeled as an uncertainty parameter (`rho`), embedding the closed-loop system into a Linear Time-Varying (LTV) differential inclusion. Linear Matrix Inequalities (LMIs) are solved dynamically to find the `P` matrix, guaranteeing absolute stability across the entire saturation polytope.
4. **Curved Path Tracking:** The exact feedback linearization states (`z`) dynamically incorporate path curvature (`kappa`), allowing the controller to track rotating geometric frames rather than just straight lines.

## Prerequisites
To run this simulation, you need:
* **MATLAB** (Optimization Toolbox required for `fmincon`)
* **YALMIP** (For defining the LMI constraints)
* **Robust Control Toolbox** (For the `lmilab` solver, or you can swap to `sedumi`)

## How to Run
Simply execute the `nmpc_dubins.m` script in MATLAB. 

The script runs in two phases:
* **Phase 1:** Dynamically defines the LMI variables, applies the Schur complement for control bounds, and solves the Semi-Definite Program to compute the `P` matrix (Quadratic Attraction Domain).
* **Phase 2:** Executes the NMPC loop over an 8-second simulation, utilizing `fmincon` to track a curved path. It outputs a live console feed of the metrics and generates two verification plots.

## Files Included
* `nmpc_dubins.m` - The core simulation script.
