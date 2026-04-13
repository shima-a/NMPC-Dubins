# Autonomous Initialization & LMI-Based NMPC for Dubins Path Tracking

This repository contains simulations of a comprehensive control architecture for a nonholonomic Dubins vehicle. It bridges finite-horizon NMPC with infinite-horizon Lyapunov stability, while solving the "Autonomous Initialization Problem" (handling random drop-offs) using online $\Theta^*$ path planning.

The architecture directly implements the theoretical framework outlined in Lev Rapoport and Shima Akbari's Statement of Work (SOW).

## Core Mathematical Framework

This framework resolves the issue of NMPC infeasibility outside of its local stable zone by chaining a global path planner with a mathematically guaranteed local optimal tracker. 

### 1. Two-Layer Online Path Planning
* **Layer 1: Online Global Routing ($\Theta^*$):** To handle arbitrary, unmapped drop-off locations, the system uses an any-angle $\Theta^*$ search. Using Bresenham's Line-of-Sight algorithm, it bypasses discrete grid nodes to generate highly optimized, collision-free linear segments.
* **Layer 2: Local B-Spline Curvature Smoothing:** Because the physical steering actuators cannot track discontinuous curvature (sharp corners), a Cubic B-Spline ($k=3$) wraps the $\Theta^*$ waypoints. This mathematically guarantees $C^2$ continuity for both heading ($\theta$) and curvature ($\kappa$).

### 2. LMI Synthesis & The Quadratic Attraction Domain (QAD)
* **Constant Upper-Bound Bellman Matrix:** A constant matrix `P` is synthesized offline to act as the NMPC's terminal penalty. This matrix defines the Quadratic Attraction Domain (QAD) ($\Omega_P = \{z \mid z^T P z \le 1\}$).
* **LTV Embedding & Actuator Limits:** The physical steering saturator is modeled as an uncertainty parameter. Linear Matrix Inequalities (LMIs) are dynamically solved using the Schur Complement to find `P`, guaranteeing that control effort never exceeds physical limits ($|u| \le \bar{u}$) while inside the QAD.

### 3. The 4-Step Control Handoff
Because the NMPC is only feasible strictly within the QAD, tracking is achieved via an automated timeline-based handoff:
1. **Planning Phase ($t=0$):** The stationary robot runs $\Theta^*$ and builds the B-Spline bridge intersecting the QAD.
2. **Approach Phase:** Outside the QAD ($z^T P z > 1$), the NMPC is OFF. A simple geometric controller steers the robot along the approach path.
3. **Mathematical Trigger:** The system continuously evaluates the exact condition $z^T P z \le 1$.
4. **NMPC Handoff:** The instant the robot enters the safe zone, the geometric controller shuts down, and the NMPC dynamically assumes control, guiding the system asymptotically to the path origin without saturation.

## Included Implementations

This repository includes implementations of the control framework to accommodate different software environments.

### 1. Python Implementation (`nmpc_dubins.py`)
This version simulates the full end-to-end pipeline. It uses `cvxpy` for the LMI synthesis, `scipy.optimize` for the nonlinear MPC loop, and `scipy.interpolate` for the B-Spline generation.
* **Prerequisites:** Python 3.x, `numpy`, `scipy`, `cvxpy`, `matplotlib`
* **How to run:** Execute the script via terminal or your preferred IDE: `python nmpc_dubins.py`

### 2. MATLAB Implementation (`nmpc_dubins.m`)
*(Note: Requires adaptation to include the newly integrated Python $\Theta^*$ search algorithm).*
* **Prerequisites:** MATLAB, Optimization Toolbox (for `fmincon`), YALMIP, and a semidefinite solver (e.g., `lmilab`, `sedumi`, or `mosek`).

## Simulation Flow
When running the Python script, the simulation automatically progresses through three phases:
* **Phase 1 (Routing):** Executes $\Theta^*$ and wraps the path in a $C^2$ continuous B-Spline.
* **Phase 2 (Synthesis):** Solves the Semi-Definite Program to compute the exact `P` matrix and define the QAD boundary.
* **Phase 3 (Handoff Simulation):** Runs the physical robot step-by-step, logging the transition from the Geometric Approach Controller to the NMPC. Finally, it generates 2 detailed verification plots displaying the track, curvature, state errors, and the exact handoff trigger time.
