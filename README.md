# Spatial NMPC for Dubins Path Tracking

This repository contains a robust, Quasi-Infinite Horizon Nonlinear Model Predictive Controller (NMPC) for a Dubins-like wheeled robot. It is designed to track arbitrary curvilinear paths (such as B-Splines, lines, and arcs) with mathematical guarantees of stability, even when the physical steering actuators are fully saturated.

This implementation transitions away from traditional time-domain Serret-Frenet (SF) formulations, which are prone to coordinate singularities, and instead utilizes a strictly spatial parameterization based on absolute stability theory (Rapoport & Morozov, IFAC 2008).

## Key Features

* **Spatial Error Parameterization:** The independent variable is the path arc length (`xi`), not time. This decouples the steering control from the vehicle's forward velocity and eliminates the standard SF singularity. The state is strictly defined by lateral deviation (`z1 = eta`) and the tangent of the heading error (`z2 = tan(psi)`).
* **Native Saturation Handling via LTV Immersion:** The physical limits of the steering actuator are mathematically baked into the control synthesis. The saturated nonlinear dynamics are immersed into a Linear Time-Varying (LTV) differential inclusion using a 4-vertex polytope.
* **Offline LMI Synthesis (Guaranteed Monotonicity):** Before the controller runs, `cvxpy` solves a Semi-Definite Program (SDP) to find a Common Quadratic Lyapunov Function (CQLF). This computes a $P$ matrix that defines a Quadratic Attraction Domain (QAD). Inside the QAD, the tracking energy is mathematically guaranteed to decrease monotonically, despite un-canceled geometric drift and active steering saturation.
* **Terminal-Only NMPC (No Stage Cost):** The controller minimizes only the terminal cost `V(z_N)`, avoiding standard Hamiltonian singular chatter through a microscopic control-rate regularization.
* **Bulletproof CasADi NLP Solver:** The online optimizer is protected against mathematical failures:
  * **Autodiff Safety:** Algebraic rewrite of the exact kinematics (`sqrt` trick) to prevent `Inf` gradients during the optimization search.
  * **Approach Angle Bounds:** Kinematic bounds (`|z2| <= 1.2`) prevent the optimizer from testing $90^\circ$ perpendicular dives when starting far from the path.
  * **Exact Penalty Handoff:** Soft $L_2$ constraints on the terminal set allow the NMPC to seamlessly compute optimal approach trajectories from far outside the QAD without returning "Infeasible" errors.
* **Strict Input Validation:** Pre-checks evaluate if requested reference paths exceed the vehicle's physical limits or the mathematical bounds of the LMI synthesis, preventing out-of-domain crashes.

## Dependencies

* Python 3.8+
* `numpy`
* `scipy`
* `matplotlib`
* `cvxpy` (for offline LMI synthesis)
* `casadi` (for the online NLP solver)

You can install all dependencies via pip:
```bash
pip install numpy scipy matplotlib cvxpy casadi
