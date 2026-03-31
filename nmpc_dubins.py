import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# =========================================================================
# Dynamic LMI-Based NMPC for Dubins Path Tracking (Curved Path)
# Python Translation of Shima Akbari's MATLAB code
# =========================================================================

# 1. Dubins Robot Physical Parameters & Constraints
v = 1.0            # Forward velocity (m/s)
H = 0.5            # Wheelbase (m)
u_max = 1.0        # Physical steering rate limit |u| <= u_max (rad/s)
kappa = 0.1        # Path curvature (1/m)

# NMPC Horizon Setup
Tp = 1.5           # Prediction Horizon (seconds)
dt = 0.1           # Sampling time
N = int(Tp / dt)   # Number of prediction steps

# =========================================================================
# 2. Dynamic LMI Synthesis (Bellman Upper Bound & LTV)
# =========================================================================
print("=================================================================")
print("--- PHASE 1: Computing Upper-Bound Bellman Matrix (P) via LMI ---")
print("=================================================================")

A = np.array([[0, 1, 0], 
              [0, 0, 1], 
              [0, 0, 0]])
B = np.array([[0], [0], [1]])

rho_min = 0.5
rho_max = 1.0

# Define CVXPY Variables for SDP
Q = cp.Variable((3, 3), symmetric=True)
Y = cp.Variable((1, 3))

# Build Constraints
constraints = [
    Q >> 1e-4 * np.eye(3),  # Positive definiteness
    Q << 100 * np.eye(3)    # Upper bound to prevent numerical explosion
]

# LMI 1 & 2: Absolute Stability across LTV Polytope Vertices
LMI1 = A @ Q + Q @ A.T + rho_min * (B @ Y + Y.T @ B.T)
LMI2 = A @ Q + Q @ A.T + rho_max * (B @ Y + Y.T @ B.T)
constraints.append(LMI1 << -1e-4 * np.eye(3))
constraints.append(LMI2 << -1e-4 * np.eye(3))

# LMI 3: Control Constraints (Schur Complement)
scale = H / (v**2)
block_matrix = cp.bmat([
    [Q, scale * Y.T],
    [scale * Y, np.array([[u_max**2]])]
])
constraints.append(block_matrix >> 0)

# Objective: Maximize Trace(Q) -> Minimize -Trace(Q)
objective = cp.Minimize(-cp.trace(Q))

# Solve the Semi-Definite Program
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS, verbose=False)

if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
    Q_opt = Q.value
    P = np.linalg.inv(Q_opt)
    print('Success! LMI Solved. Dynamic matrix P calculated:')
    print(np.round(P, 4))
else:
    raise ValueError("LMI Solver failed. Problem status:", prob.status)

# =========================================================================
# Core NMPC Functions
# =========================================================================
def get_z_state(x, v, H, kappa):
    # Maps physical Dubins coordinates to relative-degree-3 linear coordinates
    ye, th_e, alpha = x
    z1 = ye
    z2 = v * np.sin(th_e)
    z3 = v * np.cos(th_e) * ((v / H) * np.tan(alpha) - (kappa * v * np.cos(th_e)) / (1 - kappa * ye))
    return np.array([z1, z2, z3])

def nmpc_cost(U, x0, N, dt, P, v, H, kappa):
    J = 0
    x = np.copy(x0)
    # Include running cost to convexify Hamiltonian (Solve Singular Problem)
    Q_stage = np.diag([5.0, 1.0, 0.1])
    R_stage = 0.5
    
    for i in range(N):
        u = U[i]
        ye_dot = v * np.sin(x[1])
        th_dot = (v / H) * np.tan(x[2]) - kappa * (v * np.cos(x[1])) / (1 - kappa * x[0])
        x_dot = np.array([ye_dot, th_dot, u])
        x = x + x_dot * dt
        
        # Running Cost
        J += (x.T @ Q_stage @ x + R_stage * u**2) * dt
        
    # Constant Upper-Bound Bellman Terminal Cost
    z_end = get_z_state(x, v, H, kappa)
    J += z_end.T @ P @ z_end
    return J

def nmpc_qad_constraint(U, x0, N, dt, P, v, H, kappa):
    x = np.copy(x0)
    for i in range(N):
        u = U[i]
        ye_dot = v * np.sin(x[1])
        th_dot = (v / H) * np.tan(x[2]) - kappa * (v * np.cos(x[1])) / (1 - kappa * x[0])
        x_dot = np.array([ye_dot, th_dot, u])
        x = x + x_dot * dt
        
    z_end = get_z_state(x, v, H, kappa)
    # Force state to be inside ellipsoid: z^T P z - 1 <= 0
    c = z_end.T @ P @ z_end - 1.0
    # SciPy SLSQP expects inequality constraint to be non-negative (>= 0), so we return -c
    return -c 

# =========================================================================
# 3. Main NMPC Simulation Loop
# =========================================================================
x0 = np.array([0.8, 0.2, 0.0]) 

sim_time = 8.0
steps = int(sim_time / dt)

x_history = np.zeros((3, steps))
x_history[:, 0] = x0

cost_history = np.zeros(steps-1)
u_history = np.zeros(steps-1)
qad_history = np.zeros(steps-1)

x_current = np.copy(x0)

print("\n=================================================================")
print("--- PHASE 2: Executing NMPC Loop (Tracking Curved Path)       ---")
print("=================================================================")
print(" Step | Cross-Track | Optimal Cost | Control (u) | QAD Value (c<=0)")
print("-------------------------------------------------------------------")

bounds = [(-u_max, u_max) for _ in range(N)]

for k in range(steps - 1):
    u0 = np.zeros(N) # Initial guess
    
    # Define constraint dictionary for scipy
    cons = {'type': 'ineq', 'fun': lambda U: nmpc_qad_constraint(U, x_current, N, dt, P, v, H, kappa)}
    
    # Solve NLP
    res = minimize(nmpc_cost, u0, args=(x_current, N, dt, P, v, H, kappa),
                   method='SLSQP', bounds=bounds, constraints=cons, options={'disp': False})
    
    U_opt = res.x
    J_opt = res.fun
    u_applied = U_opt[0]
    
    # Evaluate QAD constraint (flip sign back for correct logging: c <= 0)
    c_val = -(nmpc_qad_constraint(U_opt, x_current, N, dt, P, v, H, kappa)) 
    
    # Store metrics
    cost_history[k] = J_opt
    u_history[k] = u_applied
    qad_history[k] = c_val
    
    print(f" {k:04d} | {x_current[0]:+0.4f} m   | {J_opt:0.4f}       | {u_applied:+0.4f} rad/s| {c_val:+0.4f}")
    
    # Simulate real system
    ye_dot = v * np.sin(x_current[1])
    th_dot = (v / H) * np.tan(x_current[2]) - kappa * (v * np.cos(x_current[1])) / (1 - kappa * x_current[0])
    x_dot = np.array([ye_dot, th_dot, u_applied])
    x_current = x_current + x_dot * dt
    
    x_history[:, k+1] = x_current

print("--- Simulation Complete ---")

# =========================================================================
# 4. Plotting Results
# =========================================================================
time_arr = np.arange(0, sim_time, dt)

plt.figure(figsize=(8, 10))

plt.subplot(3, 1, 1)
plt.plot(time_arr, x_history[0, :], 'b', linewidth=2)
plt.title('Cross-Track Error (y_e)')
plt.ylabel('Meters')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_arr, x_history[1, :], 'r', linewidth=2)
plt.title('Orientation Error (theta_e)')
plt.ylabel('Radians')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_arr, x_history[2, :], 'g', linewidth=2)
plt.title('Steering Angle (alpha)')
plt.xlabel('Time (s)')
plt.ylabel('Radians')
plt.grid(True)
plt.tight_layout()

# Metric Verification Plot
plt.figure(figsize=(8, 10))
time_steps = np.arange(1, steps)

plt.subplot(3, 1, 1)
plt.plot(time_steps, cost_history, 'k', linewidth=2)
plt.title('Optimal Bellman Cost (J*)')
plt.ylabel('Cost')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_steps, u_history, 'm', linewidth=2)
plt.axhline(u_max, color='r', linestyle='--', linewidth=1.5)
plt.axhline(-u_max, color='r', linestyle='--', linewidth=1.5)
plt.title('Applied Steering Control (u)')
plt.ylabel('rad/s')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_steps, qad_history, 'c', linewidth=2)
plt.axhline(0, color='r', linestyle='--', linewidth=1.5)
plt.title('Terminal QAD Constraint Value (z^T P z - 1)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()

plt.show()