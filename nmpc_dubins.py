import numpy as np
import cvxpy as cp
import casadi as ca
import matplotlib.pyplot as plt


# =====================================================================
# 1. LMI synthesis (SDP in Q = P^{-1})
# =====================================================================

def synthesize_P(lam, c_bar, u_bar, alpha1,
                 max_iter=50, tol=1e-6, verbose=True):
    """
    Solve 4-vertex LMI / SDP iteratively.

    Iteration on k0 is what keeps the saturator sector consistent with
    the ellipsoidal QAD {z : z^T P z <= 1}.

    Parameters
    ----------
    lam    : double pole of the linear (unsaturated) closed loop
    c_bar  : bound on |c(xi)| along the reference path
    u_bar  : saturation level on u
    alpha1 : target bound on |z1| inside QAD (must satisfy c_bar*alpha1 < 1)

    Returns
    -------
    dict with P, Q, k0, sigma0, u0, alpha1, and the design parameters.
    """
    # vector that gives sigma = d^T z
    d = np.array([lam**2, 2.0 * lam])

    # start from k0 = 1 (linear region, no saturation) and iterate
    k0 = 1.0
    P = None
    Q_val = None
    sigma0 = None
    u0 = None

    if c_bar * alpha1 >= 1.0:
        raise ValueError(
            f"alpha1 * c_bar = {alpha1*c_bar} must be < 1 "
            "(otherwise c z1 + 1 can hit zero inside QAD)."
        )

    for it in range(max_iter):
        # free margin of sigma before the saturator saturates
        u0 = u_bar * (1.0 - c_bar * alpha1) - c_bar
        if u0 <= 0.0:
            raise ValueError(
                f"u0 = {u0:.4f} <= 0. Increase u_bar, reduce alpha1 "
                "or reduce c_bar."
            )

        # current polytope vertex values
        betas = [k0, 1.0]
        gammas = [1.0 - c_bar * alpha1, 1.0 + c_bar * alpha1]

        # SDP: variables and constraints
        Q = cp.Variable((2, 2), symmetric=True)
        cons = [Q >> 1e-8 * np.eye(2)]

        # 4 vertex Lyapunov inequalities A_v Q + Q A_v^T < 0
        eps = 1e-6
        for beta in betas:
            for gamma in gammas:
                A = np.array([
                    [0.0,             gamma            ],
                    [-beta * lam**2, -2.0 * beta * lam]
                ])
                cons.append(A @ Q + Q @ A.T << -eps * np.eye(2))

        # Rectangle containment on z1: Q_{11} = e1^T Q e1 <= alpha1^2
        cons.append(Q[0, 0] <= alpha1**2)

        # maximise log det Q --> largest QAD
        prob = cp.Problem(cp.Maximize(cp.log_det(Q)), cons)
        prob.solve(solver=cp.SCS, verbose=False)

        if Q.value is None or prob.status not in ("optimal",
                                                  "optimal_inaccurate"):
            raise RuntimeError(f"SDP failed: status = {prob.status}")

        Q_val = np.array(Q.value)
        P = np.linalg.inv(Q_val)

        # sigma0 = sup { |d^T z| : z^T P z <= 1 } = sqrt( d^T Q d )
        sigma0 = float(np.sqrt(d @ Q_val @ d))
        k0_new = min(u0 / sigma0, 1.0)

        if verbose:
            print(f"  iter {it+1:02d}:  k0={k0:.5f}  "
                  f"sigma0={sigma0:.5f}  u0={u0:.5f}  "
                  f"k0_new={k0_new:.5f}")

        # stop when k0 stops moving
        if abs(k0_new - k0) < tol:
            k0 = k0_new
            break
        k0 = k0_new

    return dict(
        P=P, Q=Q_val, k0=k0, sigma0=sigma0, u0=u0,
        alpha1=alpha1, lam=lam, c_bar=c_bar, u_bar=u_bar,
    )


# =====================================================================
# 2. Continuous-xi dynamics and the nominal saturated control
# =====================================================================

def rhs(z, c_xi, u):
    """Right-hand side of the path-length-parametrised error model."""
    z1 = z[0]
    z2 = z[1]
    # dz1/dxi
    dz1 = (c_xi * z1 + 1.0) * z2
    # dz2/dxi
    dz2 = u * (c_xi * z1 + 1.0) * (1.0 + z2 ** 2) ** 1.5 \
          + c_xi * (1.0 + z2 ** 2)
    return np.array([dz1, dz2])


def nominal_u(z1, z2, c_xi, lam, u_bar):
    """
    The nominal saturated feedback-linearising control from the paper.
    Returned for diagnostic / fallback use only, NOT used to drive the
    plant during NMPC simulation.
    """
    sigma = 2.0 * lam * z2 + lam ** 2 * z1
    denom = (c_xi * z1 + 1.0) * (1.0 + z2 ** 2) ** 1.5
    # denom is strictly positive inside QAD by construction (alpha1*c_bar < 1)
    u_raw = -(sigma + c_xi * (1.0 + z2 ** 2)) / denom
    return float(np.clip(u_raw, -u_bar, u_bar))


# =====================================================================
# 3. NMPC: CasADi Opti with terminal-only cost and terminal QAD set
# =====================================================================

def build_nmpc_solver(P, u_bar, N, dxi):
    """
    Build a parametric NMPC with:
       * state  Z (2 x (N+1))
       * input  U (1 x N)
       * parameters  z0 (current error state) and c_par (N curvatures)
    Dynamics integrated in xi with RK4, one step per dxi.
    """
    opti = ca.Opti()
    Z = opti.variable(2, N + 1)
    U = opti.variable(1, N)
    z0_par = opti.parameter(2)
    c_par = opti.parameter(N)
    Pm = ca.DM(P)

    # initial condition
    opti.subject_to(Z[:, 0] == z0_par)

    # RK4 in xi for every step
    for i in range(N):
        c_i = c_par[i]
        u_i = U[0, i]
        # control saturation
        opti.subject_to(opti.bounded(-u_bar, u_i, u_bar))

        # control saturation
        opti.subject_to(opti.bounded(-u_bar, u_i, u_bar))

        # --- ROBUSTNESS FIXES FOR CASADI ---
        # Prevent the car from turning too sharply (limit to ~50 degrees).
        # This completely stops V(z) from spiking when starting far away, 
        # and inherently prevents the solver from encountering Inf math errors.
        opti.subject_to(opti.bounded(-1.2, Z[1, i], 1.2))
        
        # Prevent the NLP solver from exploring regions past the center of curvature
        opti.subject_to(c_i * Z[0, i] + 1.0 >= 0.05)

        # terminal QAD constraint
        zN = Z[:, N]
        opti.subject_to(opti.bounded(-1.2, zN[1], 1.2)) # Apply heading bound to the final node
        
        # Smooth L2 slack variable to prevent Infeasible crashes from 8m away
        slack = opti.variable()
        opti.subject_to(slack >= 0)
        opti.subject_to(zN.T @ Pm @ zN <= 1.0 + slack)

        dU = ca.diff(U)
        opti.minimize(zN.T @ Pm @ zN + 1e4 * (slack ** 2) + 0.1 * ca.sumsqr(dU))


        def f(zz, c_val=c_i, u_val=u_i):
            zz1 = zz[0]
            zz2 = zz[1]
            d1 = (c_val * zz1 + 1.0) * zz2
            d2 = u_val * (c_val * zz1 + 1.0) * (1.0 + zz2 ** 2) ** 1.5 \
                 + c_val * (1.0 + zz2 ** 2)
            return ca.vertcat(d1, d2)

        k1 = f(Z[:, i])
        k2 = f(Z[:, i] + 0.5 * dxi * k1)
        k3 = f(Z[:, i] + 0.5 * dxi * k2)
        k4 = f(Z[:, i] + dxi * k3)
        z_next = Z[:, i] + (dxi / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        opti.subject_to(Z[:, i + 1] == z_next)

        # control saturation
        opti.subject_to(opti.bounded(-u_bar, u_i, u_bar))

    # terminal QAD constraint
    zN = Z[:, N]
    opti.subject_to(zN.T @ Pm @ zN <= 1.0)

    # terminal-only cost
    opti.minimize(zN.T @ Pm @ zN)

    p_opts = {'print_time': 0}
    s_opts = {'print_level': 0, 'max_iter': 300, 'tol': 1e-8,
              'sb': 'yes'}
    opti.solver('ipopt', p_opts, s_opts)

    return opti, Z, U, z0_par, c_par


# =====================================================================
# 4. Closed-loop simulation
# =====================================================================

def simulate(syn, c_profile, z_init, N, dxi, Xi_final):
    """
    Run closed-loop NMPC along xi from 0 to Xi_final, starting from z_init.
    c_profile : callable xi -> kappa(xi), |kappa| <= c_bar.
    """
    P = syn["P"]
    u_bar = syn["u_bar"]
    lam = syn["lam"]

    opti, Z, U, z0_par, c_par = build_nmpc_solver(P, u_bar, N, dxi)

    xi_traj = [0.0]
    z_traj = [np.array(z_init, dtype=float)]
    u_traj = []
    V_traj = [float(z_traj[0] @ P @ z_traj[0])]

    z = np.array(z_init, dtype=float)
    xi = 0.0

    # warm starts
    U_warm = np.zeros(N)
    Z_warm = np.tile(z.reshape(2, 1), (1, N + 1))

    while xi < Xi_final - 1e-9:
        # fill in the N curvatures along the horizon
        c_vals = np.array([c_profile(xi + k * dxi) for k in range(N)])
        opti.set_value(z0_par, z)
        opti.set_value(c_par, c_vals)
        opti.set_initial(Z, Z_warm)
        opti.set_initial(U, U_warm)

        try:
            sol = opti.solve()
            u_now = float(sol.value(U[0, 0]))
            U_sol = sol.value(U)
            Z_sol = sol.value(Z)
            # shift warm start
            U_warm = np.concatenate([U_sol[1:], [0.0]])
            Z_warm = np.column_stack([Z_sol[:, 1:], Z_sol[:, -1]])
        except Exception:
            # fallback: nominal saturated control (only for robustness of
            # the numerical experiment; NOT part of the theory)
            u_now = nominal_u(z[0], z[1], c_profile(xi), lam, u_bar)
            U_warm = np.zeros(N)
            Z_warm = np.tile(z.reshape(2, 1), (1, N + 1))

        # step the real plant one dxi with RK4 using u_now
        c0 = c_profile(xi)
        k1 = rhs(z,                    c0, u_now)
        k2 = rhs(z + 0.5 * dxi * k1,   c0, u_now)
        k3 = rhs(z + 0.5 * dxi * k2,   c0, u_now)
        k4 = rhs(z + dxi * k3,         c0, u_now)
        z = z + (dxi / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xi = xi + dxi

        xi_traj.append(xi)
        z_traj.append(z.copy())
        u_traj.append(u_now)
        V_traj.append(float(z @ P @ z))

    return (np.array(xi_traj), np.array(z_traj),
            np.array(u_traj), np.array(V_traj))


# =====================================================================
# 5. Main: synthesise P, then run a few scenarios
# =====================================================================

if __name__ == "__main__":

    # ------------------------ design parameters ----------------------
    lam = 1.0
    c_bar = 0.3           # Set maximum curvature bound to 0.5 to match the test case
    u_bar = 1.0
    alpha1 = 0.8          # Target bound for |z1| inside QAD. Must satisfy alpha1 * c_bar < 1

    # ------------------------ synthesise P ---------------------------
    syn = synthesize_P(lam, c_bar, u_bar, alpha1, verbose=True)
    P = syn["P"]
    Q = syn["Q"]
    print("\nLMI synthesis converged:")
    print(f"  alpha1 = {syn['alpha1']:.4f}")
    print(f"  u0     = {syn['u0']:.4f}")
    print(f"  sigma0 = {syn['sigma0']:.4f}")
    print(f"  k0     = {syn['k0']:.5f}")
    print(f"  P =\n{P}")
    print(f"  eigvals(P) = {np.linalg.eigvalsh(P)}")

    # ------------------------ test curvature ------------------------
    kappa_val = -0.2
    def c_profile_const(xi):
        return kappa_val  # Constant curvature test

    # ------------------------ NMPC horizon ---------------------------
    N = 30
    dxi = 0.1           # -> horizon length N*dxi = 3 m
    Xi_final = 40.0

    # ------------------------ run specific initial condition ---------
    # Initial Condition
    z0 = np.array([0.0, np.tan(np.radians(75))])


    #-----------------------------------------------------------------
    # =========================================================================
    # INPUT VALIDATION (Prevents Impossible Math/Physics)
    # =========================================================================
    
    # 1. Physical Impossibility Check (Path is sharper than the steering wheel can turn)
    if abs(kappa_val) > u_bar:
        raise ValueError(
            f"\n[!] PHYSICAL LIMIT ERROR:\n"
            f"The test curvature (|kappa| = {abs(kappa_val)}) exceeds the physical "
            f"steering capacity of the vehicle (u_bar = {u_bar}). The car physically cannot make this turn."
        )

    # 2. Mathematical Theorem Check (Testing outside the LMI synthesis bounds)
    if abs(kappa_val) > c_bar:
        raise ValueError(
            f"\n[!] THEOREM VIOLATION:\n"
            f"The test curvature (|kappa| = {abs(kappa_val)}) is greater than the maximum "
            f"design curvature used for the LMI synthesis (c_bar = {c_bar}). "
            f"monotonicity theorem only holds if |kappa| <= c_bar. Increase c_bar and rerun."
        )

    # 3. Coordinate Singularity Check (Starting past the center of the curve)
    if 1.0 + kappa_val * z0[0] <= 0.05:
        raise ValueError(
            f"\n[!] GEOMETRIC SINGULARITY:\n"
            f"The initial condition z0={z0} with kappa={kappa_val} is outside the valid "
            f"tubular neighborhood. The coordinate frame folds on itself here."
        )

    # 4. Infinite Heading Check (Starting perfectly perpendicular to the path)
    if abs(z0[1]) > 5.0:  # ~78 degrees
        raise ValueError(
            f"\n[!] HEADING SINGULARITY:\n"
            f"The initial heading error is too close to 90 degrees (z2 = tan(psi) = {z0[1]:.2f}). "
            f"The spatial model requires forward progression along the path."
        )
    # =========================================================================
    #------------------------------------------------------------------

    V0 = float(z0 @ P @ z0)
    print(f"\n[Simulation]  z0 = {z0},   V(z0) = {V0:.4f}   "
          f"({'inside' if V0 <= 1.0 else 'outside'} QAD)")

    xi_t, z_t, u_t, V_t = simulate(syn, c_profile_const, z0, N, dxi, Xi_final)

    # quick monotonicity diagnostic once inside QAD
    in_qad = V_t <= 1.0 + 1e-9
    if in_qad.any():
        idx = np.where(in_qad)[0]
        dV = np.diff(V_t[idx[0]:])
        max_inc = dV.max() if len(dV) > 0 else 0.0
        print(f"  max increase of V after entering QAD: {max_inc:+.2e}")

  # plotting
    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    
    axs[0].plot(xi_t, z_t[:, 0], label=r"$z_1=\eta$")
    axs[0].plot(xi_t, z_t[:, 1], label=r"$z_2=\tan\psi$")
    axs[0].set_ylabel("state")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(xi_t[:-1], u_t)
    axs[1].axhline(+u_bar, color='r', ls='--', lw=0.8)
    axs[1].axhline(-u_bar, color='r', ls='--', lw=0.8)
    axs[1].set_ylabel("u")
    axs[1].grid(True)

    # Subplot 3: V(z) with QAD entry line and green background
    axs[2].plot(xi_t, V_t, 'b', lw=1.5)
    axs[2].axhline(1.0, color='r', ls='--', lw=1.0, label="QAD Boundary")
    
    # Check if and where it enters the QAD
    in_qad_mask = V_t <= 1.0 + 1e-9
    if in_qad_mask.any():
        idx_enter = np.where(in_qad_mask)[0][0]
        xi_enter = xi_t[idx_enter]
        
        # Vertical dashed line
        axs[2].axvline(xi_enter, color='g', linestyle='--', lw=1.5, label='Entered QAD')
        # Green background shading
        axs[2].axvspan(xi_enter, xi_t[-1], facecolor='g', alpha=0.15)

    axs[2].set_ylabel(r"$V = z^\top P z$")
    axs[2].set_xlabel(r"$\xi$ (Path Length)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True)

    fig.suptitle(f"Test Case: kappa={kappa_val}, z0={z0}")
    plt.tight_layout()
    plt.savefig("simulation_result.png", dpi=130)
    print("  saved plot -> simulation_result.png")
    plt.show()