% =========================================================================
% Dynamic LMI-Based NMPC for Dubins Path Tracking (Curved Path)
% Shima Akbari
% Last Modified: 24 March 2026
% =========================================================================

clear; 
clc; 
close all;

%% 1. Dubins Robot Physical Parameters & Constraints
v = 1.0;            % Forward velocity (m/s)
H = 0.5;            % Wheelbase (m)
u_max = 1.0;        % Physical steering rate limit |u| <= u_max (rad/s)
kappa = 0.1;        % Path curvature (1/m). 0.1 = tracking a 10m radius curve.

% NMPC Horizon Setup
Tp = 1.5;           % Prediction Horizon (seconds)
dt = 0.1;           % Sampling time
N = round(Tp / dt); % Number of prediction steps

%% 2. Dynamic LMI Synthesis (SOW Point 2 & 3: Bellman Upper Bound & LTV)
% We extract the relative-degree-3 cross-track error subsystem.
A = [0, 1, 0; 
     0, 0, 1; 
     0, 0, 0];
B = [0; 0; 1];

% SOW Point 3: The saturator "spoils" the feedback linearization. 
% We model this by embedding the system into an LTV differential inclusion
rho_min = 0.5; % Minimum saturation ratio (50% of requested control)
rho_max = 1.0; % Maximum saturation ratio (100% - no saturation)

disp('=================================================================');
disp('--- PHASE 1: Computing Upper-Bound Bellman Matrix (P) via LMI ---');
disp('=================================================================');

% Define YALMIP SDP Variables (Q = P^-1, Y = K*Q)
Q = sdpvar(3, 3, 'symmetric');
Y = sdpvar(1, 3);

% Build LMI Constraints
Constraints = [Q >= 1e-4 * eye(3)]; % Q must be strictly positive definite
Constraints = [Constraints, Q <= 100 * eye(3)]; % Bounds the maximum eigenvalues of Q

% LMI 1 & 2: Absolute Stability across the LTV Polytope (Vertices of rho)
Constraints = [Constraints, A*Q + Q*A' + rho_min*(B*Y + Y'*B') <= -1e-4*eye(3)];
Constraints = [Constraints, A*Q + Q*A' + rho_max*(B*Y + Y'*B') <= -1e-4*eye(3)];

% LMI 3: Control Constraints (Schur Complement)
scale = H / (v^2); % Physical-to-Virtual mapping factor
Constraints = [Constraints, [Q, scale*Y'; scale*Y, u_max^2] >= 0];

% Objective: Maximize Trace(Q) to maximize the Attraction Domain volume
Objective = -trace(Q);

% Solve the Semi-Definite Program (SDP) dynamically
options = sdpsettings('solver', 'lmilab', 'verbose', 1);
sol = optimize(Constraints, Objective, options);

if sol.problem == 0
    Q_opt = value(Q);
    Y_opt = value(Y);
    
    % The constant upper-bound Bellman matrix P
    P = inv(Q_opt); 
    disp('Success! LMI Solved. Dynamic matrix P calculated:');
    disp(P);
else
    disp(sol.info);
    error('LMI Solver failed.');
end

%% 3. NMPC Simulation Setup
% Initial Physical State: [cross-track error; orientation error; steering angle]
x0 = [0.8; 0.2; 0.0]; 

sim_time = 8;
steps = round(sim_time / dt);
x_history = zeros(3, steps);
x_history(:,1) = x0;

% Metric Tracking Arrays
cost_history = zeros(1, steps-1); % Tracks optimal cost J*
u_history    = zeros(1, steps-1); % Tracks applied steering rate u
qad_history  = zeros(1, steps-1); % Tracks terminal constraint value c (must be <= 0)

% fmincon Options
opt = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');

%% 4. Main NMPC Control Loop
x_current = x0;

disp('=================================================================');
disp('--- PHASE 2: Executing NMPC Loop (Tracking Curved Path)       ---');
disp('=================================================================');
fprintf(' Step | Cross-Track (y_e) | Optimal Cost (J*) | Control (u) | QAD Value (c <= 0)\n');
fprintf('--------------------------------------------------------------------------------\n');

for k = 1:steps-1
    u0 = zeros(N, 1); % Initial guess
    
    % Physical control bounds
    lb = -u_max * ones(N, 1);
    ub =  u_max * ones(N, 1);
    
    % Anonymous functions for MPC Cost and QAD Constraints (Now passing kappa)
    cost_func = @(U) nmpc_cost(U, x_current, N, dt, P, v, H, kappa);
    nonlcon_func = @(U) nmpc_qad_constraint(U, x_current, N, dt, P, v, H, kappa);
    
    % Solve finite-horizon optimal control problem
    [U_opt, J_opt] = fmincon(cost_func, u0, [], [], [], [], lb, ub, nonlcon_func, opt);
    
    % Extract applied control and calculate terminal constraint value
    u_applied = U_opt(1);
    [c_val, ~] = nonlcon_func(U_opt); % Evaluate QAD constraint at optimal sequence
    
    % Store Metrics
    cost_history(k) = J_opt;
    u_history(k)    = u_applied;
    qad_history(k)  = c_val;
    
    % Print Live Diagnostics
    fprintf(' %04d | %+0.4f m       | %0.4f           | %+0.4f rad/s| %+0.4f\n', ...
            k, x_current(1), J_opt, u_applied, c_val);
    
    % Simulate the real nonlinear Dubins kinematics for a CURVED path
    ye_dot = v * sin(x_current(2));
    th_dot = (v / H) * tan(x_current(3)) - kappa * (v * cos(x_current(2))) / (1 - kappa * x_current(1));
    al_dot = u_applied;
    
    x_current = x_current + [ye_dot; th_dot; al_dot] * dt;
    x_history(:, k+1) = x_current;
end
disp('--- Simulation Complete ---');

%% 5. Plotting Results
% FIGURE 1: Physical System States
figure('Name', 'Dubins Curved Path Tracking via NMPC', 'Position', [100, 100, 600, 800]);
subplot(3,1,1); plot(0:dt:sim_time-dt, x_history(1,:), 'b', 'LineWidth', 2); 
title('Cross-Track Error (y_e)'); ylabel('Meters'); grid on;

subplot(3,1,2); plot(0:dt:sim_time-dt, x_history(2,:), 'r', 'LineWidth', 2); 
title('Orientation Error (\theta_e)'); ylabel('Radians'); grid on;

subplot(3,1,3); plot(0:dt:sim_time-dt, x_history(3,:), 'g', 'LineWidth', 2); 
title('Steering Angle (\alpha)'); xlabel('Time (s)'); ylabel('Radians'); grid on;

% FIGURE 2: Analytical Metrics
figure('Name', 'NMPC Analytical Metrics', 'Position', [750, 100, 600, 800]);
subplot(3,1,1); plot(1:steps-1, cost_history, 'k', 'LineWidth', 2);
title('Optimal Bellman Cost (J*)'); ylabel('Cost'); grid on;
subtitle('Verification of Lyapunov monotonic decrease');

subplot(3,1,2); plot(1:steps-1, u_history, 'm', 'LineWidth', 2);
yline(u_max, 'r--', 'LineWidth', 1.5); yline(-u_max, 'r--', 'LineWidth', 1.5);
title('Applied Steering Control (u)'); ylabel('rad/s'); grid on;
subtitle('Verification of physical saturator bounds');

subplot(3,1,3); plot(1:steps-1, qad_history, 'c', 'LineWidth', 2);
yline(0, 'r--', 'LineWidth', 1.5);
title('Terminal QAD Constraint Value (z^T P z - 1)'); xlabel('Time Step'); ylabel('Value'); grid on;
subtitle('Verification of Attraction Domain entry (Must remain \leq 0)', 'Interpreter', 'tex');

%% =========================================================================
% Core NMPC Functions
% =========================================================================

function z = get_z_state(x, v, H, kappa)
    % Maps physical Dubins coordinates to relative-degree-3 linear coordinates
    % Incorporates path curvature (\kappa)
    ye = x(1); th_e = x(2); alpha = x(3);
    
    z1 = ye;
    z2 = v * sin(th_e);
    
    % Exact feedback linearization state z3, updated for curve tracking
    z3 = v * cos(th_e) * ((v / H) * tan(alpha) - (kappa * v * cos(th_e)) / (1 - kappa * ye)); 
    
    z = [z1; z2; z3];
end

function J = nmpc_cost(U, x0, N, dt, P, v, H, kappa)
    J = 0;
    x = x0;
    % SOW Point 1: Include running cost (R) to convexify Hamiltonian
    Q_stage = diag([5, 1, 0.1]); 
    R_stage = 0.5;
    
    for i = 1:N
        u = U(i);
        
        % Curved kinematics
        ye_dot = v * sin(x(2));
        th_dot = (v / H) * tan(x(3)) - kappa * (v * cos(x(2))) / (1 - kappa * x(1));
        
        x_dot = [ye_dot; th_dot; u];
        x = x + x_dot * dt;
        
        % Running Cost
        J = J + (x'*Q_stage*x + R_stage*u^2) * dt;
    end
    
    % SOW Point 2: Apply the Constant Upper-Bound Bellman Terminal Cost
    z_end = get_z_state(x, v, H, kappa);
    J = J + z_end' * P * z_end;
end

function [c, ceq] = nmpc_qad_constraint(U, x0, N, dt, P, v, H, kappa)
    x = x0;
    for i = 1:N
        u = U(i);
        
        % Curved kinematics
        ye_dot = v * sin(x(2));
        th_dot = (v / H) * tan(x(3)) - kappa * (v * cos(x(2))) / (1 - kappa * x(1));
        
        x_dot = [ye_dot; th_dot; u];
        x = x + x_dot * dt;
    end
    
    % SOW Point 3: Quadratic Attraction Domain (QAD) Enforcement
    z_end = get_z_state(x, v, H, kappa);
    
    % Force the terminal state to be inside the ellipsoid: z^T * P * z <= 1
    c = z_end' * P * z_end - 1.0; 
    ceq = [];
end