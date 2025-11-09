"""
Functional tests for trajectory_opt_with_contact package.
Tests basic functionality without running full optimization.
"""

import torch
import os
import numpy as np
import time
from trajectory_opt_with_contact import (
    step_square, step_square_pos_ip, IPMOptions, rollout, ContactQPSolver, visualize_result,step_square_pos_ip_lin
)


# ===========================================================
# 1. Setup experiment parameters
# ===========================================================
device = torch.device("cuda")
dtype = torch.float32

# Physical parameters
m = 1.0
Izz = 1.0 / 6.0
half = 0.1
mu = 0.6
h = 0.02
horizon = 30 #100 #150

# Initial state
q0 = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
v0 = torch.zeros(3, dtype=dtype, device=device)
pr0 = torch.tensor([0.0, -0.15], dtype=dtype, device=device)

# Goal state
goal = torch.tensor([0.3, 0.0, 0.0], dtype=dtype, device=device)

# Control sequence: constant push forward
u_seq = torch.zeros(horizon, 2, dtype=dtype, device=device)
u_seq[:, 1] = 0.25  # push upward along y
u_seq[:, 0] = 0.05

# Output directories
os.makedirs("results", exist_ok=True)

# ===========================================================
# 2. Run simulation with CVXPY solver
# ===========================================================
print("\n=== Running simulation with CVXPY solver ===")
solver_cvx = ContactQPSolver(mu=mu, n_contacts=1, backend="cvxpy")

loss_cvx, q_final_cvx, lam_hist_cvx, phi_hist_cvx, q_hist_cvx, pusher_hist_cvx = rollout(
    u_seq, q0, v0, pr0, horizon, h, m, Izz, half, mu, goal, qp_solver=solver_cvx, device=device
)

result_cvx = {
    "trajectory": q_hist_cvx.cpu().numpy(),
    "pusher_trajectory": pusher_hist_cvx.cpu().numpy(),
    "contact_forces": lam_hist_cvx.cpu().numpy()/h,
    "signed_distances": phi_hist_cvx.cpu().numpy(),
    "u_seq": u_seq.cpu().numpy(),
}

visualize_result(
    result_cvx,
    goal.cpu().numpy(),
    half_size=half,
    save_trajectory="results/traj_cvx.png",
    save_animation="results/traj_cvx.mp4",
    save_analysis="results/analysis_cvx.png",
    xlim=(-0.3, 0.6),
    ylim=(-0.3, 0.6),
)

# ===========================================================
# 3. Run simulation with interior point solver
# ===========================================================
print("\n=== Running simulation with IP solver ===")
solver_ip = ContactQPSolver(mu=mu, n_contacts=1, backend="qpth")

# We'll use the same rollout function but with step_square_ip inside
qs_ip = [q0]
vs_ip = [v0]
prs_ip = [pr0]
lams_ip = []
phis_ip = []

start_time = time.time()

q, v, pr = q0.clone(), v0.clone(), pr0.clone()
for k in range(horizon):
    # ========================= Small demo usage (pseudo) ========================= #
    # damping is implemented differently (apply damping force instead of damping on vel)
    # Also thye lam here is actual force, not impulse
    q_next, v_next, pr_next, lam, phi = step_square_pos_ip(
       q, v, pr, u_seq[k], h=h, m=m, Izz=Izz, half=half, mu=mu,
       skip_solving_threshold = 0.3,
        ipm_opts=IPMOptions(target_mu=1e-4, max_newton=20, tol=1e-4, smooth_sdf=50.0,
        enable_viscous_ground_friction=True,
        c_lin=8.0,          
        c_ang=8.0 * half     
        ))

    # q_next, v_next, pr_next, lam, phi = step_square_pos_ip_lin(
    #    q, v, pr, u_seq[k], h=h, m=m, Izz=Izz, half=half, mu=mu,
    #    skip_solving_threshold = 0.3,
    #     ipm_opts=IPMOptions(target_mu=1e-4, max_newton=20, tol=1e-5, smooth_sdf=50.0,
    #     enable_viscous_ground_friction=True,
    #     c_lin=8.0,          
    #     c_ang=8.0 * half     
    #     ), relinearize_each_newton = False)

    qs_ip.append(q_next)
    vs_ip.append(v_next)
    prs_ip.append(pr_next)
    lams_ip.append(lam)
    phis_ip.append(phi)
    q, v, pr = q_next, v_next, pr_next

print(f'Time elapsed: {time.time() - start_time}')

result_ip = {
    "trajectory": torch.stack(qs_ip).detach().cpu().numpy(),
    "pusher_trajectory": torch.stack(prs_ip).cpu().numpy(),
    "contact_forces": torch.stack(lams_ip).detach().cpu().numpy(),
    "signed_distances": torch.tensor(phis_ip).cpu().numpy(),
    "u_seq": u_seq.detach().cpu().numpy(),
}

visualize_result(
    result_ip,
    goal.cpu().numpy(),
    half_size=half,
    save_trajectory="results/traj_ip.png",
    save_animation="results/traj_ip.mp4",
    save_analysis="results/analysis_ip.png",
    xlim=(-0.3, 0.6),
    ylim=(-0.3, 0.6),
)
exit()
# ===========================================================
# 4. Compare results numerically
# ===========================================================
print("\n=== Comparison Summary ===")
print(f"Final position CVXPY: {q_final_cvx}")
print(f"Final position QPTH : {q}")
print(f"Difference (L2 norm) : {(q_final_cvx - q).norm().item():.6f}")

print("✓ All tests complete. Check 'results/' folder for videos and plots.")