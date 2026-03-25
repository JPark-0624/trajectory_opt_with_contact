"""
Example: IRL Feature Matching Test

Step 1. SQP SS로 known w_true로 demo trajectory 생성
Step 2. IRLSolver.fit_feature_matching() 실행
Step 3. w_recovered vs w_true 비교

목적: inner solver 없이 outer loop (weight optimization) 구조가
       올바르게 작동하는지 검증.
"""

import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_opt_with_contact.single_shooting_SQP_moreau import (
    SingleShootingSQPGaussNewton, SQPConfig, CostWeights
)
from trajectory_opt_with_contact.dynamics import step_square_pos_ip, IPMOptions
from trajectory_opt_with_contact.irl_sqp import IRLSolver, IRLConfig, pack_demo

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

MASS        = 1.0
SIDE        = 0.2
MU          = 0.5
HORIZON     = 30
DT          = 0.05

ipm_opts = IPMOptions(
    target_mu=1e-4, max_newton=20, tol=1e-6,
    enable_viscous_ground_friction=True,
    smooth_sdf=50.0, c_lin=1.0, c_ang=0.00667
)

sqp_solver = SingleShootingSQPGaussNewton(
    mass=MASS, side_length=SIDE, mu=MU,
    horizon=HORIZON, dt=DT, device=device,
    dynamics_module=step_square_pos_ip,
    ipmOpts=ipm_opts,
)

# ---------------------------------------------------------------------------
# Step 1: Generate demo with known weights
# ---------------------------------------------------------------------------

q0      = [0.0, 0.0, 0.0]
v0      = [0.0, 0.0, 0.0]
pusher0 = [-0.2, 0.0]
goal = [0.5, 0.2, 0.3]

# True weights (normalized internally by IRL, but we set absolute scale here)
w_true = CostWeights(
    wTargetXY    = 20.0,
    wTargetOrient = 1.0,
    wObjVel      = 1.0,
    wControl     = 0.01,
)

print("\n" + "="*60)
print("Step 1: Generating demo trajectory")
print("="*60)
print(f"True weights:")
print(f"  wTargetXY   : {w_true.wTargetXY}")
print(f"  wTargetOrient: {w_true.wTargetOrient}")
print(f"  wObjVel     : {w_true.wObjVel}")
print(f"  wControl    : {w_true.wControl}")

cfg_demo = SQPConfig(
    maxIters=50,
    tol=1e-4,
    use_gauss_newton=True,  # ⭐ Enable Gauss-Newton Hessian (P = J^T J)
    hessian_regularization=1e-6,  # Regularization λ for P + λI
    use_line_search=True,
    line_search_max_iters=10,
    line_search_beta=0.5,
    u_min=-0.5,
    u_max=0.5,
    use_trust_region=True,
    trust_region_iters=3,
    trust_region_size=0.5,
    use_mu_scheduling=False,
    mu_schedule_type='stepwise',
    mu_start=1e-3,
    mu_end=1e-6,
    u_init_mode='zero'
)

result_demo = sqp_solver.optimize(
    q0=q0, v0=v0, pusher0=pusher0, goal=goal,
    cfg=cfg_demo, w=w_true, verbose=True,
)

u_demo = result_demo['u_seq']
print(f"\nDemo generated. Final loss: {result_demo['loss']:.4f}")
print(f"Position error: {result_demo['stationarityInfo']['final_terminal_error']:.4f}m")

demo = pack_demo(q0, v0, pusher0, goal, u_demo)

# ---------------------------------------------------------------------------
# Step 2: Feature matching IRL
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("Step 2: Feature Matching IRL")
print("="*60)

irl = IRLSolver(
    mass=MASS, side_length=SIDE, mu=MU,
    horizon=HORIZON, dt=DT,
    device=device,
    dynamics_solver='IP',
    ipm_opts=ipm_opts,
)

cfg_irl = IRLConfig(
    max_outer_iters=1,
    lr_weights=0.1,
    weight_tol=1e-4,
    max_inner_iters=20,   # Adam steps for inner solve in feature matching
    verbose=True,
    log_every=5,
)

result_irl = irl.fit_feature_matching(demo, cfg=cfg_irl)

# ---------------------------------------------------------------------------
# Step 2b: Control Matching QP IRL (main test)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("Step 2b: Control Matching QP IRL")
print("="*60)

irl_cm = IRLSolver(
    mass=MASS, side_length=SIDE, mu=MU,
    horizon=HORIZON, dt=DT,
    device=device,
    dynamics_solver='IP',
    ipm_opts=ipm_opts,
    sqp_solver=sqp_solver,
)

cfg_cm = IRLConfig(
    max_outer_iters=20,
    max_inner_iters=30,
    weight_tol=1e-4,
    hessian_reg=1e-4,
    warm_start_inner=True,
    verbose=True,
    log_every=1,
)

# Uniform weight init
w_init = torch.ones(4, dtype=torch.float64) / 4

result_cm = irl_cm.fit_control_matching_qp(demo, w_init=w_init, cfg=cfg_cm)

# ---------------------------------------------------------------------------
# Step 3: Compare weights
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("Step 3: Weight Comparison")
print("="*60)

w_true_vec = torch.tensor(
    [w_true.wTargetXY, w_true.wTargetOrient, w_true.wObjVel, w_true.wControl],
    dtype=torch.float64
)
w_true_norm = w_true_vec / w_true_vec.sum()
names = IRLSolver.WEIGHT_NAMES

for label, w_rec in [("Feature Matching", result_irl['w_recovered']),
                     ("Control Matching QP", result_cm['w_recovered'])]:
    print(f"\n--- {label} ---")
    print(f"{'Name':12s} {'True (norm)':>12s} {'Recovered':>12s} {'Abs Error':>12s}")
    print("-" * 52)
    for name, wt, wr in zip(names, w_true_norm, w_rec.cpu()):
        err = abs(wt.item() - wr.item())
        print(f"{name:12s} {wt.item():12.6f} {wr.item():12.6f} {err:12.6f}")
    l2 = torch.norm(w_rec.cpu() - w_true_norm).item()
    print(f"L2 error: {l2:.6f}")

# ---------------------------------------------------------------------------
# Convergence plot (optional)
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, (label, res) in enumerate([("Feature Matching", result_irl),
                                         ("Control Matching QP", result_cm)]):
        axes[row, 0].semilogy(res['loss_history'])
        axes[row, 0].set_title(f'{label} - Loss')
        axes[row, 0].set_xlabel('Outer iteration')
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].semilogy(res['grad_norm_history'])
        axes[row, 1].set_title(f'{label} - Gradient Norm')
        axes[row, 1].set_xlabel('Outer iteration')
        axes[row, 1].grid(True, alpha=0.3)

        w_hist = np.array(res['w_history'])
        for i, name in enumerate(names):
            axes[row, 2].plot(w_hist[:, i], label=name)
        for i, wt in enumerate(w_true_norm):
            axes[row, 2].axhline(wt.item(), linestyle='--', alpha=0.4)
        axes[row, 2].set_title(f'{label} - Weight Trajectory')
        axes[row, 2].set_xlabel('Outer iteration')
        axes[row, 2].legend(fontsize=8)
        axes[row, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('irl_results.png', dpi=150)
    print("\nPlot saved: irl_results.png")

except ImportError:
    print("\n(matplotlib not available, skipping plot)")