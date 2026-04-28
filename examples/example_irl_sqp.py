"""
Example: IRL — KKT Residual + Control Matching EG

두 방법을 순서대로 실행하고 결과를 비교한다.
  1. fit_weights_kkt_residual  : single QP, inner loop 없음 (빠름, warm-start용)
  2. fit_control_matching_eg   : bilevel IRL, EG outer + SQP inner

Usage:
    # 1. 데이터셋 생성 (최초 1회)
    python generate_irl_dataset.py --out_dir ./data/irl

    # 2. IRL 실행
    python example_irl_sqp.py --data_dir ./data/irl
"""

import torch
import numpy as np
import argparse, json, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_opt_with_contact.single_shooting_SQP_moreau import (
    SingleShootingSQPGaussNewton, CostWeights
)
from trajectory_opt_with_contact.dynamics import step_square_pos_ip, IPMOptions
from trajectory_opt_with_contact.irl_sqp import IRLSolver, IRLConfig
from trajectory_opt_with_contact.utils import load_demo_npz

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',    type=str,   default='./data/irl')
parser.add_argument('--outer_iters', type=int,   default=50)
parser.add_argument('--inner_iters', type=int,   default=200)
parser.add_argument('--lr',          type=float, default=0.1)
parser.add_argument('--weight_tol',  type=float, default=1e-6)
parser.add_argument('--skip_eg',     action='store_true',
                    help='KKT residual only (skip bilevel EG)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

meta_path = os.path.join(args.data_dir, 'metadata.json')
assert os.path.exists(meta_path), \
    f"Dataset not found at {meta_path}. Run generate_irl_dataset.py first."

with open(meta_path) as f:
    metadata = json.load(f)

w_true = np.array(metadata['w_true_norm'])
wnames = metadata['weight_names']
phys   = metadata['physics']

print(f"\n{'='*60}")
print(f"IRL Example")
print(f"{'='*60}")
print(f"w_true: {dict(zip(wnames, w_true.round(4)))}")

# Load all converged demos
all_demos = []
for d in metadata['demos']:
    p = os.path.join(args.data_dir, f"{d['name']}.npz")
    if d['inner_converged']:
        all_demos.append(load_demo_npz(p))
        print(f"  + {d['name']:20s} proj_grad={d['final_proj_grad_norm']:.2e}")
    else:
        print(f"  - {d['name']:20s} skipped (not converged)")

assert len(all_demos) > 0, "No converged demos found."
print(f"Using {len(all_demos)}/{len(metadata['demos'])} demos")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

ipm_cfg  = metadata['ipm_opts']
ipm_opts = IPMOptions(
    target_mu  = ipm_cfg['target_mu'],
    max_newton = ipm_cfg['max_newton'],
    tol        = ipm_cfg['tol'],
    enable_viscous_ground_friction = ipm_cfg['enable_viscous_ground_friction'],
    smooth_sdf = ipm_cfg['smooth_sdf'],
    c_lin      = ipm_cfg['c_lin'],
    c_ang      = ipm_cfg['c_ang'],
)

sqp_solver = SingleShootingSQPGaussNewton(
    mass=phys['mass'], side_length=phys['side'], mu=phys['mu'],
    horizon=phys['horizon'], dt=phys['dt'], device=device,
    dynamics_module=step_square_pos_ip, ipmOpts=ipm_opts,
)

irl = IRLSolver(
    mass=phys['mass'], side_length=phys['side'], mu=phys['mu'],
    horizon=phys['horizon'], dt=phys['dt'],
    device=device, ipm_opts=ipm_opts,
    sqp_solver=sqp_solver,
)

# ---------------------------------------------------------------------------
# Method 1: KKT Residual
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("Method 1: KKT Residual")
print(f"{'='*60}")

result_kkt = irl.fit_weights_kkt_residual(all_demos, verbose=True)
w_kkt = result_kkt['w_recovered'].cpu().numpy()

# ---------------------------------------------------------------------------
# Method 2: Control Matching EG  (KKT result as warm start)
# ---------------------------------------------------------------------------

if not args.skip_eg:
    print(f"\n{'='*60}")
    print("Method 2: Control Matching EG")
    print(f"{'='*60}")

    cfg = IRLConfig(
        max_outer_iters      = args.outer_iters,
        max_inner_iters      = args.inner_iters,
        lr_weights           = args.lr,
        weight_tol           = args.weight_tol,
        lr_patience          = 5,
        lr_factor            = 0.5,
        lr_min               = 1e-5,
        inner_cost_tol       = 1e-7,
        inner_proj_grad_tol  = 1e-4,
        warm_start_inner     = True,
        hessian_reg          = 1e-2,
        u_min                = -0.5,
        u_max                =  0.5,
        verbose              = True,
        log_every            = 1,
    )

    result_eg = irl.fit_control_matching_eg(
        all_demos,
        w_init=result_kkt['w_recovered'].clone(),
        cfg=cfg,
    )
    w_eg = result_eg['w_recovered'].cpu().numpy()

# ---------------------------------------------------------------------------
# Results comparison
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("Weight Comparison")
print(f"{'='*60}")
header = f"{'Name':12s} {'True':>10s} {'KKT':>10s}"
if not args.skip_eg:
    header += f" {'EG':>10s}"
print(header)
print("-" * len(header))

for i, name in enumerate(wnames):
    row = f"{name:12s} {w_true[i]:10.4f} {w_kkt[i]:10.4f}"
    if not args.skip_eg:
        row += f" {w_eg[i]:10.4f}"
    print(row)

print(f"\nL2 error  KKT: {np.linalg.norm(w_kkt - w_true):.4f}", end='')
if not args.skip_eg:
    print(f"   EG: {np.linalg.norm(w_eg - w_true):.4f}", end='')
print()

if not args.skip_eg:
    n_conv  = sum(result_eg['inner_converged_history'])
    n_total = len(result_eg['inner_converged_history'])
    pg      = result_eg['inner_proj_grad_history']
    print(f"\nInner convergence: {n_conv}/{n_total} iters converged")
    print(f"  proj_grad range: [{min(pg):.2e}, {max(pg):.2e}]")

# ---------------------------------------------------------------------------
# Plot (EG iteration history + final bar chart)
# ---------------------------------------------------------------------------

if args.skip_eg:
    sys.exit(0)

try:
    import matplotlib.pyplot as plt

    iters  = range(len(result_eg['loss_history']))
    w_hist = np.array(result_eg['w_history'])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('IRL Control Matching EG', fontsize=13)

    # Outer loss
    axes[0, 0].semilogy(iters, result_eg['loss_history'])
    axes[0, 0].set_title('Outer Loss ||u*−u_demo||²')
    axes[0, 0].set_xlabel('Outer iter')
    axes[0, 0].grid(True, alpha=0.3)

    # Outer grad norm
    axes[0, 1].semilogy(iters, result_eg['grad_norm_history'])
    axes[0, 1].set_title('|grad_w| (IFT analytical)')
    axes[0, 1].set_xlabel('Outer iter')
    axes[0, 1].grid(True, alpha=0.3)

    # Inner proj grad norm
    axes[0, 2].semilogy(iters, result_eg['inner_proj_grad_history'], 'r-o', ms=3)
    axes[0, 2].axhline(cfg.inner_proj_grad_tol, ls='--', color='gray',
                       alpha=0.6, label=f'tol={cfg.inner_proj_grad_tol:.0e}')
    axes[0, 2].set_title('Inner Proj Grad Norm (SQP)')
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('Outer iter')
    axes[0, 2].grid(True, alpha=0.3)

    # Weight trajectory
    for i, name in enumerate(wnames):
        axes[1, 0].plot(iters, w_hist[:, i], label=name)
    for wt in w_true:
        axes[1, 0].axhline(wt, ls='--', alpha=0.3, color='gray')
    axes[1, 0].set_title('Weight Trajectory')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_xlabel('Outer iter')
    axes[1, 0].grid(True, alpha=0.3)

    # Control error + LR
    ax_ctrl = axes[1, 1]
    ax_ctrl.plot(iters, result_eg['control_error_history'])
    ax_ctrl.set_title('Control Error + LR')
    ax_ctrl.set_xlabel('Outer iter')
    ax_ctrl.grid(True, alpha=0.3)
    ax_lr = ax_ctrl.twinx()
    ax_lr.semilogy(iters, result_eg['lr_history'], 'g--', alpha=0.5)
    ax_lr.set_ylabel('lr', color='g')

    # Final weight bar chart (True / KKT / EG)
    x = np.arange(len(wnames))
    w = 0.25
    axes[1, 2].bar(x - w, w_true, w, label='True',  alpha=0.85)
    axes[1, 2].bar(x,     w_kkt,  w, label='KKT',   alpha=0.85)
    axes[1, 2].bar(x + w, w_eg,   w, label='EG',    alpha=0.85)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(wnames, rotation=15, fontsize=8)
    axes[1, 2].set_title('Final Weights')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = 'irl_result.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved: {out}")

except ImportError:
    print("\n(matplotlib not available)")