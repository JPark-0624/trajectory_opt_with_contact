"""
IRL Dataset Generator

Known w_true로 SQP SS를 실행하여 demo trajectory를 생성하고 저장합니다.
IRL 테스트 시 매번 재생성하지 않고 이 파일을 불러씁니다.

Usage:
    python generate_irl_dataset.py [--out_dir ./data/irl]
"""

import torch
import numpy as np
import argparse
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_opt_with_contact.single_shooting_SQP_moreau import (
    SingleShootingSQPGaussNewton, SQPConfig, CostWeights
)
from trajectory_opt_with_contact.dynamics import step_square_pos_ip, IPMOptions
from trajectory_opt_with_contact.utils import save_demo_npz

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='./data/irl')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Physical params (fixed across all demos)
MASS   = 1.0
SIDE   = 0.2
MU     = 0.5
DT     = 0.05
HORIZON = 30

ipm_opts = IPMOptions(
    target_mu=1e-4, max_newton=20, tol=1e-6,
    enable_viscous_ground_friction=True,
    smooth_sdf=50.0, c_lin=1.0, c_ang=0.00667
)

sqp = SingleShootingSQPGaussNewton(
    mass=MASS, side_length=SIDE, mu=MU,
    horizon=HORIZON, dt=DT, device=device,
    dynamics_module=step_square_pos_ip,
    ipmOpts=ipm_opts,
)

sqp_cfg = SQPConfig(
    maxIters=60,
    cost_tol=1e-7,
    proj_grad_tol=1e-4,
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

# True weights (to recover via IRL)
W_TRUE = CostWeights(
    wTargetXY    = 20.0,
    wTargetOrient = 1.0,
    wObjVel      = 1.0,
    wControl     = 0.01,
    wContact     = 3.0
)
w_true_vec = np.array([W_TRUE.wTargetXY, W_TRUE.wTargetOrient,
                        W_TRUE.wObjVel,   W_TRUE.wControl, W_TRUE.wContact])
w_true_norm = w_true_vec / w_true_vec.sum()

# ---------------------------------------------------------------------------
# Demo scenarios
# Different (q0, pusher0, goal) combinations for identifiability
# ---------------------------------------------------------------------------

scenarios = [
    {
        'name': 'diagonal_push',
        'q0':      [0.0,  0.0, 0.0],
        'v0':      [0.0,  0.0, 0.0],
        'pusher0': [-0.1, -0.3],
        'goal':    [0.2,  0.5, 0.3],
    },
    {
        'name': 'x_axis_push',
        'q0':      [0.0,  0.0, 0.0],
        'v0':      [0.0,  0.0, 0.0],
        'pusher0': [-0.3, 0.0],
        'goal':    [0.5,  0.0, 0.0],
    },
    {
        'name': 'angled_push',
        'q0':      [0.0,  0.0,  0.0],
        'v0':      [0.0,  0.0,  0.0],
        'pusher0': [0.0, -0.3],
        'goal':    [0.2,  0.5, -0.3],
    },
    {
        'name': 'side_push',
        'q0':      [0.0,  0.0, 0.0],
        'v0':      [0.0,  0.0, 0.0],
        'pusher0': [-0.2, 0.0],
        'goal':    [0.5,  0.2, 0.3],
    },
]

# ---------------------------------------------------------------------------
# Generate and save
# ---------------------------------------------------------------------------
 
metadata = {
    'w_true': w_true_vec.tolist(),
    'w_true_norm': w_true_norm.tolist(),
    'weight_names': ['w_target', 'w_orient', 'w_v', 'w_ctrl', 'w_contact'],
    'w_true_named': {
        'w_target':  float(W_TRUE.wTargetXY),
        'w_orient':  float(W_TRUE.wTargetOrient),
        'w_v':       float(W_TRUE.wObjVel),
        'w_ctrl':    float(W_TRUE.wControl),
        'w_contact': float(W_TRUE.wContact),
    },
    'physics': {
        'mass': MASS, 'side': SIDE, 'mu': MU,
        'dt': DT, 'horizon': HORIZON,
    },
    'ipm_opts': {
        'target_mu': ipm_opts.target_mu,
        'max_newton': ipm_opts.max_newton,
        'tol': ipm_opts.tol,
        'enable_viscous_ground_friction': ipm_opts.enable_viscous_ground_friction,
        'smooth_sdf': ipm_opts.smooth_sdf,
        'c_lin': ipm_opts.c_lin,
        'c_ang': ipm_opts.c_ang,
    },
    'sqp_cfg': {
        'maxIters': sqp_cfg.maxIters,
        'cost_tol': sqp_cfg.cost_tol,
        'proj_grad_tol': sqp_cfg.proj_grad_tol,
        'tol': sqp_cfg.tol,
        'u_min': sqp_cfg.u_min,
        'u_max': sqp_cfg.u_max,
    },
    'demos': [],
}
 
for sc in scenarios:
    print(f"\n{'='*60}")
    print(f"Generating demo: {sc['name']}")
    print(f"  q0={sc['q0']}, goal={sc['goal']}")
    print(f"{'='*60}")
 
    result = sqp.optimize(
        q0=sc['q0'], v0=sc['v0'],
        pusher0=sc['pusher0'], goal=sc['goal'],
        cfg=sqp_cfg, w=W_TRUE, verbose=True,
    )
 
    inner_grad_norm      = result['stationarityInfo']['final_grad_norm']
    inner_proj_grad_norm = result['stationarityInfo']['final_proj_grad_norm']
    inner_converged      = result['stationarityInfo']['converged']
 
    print(f"\n  → grad_norm={inner_grad_norm:.3e}  (projected: {inner_proj_grad_norm:.3e}), converged={inner_converged}")
    print(f"  → loss={result['loss']:.4f}, pos_err={result['stationarityInfo']['final_terminal_error']:.4f}m")
 
    if not inner_converged:
        print(f"  ⚠ Skipping '{sc['name']}' — SQP did not converge")
        continue
 
    # Pack demo
    demo = {
        'q0':      np.array(sc['q0']),
        'v0':      np.array(sc['v0']),
        'pusher0': np.array(sc['pusher0']),
        'goal':    np.array(sc['goal']),
        'u_demo':  result['u_seq'],          # (T, 2) numpy
        'obstacle_pos': None,
    }
 
    # Save
    path = os.path.join(args.out_dir, f"{sc['name']}.npz")
    save_demo_npz(demo, path)
    print(f"  → Saved: {path}")
 
    metadata['demos'].append({
        'name': sc['name'],
        'path': path,
        'q0': sc['q0'],
        'v0': sc['v0'],
        'pusher0': sc['pusher0'],
        'goal': sc['goal'],
        'final_grad_norm': float(inner_grad_norm),
        'final_proj_grad_norm': float(inner_proj_grad_norm),
        'inner_converged': bool(inner_converged),
        'final_loss': float(result['loss']),
        'pos_error': float(result['stationarityInfo']['final_terminal_error']),
    })
 
# Save metadata
meta_path = os.path.join(args.out_dir, 'metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"\nMetadata saved: {meta_path}")
 
# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
 
print(f"\n{'='*60}")
print("Dataset Summary")
print(f"{'='*60}")
print(f"w_true (normalized): {dict(zip(metadata['weight_names'], w_true_norm.round(4)))}")
print(f"Saved {len(metadata['demos'])}/{len(scenarios)} demos\n")
for d in metadata['demos']:
    print(f"  ✓ {d['name']:20s} | grad={d['final_grad_norm']:.2e} (proj={d['final_proj_grad_norm']:.2e}) | loss={d['final_loss']:.4f}")
 
skipped = [sc['name'] for sc in scenarios
           if sc['name'] not in [d['name'] for d in metadata['demos']]]
for name in skipped:
    print(f"  ✗ {name:20s} | skipped (not converged)")
 