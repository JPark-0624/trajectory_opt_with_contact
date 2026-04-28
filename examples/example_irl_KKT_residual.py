"""
IRL KKT Residual — Trajectory Comparison

w_true vs w_recovered로 최적화한 trajectory를 비교합니다.

Animation: 왼쪽 = demo (w_true), 오른쪽 = recovered (w_recovered)
각 demo마다 animation, trajectory plot, analysis plot을 저장합니다.
모든 파일에 동일한 time signature를 붙여 구분합니다.

Usage:
    python example_irl_KKT_trajectory_comparison.py --data_dir ./data/irl
    python example_irl_KKT_trajectory_comparison.py --data_dir ./data/irl \\
        --w_recovered 0.915 0.047 0.010 0.002 0.026
"""

import torch
import numpy as np
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_opt_with_contact.single_shooting_SQP_moreau import (
    SingleShootingSQPGaussNewton, SQPConfig, CostWeights
)
from trajectory_opt_with_contact.dynamics import step_square_pos_ip, IPMOptions
from trajectory_opt_with_contact.utils import load_demo_npz
from trajectory_opt_with_contact.visualizer import TrajectoryVisualizer

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/irl')
parser.add_argument('--w_recovered', type=float, nargs=5,
                    default=[0.915429, 0.046885, 0.010116, 0.001647, 0.025923],
                    metavar=('w_target', 'w_orient', 'w_v', 'w_ctrl', 'w_contact'),
                    help='Recovered weights from KKT residual IRL')
args = parser.parse_args()

# 모든 파일에 공통으로 사용할 time signature
TS = int(time.time())

# ---------------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------------

with open(os.path.join(args.data_dir, 'metadata.json')) as f:
    metadata = json.load(f)

phys     = metadata['physics']
ipm_cfg  = metadata['ipm_opts']
sqp_meta = metadata['sqp_cfg']
w_true_norm      = np.array(metadata['w_true_norm'])
weight_names     = metadata['weight_names']
w_recovered_vals = np.array(args.w_recovered)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Time signature: {TS}")
print(f"Device: {device}")
print(f"\nw_true:      {dict(zip(weight_names, w_true_norm.round(4)))}")
print(f"w_recovered: {dict(zip(weight_names, w_recovered_vals.round(4)))}")
print(f"L2 error:    {np.linalg.norm(w_recovered_vals - w_true_norm):.4f}")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

ipm_opts = IPMOptions(
    target_mu  = ipm_cfg['target_mu'],
    max_newton = ipm_cfg['max_newton'],
    tol        = ipm_cfg['tol'],
    enable_viscous_ground_friction = ipm_cfg['enable_viscous_ground_friction'],
    smooth_sdf = ipm_cfg['smooth_sdf'],
    c_lin      = ipm_cfg['c_lin'],
    c_ang      = ipm_cfg['c_ang'],
)

sqp = SingleShootingSQPGaussNewton(
    mass        = phys['mass'],
    side_length = phys['side'],
    mu          = phys['mu'],
    horizon     = phys['horizon'],
    dt          = phys['dt'],
    device      = device,
    dynamics_module = step_square_pos_ip,
    ipmOpts     = ipm_opts,
)

sqp_cfg = SQPConfig(
    maxIters               = sqp_meta['maxIters'],
    cost_tol               = sqp_meta['cost_tol'],
    proj_grad_tol          = sqp_meta['proj_grad_tol'],
    use_gauss_newton       = True,
    hessian_regularization = 1e-6,
    use_line_search        = True,
    line_search_max_iters  = 10,
    line_search_beta       = 0.5,
    u_min                  = sqp_meta['u_min'],
    u_max                  = sqp_meta['u_max'],
    use_trust_region       = True,
    trust_region_iters     = 3,
    trust_region_size      = 0.5,
    use_mu_scheduling      = False,
    u_init_mode            = 'zero',
)

def make_cw(w_arr):
    return CostWeights(
        wTargetXY    = float(w_arr[0]),
        wTargetOrient= float(w_arr[1]),
        wObjVel      = float(w_arr[2]),
        wControl     = float(w_arr[3]),
        wContact     = float(w_arr[4]),
    )

W_TRUE      = make_cw(w_true_norm)
W_RECOVERED = make_cw(w_recovered_vals)

viz  = TrajectoryVisualizer(half_size=sqp.half)
half = sqp.half
T    = phys['horizon']
dt   = phys['dt']
ts   = np.arange(T + 1) * dt

# ---------------------------------------------------------------------------
# Per-demo: optimize + visualize
# ---------------------------------------------------------------------------

summary      = []
all_plot_data = []

for d in metadata['demos']:
    if not d['inner_converged']:
        print(f"\nSkipping {d['name']} (not converged)")
        continue

    name    = d['name']
    q0      = d['q0']
    v0      = d['v0']
    pusher0 = d['pusher0']
    goal    = d['goal']

    print(f"\n{'='*60}")
    print(f"Demo: {name}  [ts={TS}]")
    print(f"{'='*60}")

    # Optimize with w_true (= demo reference)
    print(f"  Optimizing with w_true...")
    result_true = sqp.optimize(
        q0=q0, v0=v0, pusher0=pusher0, goal=goal,
        cfg=sqp_cfg, w=W_TRUE, verbose=False,
    )

    # Optimize with w_recovered
    print(f"  Optimizing with w_recovered...")
    result_rec = sqp.optimize(
        q0=q0, v0=v0, pusher0=pusher0, goal=goal,
        cfg=sqp_cfg, w=W_RECOVERED, verbose=False,
    )

    # Metrics
    u_true = result_true['u_seq']
    u_rec  = result_rec['u_seq']
    q_true = result_true['trajectory']
    q_rec  = result_rec['trajectory']

    ctrl_err     = float(np.linalg.norm(u_rec - u_true))
    pos_err_traj = np.linalg.norm(q_rec - q_true, axis=1)
    pos_err_term = float(pos_err_traj[-1])
    proj_true    = result_true['stationarityInfo']['final_proj_grad_norm']
    proj_rec     = result_rec['stationarityInfo']['final_proj_grad_norm']

    print(f"  w_true:      proj={proj_true:.2e}, "
          f"term_err={result_true['stationarityInfo']['final_terminal_error']:.4f}m")
    print(f"  w_recovered: proj={proj_rec:.2e}, "
          f"term_err={result_rec['stationarityInfo']['final_terminal_error']:.4f}m")
    print(f"  ||u_rec - u_true|| = {ctrl_err:.4f}")
    print(f"  pos_err_term       = {pos_err_term:.4f}m")

    summary.append({
        'name': name, 'ctrl_err': ctrl_err,
        'pos_err_term': pos_err_term,
        'proj_true': proj_true, 'proj_rec': proj_rec,
        'term_true': result_true['stationarityInfo']['final_terminal_error'],
        'term_rec':  result_rec['stationarityInfo']['final_terminal_error'],
    })

    all_plot_data.append({
        'name': name, 'goal': goal,
        'q_true': q_true, 'q_rec': q_rec,
        'pr_true': result_true['pusher_trajectory'],
        'pr_rec':  result_rec['pusher_trajectory'],
        'u_true': u_true, 'u_rec': u_rec,
        'pos_err_traj': pos_err_traj,
    })

    # ------------------------------------------------------------------
    # Visualize: left = demo (w_true), right = recovered (w_recovered)
    #
    # animate_trajectory reads:
    #   result['initial_trajectory']        → left panel
    #   result['trajectory']                → right panel
    #
    # result_rec에 demo trajectory를 initial로 삽입
    # ------------------------------------------------------------------
    result_for_viz = dict(result_rec)
    result_for_viz['initial_trajectory']          = result_true['trajectory']
    result_for_viz['initial_velocity_trajectory'] = result_true['velocity_trajectory']
    result_for_viz['initial_pusher_trajectory']   = result_true['pusher_trajectory']

    xlim = (-0.6, 1.0)
    ylim = (-0.6, 1.0)

    f_traj = f"traj_{name}_{TS}.png"
    f_anim = f"anim_{name}_{TS}.mp4"
    f_anal = f"analysis_{name}_{TS}.png"

    try:
        viz.plot_trajectory(result_for_viz, goal,
                            xlim=xlim, ylim=ylim, save_path=f_traj)
        viz.plot_analysis(result_for_viz, goal, save_path=f_anal)
        viz.animate_trajectory(result_for_viz, goal,
                               save_path=f_anim, fps=30,
                               obstacle_pos=None,
                               xlim=xlim, ylim=ylim)
        print(f"  ✓ Saved: {f_traj}, {f_anal}, {f_anim}")
    except Exception as e:
        print(f"  △ Visualization error: {e}")

# ---------------------------------------------------------------------------
# Summary comparison plot (all demos, 3 columns)
# ---------------------------------------------------------------------------

n_demos = len(all_plot_data)
c_true  = '#2196F3'
c_rec   = '#FF5722'

fig, axes = plt.subplots(n_demos, 3, figsize=(14, 4 * n_demos))
if n_demos == 1:
    axes = axes[np.newaxis, :]
fig.suptitle(
    f'IRL KKT Residual — Demo (w_true) vs Recovered (w_rec)  [ts={TS}]\n'
    f'w_rec={dict(zip(weight_names, w_recovered_vals.round(3)))}',
    fontsize=11, fontweight='bold'
)

for row, pd in enumerate(all_plot_data):
    ax_xy   = axes[row, 0]
    ax_pos  = axes[row, 1]
    ax_ctrl = axes[row, 2]

    goal_arr = np.array(pd['goal'])
    q_t  = pd['q_true']
    q_r  = pd['q_rec']
    t_c  = np.arange(T) * dt

    # --- XY trajectory ---
    ax_xy.set_aspect('equal')
    ax_xy.plot(q_t[:, 0], q_t[:, 1], '-',  color=c_true, lw=1.8, label='demo (w_true)')
    ax_xy.plot(q_r[:, 0], q_r[:, 1], '--', color=c_rec,  lw=1.8, label='recovered')
    ax_xy.plot(pd['pr_true'][:, 0], pd['pr_true'][:, 1],
               '-',  color=c_true, lw=0.8, alpha=0.35)
    ax_xy.plot(pd['pr_rec'][:, 0],  pd['pr_rec'][:, 1],
               '--', color=c_rec,  lw=0.8, alpha=0.35)
    goal_patch = mpatches.Rectangle(
        (goal_arr[0]-half, goal_arr[1]-half), 2*half, 2*half,
        lw=1.5, edgecolor='gold', facecolor='none', linestyle='--', label='goal'
    )
    ax_xy.add_patch(goal_patch)
    ax_xy.set_title(f"{pd['name']}\nXY trajectory", fontsize=9)
    ax_xy.legend(fontsize=7, loc='upper left')
    ax_xy.set_xlabel('x [m]', fontsize=8)
    ax_xy.set_ylabel('y [m]', fontsize=8)
    ax_xy.grid(True, alpha=0.3)

    # --- Position error over time ---
    ax_pos.plot(ts, pd['pos_err_traj'], color='purple', lw=1.5)
    ax_pos.axhline(pd['pos_err_traj'][-1], color='purple', ls='--', alpha=0.5,
                   label=f"terminal={pd['pos_err_traj'][-1]:.3f}m")
    ax_pos.set_title('||q_rec - q_true|| over time', fontsize=9)
    ax_pos.set_xlabel('time [s]', fontsize=8)
    ax_pos.set_ylabel('[m]', fontsize=8)
    ax_pos.legend(fontsize=7)
    ax_pos.grid(True, alpha=0.3)

    # --- Control comparison ---
    u_t = pd['u_true']
    u_r = pd['u_rec']
    ax_ctrl.plot(t_c, u_t[:, 0], '-',  color=c_true, lw=1.2, label='u_x demo')
    ax_ctrl.plot(t_c, u_t[:, 1], '--', color=c_true, lw=1.2, label='u_y demo')
    ax_ctrl.plot(t_c, u_r[:, 0], '-',  color=c_rec,  lw=1.2, label='u_x rec')
    ax_ctrl.plot(t_c, u_r[:, 1], '--', color=c_rec,  lw=1.2, label='u_y rec')
    ax_ctrl.set_title('Control trajectories', fontsize=9)
    ax_ctrl.set_xlabel('time [s]', fontsize=8)
    ax_ctrl.set_ylabel('[m/s]', fontsize=8)
    ax_ctrl.legend(fontsize=6, ncol=2)
    ax_ctrl.grid(True, alpha=0.3)

plt.tight_layout()
f_summary = f"summary_comparison_{TS}.png"
plt.savefig(f_summary, dpi=150, bbox_inches='tight')
print(f"\nSummary plot saved: {f_summary}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print(f"\n{'='*65}")
print("Summary")
print(f"{'='*65}")
print(f"{'Demo':20s} {'ctrl_err':>10s} {'pos_term':>10s} "
      f"{'term_true':>10s} {'term_rec':>10s}")
print("-" * 65)
for s in summary:
    print(f"{s['name']:20s} {s['ctrl_err']:10.4f} {s['pos_err_term']:10.4f}m "
          f"{s['term_true']:10.4f}m {s['term_rec']:10.4f}m")

print(f"\nAll files share time signature: {TS}")
print(f"  Animation:  anim_{{name}}_{TS}.mp4")
print(f"  Trajectory: traj_{{name}}_{TS}.png")
print(f"  Analysis:   analysis_{{name}}_{TS}.png")
print(f"  Summary:    summary_comparison_{TS}.png")