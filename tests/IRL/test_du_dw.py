"""
∂u*/∂w 비교: Analytical IFT vs Autograd (moreau K-step unrolling)

검증 목적:
  - Analytical IFT (-P^{-1} M)의 수식 정확성을 moreau autograd로 검증
  - moreau는 참값으로 가정, 우리 수식을 검증하는 것

구조:
  Method 1 (IFT):     u_demo 로드 → compute_gauss_newton_hessian() → -P^{-1} M
  Method 2 (Autograd): optimize_differentiable(K, zero_init, W_TRUE_NORM) → jacobian

수렴 확인:
  ||u_K - u_demo|| < tol  →  autograd Jacobian이 유효한 비교 대상

Usage:
    python test_ift_vs_autograd.py --data_dir ./data/irl --autograd_K 40
"""

import torch
import numpy as np
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_opt_with_contact.single_shooting_SQP_moreau import (
    SingleShootingSQPGaussNewton, SQPConfig, CostWeights
)
from trajectory_opt_with_contact.dynamics import step_square_pos_ip, IPMOptions
from trajectory_opt_with_contact.utils import load_demo_npz

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',   type=str, default='./data/irl')
parser.add_argument('--autograd_K', type=int, default=40,
                    help='SQP iterations for optimize_differentiable. '
                         'Increase until final proj_grad < 1e-4')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------------

meta_path = os.path.join(args.data_dir, 'metadata.json')
assert os.path.exists(meta_path), f"metadata.json not found at {args.data_dir}"

with open(meta_path) as f:
    metadata = json.load(f)

phys         = metadata['physics']
ipm_cfg      = metadata['ipm_opts']
sqp_meta     = metadata['sqp_cfg']
w_true_norm  = np.array(metadata['w_true_norm'])
weight_names = metadata['weight_names']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"w_true (normalized): {dict(zip(weight_names, w_true_norm.round(4)))}")
print(f"Autograd K: {args.autograd_K}")

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

# Normalized w — same scale for both IFT and autograd
W_TRUE_NORM = CostWeights(
    wTargetXY    = float(w_true_norm[0]),
    wTargetOrient= float(w_true_norm[1]),
    wObjVel      = float(w_true_norm[2]),
    wControl     = float(w_true_norm[3]),
    wContact     = float(w_true_norm[4]),
)

# W_TRUE: unnormalized (demo 생성 조건과 동일)
w_true_named = metadata['w_true_named']
w_true_unnorm = np.array([
    w_true_named['w_target'], w_true_named['w_orient'],
    w_true_named['w_v'],      w_true_named['w_ctrl'],
    w_true_named['w_contact'],
])

N_w = len(weight_names)
n   = phys['horizon'] * 2
reg = sqp_cfg.hessian_regularization

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cos_sim(A, B):
    a, b = A.flatten(), B.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

def col_cos(A, B):
    return [(A[:, i] @ B[:, i] /
             (A[:, i].norm() * B[:, i].norm() + 1e-12)).item()
            for i in range(A.shape[1])]

def make_cw(w_):
    return CostWeights(
        wTargetXY    = w_[0],
        wTargetOrient= w_[1],
        wObjVel      = w_[2],
        wControl     = w_[3],
        wContact     = w_[4],
    )

# ---------------------------------------------------------------------------
# Per-demo comparison
# ---------------------------------------------------------------------------

results = []

for d in metadata['demos']:
    if not d['inner_converged']:
        print(f"\nSkipping {d['name']} (not converged in metadata)")
        continue

    name    = d['name']
    q0      = d['q0']
    v0      = d['v0']
    pusher0 = d['pusher0']
    goal    = d['goal']

    print(f"\n{'='*65}")
    print(f"Demo: {name}")
    print(f"  q0={q0}, goal={goal}")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # Load u_demo from file  (수렴된 u*, optimize() 재실행 불필요)
    # ------------------------------------------------------------------
    npz_path = os.path.join(args.data_dir, f"{name}.npz")
    demo     = load_demo_npz(npz_path)
    u_demo   = torch.tensor(demo['u_demo'], dtype=torch.float64, device=device)

    # tensor versions — list로 넘어오면 dynamics에서 AttributeError 발생
    q0_t      = torch.tensor(q0,      dtype=torch.float64, device=device)
    v0_t      = torch.tensor(v0,      dtype=torch.float64, device=device)
    pusher0_t = torch.tensor(pusher0, dtype=torch.float64, device=device)
    goal_t    = torch.tensor(goal,    dtype=torch.float64, device=device)

    print(f"  u_demo loaded: shape={tuple(u_demo.shape)}, "
          f"range=[{u_demo.min():.3f}, {u_demo.max():.3f}]")
    print(f"  metadata proj_grad={d['final_proj_grad_norm']:.2e}")

    # ------------------------------------------------------------------
    # Method 1: Analytical IFT
    # ------------------------------------------------------------------
    # Method 1: Analytical IFT (W_TRUE unnormalized — demo 생성 조건과 동일)
    W_TRUE = CostWeights(
        wTargetXY    = float(w_true_unnorm[0]),
        wTargetOrient= float(w_true_unnorm[1]),
        wObjVel      = float(w_true_unnorm[2]),
        wControl     = float(w_true_unnorm[3]),
        wContact     = float(w_true_unnorm[4]),
    )
    with torch.no_grad():
        P_gn, _, _, J_r, r = sqp.compute_gauss_newton_hessian(
            u_demo, q0_t, v0_t, pusher0_t, goal_t, W_TRUE,
            regularization=0.0,
        )

    T    = phys['horizon']
    n_res = r.shape[0]
    block_sizes  = [n, 2, 1, 3, T]
    block_w_vals = [W_TRUE.wControl, W_TRUE.wTargetXY,
                    W_TRUE.wTargetOrient, W_TRUE.wObjVel,
                    W_TRUE.wContact]
    block_col    = [3, 0, 1, 2, 4]

    # M[:,col] = J_r_block^T r_block / w_i
    # Derivation: d/dw_i [J_r^T r] = d/dw_i [w_i * J_fi^T fi] = J_fi^T fi = J_r_i^T r_i / w_i
    M = torch.zeros(n, N_w, dtype=torch.float64, device=device)
    idx = 0
    for i, bs in enumerate(block_sizes):
        wi  = float(block_w_vals[i]) if not isinstance(block_w_vals[i], torch.Tensor) \
              else block_w_vals[i].item()
        wi  = max(wi, 1e-8)
        col = block_col[i]
        M[:, col] = J_r[idx:idx+bs, :].T @ r[idx:idx+bs] / wi
        idx += bs
    P_reg = P_gn + reg * torch.eye(n, dtype=torch.float64, device=device)
    du_dw_ift = -torch.linalg.solve(P_reg, M).cpu()  # (n, N_w)

    print(f"\n  [IFT]     ||J|| = {du_dw_ift.norm():.4e}")

    # ------------------------------------------------------------------
    # Method 2: Autograd (W_TRUE unnormalized, zero init, K steps)
    #
    # demo 생성과 동일한 W_TRUE로 수렴 → u_demo와 같은 u*에 도달
    # zero init에서 전체 최적화 경로가 graph에 포함
    # ------------------------------------------------------------------
    w_t = torch.tensor(w_true_unnorm, dtype=torch.float64,
                       device=device, requires_grad=True)

    def u_from_w(w_):
        u_K, _ = sqp.optimize_differentiable(
            q0=q0_t, v0=v0_t, pusher0=pusher0_t, goal=goal_t,
            w=make_cw(w_),
            K=args.autograd_K,
            reg=reg,
            u_init=None,   # zero init
            verbose=False,
        )
        return u_K.flatten()

    # 수렴 확인 (no_grad)
    with torch.no_grad():
        _, final_proj = sqp.optimize_differentiable(
            q0=q0_t, v0=v0_t, pusher0=pusher0_t, goal=goal_t,
            w=make_cw(w_t), K=args.autograd_K, reg=reg,
            u_init=None, verbose=False,
        )
    converged_auto = final_proj < 5e-4
    print(f"\n  [Autograd] K={args.autograd_K}, W_TRUE, final proj_grad={final_proj:.3e}  "
          f"({'✓ converged' if converged_auto else '△ not converged — increase K'})")
    if not converged_auto:
        print(f"  ⚠ Skipping Jacobian")
        continue

    print(f"  [Autograd] computing Jacobian...")
    du_dw_auto = torch.autograd.functional.jacobian(
        u_from_w,
        w_t,
        create_graph=False,
        vectorize=False,
        strategy='reverse-mode',
    ).detach().cpu()  # (n, N_w)

    print(f"  [Autograd] ||J|| = {du_dw_auto.norm():.4e}")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    overall_cos = cos_sim(du_dw_ift, du_dw_auto)
    per_col     = col_cos(du_dw_ift, du_dw_auto)

    print(f"\n  cos(IFT, Autograd) overall = {overall_cos:.4f}")
    print(f"  ||IFT|| / ||Auto|| = {du_dw_ift.norm().item() / (du_dw_auto.norm().item() + 1e-12):.4f}")
    print(f"  Per-column cosine similarity:")
    for wn, c in zip(weight_names, per_col):
        print(f"    {wn:12s}: {c:+.4f}")

    results.append({
        'name':        name,
        'norm_ift':    du_dw_ift.norm().item(),
        'norm_auto':   du_dw_auto.norm().item(),
        'cos_overall': overall_cos,
        'cos_per_col': per_col,
        'J_ift':       du_dw_ift.numpy(),
        'J_auto':      du_dw_auto.numpy(),
    })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*65}")
print("Summary")
print(f"{'='*65}")
print(f"{'Demo':20s} {'||IFT||':>10s} {'||Auto||':>10s} "
      f"{'ratio':>8s} {'cos':>8s}")
print("-" * 65)
for r in results:
    print(f"{r['name']:20s} {r['norm_ift']:10.4e} {r['norm_auto']:10.4e} "
          f"{r['norm_ift']/(r['norm_auto']+1e-12):8.4f} {r['cos_overall']:8.4f}")

if results:
    mean_cos = np.mean([r['cos_overall'] for r in results])
    print(f"\nMean cos(IFT, Autograd): {mean_cos:.4f}")

    print(f"\nPer-column mean cosine similarity:")
    per_col_all = np.array([r['cos_per_col'] for r in results])
    for i, wn in enumerate(weight_names):
        print(f"  {wn:12s}: {per_col_all[:, i].mean():+.4f}  "
              f"(min={per_col_all[:, i].min():+.4f}, "
              f"max={per_col_all[:, i].max():+.4f})")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_demos = len(results)
    if n_demos == 0:
        raise RuntimeError("No results")

    # ----------------------------------------------------------------
    # Figure layout:
    # Row per demo:
    #   col 0: per-weight cosine bar chart
    #   col 1: norm ratio bar chart
    #   col 2: IFT Jacobian heatmap  (n x N_w)
    #   col 3: Autograd Jacobian heatmap
    #   col 4: difference heatmap  (IFT - Auto) / max(|IFT|)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(
        n_demos, 5,
        figsize=(20, 4 * n_demos),
        gridspec_kw={'width_ratios': [2, 1, 3, 3, 3]},
    )
    if n_demos == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'IFT vs Autograd  ∂u*/∂w  (K={args.autograd_K})',
                 fontsize=13, fontweight='bold')

    cmap_j   = 'RdBu_r'
    cmap_err = 'RdYlGn_r'

    for di, r in enumerate(results):
        J_ift  = r['J_ift']   # (n, N_w)
        J_auto = r['J_auto']  # (n, N_w)
        name_d = r['name']

        # ---- col 0: per-weight cosine ----
        ax = axes[di, 0]
        colors = ['green' if c > 0.9 else 'orange' if c > 0.7 else 'red'
                  for c in r['cos_per_col']]
        ax.barh(weight_names, r['cos_per_col'], color=colors)
        ax.axvline(0.9, ls='--', color='green', alpha=0.5, lw=1)
        ax.axvline(0.0, ls='-',  color='black', lw=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_title(f'{name_d}\nPer-weight cos(IFT, Auto)={r["cos_overall"]:.3f}')
        ax.set_xlabel('Cosine similarity')
        ax.grid(True, axis='x', alpha=0.3)
        for i, c in enumerate(r['cos_per_col']):
            ax.text(min(c, 1.05), i, f'{c:.3f}', va='center', fontsize=7)

        # ---- col 1: norm ratio per weight ----
        ax = axes[di, 1]
        norm_ift_col  = [np.linalg.norm(J_ift[:,  j]) for j in range(N_w)]
        norm_auto_col = [np.linalg.norm(J_auto[:, j]) for j in range(N_w)]
        x = np.arange(N_w)
        ax.bar(x - 0.2, norm_ift_col,  0.4, label='IFT',  alpha=0.8)
        ax.bar(x + 0.2, norm_auto_col, 0.4, label='Auto', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([wn.replace('w_','') for wn in weight_names],
                           rotation=30, fontsize=7)
        ax.set_title('Column norms')
        ax.legend(fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)

        # ---- col 2,3: Jacobian heatmaps ----
        vmax = max(np.abs(J_ift).max(), np.abs(J_auto).max())
        vmax = vmax if vmax > 0 else 1.0

        for col_idx, (J, label) in enumerate([(J_ift, 'IFT'), (J_auto, 'Autograd')]):
            ax = axes[di, 2 + col_idx]
            im = ax.imshow(J, aspect='auto', cmap=cmap_j,
                           vmin=-vmax, vmax=vmax)
            ax.set_title(f'{label}  ||J||={np.linalg.norm(J):.2e}')
            ax.set_xlabel('Weight index')
            ax.set_ylabel('Control index')
            ax.set_xticks(range(N_w))
            ax.set_xticklabels([wn.replace('w_','') for wn in weight_names],
                               rotation=30, fontsize=7)
            plt.colorbar(im, ax=ax, shrink=0.8)

        # ---- col 4: relative difference heatmap ----
        ax   = axes[di, 4]
        diff = (J_ift - J_auto) / (vmax + 1e-12)
        im   = ax.imshow(diff, aspect='auto', cmap=cmap_err,
                         vmin=-1, vmax=1)
        ax.set_title(f'(IFT - Auto) / max|IFT|\n'
                     f'ratio={r['norm_ift']/(r['norm_auto']+1e-12):.4f}')
        ax.set_xlabel('Weight index')
        ax.set_ylabel('Control index')
        ax.set_xticks(range(N_w))
        ax.set_xticklabels([wn.replace('w_','') for wn in weight_names],
                           rotation=30, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out = 'ift_vs_autograd_result.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out}")

except Exception as e:
    print(f"\n(Visualization skipped: {e})")


print(f"\n{'='*65}")
print("Summary")
print(f"{'='*65}")
print(f"{'Demo':20s} {'||IFT||':>10s} {'||Auto||':>10s} "
      f"{'ratio':>8s} {'cos':>8s}")
print("-" * 65)
for r in results:
    print(f"{r['name']:20s} {r['norm_ift']:10.4e} {r['norm_auto']:10.4e} "
          f"{r['norm_ift']/(r['norm_auto']+1e-12):8.4f} {r['cos_overall']:8.4f}")

if results:
    mean_cos = np.mean([r['cos_overall'] for r in results])
    print(f"\nMean cos(IFT, Autograd): {mean_cos:.4f}")

    print(f"\nPer-column mean cosine similarity:")
    per_col_all = np.array([r['cos_per_col'] for r in results])
    for i, wn in enumerate(weight_names):
        print(f"  {wn:12s}: {per_col_all[:, i].mean():+.4f}  "
              f"(min={per_col_all[:, i].min():+.4f}, "
              f"max={per_col_all[:, i].max():+.4f})")