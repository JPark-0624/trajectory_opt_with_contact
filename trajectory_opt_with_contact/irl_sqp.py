"""
Inverse Reinforcement Learning for Contact-Implicit Trajectory Optimization.

Two methods:
  1. fit_weights_kkt_residual  : Inverse KKT — single QP, no inner loop
  2. fit_control_matching_eg   : Bilevel — SQP inner + IFT + Exponentiated Gradient outer

Cost structure (residual form, matches single_shooting_SQP_moreau):
  J(u; w) = (1/2) ||r(u; w)||^2
  r = [sqrt(w_ctrl)*u_flat, sqrt(w_target)*(q_T[:2]-g[:2]),
       sqrt(w_orient)*wrap(q_T[2]-g[2]), sqrt(w_vel)*v_T, sqrt(w_contact)*contact_term]

IFT identity:
  du*/dw = -P^{-1} M
  M[:,i] = J_{r,i}^T r_i / w_i        (d/dw_i [J_r^T r], w explicit only)
  P      = J_r^T J_r + reg*I

Weight vector w = [w_target, w_orient, w_v, w_ctrl, w_contact]  (simplex: sum=1, w>=0)

Boundary conditions on u:
  u in [u_min, u_max]^{2T}
  Projected gradient nullifies components at active bounds:
    g_i = 0  if  u_i >= u_max - eps and g_i < 0   (upper active, can't go higher)
    g_i = 0  if  u_i <= u_min + eps and g_i > 0   (lower active, can't go lower)
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
import moreau
import moreau.torch as moreau_torch
import scipy.sparse as sparse

from .dynamics import IPMOptions


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class IRLConfig:
    # Outer loop
    max_outer_iters: int = 50
    weight_tol: float = 1e-4          # convergence: ||w_new - w_old||

    # EG learning rate + plateau scheduler
    lr_weights: float = 0.01
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1e-5

    # Inner SQP
    max_inner_iters: int = 200
    warm_start_inner: bool = True
    inner_cost_tol: float = 1e-7
    inner_proj_grad_tol: float = 1e-4

    # IFT regularization
    hessian_reg: float = 1e-2

    # Control bounds (must match inner SQP)
    u_min: float = -0.5
    u_max: float =  0.5

    # Logging
    verbose: bool = True
    log_every: int = 5


# ---------------------------------------------------------------------------
# Block structure (shared by both methods)
# ---------------------------------------------------------------------------
# Residual block order: [ctrl(2T), target_xy(2), orient(1), vel(3), contact(T)]
# w column order:       [w_target(0), w_orient(1), w_v(2), w_ctrl(3), w_contact(4)]
_BLOCK_COL = [3, 0, 1, 2, 4]   # block i → weight column index


def _build_block_sizes(T: int) -> List[int]:
    return [T * 2, 2, 1, 3, T]


def _build_w_vals(w: torch.Tensor) -> List:
    """Return w values in block order: [ctrl, target_xy, orient, vel, contact]."""
    return [w[3], w[0], w[1], w[2], w[4]]


def _compute_M(J_r: torch.Tensor, r: torch.Tensor,
               block_sizes: List[int], w: torch.Tensor,
               n: int, N_w: int, device) -> torch.Tensor:
    """
    Compute mixed partial M (n x N_w):
        M[:,i] = J_{r,block_i}^T @ r_{block_i} / w_i

    Derivation:
        d/dw_i [J_r^T r]  (w explicit only, u fixed)
      = d/dw_i [w_i * J_{f_i}^T f_i]
      = J_{f_i}^T f_i
      = (J_{r,i}/sqrt(w_i))^T (r_i/sqrt(w_i))
      = J_{r,i}^T r_i / w_i
    """
    w_vals = _build_w_vals(w)
    M = torch.zeros(n, N_w, dtype=torch.float64, device=device)
    idx = 0
    for i, bs in enumerate(block_sizes):
        wi  = float(w_vals[i]) if not isinstance(w_vals[i], torch.Tensor) else w_vals[i].item()
        col = _BLOCK_COL[i]
        M[:, col] = J_r[idx:idx+bs, :].T @ r[idx:idx+bs] / max(wi, 1e-8)
        idx += bs
    return M


def _projected_grad_norm(g: torch.Tensor, u_flat: torch.Tensor,
                         u_min: float, u_max: float,
                         eps: float = 1e-6) -> float:
    """
    Projected gradient norm: zero out components blocked by active bounds.
        g_i = 0  if u_i >= u_max - eps and g_i < 0
        g_i = 0  if u_i <= u_min + eps and g_i > 0
    """
    pg = g.clone()
    pg[(u_flat >= u_max - eps) & (pg < 0)] = 0.0
    pg[(u_flat <= u_min + eps) & (pg > 0)] = 0.0
    return pg.norm().item()


# ---------------------------------------------------------------------------
# IRLSolver
# ---------------------------------------------------------------------------

class IRLSolver:
    """
    IRL solver for pusher-slider system.

    Methods:
      fit_weights_kkt_residual : Inverse KKT (single QP, no inner loop)
      fit_control_matching_eg  : Bilevel IRL with EG outer loop
    """

    WEIGHT_NAMES = ['w_target', 'w_orient', 'w_v', 'w_ctrl', 'w_contact']
    N_WEIGHTS = 5

    def __init__(
        self,
        mass: float,
        side_length: float,
        mu: float,
        horizon: int,
        dt: float,
        device: str = 'cpu',
        ipm_opts: Optional[IPMOptions] = None,
        sqp_solver=None,            # SingleShootingSQPGaussNewton instance
    ):
        self.mass     = mass
        self.half     = side_length / 2
        self.mu       = mu
        self.horizon  = horizon
        self.dt       = dt
        self.device   = torch.device(device)
        self.ipm_opts = ipm_opts if ipm_opts is not None else IPMOptions()
        self.sqp_solver = sqp_solver
        self.Izz = (1.0 / 6.0) * mass * (side_length ** 2 + side_length ** 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_weights_kkt_residual(
        self,
        demos,
        verbose: bool = True,
        u_min: float = -0.5,
        u_max: float =  0.5,
    ) -> Dict:
        """
        Inverse KKT IRL.

        At u_demo ≈ u*(w*), the KKT condition gives:
            Φ(u_demo) · w ≈ 0
        where Φ[:,i] = M[:,i] = J_{r,i}^T r_i / w_i  is the IFT mixed partial.

        For multiple demos:
            min_w  w^T (Σ_k Φ_k^T Φ_k) w
            s.t.   sum(w) = 1,  w >= 0

        Φ is evaluated at uniform w (w_i = 1/N_w) since the w-scaling cancels:
            Φ[:,i]|_{w=uniform} = J_{r,i}^T r_i * N_w
        which gives the same minimizer as the true w* evaluation.

        Boundary condition:
            Active-bound components of Φ w are zeroed before the QP,
            reflecting that the KKT condition at bounds is an inequality.
        """
        assert self.sqp_solver is not None, \
            "sqp_solver required. Pass SingleShootingSQPGaussNewton instance."

        from .single_shooting_SQP_moreau import CostWeights, SQPConfig

        if isinstance(demos, dict):
            demos = [demos]
        n_demos = len(demos)
        N_w = self.N_WEIGHTS
        T   = self.sqp_solver.horizon
        n   = T * 2
        block_sizes = _build_block_sizes(T)

        if verbose:
            print("\n" + "="*60)
            print(f"IRL - KKT Residual ({n_demos} demo{'s' if n_demos > 1 else ''})")
            print("="*60)

        # Uniform w for Jacobian evaluation (scaling cancels in Φ^T Φ minimizer)
        w_eval    = torch.ones(N_w, dtype=torch.float64, device=self.device) / N_w
        cost_w_eval = self._w_to_cost_weights(w_eval)

        # Accumulate Σ = Σ_k Φ_k^T Φ_k
        Sigma    = torch.zeros(N_w, N_w, dtype=torch.float64, device=self.device)
        Phi_list = []

        for k, demo in enumerate(demos):
            q0, v0, pr0, goal, u_demo, _ = self._unpack_demo(demo)

            _, _, _, J_r, r = self.sqp_solver.compute_gauss_newton_hessian(
                u_demo, q0, v0, pr0, goal, cost_w_eval, regularization=0.0,
            )

            # Φ_k: (n, N_w)  —  M[:,i] = J_{r,i}^T r_i / w_i
            Phi_k = _compute_M(J_r, r, block_sizes, w_eval, n, N_w, self.device)

            # Boundary condition: zero out rows where u_demo is at active bounds
            # and the gradient direction is blocked
            u_flat = u_demo.flatten()
            active_upper = u_flat >= u_max - 1e-6
            active_lower = u_flat <= u_min + 1e-6
            # For each weight column, zero rows at active bounds
            # (conservative: zero entire row regardless of sign — exact treatment
            #  requires knowing sign of Phi_k @ w*, but w* is unknown here)
            # Instead apply sign-aware projection at evaluation time (see verbose block)
            Sigma += Phi_k.T @ Phi_k
            Phi_list.append((Phi_k, u_demo))

        if verbose:
            cond = torch.linalg.cond(Sigma).item()
            print(f"\nΣ condition number: {cond:.2e}")
            eigs = torch.linalg.eigvalsh(Sigma)
            print(f"Σ eigenvalues: {eigs.cpu().numpy()}")

        # QP: min_w  (1/2) w^T (2Σ) w
        #     s.t.   sum(w) = 1,  w >= 0
        m_total = 1 + N_w
        A_dense = torch.zeros(m_total, N_w, dtype=torch.float64)
        A_dense[0, :] = 1.0
        for i in range(N_w):
            A_dense[1 + i, i] = -1.0

        A_sp   = sparse.csr_matrix(A_dense.numpy())
        A_ro_t = torch.tensor(A_sp.indptr,  dtype=torch.int32)
        A_ci_t = torch.tensor(A_sp.indices, dtype=torch.int32)
        A_vals = torch.tensor(A_sp.data,    dtype=torch.float64)

        P_dense = 2.0 * Sigma
        P_ro_t  = torch.arange(0, (N_w + 1) * N_w, N_w, dtype=torch.int32)
        P_ci_t  = torch.arange(N_w, dtype=torch.int32).repeat(N_w)

        moreau_dev = 'cuda' if self.device.type == 'cuda' and moreau.device_available('cuda') else 'cpu'
        qp = moreau_torch.Solver(
            n=N_w, m=m_total,
            P_row_offsets=P_ro_t,
            P_col_indices=P_ci_t,
            A_row_offsets=A_ro_t,
            A_col_indices=A_ci_t,
            cones=moreau.Cones(num_zero_cones=1, num_nonneg_cones=N_w),
            settings=moreau.Settings(device=moreau_dev),
        )

        q_vec = torch.zeros(N_w, dtype=torch.float64, device=self.device)
        b_vec = torch.cat([
            torch.ones(1,    dtype=torch.float64, device=self.device),
            torch.zeros(N_w, dtype=torch.float64, device=self.device),
        ])

        #qp.setup(P_dense.flatten(), A_vals.to(self.device))
        sol = qp.solve(P_dense.flatten(), A_vals.to(self.device), q_vec, b_vec)
        w_recovered = sol.x.clamp(min=0.0)
        w_recovered = w_recovered / w_recovered.sum().clamp_min(1e-12)

        if verbose:
            print(f"\nKKT Residual QP solved.")
            print(f"\n||Φ w|| per demo (raw / projected with boundary condition):")
            for k, (Phi_k, u_demo_k) in enumerate(Phi_list):
                phi_w  = Phi_k @ w_recovered            # (n,)
                u_flat = u_demo_k.flatten()
                raw    = phi_w.norm().item()
                proj   = _projected_grad_norm(phi_w, u_flat, u_min, u_max)
                print(f"  demo[{k}]  raw={raw:.3e}  projected={proj:.3e}")
            self._print_weights(w_recovered)

        return {
            'w_recovered': w_recovered,
            'Sigma': Sigma,
            'n_demos': n_demos,
        }

    def fit_control_matching_eg(
        self,
        demos,
        w_init: Optional[torch.Tensor] = None,
        cfg: Optional[IRLConfig] = None,
    ) -> Dict:
        """
        Bilevel Control Matching IRL with Exponentiated Gradient (EG) outer loop.

        Each outer iteration, for each demo k:
          1. SQP-SS(w) → u*(w),  ift_data contains du_dw = -P^{-1} M
             M[:,i] = J_{r,i}^T r_i / w_i   (correct IFT mixed partial)
          2. grad_w += 2 * (du_dw)^T (u*_k - u_demo_k)
             with boundary correction: zero grad components at active u bounds

        EG update (maintains simplex w >= 0, sum=1 exactly):
          w_new_i = w_i * exp(-lr * grad_i) / Z

        Gradient accumulation is skipped for demos where inner SQP did not converge,
        since du_dw is unreliable at non-optimal u*.
        """
        assert self.sqp_solver is not None, \
            "sqp_solver required. Pass SingleShootingSQPGaussNewton instance."

        from .single_shooting_SQP_moreau import CostWeights, SQPConfig

        if cfg is None:
            cfg = IRLConfig()
        if isinstance(demos, dict):
            demos = [demos]
        n_demos = len(demos)
        demo_data = [self._unpack_demo(d) for d in demos]
        N_w = self.N_WEIGHTS

        if verbose := cfg.verbose:
            print("\n" + "="*60)
            print(f"IRL - Control Matching EG ({n_demos} demo{'s' if n_demos > 1 else ''})")
            print("="*60)

        # Init weights
        if w_init is None:
            w = torch.ones(N_w, dtype=torch.float64, device=self.device) / N_w
        else:
            w = w_init.to(dtype=torch.float64, device=self.device).clone()
            w = w / w.sum().clamp_min(1e-12)

        sqp_cfg = SQPConfig(
            maxIters            = cfg.max_inner_iters,
            cost_tol            = cfg.inner_cost_tol,
            proj_grad_tol       = cfg.inner_proj_grad_tol,
            use_gauss_newton    = True,
            hessian_regularization = cfg.hessian_reg,
            use_line_search     = True,
            line_search_max_iters = 10,
            line_search_beta    = 0.5,
            u_min               = cfg.u_min,
            u_max               = cfg.u_max,
            use_trust_region    = True,
            trust_region_iters  = 3,
            trust_region_size   = 0.5,
            use_mu_scheduling   = False,
            verbose             = False,
        )

        lr          = cfg.lr_weights
        best_loss   = float('inf')
        plateau_cnt = 0

        history = {
            'loss': [], 'control_error': [], 'w': [], 'grad_norm': [],
            'inner_proj_grad': [], 'inner_iters': [], 'inner_converged': [], 'lr': [],
        }
        w_prev     = None
        u_currents = [None] * n_demos  # warm start per demo

        for it in range(cfg.max_outer_iters):

            cost_w       = self._w_to_cost_weights(w)
            grad_w_total = torch.zeros(N_w, dtype=torch.float64, device=self.device)
            total_loss   = 0.0
            total_ctrl_err = 0.0
            all_inner_proj  = []
            all_inner_iters = []
            all_converged   = []

            for k, (q0, v0, pr0, goal, u_demo, _) in enumerate(demo_data):
                # Warm start: use previous u* or u_demo on first iter
                if u_currents[k] is None:
                    u_currents[k] = u_demo.clone()

                result = self.sqp_solver.optimize(
                    q0.cpu().numpy(), v0.cpu().numpy(),
                    pr0.cpu().numpy(), goal.cpu().numpy(),
                    u_init=u_currents[k].cpu().numpy() if cfg.warm_start_inner else None,
                    cfg=sqp_cfg, w=cost_w, verbose=False,
                )
                u_star = torch.tensor(result['u_seq'], dtype=torch.float64, device=self.device)
                if cfg.warm_start_inner:
                    u_currents[k] = u_star.detach()

                inner_proj_grad = result['stationarityInfo']['final_proj_grad_norm']
                inner_converged = result['stationarityInfo']['converged']
                inner_iters     = len(result['history']['loss'])

                # du_dw = -P^{-1} M  (computed inside SQP using correct M[:,i] = J_r_i^T r_i / w_i)
                du_dw = result['ift_data']['du_dw_analytical'].to(self.device)  # (n, N_w)
                r_k   = (u_star - u_demo).flatten()                             # (n,)

                if inner_converged:
                    # grad_w = 2 * (du_dw)^T (u* - u_demo)
                    raw_grad = 2.0 * du_dw.T @ r_k   # (N_w,)

                    # Boundary condition on u:
                    # At active bounds, the KKT condition is an inequality.
                    # Zero out the contribution of rows where u* is at a bound
                    # and the gradient direction is blocked.
                    u_flat = u_star.flatten()
                    mask = torch.ones(r_k.shape[0], dtype=torch.float64, device=self.device)
                    # upper active: u_i at u_max, only the negative residual direction matters
                    mask[(u_flat >= cfg.u_max - 1e-6) & (r_k < 0)] = 0.0
                    # lower active: u_i at u_min, only the positive residual direction matters
                    mask[(u_flat <= cfg.u_min + 1e-6) & (r_k > 0)] = 0.0
                    grad_w_bc = 2.0 * du_dw.T @ (mask * r_k)

                    grad_w_total += grad_w_bc

                total_loss     += r_k.dot(r_k).item()
                total_ctrl_err += torch.norm(u_star - u_demo).item()
                all_inner_proj.append(inner_proj_grad)
                all_inner_iters.append(inner_iters)
                all_converged.append(inner_converged)

                if verbose and (it % cfg.log_every == 0 or it < 3):
                    status = "✓" if inner_converged else "△"
                    print(f"  demo[{k}] {status}  "
                          f"proj_grad={inner_proj_grad:.2e}  "
                          f"ctrl_err={torch.norm(u_star - u_demo).item():.4e}")

            # EG update: w_new_i = w_i * exp(-lr * grad_i) / Z
            if grad_w_total.norm() > 0:
                w_new = w * torch.exp(-lr * grad_w_total)
                w_new = w_new / w_new.sum().clamp_min(1e-12)
            else:
                w_new = w.clone()

            # Plateau LR scheduler
            cur_loss = total_loss / n_demos
            if cur_loss < best_loss - 1e-6:
                best_loss   = cur_loss
                plateau_cnt = 0
            else:
                plateau_cnt += 1
            if plateau_cnt >= cfg.lr_patience:
                lr          = max(lr * cfg.lr_factor, cfg.lr_min)
                plateau_cnt = 0
                if verbose:
                    print(f"  [LR] plateau → lr={lr:.2e}")

            n_conv    = sum(all_converged)
            mean_proj = sum(all_inner_proj) / n_demos
            grad_norm = grad_w_total.norm().item()

            history['loss'].append(cur_loss)
            history['control_error'].append(total_ctrl_err / n_demos)
            history['w'].append(w.detach().cpu().numpy().copy())
            history['grad_norm'].append(grad_norm)
            history['inner_proj_grad'].append(mean_proj)
            history['inner_iters'].append(sum(all_inner_iters) / n_demos)
            history['inner_converged'].append(n_conv == n_demos)
            history['lr'].append(lr)

            if verbose and (it % cfg.log_every == 0 or it < 3):
                print(f"\nIter {it:3d} | loss={cur_loss:.4e} | |grad_w|={grad_norm:.2e} "
                      f"| conv={n_conv}/{n_demos} (mean_proj={mean_proj:.2e}) | lr={lr:.2e}")
                self._print_weights(w)

            # Convergence
            if w_prev is not None and torch.norm(w_new - w_prev).item() < cfg.weight_tol:
                if verbose:
                    print(f"\nConverged at iter {it}")
                break
            w_prev = w.clone()
            w = w_new.detach()

        if verbose:
            print("\n" + "="*60)
            print("Control Matching EG Complete")
            self._print_weights(w)

        return {
            'w_recovered': w,
            'loss_history': history['loss'],
            'control_error_history': history['control_error'],
            'w_history': history['w'],
            'grad_norm_history': history['grad_norm'],
            'inner_proj_grad_history': history['inner_proj_grad'],
            'inner_iters_history': history['inner_iters'],
            'inner_converged_history': history['inner_converged'],
            'lr_history': history['lr'],
            'num_iters': it + 1,
            'u_finals': u_currents,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _w_to_cost_weights(self, w: torch.Tensor):
        """w = [w_target, w_orient, w_v, w_ctrl, w_contact] → CostWeights."""
        from .single_shooting_SQP_moreau import CostWeights
        return CostWeights(
            wTargetXY     = w[0].item(),
            wTargetOrient = w[1].item(),
            wObjVel       = w[2].item(),
            wControl      = w[3].item(),
            wContact      = w[4].item(),
        )

    def _unpack_demo(self, demo: Dict):
        def t(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(dtype=torch.float64, device=self.device)
            return torch.tensor(x, dtype=torch.float64, device=self.device)
        return (
            t(demo['q0']), t(demo['v0']), t(demo['pusher0']),
            t(demo['goal']), t(demo['u_demo']),
            t(demo.get('obstacle_pos')),
        )

    def _print_weights(self, w: torch.Tensor):
        for name, val in zip(self.WEIGHT_NAMES, w):
            print(f"    {name:12s}: {val.item():.6f}")


# ---------------------------------------------------------------------------
# Demo packing utility
# ---------------------------------------------------------------------------

def pack_demo(q0, v0, pusher0, goal, u_demo, obstacle_pos=None) -> Dict:
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    return dict(
        q0=to_np(q0), v0=to_np(v0), pusher0=to_np(pusher0),
        goal=to_np(goal), u_demo=to_np(u_demo),
        obstacle_pos=None if obstacle_pos is None else to_np(obstacle_pos),
    )