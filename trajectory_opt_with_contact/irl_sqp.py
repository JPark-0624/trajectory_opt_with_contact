"""
Inverse Reinforcement Learning for Contact-Implicit Trajectory Optimization.

Two-phase design:
  Phase 1 - Feature Matching    : validates outer loop structure (no inner loop grad needed)
  Phase 2 - Control Matching QP : SQP SS inner solver + IFT + moreau outer QP

Cost structure (matches dynamics.rollout):
  J(u; w) = w_target * ||q_T[:2] - goal[:2]||²
           + w_orient * wrap(q_T[2] - goal[2])²
           + w_v      * ||v_T||²
           + w_ctrl   * ||u||²

Weight vector w = [w_target, w_orient, w_v, w_ctrl]  (4-dim, no obstacle)
Parameterized as w = softplus(theta) / sum(softplus(theta))  (positive, normalized)
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import moreau
import moreau.torch as moreau_torch
from scipy import sparse

from .dynamics import rollout, IPMOptions


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class IRLConfig:
    # Outer loop
    max_outer_iters: int = 50
    lr_weights: float = 0.05
    weight_tol: float = 1e-4       # convergence: ||w_new - w_old||

    # Inner loop (used only in control matching)
    max_inner_iters: int = 30      # SQP SS iterations per outer step
    warm_start_inner: bool = True

    # IFT (control matching only)
    hessian_reg: float = 1e-4
    cg_max_iters: int = 50

    # Logging
    verbose: bool = True
    log_every: int = 5


# ---------------------------------------------------------------------------
# IRLSolver
# ---------------------------------------------------------------------------

class IRLSolver:
    """
    IRL solver for pusher-slider system.

    Supports two modes:
      - feature_matching : outer loop only, validates optimization structure
      - control_matching : bilevel, SQP SS inner loop + IFT gradient
    """

    WEIGHT_NAMES = ['w_target', 'w_orient', 'w_v', 'w_ctrl']
    N_WEIGHTS = 4

    def __init__(
        self,
        # Physical parameters
        mass: float,
        side_length: float,
        mu: float,
        horizon: int,
        dt: float,
        device: str = 'cpu',
        dynamics_solver: str = 'IP',
        ipm_opts: Optional[IPMOptions] = None,
        # Inner SQP solver (required for control_matching)
        sqp_solver=None,
    ):
        self.mass = mass
        self.half = side_length / 2
        self.mu = mu
        self.horizon = horizon
        self.dt = dt
        self.device = torch.device(device)
        self.dynamics_solver = dynamics_solver
        self.ipm_opts = ipm_opts if ipm_opts is not None else IPMOptions()
        self.sqp_solver = sqp_solver  # SingleShootingSQPGaussNewton instance

        self.Izz = (1.0 / 6.0) * mass * (side_length ** 2 + side_length ** 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_feature_matching(
        self,
        demo: Dict,
        w_init: Optional[torch.Tensor] = None,
        cfg: Optional[IRLConfig] = None,
    ) -> Dict:
        """
        Phase 1: Feature matching IRL.

        Minimizes: ||phi(u_demo; w) - phi_demo||^2
        where phi(u; w) = [w_target*goal_term, w_orient*orient_term, w_v*v_term, w_ctrl*ctrl_term]

        u_demo is fixed — no inner loop needed.
        Only validates that outer weight optimization converges.
        """
        if cfg is None:
            cfg = IRLConfig()

        q0, v0, pr0, goal, u_demo, _ = self._unpack_demo(demo)

        # Demo features (unweighted, computed once)
        phi_demo = self._compute_features(u_demo, q0, v0, pr0, goal)

        if cfg.verbose:
            print("\n" + "="*60)
            print("IRL - Feature Matching")
            print("="*60)
            print("Demo features (unweighted):")
            for name, val in zip(self.WEIGHT_NAMES, phi_demo):
                print(f"  {name:12s}: {val.item():.6e}")

        # Init theta
        theta = self._init_theta(w_init)
        opt = torch.optim.Adam([theta], lr=cfg.lr_weights)

        history = {'loss': [], 'w': [], 'grad_norm': []}
        w_prev = None

        for it in range(cfg.max_outer_iters):
            opt.zero_grad()

            w = self._weights_from_theta(theta)

            # Weighted feature matching loss: ||w * phi_demo - phi_demo||^2
            # = ||(w - 1) * phi_demo||^2  ... but this is trivially minimized at w=1
            # Correct formulation: match feature expectations
            # phi_learned(w) = w * phi_demo  (since u is fixed to u_demo)
            # target: phi_demo (i.e. want w * phi_demo = phi_demo → w = 1 uniformly)
            # Better: use the IRL loss from Abbeel & Ng —
            # maximize w^T (phi_demo - phi_pi(w))
            # Here phi_pi(w) = phi_demo (same u), so this degenerates.
            #
            # Practical test formulation: perturb u with inner Adam steps, then match.
            # For pure outer-loop test, we optimize:
            #   loss = ||w - w_uniform||^2  is trivial.
            #
            # Instead: run K Adam steps on u with current w, compare features.
            u_opt = self._solve_inner_adam(
                w.detach(), q0, v0, pr0, goal,
                u_init=u_demo.clone(),
                max_iters=cfg.max_inner_iters,
            )

            phi_opt = self._compute_features_differentiable(u_opt.detach(), q0, v0, pr0, goal, w)
            loss = torch.sum((phi_opt - phi_demo) ** 2)

            loss.backward()
            grad_norm = theta.grad.norm().item() if theta.grad is not None else 0.0
            opt.step()

            history['loss'].append(loss.item())
            history['w'].append(w.detach().cpu().numpy().copy())
            history['grad_norm'].append(grad_norm)

            if cfg.verbose and (it % cfg.log_every == 0 or it < 3):
                print(f"\nIter {it:3d} | loss={loss.item():.4e} | grad={grad_norm:.2e}")
                self._print_weights(w)

            # Convergence
            if w_prev is not None:
                if torch.norm(w - w_prev).item() < cfg.weight_tol:
                    if cfg.verbose:
                        print(f"\nConverged at iter {it} (weight change < {cfg.weight_tol})")
                    break
            w_prev = w.detach().clone()

        w_final = self._weights_from_theta(theta).detach()
        if cfg.verbose:
            print("\n" + "="*60)
            print("Feature Matching Complete")
            self._print_weights(w_final)

        return {
            'w_recovered': w_final,
            'loss_history': history['loss'],
            'w_history': history['w'],
            'grad_norm_history': history['grad_norm'],
            'num_iters': it + 1,
        }

    def fit_control_matching(
        self,
        demo: Dict,
        w_init: Optional[torch.Tensor] = None,
        cfg: Optional[IRLConfig] = None,
    ) -> Dict:
        """
        Phase 2: Control matching IRL with IFT.

        Minimizes: ||u*(w) - u_demo||^2
        where u*(w) = SQP_SS(w)  (inner solver)
        Gradient: d_loss/dw via implicit function theorem.
        """
        assert self.sqp_solver is not None, \
            "sqp_solver required for control_matching. Pass SingleShootingSQPGaussNewton instance."

        if cfg is None:
            cfg = IRLConfig()

        q0, v0, pr0, goal, u_demo, _ = self._unpack_demo(demo)

        if cfg.verbose:
            print("\n" + "="*60)
            print("IRL - Control Matching (IFT)")
            print("="*60)

        theta = self._init_theta(w_init)
        opt = torch.optim.Adam([theta], lr=cfg.lr_weights)

        history = {'loss': [], 'control_error': [], 'w': [], 'grad_norm': []}
        w_prev = None
        u_current = None  # warm start

        for it in range(cfg.max_outer_iters):
            w = self._weights_from_theta(theta)

            # --- Inner solve (no gradient through this) ---
            from .single_shooting_SQP_moreau import CostWeights, SQPConfig
            cost_w = self._w_to_cost_weights(w.detach())
            sqp_cfg = SQPConfig(maxIters=cfg.max_inner_iters, verbose=False)

            result = self.sqp_solver.optimize(
                q0.cpu().numpy(), v0.cpu().numpy(),
                pr0.cpu().numpy(), goal.cpu().numpy(),
                u_init=u_current.cpu().numpy() if (cfg.warm_start_inner and u_current is not None) else None,
                cfg=sqp_cfg,
                w=cost_w,
                verbose=False,
            )
            u_star = torch.tensor(result['u_seq'], dtype=torch.float64, device=self.device)

            if cfg.warm_start_inner:
                u_current = u_star.detach()

            # --- IFT gradient: d||u*-u_demo||^2 / dw ---
            grad_w = self._ift_gradient(u_star, u_demo, w, q0, v0, pr0, goal, cfg)

            control_error = torch.norm(u_star - u_demo).item()
            loss_val = torch.sum((u_star - u_demo) ** 2).item()

            history['loss'].append(loss_val)
            history['control_error'].append(control_error)
            history['w'].append(w.detach().cpu().numpy().copy())
            history['grad_norm'].append(grad_w.norm().item())

            if cfg.verbose and (it % cfg.log_every == 0 or it < 3):
                print(f"\nIter {it:3d} | ctrl_err={control_error:.4e} | grad={grad_w.norm().item():.2e}")
                self._print_weights(w)

            # Convergence
            if w_prev is not None:
                if torch.norm(w - w_prev).item() < cfg.weight_tol:
                    if cfg.verbose:
                        print(f"\nConverged at iter {it}")
                    break
            w_prev = w.detach().clone()

            # Manual gradient step via theta
            opt.zero_grad()
            theta.grad = self._grad_w_to_theta(grad_w, theta)
            opt.step()

        w_final = self._weights_from_theta(theta).detach()
        if cfg.verbose:
            print("\n" + "="*60)
            print("Control Matching Complete")
            self._print_weights(w_final)

        return {
            'w_recovered': w_final,
            'loss_history': history['loss'],
            'control_error_history': history['control_error'],
            'w_history': history['w'],
            'grad_norm_history': history['grad_norm'],
            'num_iters': it + 1,
            'u_final': u_current,
        }

    def fit_control_matching_qp(
        self,
        demo: Dict,
        w_init: Optional[torch.Tensor] = None,
        cfg: Optional[IRLConfig] = None,
    ) -> Dict:
        """
        Control matching IRL: IFT gradient + moreau outer Gauss-Newton QP.

        Each outer iteration:
          1. SQP SS(w_k)  →  u*(w_k), P, J_r, r       [inner solve, ~4s]
          2. dr_dw  analytical  (∂r_i/∂w_i = r_i / 2w_i)
          3. M = J_r^T @ dr_dw                         [free]
          4. J_outer = -solve(P + reg*I, M)             [linalg.solve, 120×4]
          5. r_outer = u*(w_k) - u_demo
          6. outer QP via moreau → δw                  [~0.01s]
          7. w_{k+1} = clamp_positive(w_k + δw), renormalize

        Outer QP (Gauss-Newton linearization):
          minimize_δw  (1/2)||J_outer δw + r_outer||²
          s.t.  sum(δw) = 0          (simplex, zero cone)
                w_k + δw >= 0        (positivity, nonneg cone)
        """
        assert self.sqp_solver is not None, \
            "sqp_solver required. Pass SingleShootingSQPGaussNewton instance."

        if cfg is None:
            cfg = IRLConfig()

        q0, v0, pr0, goal, u_demo, _ = self._unpack_demo(demo)
        n = self.horizon * 2
        N_w = self.N_WEIGHTS

        if cfg.verbose:
            print("\n" + "="*60)
            print("IRL - Control Matching QP (IFT + moreau outer)")
            print("="*60)

        # Init weights (uniform)
        if w_init is None:
            w = torch.ones(N_w, dtype=torch.float64, device=self.device) / N_w
        else:
            w = w_init.to(dtype=torch.float64, device=self.device).clone()
            w = w / w.sum().clamp_min(1e-12)

        # Pre-build outer moreau QP (structure fixed: N_w vars, 1+N_w constraints)
        # Row 0 (zero cone):    [1, 1, 1, 1] δw = 0   (sum preserved)
        # Rows 1..N_w (nonneg): [-I] δw + s = w_k      (positivity)
        A_dense = torch.zeros(1 + N_w, N_w, dtype=torch.float64)
        A_dense[0, :] = 1.0
        for i in range(N_w):
            A_dense[1 + i, i] = -1.0
        A_sp = sparse.csr_matrix(A_dense.numpy())
        A_ro_t = torch.tensor(A_sp.indptr,  dtype=torch.int32)
        A_ci_t = torch.tensor(A_sp.indices, dtype=torch.int32)
        A_vals_fixed = torch.tensor(A_sp.data, dtype=torch.float64)  # fixed every iter

        P_ro_t = torch.arange(0, (N_w + 1) * N_w, N_w, dtype=torch.int32)
        P_ci_t = torch.arange(N_w, dtype=torch.int32).repeat(N_w)

        cones_outer = moreau.Cones(num_zero_cones=1, num_nonneg_cones=N_w)
        moreau_dev = 'cuda' if self.device.type == 'cuda' and moreau.device_available('cuda') else 'cpu'
        outer_qp = moreau_torch.Solver(
            n=N_w, m=1 + N_w,
            P_row_offsets=P_ro_t,
            P_col_indices=P_ci_t,
            A_row_offsets=A_ro_t,
            A_col_indices=A_ci_t,
            cones=cones_outer,
            settings=moreau.Settings(device=moreau_dev),
        )

        history = {'loss': [], 'control_error': [], 'w': [], 'grad_norm': []}
        w_prev = None
        u_current = None

        from .single_shooting_SQP_moreau import CostWeights, SQPConfig

        for it in range(cfg.max_outer_iters):

            # Step 1: inner SQP SS
            cost_w = self._w_to_cost_weights(w)
            sqp_cfg = SQPConfig(
                maxIters=cfg.max_inner_iters,
                cost_tol=1e-6, tol=1e-4,
                verbose=False,
            )
            result = self.sqp_solver.optimize(
                q0.cpu().numpy(), v0.cpu().numpy(),
                pr0.cpu().numpy(), goal.cpu().numpy(),
                u_init=u_current.cpu().numpy() if (cfg.warm_start_inner and u_current is not None) else None,
                cfg=sqp_cfg, w=cost_w, verbose=False,
            )
            u_star = torch.tensor(result['u_seq'], dtype=torch.float64, device=self.device)
            if cfg.warm_start_inner:
                u_current = u_star.detach()

            P_inn   = result['ift_data']['P'].to(self.device)    # (n, n)
            J_r     = result['ift_data']['J_r'].to(self.device)  # (n_res, n)
            r_res   = result['ift_data']['r'].to(self.device)    # (n_res,)

            # Step 2: dr_dw analytical
            # Residual blocks: [ctrl: T*2, target_xy: 2, orient: 1, vel: 3]
            # r_block_i = sqrt(w_i) * f_i(u)  →  dr/dw_i = r_block_i / (2*w_i)
            n_res = r_res.shape[0]
            dr_dw = torch.zeros(n_res, N_w, dtype=torch.float64, device=self.device)
            block_sizes = [n, 2, 1, 3]  # ctrl, target_xy, orient, vel
            idx = 0
            for i, bs in enumerate(block_sizes):
                wi = w[i].clamp(min=1e-8)
                dr_dw[idx:idx+bs, i] = r_res[idx:idx+bs] / (2.0 * wi)
                idx += bs

            # Step 3: M = J_r^T @ dr_dw  (n × N_w)
            M = J_r.T @ dr_dw

            # Step 4: J_outer = -P^{-1} M
            reg = cfg.hessian_reg * torch.eye(n, dtype=torch.float64, device=self.device)
            J_outer = -torch.linalg.solve(P_inn + reg, M)  # (n, N_w)

            # Step 5: r_outer
            r_outer = (u_star - u_demo).flatten()  # (n,)

            # Step 6: outer QP
            P_out  = J_outer.T @ J_outer          # (N_w, N_w)
            q_out  = J_outer.T @ r_outer           # (N_w,)
            b_out  = torch.cat([
                torch.zeros(1, dtype=torch.float64, device=self.device),
                w.detach(),
            ])
            outer_qp.setup(P_out.flatten(), A_vals_fixed.to(self.device))
            sol = outer_qp.solve(q_out, b_out)
            delta_w = sol.x  # (N_w,)

            # Step 7: update w
            w_new = (w + delta_w).clamp(min=0.0)
            w_new = w_new / w_new.sum().clamp_min(1e-12)

            # Logging
            ctrl_err  = torch.norm(u_star - u_demo).item()
            loss_val  = r_outer.dot(r_outer).item()
            grad_norm = torch.norm(q_out).item()

            history['loss'].append(loss_val)
            history['control_error'].append(ctrl_err)
            history['w'].append(w.detach().cpu().numpy().copy())
            history['grad_norm'].append(grad_norm)

            if cfg.verbose and (it % cfg.log_every == 0 or it < 3):
                print(f"\nIter {it:3d} | ctrl_err={ctrl_err:.4e} | |q|={grad_norm:.2e} | δw_norm={torch.norm(delta_w).item():.2e}")
                self._print_weights(w)

            # Convergence
            if w_prev is not None:
                if torch.norm(w_new - w_prev).item() < cfg.weight_tol:
                    if cfg.verbose:
                        print(f"\nConverged at iter {it}")
                    break
            w_prev = w.clone()
            w = w_new.detach()

        if cfg.verbose:
            print("\n" + "="*60)
            print("Control Matching QP Complete")
            self._print_weights(w)

        return {
            'w_recovered': w,
            'loss_history': history['loss'],
            'control_error_history': history['control_error'],
            'w_history': history['w'],
            'grad_norm_history': history['grad_norm'],
            'num_iters': it + 1,
            'u_final': u_current,
        }

    # ------------------------------------------------------------------
    # Weight parameterization
    # ------------------------------------------------------------------

    def _init_theta(self, w_init: Optional[torch.Tensor]) -> torch.Tensor:
        """theta: unconstrained parameter. w = softplus(theta) / sum(softplus(theta))"""
        if w_init is None:
            theta = torch.zeros(self.N_WEIGHTS, dtype=torch.float64, device=self.device)
        else:
            # Inverse softplus: theta = log(exp(w) - 1)
            w = w_init.to(dtype=torch.float64, device=self.device).clamp(min=1e-6)
            theta = torch.log(torch.expm1(w))
        return theta.requires_grad_(True)

    def _weights_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """theta → normalized positive weights via softplus + L1 normalize."""
        raw = F.softplus(theta)
        return raw / raw.sum().clamp_min(1e-12)

    def _grad_w_to_theta(self, grad_w: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Chain rule: d_loss/d_theta = d_loss/dw * dw/d_theta
        dw/d_theta = d/d_theta [ softplus(theta) / sum(softplus(theta)) ]
                   = Jacobian of normalization * diag(sigmoid(theta))
        """
        raw = F.softplus(theta.detach())
        s = raw.sum()
        sig = torch.sigmoid(theta.detach())  # d softplus / d theta

        # Jacobian of normalize(raw) w.r.t. raw: (I*s - raw*1^T) / s^2
        # Then chain with sig: J_theta[i,j] = sig[j] * (delta_ij * s - raw[i]) / s^2
        J = (torch.diag(s.expand(self.N_WEIGHTS)) - raw.unsqueeze(1)) / (s ** 2)
        J = J * sig.unsqueeze(0)  # (N, N)

        return (J @ grad_w.unsqueeze(1)).squeeze(1)

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    def _rollout(self, u_seq, q0, v0, pr0, goal, w_target=1., w_orient=1., w_v=1., w_ctrl=1.):
        """Thin wrapper around dynamics.rollout."""
        return rollout(
            u_seq, q0, v0, pr0,
            self.horizon, self.dt,
            self.mass, self.Izz, self.half, self.mu, goal,
            w_target=w_target, w_orient=w_orient, w_v=w_v, w_ctrl=w_ctrl,
            w_obs=0.0,
            dynamics_solver=self.dynamics_solver,
            ipm_opts=self.ipm_opts,
            device=self.device,
        )

    def _compute_features(self, u_seq, q0, v0, pr0, goal) -> torch.Tensor:
        """Compute unweighted feature vector phi(u) = [goal, orient, vel, ctrl]."""
        with torch.no_grad():
            _, _, _, _, _, _, goal_term, orient_term, ctrl_term, v_term, _, _ = \
                self._rollout(u_seq, q0, v0, pr0, goal)
        return torch.stack([goal_term, orient_term, v_term, ctrl_term]).detach()

    def _compute_features_differentiable(self, u_seq, q0, v0, pr0, goal, w) -> torch.Tensor:
        """
        Weighted features differentiable w.r.t. w.
        u_seq should be detached — gradient flows only through w.
        """
        _, _, _, _, _, _, goal_term, orient_term, ctrl_term, v_term, _, _ = \
            self._rollout(u_seq, q0, v0, pr0, goal)
        phi = torch.stack([goal_term, orient_term, v_term, ctrl_term])
        return w * phi  # element-wise: gradient w.r.t. w is just phi

    # ------------------------------------------------------------------
    # Inner solvers
    # ------------------------------------------------------------------

    def _solve_inner_adam(self, w, q0, v0, pr0, goal, u_init=None, max_iters=50) -> torch.Tensor:
        """Simple Adam inner solver (for feature matching test)."""
        if u_init is None:
            u_seq = torch.zeros(self.horizon, 2, dtype=torch.float64, device=self.device)
        else:
            u_seq = u_init.detach().clone()
        u_seq = u_seq.requires_grad_(True)
        opt = torch.optim.Adam([u_seq], lr=0.01)

        for _ in range(max_iters):
            opt.zero_grad()
            loss, *_ = self._rollout(u_seq, q0, v0, pr0, goal,
                                     w_target=w[0].item(), w_orient=w[1].item(),
                                     w_v=w[2].item(), w_ctrl=w[3].item())
            loss.backward()
            opt.step()

        return u_seq.detach()

    # ------------------------------------------------------------------
    # IFT gradient (for control matching)
    # ------------------------------------------------------------------

    def _ift_gradient(self, u_star, u_demo, w, q0, v0, pr0, goal, cfg) -> torch.Tensor:
        """
        Compute d||u*-u_demo||^2 / dw via IFT.

        d_loss/dw = 2(u*-u_demo)^T * du*/dw
        du*/dw = -H^{-1} * (d^2 J / du dw)   [IFT]
        H = d^2 J / du^2  at u*
        """
        u_req = u_star.detach().requires_grad_(True)
        w_req = w.detach().requires_grad_(True)

        # Inner loss at u* with current w
        L, *_ = self._rollout(u_req, q0, v0, pr0, goal,
                               w_target=w_req[0], w_orient=w_req[1],
                               w_v=w_req[2], w_ctrl=w_req[3])

        grad_w = torch.zeros(self.N_WEIGHTS, dtype=torch.float64, device=self.device)
        dloss_du = 2.0 * (u_star - u_demo)  # (T, 2)

        for i in range(self.N_WEIGHTS):
            # Mixed derivative: d^2 J / (du * dw_i)
            dL_dwi = torch.autograd.grad(L, w_req, create_graph=True, retain_graph=True)[0][i]
            mixed = torch.autograd.grad(dL_dwi, u_req, retain_graph=True)[0]  # (T, 2)

            # Solve H * x = -mixed  via CG (H implicit via hvp)
            def hvp(v):
                g = torch.autograd.grad(L, u_req, create_graph=True, retain_graph=True)[0]
                Hv = torch.autograd.grad(g, u_req, grad_outputs=v.reshape_as(g),
                                         retain_graph=True)[0]
                return Hv.reshape(-1) + cfg.hessian_reg * v

            x = self._cg(hvp, -mixed.reshape(-1), cfg.cg_max_iters)  # (T*2,)
            du_dwi = x.reshape_as(u_star)

            grad_w[i] = (dloss_du * du_dwi).sum()

        return grad_w

    def _cg(self, A_fn, b: torch.Tensor, max_iters: int, tol: float = 1e-8) -> torch.Tensor:
        """Conjugate gradient: solve A x = b."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs = torch.dot(r, r)

        for _ in range(max_iters):
            Ap = A_fn(p)
            alpha = rs / (torch.dot(p, Ap) + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if rs_new.sqrt() < tol:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new

        return x

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _w_to_cost_weights(self, w: torch.Tensor):
        """Convert normalized weight tensor to CostWeights dataclass."""
        from .single_shooting_SQP_moreau import CostWeights
        return CostWeights(
            wTargetXY=w[0].item(),
            wTargetOrient=w[1].item(),
            wObjVel=w[2].item(),
            wControl=w[3].item(),
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