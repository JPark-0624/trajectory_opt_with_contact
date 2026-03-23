"""
Single Shooting SQP with Explicit Gauss-Newton Hessian

Implementation using residual Jacobian for P = J^T J approximation.
Matches single_shooting_SQP_moreau.py structure exactly.
"""

import torch
import numpy as np
from scipy import sparse
import moreau
import moreau.torch as moreau_torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from trajectory_opt_with_contact.dynamics import IPMOptions
import time
import math

def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to (-pi, pi]. Works elementwise."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class SQPConfig:
    """SQP optimizer configuration"""
    maxIters: int = 20
    tol: float = 1e-4        # gradient norm convergence
    cost_tol: float = 1e-6   # cost change convergence
    
    # Line search
    use_line_search: bool = True
    line_search_max_iters: int = 10
    line_search_beta: float = 0.5
    line_search_c1: float = 1e-4
    line_search_ftol: float = 1e-8  # accept if cost_trial <= cost_curr + ftol
    
    # Control bounds
    u_min: float = -0.3
    u_max: float = 0.3
    
    # Trust region (early iterations)
    use_trust_region: bool = True
    trust_region_iters: int = 3
    trust_region_size: float = 0.5
    
    # Gauss-Newton Hessian
    use_gauss_newton: bool = True
    hessian_regularization: float = 1e-6

    # Initial guess ('geometric' or 'zero')
    u_init_mode: str = 'geometric'
    
    # IPM target_mu scheduling
    use_mu_scheduling: bool = False  # Enable scheduling
    mu_schedule_type: str = 'exponential'  # 'linear', 'exponential', 'stepwise', 'adaptive'
    mu_start: float = 1e-2  # Initial (loose)
    mu_end: float = 1e-5    # Final (tight)


@dataclass
class CostWeights:
    """Cost function weights.
    Fields accept float or torch.Tensor (scalar).
    Pass torch.Tensor with requires_grad=True to enable gradient flow w -> u*(w).
    """
    wControl: object = 0.1
    wControlSmooth: object = 0.0
    wObjVel: object = 1.0  # Terminal velocity
    wTargetXY: object = 20.0
    wTargetOrient: object = 0.5


class SingleShootingSQPGaussNewton:
    """
    Single Shooting SQP with explicit Gauss-Newton Hessian.
    
    Uses residual formulation:
        f(u) = (1/2) ||r(u)||²
    
    Hessian approximation:
        P = J^T J  where J = ∂r/∂u
    
    Matches SingleShootingSQP API exactly.
    """
    
    def __init__(
        self,
        mass: float,
        side_length: float,
        mu: float,
        horizon: int,
        dt: float,
        device: str = "cpu",
        dynamics_module = None,
        ipmOpts: Optional[IPMOptions] = None,
    ):
        self.mass = mass
        self.side = side_length
        self.half = side_length / 2
        self.mu = mu
        self.horizon = horizon
        self.dt = dt
        self.device = torch.device(device)

        self.ipmOpts = ipmOpts if ipmOpts is not None else IPMOptions()
        self.dynamics = dynamics_module
        self.Izz = (1.0/6.0) * mass * (side_length**2 + side_length**2)

        # Pre-build moreau.torch.Solver with fixed sparsity structure.
        # A (box constraints) structure is fixed for all iterations.
        # P (Gauss-Newton Hessian) is dense n x n — structure also fixed, only values change.
        self._init_moreau_torch_solver(device)
    
    def _init_moreau_torch_solver(self, device: str):
        """
        Pre-build moreau.torch.Solver with fixed sparsity structure.

        P: dense n x n (full symmetric) — structure fixed, values updated each iter.
        A: box constraints — structure fixed, values updated each iter (u_curr changes).

        moreau.torch.Solver requires CSR sparsity indices at construction time,
        so we register them once here.
        """
        T = self.horizon
        n = T * 2        # primal vars
        m = T * 2 * 2    # constraints (lower + upper per control dim)

        # --- P sparsity: dense n x n, full symmetric (both triangles required) ---
        # Row i has entries at columns 0..n-1
        P_row_offsets = torch.arange(0, (n + 1) * n, n, dtype=torch.int32)  # [0, n, 2n, ...]
        P_col_indices = torch.arange(n, dtype=torch.int32).repeat(n)         # [0,1,...,n-1, 0,1,...] x n

        # --- A sparsity: box constraints, one nonzero per row ---
        # Lower bound row t*4+d*2+0: -delta_u[t,d]  → col = t*2+d, val = -1
        # Upper bound row t*4+d*2+1: +delta_u[t,d]  → col = t*2+d, val = +1
        A_row_offsets = torch.arange(0, m + 1, dtype=torch.int32)  # one nnz per row
        A_col_indices = torch.zeros(m, dtype=torch.int32)
        for t in range(T):
            for d in range(2):
                idx = t * 2 + d
                A_col_indices[idx * 2 + 0] = idx  # lower bound row
                A_col_indices[idx * 2 + 1] = idx  # upper bound row

        cones = moreau.Cones(num_zero_cones=0, num_nonneg_cones=m)

        moreau_device = 'cuda' if torch.device(device).type == 'cuda' and moreau.device_available('cuda') else 'cpu'
        settings = moreau.Settings(device=moreau_device)

        self._moreau_torch_solver = moreau_torch.Solver(
            n=n, m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
            settings=settings,
        )
        self._moreau_n = n
        self._moreau_m = m

    def forward_simulate(
        self,
        q0: torch.Tensor,
        v0: torch.Tensor,
        pusher0: torch.Tensor,
        u: torch.Tensor,
        store_contact_data: bool = False,
    ) -> Tuple:
        """Forward simulation from initial state with controls u."""
        T = self.horizon
        
        qs = [q0]
        vs = [v0]
        prs = [pusher0]
        
        if store_contact_data:
            contact_forces = []
            signed_distances = []
        
        q, v, pr = q0, v0, pusher0
        z_prev = None
        for t in range(T):
            q_next, v_next, pr_next, lam, phi, z_prev = self.dynamics(
                qk=q, vk=v, 
                pusher_pos=pr, 
                u_push=u[t],
                h=self.dt,
                m=self.mass,
                Izz=self.Izz,
                half=self.half,
                mu=self.mu,
                ipm_opts=self.ipmOpts,
                z_prev=z_prev,
                skip_solving_threshold=100.0
            )
            
            qs.append(q_next)
            vs.append(v_next)
            prs.append(pr_next)
            
            if store_contact_data:
                contact_forces.append(lam)
                signed_distances.append(phi)
            
            q, v, pr = q_next, v_next, pr_next
        
        terminalVelEnergy = (v ** 2).sum()
        
        qs_stacked = torch.stack(qs)
        vs_stacked = torch.stack(vs)
        prs_stacked = torch.stack(prs)
        
        if store_contact_data:
            contact_forces_stacked = torch.stack(contact_forces)
            signed_distances_stacked = torch.stack(signed_distances)
            return qs_stacked, vs_stacked, prs_stacked, terminalVelEnergy, \
                   contact_forces_stacked, signed_distances_stacked
        else:
            return qs_stacked, vs_stacked, prs_stacked, terminalVelEnergy, None, None
    
    def compute_residuals(
        self,
        u: torch.Tensor,
        q0: torch.Tensor,
        v0: torch.Tensor,
        pusher0: torch.Tensor,
        goal: torch.Tensor,
        w: CostWeights,
    ) -> torch.Tensor:
        """
        Compute residual vector for Gauss-Newton.
        
        Residual structure:
            r = [r_control, r_target_xy, r_orient, r_vel]
        
        where each component is scaled by sqrt(weight) so that:
            f(u) = (1/2) ||r(u)||² = sum of weighted squared terms
        """
        # Forward simulate
        qs, vs, _, terminalVelEnergy, _, _ = self.forward_simulate(
            q0, v0, pusher0, u, store_contact_data=False
        )
        
        qFinal = qs[-1]
        vFinal = vs[-1]
        
        def _as_tensor(w_val):
            """Convert weight (float or Tensor) to scalar tensor, preserving grad."""
            if isinstance(w_val, torch.Tensor):
                return w_val.to(dtype=u.dtype, device=self.device)
            return torch.tensor(w_val, dtype=u.dtype, device=self.device)

        # --- Control residuals ---
        # r_control = sqrt(wControl) * u
        r_control = torch.sqrt(_as_tensor(w.wControl)) * u.flatten()
        
        # --- Target XY residuals ---
        # r_target = sqrt(wTargetXY) * (q_final[:2] - goal[:2])
        rXY = qFinal[:2] - goal[:2]
        r_target = torch.sqrt(_as_tensor(w.wTargetXY)) * rXY
        
        # --- Orientation residual ---
        # r_orient = sqrt(wTargetOrient) * wrap(q_final[2] - goal[2])
        rTheta = wrap_to_pi(qFinal[2] - goal[2])
        r_orient = torch.sqrt(_as_tensor(w.wTargetOrient)) * rTheta.unsqueeze(0)
        
        # --- Terminal velocity residuals ---
        # r_vel = sqrt(wObjVel) * v_final
        r_vel = torch.sqrt(_as_tensor(w.wObjVel)) * vFinal
        
        # Concatenate all residuals
        r = torch.cat([r_control, r_target, r_orient, r_vel])
        
        return r
    
    def compute_gauss_newton_hessian(
        self,
        u: torch.Tensor,
        q0: torch.Tensor,
        v0: torch.Tensor,
        pusher0: torch.Tensor,
        goal: torch.Tensor,
        w: CostWeights,
        regularization: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute Gauss-Newton Hessian P = J^T J and gradient g.
        
        Returns:
            P: Hessian matrix (n, n) where n = 2T
            g: Gradient vector (n,)
            timing: Dict with timing breakdown
        """
        timing = {}
        
        # Require gradient for Jacobian computation
        u_leaf = u.detach().requires_grad_(True)
        
        with torch.enable_grad():
            # Compute residuals
            t0 = time.time()
            r = self.compute_residuals(u_leaf, q0, v0, pusher0, goal, w)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()  # ⭐ Wait for GPU completion
            timing['residual_eval'] = time.time() - t0
            
            # Jacobian: J = ∂r/∂u (VECTORIZED for speed!)
            t0 = time.time()
            J = torch.autograd.functional.jacobian(
                lambda u_: self.compute_residuals(
                    u_.reshape(self.horizon, 2), q0, v0, pusher0, goal, w
                ),
                u_leaf.flatten(),
                create_graph=False,
                vectorize=True,  # ⭐ CRITICAL: 20× speedup!
                strategy='reverse-mode',  # Leverages IFT backward
            )
            if self.device.type == 'cuda':
                torch.cuda.synchronize()  # ⭐ Wait for GPU completion
            timing['jacobian_computation'] = time.time() - t0
        
        # Gradient: g = J^T r
        t0 = time.time()
        g = J.T @ r
        if self.device.type == 'cuda':
            torch.cuda.synchronize()  # ⭐ Wait for GPU completion
        timing['gradient_assembly'] = time.time() - t0
        
        # Gauss-Newton Hessian: P = J^T J
        t0 = time.time()
        P = J.T @ J
        
        # Add regularization for numerical stability
        if regularization > 0:
            P = P + regularization * torch.eye(P.shape[0], device=self.device, dtype=P.dtype)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()  # ⭐ Wait for GPU completion
        timing['hessian_assembly'] = time.time() - t0
        
        return P, g, timing
    
    def build_qp_matrices(
        self,
        u_curr: torch.Tensor,
        q0: torch.Tensor,
        v0: torch.Tensor,
        pusher0: torch.Tensor,
        goal: torch.Tensor,
        w: CostWeights,
        cfg: SQPConfig,
    ) -> Tuple:
        """
        Build QP matrices for DELTA formulation with Gauss-Newton Hessian.
        
        QP solves:
            minimize  (1/2) δu' P δu + g' δu
            s.t.      u_min ≤ u_curr + δu ≤ u_max
        
        Returns: P, q, A, b, cones, timing (matching Moreau solver interface)
        """
        T = self.horizon
        n = T * 2
        
        timing = {}
        
        # --- COMPUTE HESSIAN AND GRADIENT ---
        if cfg.use_gauss_newton:
            # Gauss-Newton: P = J^T J
            P_torch, g_torch, hess_timing = self.compute_gauss_newton_hessian(
                u_curr, q0, v0, pusher0, goal, w,
                regularization=cfg.hessian_regularization
            )
            timing.update(hess_timing)
            
            t0 = time.time()
            P_np = P_torch.detach().cpu().numpy()
            g_np = g_torch.detach().cpu().numpy()
            timing['hessian_to_numpy'] = time.time() - t0
        else:
            # Fallback: Identity Hessian (steepest descent)
            t0 = time.time()
            # Compute gradient via autograd
            u_ad = u_curr.clone().detach().requires_grad_(True)
            qs, vs, prs, termVel, _, _ = self.forward_simulate(q0, v0, pusher0, u_ad, False)
            
            # Compute loss
            controlEnergy = (u_ad ** 2).sum()
            rXY = qs[-1][:2] - goal[:2]
            targetCost = (rXY ** 2).sum()
            rTh = wrap_to_pi(qs[-1][2] - goal[2])
            orientCost = rTh ** 2
            
            loss = (
                w.wControl * controlEnergy +
                w.wObjVel * termVel +
                w.wTargetXY * targetCost +
                w.wTargetOrient * orientCost
            )
            loss.backward()
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            g_np = u_ad.grad.cpu().numpy().flatten()
            P_np = np.eye(n) * w.wControl  # Simple diagonal
            timing['identity_hessian'] = time.time() - t0
        
        # --- CONSTRAINTS (box constraints on delta) ---
        t0 = time.time()
        # u_min <= u_curr + delta_u <= u_max
        # Lower: -delta_u[i] + s = u_curr[i] - u_min  (s >= 0)
        # Upper: +delta_u[i] + s = u_max - u_curr[i]  (s >= 0)
        # A has one nnz per row. For each control dim i:
        #   row 2i+0 (lower): A[2i,   i] = -1
        #   row 2i+1 (upper): A[2i+1, i] = +1
        # So A_values = [-1, +1, -1, +1, ...] of length m = n*2
        u_curr_flat = u_curr.detach().flatten()
        A_values = torch.tensor([-1.0, 1.0], dtype=torch.float64, device=self.device).repeat(n)
        b_lower = u_curr_flat - cfg.u_min
        b_upper = cfg.u_max - u_curr_flat
        b_vec = torch.stack([b_lower, b_upper], dim=1).flatten()  # interleave: [lo0,hi0,lo1,hi1,...]

        timing['constraint_assembly'] = time.time() - t0

        # --- SOLVE QP with moreau.torch (gradient-enabled) ---
        t0 = time.time()
        # P_values: flatten row-major for dense symmetric matrix
        if cfg.use_gauss_newton:
            P_values = P_torch.flatten()  # shape (n*n,), full symmetric
        else:
            P_values = torch.eye(n, dtype=torch.float64, device=self.device).flatten()
        g_vec = g_torch  # shape (n,), torch.Tensor with grad

        self._moreau_torch_solver.setup(P_values, A_values)
        solution = self._moreau_torch_solver.solve(g_vec, b_vec)
        timing['qp_solve'] = time.time() - t0

        delta_u = solution.x.reshape(self.horizon, 2)  # gradient flows through g_vec -> w

        return delta_u, g_vec, timing
    
    def evaluate_cost(
        self,
        u: torch.Tensor,
        q0: torch.Tensor,
        v0: torch.Tensor,
        pusher0: torch.Tensor,
        goal: torch.Tensor,
        w: CostWeights,
    ) -> Dict:
        """Evaluate cost and return detailed breakdown."""
        qs, vs, prs, terminalVelEnergy, _, _ = self.forward_simulate(
            q0, v0, pusher0, u, store_contact_data=False
        )
        
        # Cost components
        controlEnergy = (u ** 2).sum()
        controlSmooth = ((u[1:] - u[:-1]) ** 2).sum() if len(u) > 1 else torch.zeros((), device=self.device)
        
        qFinal = qs[-1]
        rXY = qFinal[:2] - goal[:2]
        targetCost = (rXY ** 2).sum()
        
        rTh = wrap_to_pi(qFinal[2] - goal[2])
        orientCost = rTh ** 2
        
        terminalVelCost = terminalVelEnergy
        
        loss = (
            w.wControl * controlEnergy +
            w.wControlSmooth * controlSmooth +
            w.wObjVel * terminalVelCost +
            w.wTargetXY * targetCost +
            w.wTargetOrient * orientCost
        )
        
        return {
            "loss": loss,
            "qs": qs,
            "vs": vs,
            "prs": prs,
            "controlEnergy": controlEnergy,
            "controlSmooth": controlSmooth,
            "terminalXYNormSq": targetCost,
            "terminalOrientNormSq": orientCost,
            "terminalVelEnergy": terminalVelEnergy,
            "terminalXYNorm": torch.norm(rXY),
            "terminalThetaAbs": torch.abs(rTh),
        }
    
    def optimize(
        self,
        q0,
        v0,
        pusher0,
        goal,
        u_init=None,
        cfg: Optional[SQPConfig] = None,
        w: Optional[CostWeights] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Main SQP optimization loop with Gauss-Newton Hessian.
        
        Matches SingleShootingSQP.optimize() API exactly.
        """
        # Convert inputs to tensors
        q0 = self._to_tensor(q0)
        v0 = self._to_tensor(v0)
        pusher0 = self._to_tensor(pusher0)
        goal = self._to_tensor(goal)
        
        # Default configs
        if cfg is None:
            cfg = SQPConfig()
        if w is None:
            w = CostWeights()
        
        # Initialize controls
        if u_init is None:
            if cfg.u_init_mode == 'zero':
                u_curr = torch.zeros(self.horizon, 2, dtype=torch.float64, device=self.device)
            else:  # 'geometric'
                u_curr_np = self.compute_geometric_initial_trajectory(
                    robot_pos=pusher0.detach().cpu().numpy(),
                    box_pos=q0.detach().cpu().numpy(),
                    goal_pos=goal.detach().cpu().numpy()
                )
                u_curr = self._to_tensor(u_curr_np)
            u_init_eval = u_curr
        else:
            u_curr = self._to_tensor(u_init)
            u_init_eval = u_curr
        
        if verbose:
            hess_type = "Gauss-Newton (P = J^T J)" if cfg.use_gauss_newton else "Identity (P = I)"
            print("\n" + "="*70)
            print(f"Single Shooting SQP with {hess_type}")
            print("="*70)
            print(f"Horizon: {self.horizon}, dt: {self.dt}")
            print(f"Control bounds: [{cfg.u_min}, {cfg.u_max}]")
            print(f"Max iterations: {cfg.maxIters}")
            if cfg.use_gauss_newton:
                print(f"Hessian regularization: {cfg.hessian_regularization}")
            
            # GPU acceleration info
            print(f"\n🚀 Acceleration:")
            print(f"  PyTorch device: {self.device}")
            if moreau.device_available('cuda'):
                print(f"  Moreau CUDA: available ✓")
            else:
                print(f"  Moreau CUDA: not available (using CPU)")
            print("="*70)
        
        cost_curr = None
        start_time = time.time()
        
        # History tracking
        history = {
            'loss': [],
            'grad_norm': [],
            'step_norm': [],
            'cost_change': [],
            'alpha': [],
            # Timing breakdown per iteration
            'time_hessian': [],
            'time_qp': [],
            'time_line_search': [],
            'time_total_iter': [],
            # IPM target_mu tracking
            'target_mu': [],
        }
        
        # Track previous gradient norm for adaptive scheduling
        grad_norm_prev = None
        
        # SQP loop
        for iteration in range(cfg.maxIters):
            iter_start_time = time.time()
            
            # --- Apply target_mu scheduling ---
            if cfg.use_mu_scheduling:
                target_mu = self._schedule_target_mu(
                    iteration, cfg, 
                    grad_norm=history['grad_norm'][-1] if iteration > 0 else None,
                    grad_norm_prev=grad_norm_prev
                )
                # Update IPM options for this iteration
                self.ipmOpts.target_mu = target_mu
                history['target_mu'].append(target_mu)
                
                if verbose and iteration == 0:
                    print(f"\n📊 Target_mu scheduling: {cfg.mu_schedule_type}")
                    print(f"   Start: {cfg.mu_start:.1e} → End: {cfg.mu_end:.1e}")
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"SQP Iteration {iteration+1}/{cfg.maxIters}")
                if cfg.use_mu_scheduling:
                    print(f"  IPM target_mu: {self.ipmOpts.target_mu:.2e}")
                print(f"{'='*70}")
            
            # --- STEP 1: Build and solve QP ---
            hessian_start = time.time()
            delta_u, g_vec, qp_timing = self.build_qp_matrices(
                u_curr, q0, v0, pusher0, goal, w, cfg
            )
            hessian_time = time.time() - hessian_start

            # Sync GPU if needed
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            qp_time = qp_timing.get('qp_solve', 0.0)

            # Gradient norm
            grad_norm = g_vec.detach().norm().item()
            history['grad_norm'].append(grad_norm)
            
            if verbose:
                print(f"  Gradient norm: {grad_norm:.6e}")
                print(f"  ⏱ Timing:")
                print(f"    Hessian+QP build: {hessian_time:.2f}s", end="")
                if cfg.use_gauss_newton and 'jacobian_computation' in qp_timing:
                    print(f" (Jacobian: {qp_timing['jacobian_computation']:.2f}s)")
                else:
                    print()
                print(f"    QP solve:         {qp_time:.2f}s")
            if cfg.use_trust_region and iteration < cfg.trust_region_iters:
                delta_norm = torch.norm(delta_u).item()
                if delta_norm > cfg.trust_region_size:
                    delta_u = delta_u * (cfg.trust_region_size / delta_norm)
                    if verbose:
                        print(f"  Trust region active: scaled to {cfg.trust_region_size:.3f}")
            
            u_qp = u_curr + delta_u
            step_norm = torch.norm(delta_u).item()
            history['step_norm'].append(step_norm)
            
            if verbose:
                print(f"  QP step norm: {step_norm:.6f}")
            
            # --- STEP 2: Line search ---
            line_search_start = time.time()
            alpha_accepted = 1.0
            if cfg.use_line_search and cost_curr is not None:
                alpha = 1.0
                for ls_iter in range(cfg.line_search_max_iters):
                    u_trial = u_curr + alpha * delta_u
                    
                    result = self.evaluate_cost(u_trial, q0, v0, pusher0, goal, w)
                    cost_trial = result["loss"]
                    
                    # Armijo condition with tolerance (accept near-flat landscape)
                    if cost_trial <= cost_curr + cfg.line_search_ftol:
                        if verbose:
                            print(f"  Line search: α={alpha:.4f}, cost={cost_trial.item():.6f} (accepted)")
                        u_new = u_trial
                        cost_new = cost_trial
                        alpha_accepted = alpha
                        break
                    else:
                        alpha *= cfg.line_search_beta
                        if ls_iter == cfg.line_search_max_iters - 1:
                            if verbose:
                                print(f"  Line search: α={alpha:.4f}, cost={cost_trial.item():.6f} (failed, using)")
                            u_new = u_trial
                            cost_new = cost_trial
                            alpha_accepted = alpha
            else:
                # First iteration or no line search
                u_new = u_qp
                result = self.evaluate_cost(u_new, q0, v0, pusher0, goal, w)
                cost_new = result["loss"]
            
            # Ensure GPU operations complete before timing
            if self.device.type == 'cuda':
                torch.cuda.synchronize()  # ⭐ Wait for line search to finish
            line_search_time = time.time() - line_search_start
            
            # Record timing
            iter_total_time = time.time() - iter_start_time
            history['time_hessian'].append(hessian_time)
            history['time_qp'].append(qp_time)
            history['time_line_search'].append(line_search_time)
            history['time_total_iter'].append(iter_total_time)
            
            if verbose:
                print(f"    Line search:      {line_search_time:.2f}s")
                print(f"    ─────────────────────────")
                print(f"    Total iteration:  {iter_total_time:.2f}s")
            
            # Update history
            history['loss'].append(cost_new.item())
            history['alpha'].append(alpha_accepted)
            
            # Check convergence
            if cost_curr is not None:
                cost_change = abs(cost_new.item() - cost_curr.item())
                history['cost_change'].append(cost_change)
                
                if verbose:
                    delta_cost = cost_curr.item() - cost_new.item()
                    if delta_cost > 0:
                        print(f"  ✓ Cost: {cost_new.item():.6f} (decreased by {delta_cost:.6f})")
                    else:
                        print(f"  ⚠ Cost: {cost_new.item():.6f} (increased by {-delta_cost:.6f})")
                    
                    # Print breakdown
                    print(f"    Control: {w.wControl * result['controlEnergy'].item():.6f}")
                    print(f"    Target XY: {w.wTargetXY * result['terminalXYNormSq'].item():.6f}")
                    print(f"    Orient: {w.wTargetOrient * result['terminalOrientNormSq'].item():.6f}")
                    print(f"    Terminal vel: {w.wObjVel * result['terminalVelEnergy'].item():.6f}")
                    
                    # Final position vs goal
                    q_final = result['qs'][-1].detach().cpu().numpy()
                    goal_np = goal.detach().cpu().numpy()
                    print(f"  Final pos [x,y,θ]: [{q_final[0]:.4f}, {q_final[1]:.4f}, {q_final[2]:.4f}]")
                    print(f"  Goal      [x,y,θ]: [{goal_np[0]:.4f}, {goal_np[1]:.4f}, {goal_np[2]:.4f}]")
                    print(f"  Position error: {result['terminalXYNorm'].item():.4f}m")
                    print(f"  Orientation error: {result['terminalThetaAbs'].item():.4f}rad")
                
                # Convergence check
                if grad_norm < cfg.tol:
                    if verbose:
                        print(f"\n✓ Converged! Gradient norm {grad_norm:.6e} < {cfg.tol}")
                    break
                if cost_change < cfg.cost_tol:
                    if verbose:
                        print(f"\n✓ Converged! Cost change {cost_change:.2e} < {cfg.cost_tol}")
                    break
            else:
                history['cost_change'].append(0.0)
                if verbose:
                    print(f"  Initial cost: {cost_new.item():.6f}")
            
            u_curr = u_new
            cost_curr = cost_new
            
            # Update grad_norm_prev for adaptive scheduling
            grad_norm_prev = grad_norm
        
        solve_time = time.time() - start_time
        
        # --- Final evaluation with contact data ---
        qs_final, vs_final, prs_final, _, contact_forces_final, signed_distances_final = \
            self.forward_simulate(q0, v0, pusher0, u_curr, store_contact_data=True)
        
        # Compute final gradient
        _, g_final_torch, _= self.compute_gauss_newton_hessian(
            u_curr, q0, v0, pusher0, goal, w,
            regularization=cfg.hessian_regularization
        )
        final_grad_norm = torch.norm(g_final_torch).item()
        
        # # Initial trajectory
        # if u_init is None:
        #     u_init_eval = self._init_controls_simple(q0, goal)
        # else:
        #     u_init_eval = self._to_tensor(u_init)
        qs_init, vs_init, prs_init, _, _, _ = \
            self.forward_simulate(q0, v0, pusher0, u_init_eval, store_contact_data=False)
        
        # Loss components
        controlEnergy = (u_curr ** 2).sum()
        controlSmooth = ((u_curr[1:] - u_curr[:-1]) ** 2).sum() if len(u_curr) > 1 else torch.zeros((), device=self.device)
        terminalVelCost = (vs_final[-1] ** 2).sum()
        rXY = qs_final[-1][:2] - goal[:2]
        terminalXYCost = (rXY ** 2).sum()
        rTheta = wrap_to_pi(qs_final[-1][2] - goal[2])
        terminalOrientCost = rTheta ** 2
        
        loss_final = (
            w.wControl * controlEnergy +
            w.wControlSmooth * controlSmooth +
            w.wObjVel * terminalVelCost +
            w.wTargetXY * terminalXYCost +
            w.wTargetOrient * terminalOrientCost
        )
        
        if verbose:
            print(f"\n{'='*70}")
            print("Optimization Complete!")
            print(f"{'='*70}")
            print(f"Time: {solve_time:.2f}s")
            print(f"Iterations: {iteration+1}")
            print(f"Final loss: {loss_final.item():.6f}")
            print(f"Final gradient norm: {final_grad_norm:.6e}")
            print(f"Position error: {torch.norm(rXY).item():.4f}m")
            print(f"Orientation error: {abs(rTheta.item()):.4f}rad")
            print(f"{'='*70}")
            
            # Timing breakdown summary
            total_hessian = sum(history['time_hessian'])
            total_qp = sum(history['time_qp'])
            total_line_search = sum(history['time_line_search'])
            avg_iter = np.mean(history['time_total_iter'])
            
            print(f"\n⏱ Timing Breakdown:")
            print(f"{'─'*70}")
            print(f"  Total time:        {solve_time:.2f}s")
            print(f"  Avg per iteration: {avg_iter:.2f}s")
            print(f"")
            print(f"  Cumulative by component:")
            print(f"    Hessian (build):   {total_hessian:.2f}s  ({100*total_hessian/solve_time:.1f}%)")
            print(f"    QP solve:          {total_qp:.2f}s  ({100*total_qp/solve_time:.1f}%)")
            print(f"    Line search:       {total_line_search:.2f}s  ({100*total_line_search/solve_time:.1f}%)")
            print(f"")
            print(f"  Average per iteration:")
            print(f"    Hessian:     {np.mean(history['time_hessian']):.2f}s")
            print(f"    QP solve:    {np.mean(history['time_qp']):.2f}s")
            print(f"    Line search: {np.mean(history['time_line_search']):.2f}s")
            
            # Target_mu schedule summary
            if cfg.use_mu_scheduling and len(history['target_mu']) > 0:
                print(f"\n📊 Target_mu Schedule ({cfg.mu_schedule_type}):")
                print(f"{'─'*70}")
                print(f"  Initial:    {history['target_mu'][0]:.2e}")
                print(f"  Final:      {history['target_mu'][-1]:.2e}")
                print(f"  Reduction:  {history['target_mu'][0]/history['target_mu'][-1]:.1f}×")
            
            print(f"{'='*70}\n")
        
        # Return dict matching original SS SQP format
        return {
            "loss": loss_final.detach().cpu(),
            "q_final": qs_final[-1].detach().cpu().numpy(),
            "u_seq": u_curr.detach().cpu().numpy(),
            "trajectory": qs_final.detach().cpu().numpy(),
            "velocity_trajectory": vs_final.detach().cpu().numpy(),
            "pusher_trajectory": prs_final.detach().cpu().numpy(),
            "prKnots": None,
            "contact_forces": contact_forces_final.detach().cpu().numpy(),
            "signed_distances": signed_distances_final.detach().cpu().numpy(),
            
            "initial_trajectory": qs_init.cpu().numpy(),
            "initial_velocity_trajectory": vs_init.cpu().numpy(),
            "initial_pusher_trajectory": prs_init.cpu().numpy(),
            
            "loss_components": {
                "total": loss_final.item(),
                "control_energy": controlEnergy.item(),
                "control_smooth": controlSmooth.item(),
                "obj_vel": terminalVelCost.item(),
                "target_xy": terminalXYCost.item(),
                "target_orient": terminalOrientCost.item(),
                "w_control": w.wControl,
                "w_smooth": w.wControlSmooth,
                "w_objvel": w.wObjVel,
                "w_targetxy": w.wTargetXY,
                "w_orient": w.wTargetOrient,
            },
            
            "control_gradients": g_final_torch.detach().cpu().numpy().reshape(self.horizon, 2),
            
            "history": {
                'loss': np.array(history['loss']),
                'grad_norm': np.array(history['grad_norm']),
                'step_norm': np.array(history['step_norm']),
                'cost_change': np.array(history['cost_change']),
                'alpha': np.array(history['alpha']),
                # Timing data
                'time_hessian': np.array(history['time_hessian']),
                'time_qp': np.array(history['time_qp']),
                'time_line_search': np.array(history['time_line_search']),
                'time_total_iter': np.array(history['time_total_iter']),
                # IPM scheduling
                'target_mu': np.array(history['target_mu']) if history['target_mu'] else np.array([self.ipmOpts.target_mu] * len(history['loss'])),
            },
            
            "stationarityInfo": {
                "final_defect_norm": 0.0,
                "final_grad_norm": final_grad_norm,
                "final_terminal_error": torch.norm(rXY).item(),
                "converged": final_grad_norm < cfg.tol,
            },
            
            "solve_time": solve_time,
        }
    
    def _to_tensor(self, x):
        """Convert to tensor if needed"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64, device=self.device)
        else:
            x = x.to(dtype=torch.float64, device=self.device)
        return x
    
    def _init_controls_simple(self, q0, goal):
        """Simple constant velocity initialization."""
        displacement = goal[:2] - q0[:2]
        time_horizon = self.horizon * self.dt
        avg_velocity = displacement / time_horizon
        safety_factor = 0.7
        u_const = avg_velocity * safety_factor
        u_init = u_const.unsqueeze(0).repeat(self.horizon, 1)
        return u_init
    
    def _schedule_target_mu(self, iteration: int, cfg: SQPConfig, grad_norm: float = None, grad_norm_prev: float = None) -> float:
        """
        Compute target_mu for IPM based on iteration number.
        
        Continuation method: Start with loose complementarity, gradually tighten.
        
        Args:
            iteration: Current iteration (0-indexed)
            cfg: SQP configuration
            grad_norm: Current gradient norm (for adaptive scheduling)
            grad_norm_prev: Previous gradient norm (for adaptive scheduling)
        
        Returns:
            target_mu: Target complementarity tolerance
        """
        if not cfg.use_mu_scheduling:
            # Use default from IPM options
            return self.ipmOpts.target_mu
        
        max_iter = cfg.maxIters
        t = iteration / max(max_iter - 1, 1)  # Normalized time [0, 1]
        
        if cfg.mu_schedule_type == 'linear':
            # Linear interpolation
            mu = cfg.mu_start * (1 - t) + cfg.mu_end * t
            
        elif cfg.mu_schedule_type == 'exponential':
            # Exponential decay (recommended)
            mu = cfg.mu_start * (cfg.mu_end / cfg.mu_start) ** t
            
        elif cfg.mu_schedule_type == 'stepwise':
            # Step-wise schedule
            if iteration < 5:
                mu = cfg.mu_start
            elif iteration < 15:
                mu = cfg.mu_start / 10
            elif iteration < 25:
                mu = cfg.mu_start / 100
            else:
                mu = cfg.mu_end
                
        elif cfg.mu_schedule_type == 'adaptive':
            # Adaptive based on convergence rate
            if iteration == 0:
                mu = cfg.mu_start
            elif grad_norm is not None and grad_norm_prev is not None:
                # Current mu (stored in instance variable)
                mu_current = getattr(self, '_current_mu', cfg.mu_start)
                
                # If gradient decreased significantly, tighten mu
                if grad_norm < 0.1 * grad_norm_prev:
                    mu = max(mu_current / 10, cfg.mu_end)
                else:
                    mu = mu_current
                    
                # Store for next iteration
                self._current_mu = mu
            else:
                mu = getattr(self, '_current_mu', cfg.mu_start)
        else:
            raise ValueError(f"Unknown mu_schedule_type: {cfg.mu_schedule_type}")
        
        return float(mu)
    
    
    def compute_geometric_initial_trajectory(
        self,
        robot_pos,      # [x, y] - initial robot position
        box_pos,        # [x, y] or [x, y, theta] - initial box position  
        goal_pos,       # [x, y] or [x, y, theta] - goal position
    ):
        """
        Create initial trajectory based on geometric path: Robot -> Box -> Goal
        
        Uses the optimizer's horizon, dt, and half (box_half_size) automatically.
        
        Args:
            robot_pos: Initial robot position [x, y]
            box_pos: Initial box position [x, y] or [x, y, theta] (only x, y used)
            goal_pos: Goal position [x, y] or [x, y, theta] (only x, y used)
        
        Returns:
            u_init: Initial control trajectory as list [[u_x, u_y], ...] (horizon x 2)
        """
        
        # Extract x, y only (handle both [x, y] and [x, y, theta])
        robot_pos = np.array(robot_pos[:2])
        box_pos = np.array(box_pos[:2])
        goal_pos = np.array(goal_pos[:2])
        
        # Compute distances
        dist_robot_to_box = np.linalg.norm(box_pos - robot_pos)
        dist_box_to_goal = np.linalg.norm(goal_pos - box_pos)
        total_dist = dist_robot_to_box + dist_box_to_goal
        
        print(f"\n[Geometric Init] Distance analysis:")
        print(f"  Robot -> Box: {dist_robot_to_box:.4f} m")
        print(f"  Box -> Goal: {dist_box_to_goal:.4f} m")
        print(f"  Total: {total_dist:.4f} m")
        
        # Handle edge case: already at goal
        if total_dist < 1e-6:
            print(f"  Already at goal! Using zero controls.")
            return [[0.0, 0.0]] * self.horizon
        
        # Allocate timesteps proportionally to distances
        min_steps = 5
        if self.horizon < 2 * min_steps:
            steps_phase1 = self.horizon // 2
        else:
            ratio = dist_robot_to_box / total_dist
            steps_phase1 = int(self.horizon * ratio)
            steps_phase1 = max(min_steps, min(self.horizon - min_steps, steps_phase1))
        
        steps_phase2 = self.horizon - steps_phase1
        
        print(f"  Phase 1 (approach): {steps_phase1} steps")
        print(f"  Phase 2 (push): {steps_phase2} steps")
        
        # Phase 1: Robot approaches box
        dir_to_box = (box_pos - robot_pos) / (dist_robot_to_box + 1e-8)
        contact_offset = 0.0001 #self.half  # Slightly more than half size
        target_contact_pos = box_pos - dir_to_box * contact_offset
        
        displacement_phase1 = target_contact_pos - robot_pos
        time_phase1 = steps_phase1 * self.dt
        velocity_phase1 = displacement_phase1 / (time_phase1 + 1e-8)
        
        print(f"  Phase 1 velocity: [{velocity_phase1[0]:.3f}, {velocity_phase1[1]:.3f}] m/s")
        
        # Phase 2: Robot pushes box to goal
        dir_to_goal = (goal_pos - box_pos) / (dist_box_to_goal + 1e-8)
        displacement_phase2 = goal_pos - box_pos
        time_phase2 = steps_phase2 * self.dt
        velocity_phase2 = displacement_phase2 / (time_phase2 + 1e-8)
        
        # Scale down push velocity
        push_scale = 0.8
        velocity_phase2 = velocity_phase2 * push_scale
        
        print(f"  Phase 2 velocity: [{velocity_phase2[0]:.3f}, {velocity_phase2[1]:.3f}] m/s")
        
        # Create control trajectory
        u_init = []
        
        # Phase 1: Approach box
        for i in range(steps_phase1):
            u_init.append([float(velocity_phase1[0]), float(velocity_phase1[1])])
        
        # Phase 2: Push box to goal
        for i in range(steps_phase2):
            u_init.append([float(velocity_phase2[0]), float(velocity_phase2[1])])
        
        # Smooth transition (optional)
        transition_steps = min(5, steps_phase1 // 4, steps_phase2 // 4)
        if transition_steps > 0:
            for i in range(transition_steps):
                alpha = (i + 1) / (transition_steps + 1)
                idx = steps_phase1 - transition_steps + i
                if 0 <= idx < steps_phase1:
                    u_init[idx] = [
                        float((1 - alpha) * velocity_phase1[0] + alpha * velocity_phase2[0]),
                        float((1 - alpha) * velocity_phase1[1] + alpha * velocity_phase2[1])
                    ]
        
        print(f"  Generated control trajectory: {len(u_init)} x 2")
        u_magnitudes = [np.linalg.norm(u) for u in u_init]
        print(f"  Control magnitude range: [{min(u_magnitudes):.3f}, {max(u_magnitudes):.3f}]")
        
        return u_init