"""
Single Shooting SQP with Moreau Solver

Clean implementation matching the existing TrajectoryOptimizer API.
Uses delta formulation for proper SQP.
"""

import torch
import numpy as np
from scipy import sparse
import moreau
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
    tol: float = 1e-4
    
    # Line search
    use_line_search: bool = True
    line_search_max_iters: int = 10
    line_search_beta: float = 0.5
    line_search_c1: float = 1e-4
    
    # Control bounds
    u_min: float = -0.3
    u_max: float = 0.3
    
    # Trust region (early iterations)
    use_trust_region: bool = True
    trust_region_iters: int = 3
    trust_region_size: float = 0.5


@dataclass
class CostWeights:
    """Cost function weights"""
    wControl: float = 0.1
    wControlSmooth: float = 0.0
    wObjVel: float = 1.0  # Terminal velocity
    wTargetXY: float = 20.0
    wTargetOrient: float = 0.5


class SingleShootingSQP:
    """
    Single Shooting trajectory optimizer using SQP with Moreau solver.
    
    API matches existing TrajectoryOptimizer for easy comparison.
    """
    
    def __init__(
        self,
        mass: float,
        side_length: float,
        mu: float,
        horizon: int,
        dt: float,
        device: str = "cpu",
        dynamics_module = None,  # Passed in from existing code
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
        
        # Store dynamics module (step_square or step_square_IP)
        self.dynamics = dynamics_module
        
        # Moment of inertia
        self.Izz = (1.0/6.0) * mass * (side_length**2 + side_length**2)
    
    def forward_simulate(
        self,
        q0: torch.Tensor,
        v0: torch.Tensor,
        pusher0: torch.Tensor,
        u: torch.Tensor,
        store_contact_data: bool = False,
    ) -> Tuple:
        """
        Forward simulation from initial state with controls u.
        
        Args:
            store_contact_data: If True, return contact forces and signed distances
        
        Returns:
            qs: (T+1, 3) object states
            vs: (T+1, 3) object velocities  
            prs: (T+1, 2) pusher positions
            terminalVelEnergy: scalar (terminal velocity squared norm)
            contact_forces: (T, 2) if store_contact_data else None
            signed_distances: (T,) if store_contact_data else None
        """
        T = self.horizon
        
        qs = [q0]
        vs = [v0]
        prs = [pusher0]
        
        # Contact data storage
        if store_contact_data:
            contact_forces = []
            signed_distances = []
        
        q, v, pr = q0, v0, pusher0
        z_prev = None
        for t in range(T):
            # Call dynamics (matches existing API)
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
                z_prev = z_prev,
                skip_solving_threshold = 100.0
            )
            
            qs.append(q_next)
            vs.append(v_next)
            prs.append(pr_next)
            
            # Store contact data
            if store_contact_data:
                contact_forces.append(lam)
                signed_distances.append(phi)
            
            q, v, pr = q_next, v_next, pr_next
        
        # FIXED: Terminal velocity energy (not cumulative!)
        terminalVelEnergy = (v ** 2).sum()  # v is final velocity after loop
        
        qs_stacked = torch.stack(qs)
        vs_stacked = torch.stack(vs)
        prs_stacked = torch.stack(prs)
        
        if store_contact_data:
            contact_forces_stacked = torch.stack(contact_forces)
            signed_distances_stacked = torch.stack(signed_distances)
            return qs_stacked, vs_stacked, prs_stacked, terminalVelEnergy, contact_forces_stacked, signed_distances_stacked
        else:
            return qs_stacked, vs_stacked, prs_stacked, terminalVelEnergy, None, None
    
    def evaluate_cost(
        self,
        u: torch.Tensor,
        q0: torch.Tensor,
        v0: torch.Tensor,
        pusher0: torch.Tensor,
        goal: torch.Tensor,
        w: CostWeights,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate cost for given controls.
        
        Returns:
            loss: scalar
            qs: (T+1, 3) trajectory
        """
        qs, vs, prs, terminalVelEnergy, lamdas, phis = self.forward_simulate(
            q0, v0, pusher0, u, store_contact_data=True
        )
        
        # Cost components
        controlEnergy = (u ** 2).sum()
        controlSmooth = ((u[1:] - u[:-1]) ** 2).sum() if len(u) > 1 else torch.zeros((), device=self.device)
        
        # Terminal costs
        qFinal = qs[-1]
        
        # Position error
        rXY = qFinal[:2] - goal[:2]
        targetCost = (rXY ** 2).sum()
        
        # Orientation error
        terminalOrientNormSq = torch.zeros((), device=self.device)
        rTh = wrap_to_pi(qFinal[2] - goal[2])
        orientCost = rTh ** 2
        
        # Terminal velocity cost (already computed in forward_simulate)
        terminalVelCost = terminalVelEnergy
        
        loss = (
            w.wControl * controlEnergy +
            w.wControlSmooth * controlSmooth +
            w.wObjVel * terminalVelCost +
            w.wTargetXY * targetCost +
            w.wTargetOrient * orientCost
        )
        
        return {
            "qs": qs,
            "vs": vs,
            "qrobot_hist": prs,
            "lamdas": lamdas,
            "phis": phis,
            "loss": loss,
            "controlEnergy": controlEnergy,
            "controlSmooth": controlSmooth,
            "objVelEnergy": terminalVelCost,
            "terminalXYNormSq": targetCost,
            "terminalOrientNormSq": orientCost,
            "terminalXYNorm": rXY.norm(),
            "terminalThetaAbs": rTh.abs()    
        }

    
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
        Build QP matrices for DELTA formulation.
        
        QP solves:
            minimize  (1/2) δu' P δu + g' δu
            s.t.      u_min ≤ u_curr + δu ≤ u_max
        
        Returns: P, q, A, b, cones
        """
        T = self.horizon
        n = T * 2  # Total variables (flattened u)
        
        # P MATRIX (Hessian approximation)
        P_diag = np.full(n, w.wControl)
        P = sparse.diags(P_diag, format='csr')
        
        # q VECTOR (Gradient via autograd)
        u_ad = u_curr.clone().detach().requires_grad_(True)
        
        result = self.evaluate_cost(u_ad, q0, v0, pusher0, goal, w)
        loss = result["loss"]
        loss.backward()
        
        grad_u = u_ad.grad.cpu().numpy().flatten()
        q_vec = grad_u
        
        # CONSTRAINTS (box constraints on delta)
        # u_min ≤ u_curr + δu ≤ u_max
        # → (u_min - u_curr) ≤ δu ≤ (u_max - u_curr)
        
        constraint_rows = []
        constraint_rhs = []
        
        for t in range(T):
            for d in range(2):
                idx = t * 2 + d
                u_curr_val = u_curr[t, d].item()
                
                # Lower: -δu + s = u_curr - u_min
                row_lower = np.zeros(n)
                row_lower[idx] = -1.0
                constraint_rows.append(row_lower)
                constraint_rhs.append(u_curr_val - cfg.u_min)
                
                # Upper: δu + s = u_max - u_curr
                row_upper = np.zeros(n)
                row_upper[idx] = 1.0
                constraint_rows.append(row_upper)
                constraint_rhs.append(cfg.u_max - u_curr_val)
        
        A = sparse.csr_array(np.vstack(constraint_rows))
        b = np.array(constraint_rhs)
        
        cones = moreau.Cones(
            num_zero_cones=0,
            num_nonneg_cones=T * 2 * 2,
        )
        
        return P, q_vec, A, b, cones
    
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
        Main SQP optimization loop.
        
        Args match existing TrajectoryOptimizer.optimize() API.
        
        Returns:
            Dictionary with trajectories, controls, loss, etc.
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
            u_curr = self._init_controls_simple(q0, goal)
        else:
            u_curr = self._to_tensor(u_init)
        
        if verbose:
            print("\n" + "="*70)
            print("Single Shooting SQP with Moreau")
            print("="*70)
            print(f"Horizon: {self.horizon}, dt: {self.dt}")
            print(f"Control bounds: [{cfg.u_min}, {cfg.u_max}]")
            print(f"Max iterations: {cfg.maxIters}")
        
        cost_curr = None
        start_time = time.time()
        
        # History tracking (matching MS)
        history = {
            'loss': [],
            'step_norm': [],
            'cost_change': [],
            'alpha': [],
        }
        
        # SQP loop
        for iteration in range(cfg.maxIters):
            if verbose:
                print(f"\n{'='*70}")
                print(f"SQP Iteration {iteration+1}/{cfg.maxIters}")
                print(f"{'='*70}")
            
            # Build and solve QP
            P, q, A, b, cones = self.build_qp_matrices(
                u_curr, q0, v0, pusher0, goal, w, cfg
            )
            
            solver = moreau.Solver(P, q, A, b, cones=cones)
            solution = solver.solve()
            
            # Extract delta
            delta_u = torch.tensor(
                solution.x.reshape(self.horizon, 2),
                device=self.device,
                dtype=torch.float64
            )
            
            # Trust region (early iterations)
            if cfg.use_trust_region and iteration < cfg.trust_region_iters:
                delta_norm = torch.norm(delta_u)
                if delta_norm > cfg.trust_region_size:
                    scale = cfg.trust_region_size / delta_norm
                    delta_u = delta_u * scale
                    if verbose:
                        print(f"  Trust region: scaled step from {delta_norm:.3f} to {cfg.trust_region_size:.3f}")
            
            u_qp = u_curr + delta_u
            step_norm = torch.norm(delta_u).item()
            
            if verbose:
                print(f"  Step norm: {step_norm:.6f}")
            
            # Line search
            alpha_accepted = 1.0
            if cfg.use_line_search and cost_curr is not None:
                alpha = 1.0
                for ls_iter in range(cfg.line_search_max_iters):
                    u_trial = u_curr + alpha * delta_u
                    
                    result = self.evaluate_cost(
                        u_trial, q0, v0, pusher0, goal, w
                    )
                    cost_trial = result["loss"]
                    
                    # Armijo condition
                    if cost_trial < cost_curr:
                        if verbose:
                            print(f"  Line search: α={alpha:.4f}, cost={cost_trial:.6f}")
                        u_new = u_trial
                        cost_new = cost_trial
                        alpha_accepted = alpha
                        break
                    else:
                        alpha *= cfg.line_search_beta
                        if ls_iter == cfg.line_search_max_iters - 1:
                            if verbose:
                                print(f"  Line search failed, using α={alpha:.4f}")
                            u_new = u_trial
                            cost_new = cost_trial
                            alpha_accepted = alpha
            else:
                # First iteration or no line search
                u_new = u_qp
                result = self.evaluate_cost(u_new, q0, v0, pusher0, goal, w)
                cost_new = result["loss"]
            
            # Record history
            history['loss'].append(cost_new.detach().cpu().numpy())
            history['step_norm'].append(step_norm)
            history['alpha'].append(alpha_accepted)
            
            # Check convergence
            if cost_curr is not None:
                cost_change = abs(cost_new - cost_curr)
                history['cost_change'].append(cost_change)
                
                if verbose:
                    if cost_new < cost_curr:
                        print(f"  ✓ Cost: {cost_new:.6f} (decreased by {cost_curr - cost_new:.6f})")
                        # Print summary
                        print(f"Loss: {result['loss'].item():.6f}")
                        print(f"  Control energy: {w.wControl * result['controlEnergy'].item():.6f}")
                        print(f"  Control smooth: {w.wControlSmooth * result['controlSmooth'].item():.6f}")
                        print(f"  Target XY: {w.wTargetXY * result['terminalXYNormSq'].item():.6f}")
                        print(f"  Target orient: {w.wTargetOrient * result['terminalOrientNormSq'].item():.6f}")
                        print(f"Terminal error: ||rXY||={result['terminalXYNorm'].item():.6f}, |rTh|={result['terminalThetaAbs'].item():.6f}")
                        print(f"Final pose: {result['qs'][-1].detach().cpu().numpy()}")
                        print(f"Goal pose: [{goal[0].item():.4f}, {goal[1].item():.4f}, {goal[2]}]")
                        print(f"grad norm : {u_new.grad.norm().item()}")
                        
                    else:
                        print(f"  ⚠ Cost: {cost_new:.6f} (increased by {cost_new - cost_curr:.6f})")
                
                if step_norm < cfg.tol and cost_change < cfg.tol:
                    if verbose:
                        print(f"\n✓ Converged after {iteration+1} iterations!")
                    break
            else:
                history['cost_change'].append(0.0)
                if verbose:
                    print(f"  Initial cost: {cost_new:.6f}")
            
            u_curr = u_new
            cost_curr = cost_new
        
        solve_time = time.time() - start_time
        
        # Final evaluation with detailed outputs INCLUDING contact data
        result_final = self.evaluate_cost(u_curr, q0, v0, pusher0, goal, w)
        
        loss_final = result_final["loss"]
        qs_final = result_final["qs"]

        # Final forward simulation WITH contact data collection
        qs_final_full, vs_final, prs_final, _, contact_forces_final, signed_distances_final = \
            self.forward_simulate(q0, v0, pusher0, u_curr, store_contact_data=True)
        
        # Compute initial trajectory for comparison (with contact data)
        if u_init is None:
            u_init_eval = self._init_controls_simple(q0, goal)
        else:
            u_init_eval = self._to_tensor(u_init)
        qs_init, vs_init, prs_init, _, _, _ = \
            self.forward_simulate(q0, v0, pusher0, u_init_eval, store_contact_data=False)
        
        # Compute loss components individually for breakdown
        controlEnergy = (u_curr ** 2).sum()
        controlSmooth = ((u_curr[1:] - u_curr[:-1]) ** 2).sum() if len(u_curr) > 1 else torch.zeros((), device=self.device)
        terminalVelCost = (vs_final[-1] ** 2).sum()
        rXY = qs_final[-1][:2] - goal[:2]
        terminalXYCost = (rXY ** 2).sum()
        rTheta = qs_final[-1][2] - goal[2]
        while rTheta > np.pi:
            rTheta -= 2 * np.pi
        while rTheta < -np.pi:
            rTheta += 2 * np.pi
        terminalOrientCost = rTheta ** 2
        
        # Extract final gradient (if needed for diagnostics)
        u_final_ad = u_curr.clone().detach().requires_grad_(True)
        result_for_grad = self.evaluate_cost(u_final_ad, q0, v0, pusher0, goal, w)
        loss_for_grad = result_for_grad["loss"]
        loss_for_grad.backward()
        final_grad = u_final_ad.grad if u_final_ad.grad is not None else torch.zeros_like(u_curr)
        
        if verbose:
            print(f"\n{'='*70}")
            print("Optimization Complete!")
            print(f"{'='*70}")
            print(f"Time: {solve_time:.2f}s")
            print(f"Final loss: {loss_final.item():.6f}")
            print(f"Final pose: {qs_final[-1].cpu().numpy()}")
            print(f"Goal: {goal.cpu().numpy()}")
            
            pos_error = torch.norm(qs_final[-1][:2] - goal[:2]).item()
            orient_error = abs(rTheta.item())
            print(f"Position error: {pos_error:.6f}m")
            print(f"Orientation error: {orient_error:.6f}rad")
        
        # Return dict matching MS format EXACTLY
        return {
            # Optimized trajectory (matching MS field names)
            "loss": loss_final.detach().cpu(),
            "q_final": qs_final[-1].detach().cpu().numpy(),
            "u_seq": u_curr.detach().cpu().numpy(),
            "trajectory": qs_final_full.detach().cpu().numpy(),
            "velocity_trajectory": vs_final.detach().cpu().numpy(),
            "pusher_trajectory": prs_final.detach().cpu().numpy(),
            "prKnots": None,  # SS doesn't have knots
            "contact_forces": contact_forces_final.detach().cpu().numpy(),  # REAL DATA!
            "signed_distances": signed_distances_final.detach().cpu().numpy(),  # REAL DATA!
            
            # Initial trajectory (for visualization comparison)
            "initial_trajectory": qs_init.cpu().numpy(),
            "initial_velocity_trajectory": vs_init.cpu().numpy(),
            "initial_pusher_trajectory": prs_init.cpu().numpy(),
            
            # Loss components (matching MS structure)
            "loss_components": {
                "total": loss_final.item(),
                "control_energy": controlEnergy.item(),
                "control_smooth": controlSmooth.item(),
                "obj_vel": terminalVelCost.item(),  # Terminal velocity (not cumulative)
                "target_xy": terminalXYCost.item(),
                "target_orient": terminalOrientCost.item(),
                # Weights for legend
                "w_control": w.wControl,
                "w_smooth": w.wControlSmooth,
                "w_objvel": w.wObjVel,
                "w_targetxy": w.wTargetXY,
                "w_orient": w.wTargetOrient,
            },
            
            # Final control gradient
            "control_gradients": final_grad.detach().cpu().numpy(),
            
            # History and diagnostics
            "history": {
                'loss': np.array(history['loss']),
                'step_norm': np.array(history['step_norm'].detach.cpu().numpy()),
                'cost_change': np.array(history['cost_change']),
                'alpha': np.array(history['alpha']),
            },
            "stationarityInfo": {
                "final_defect_norm": 0.0,  # SS has no defect (exact forward sim)
                "final_terminal_error": torch.norm(rXY).item(),
                "converged": step_norm < cfg.tol if iteration > 0 else False,
            },
            
            # Additional SS-specific info
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
        """
        Simple constant velocity initialization.
        Conservative to avoid overshoot.
        """
        # Required displacement
        displacement = goal[:2] - q0[:2]
        time_horizon = self.horizon * self.dt
        
        # Average velocity needed
        avg_velocity = displacement / time_horizon
        
        # Safety factor (conservative)
        safety_factor = 0.7
        u_const = avg_velocity * safety_factor
        
        # Constant control
        u_init = u_const.unsqueeze(0).repeat(self.horizon, 1)
        
        return u_init
