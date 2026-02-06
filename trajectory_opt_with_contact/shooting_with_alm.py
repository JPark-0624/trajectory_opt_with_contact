"""
Shooting-based Trajectory Optimization with Augmented Lagrangian Method (ALM)

Decision variables: Robot position trajectory
Constraints: Velocity magnitude inequality constraints
Method: ALM with nested optimization (outer: dual update, inner: primal optimization)
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
from .dynamics import rollout
from .qp_solver import ContactQPSolver


class ShootingWithALM:
    """
    Single-shooting trajectory optimizer with position control and velocity constraints.
    
    Key features:
    - Decision variables: Robot positions u_seq [H, 2]
    - Dynamics: Forward simulation via rollout (shooting method)
    - Constraints: Velocity magnitude ||v_k|| ≤ v_max (inequality)
    - Optimization: ALM (Augmented Lagrangian Method)
    
    ALM Structure:
        Outer loop: Update Lagrange multipliers (mu) and penalty parameter (rho)
        Inner loop: Optimize u_seq with fixed mu, rho using Adam or LBFGS
    """
    
    def __init__(self, 
                 mass: float = 1.0,
                 side_length: float = 0.2,
                 mu_friction: float = 0.5,
                 horizon: int = 100,
                 dt: float = 0.05,
                 device: str = 'cuda',
                 dynamics_solver: str = 'IP',
                 qp_solver = None,
                 # Inner optimizer
                 use_second_order: bool = False,
                 lbfgs_inner_steps: int = 20,
                 lbfgs_history: int = 20,
                 # ALM parameters
                 alm_rho_init: float = 1.0,
                 alm_rho_max: float = 1e6,
                 alm_eta: float = 10.0,
                 alm_target_tol: float = 1e-4,
                 alm_outer_iters: int = 10,
                 # Velocity constraint
                 v_max: float = 1.0,
                 # Optional: acceleration constraint
                 enable_accel_constraint: bool = False,
                 a_max: float = 5.0):
        
        self.m = mass
        self.side = side_length
        self.half = side_length / 2
        self.mu = mu_friction
        self.horizon = horizon
        self.dt = dt
        self.Izz = (1.0/6.0) * mass * (side_length**2 + side_length**2)
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            device = 'cpu'
        self.device = torch.device(device)
        
        # Dynamics
        self.dynamics_solver = dynamics_solver
        self.qp_solver = qp_solver
        
        # Inner optimizer
        self.use_second_order = use_second_order
        self.lbfgs_inner_steps = lbfgs_inner_steps
        self.lbfgs_history = lbfgs_history
        
        # ALM parameters
        self.alm_rho_init = alm_rho_init
        self.alm_rho_max = alm_rho_max
        self.alm_eta = alm_eta
        self.alm_target_tol = alm_target_tol
        self.alm_outer_iters = alm_outer_iters
        
        # Constraints
        self.v_max = v_max
        self.enable_accel_constraint = enable_accel_constraint
        self.a_max = a_max
        
        # Lagrange multipliers (initialized in optimize())
        self._mu_vel = None  # For velocity constraints [H]
        self._mu_accel = None  # For acceleration constraints [H-1]
    
    def _to_tensor(self, x, requires_grad=False):
        """Convert to torch tensor on device"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double, device=self.device)
        else:
            x = x.to(dtype=torch.double, device=self.device)
        x.requires_grad = requires_grad
        return x
    
    def _compute_velocity_constraints(self, 
                                     u_seq: torch.Tensor, 
                                     pr0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity constraint violations.
        
        Args:
            u_seq: Robot positions [H, 2]
            pr0: Initial robot position [2]
        
        Returns:
            v_seq: Velocities [H, 2]
            c_vel: Constraint violations [H] (c_k = ||v_k|| - v_max)
        """
        # Compute velocities: v_k = (u_k - u_{k-1}) / dt
        u_full = torch.cat([pr0.unsqueeze(0), u_seq], dim=0)  # [H+1, 2]
        v_seq = (u_full[1:] - u_full[:-1]) / self.dt  # [H, 2]
        
        # Constraint: c_k = ||v_k|| - v_max ≤ 0
        v_magnitude = torch.norm(v_seq, dim=1)  # [H]
        c_vel = v_magnitude - self.v_max  # [H]
        
        return v_seq, c_vel
    
    def _compute_acceleration_constraints(self, 
                                         v_seq: torch.Tensor) -> torch.Tensor:
        """
        Compute acceleration constraint violations.
        
        Args:
            v_seq: Velocities [H, 2]
        
        Returns:
            c_accel: Constraint violations [H-1] (c_k = ||a_k|| - a_max)
        """
        # Compute accelerations: a_k = (v_{k+1} - v_k) / dt
        a_seq = (v_seq[1:] - v_seq[:-1]) / self.dt  # [H-1, 2]
        
        # Constraint: c_k = ||a_k|| - a_max ≤ 0
        a_magnitude = torch.norm(a_seq, dim=1)  # [H-1]
        c_accel = a_magnitude - self.a_max  # [H-1]
        
        return c_accel
    
    def optimize(self,
                 q0,  # Initial box configuration [x, y, theta]
                 v0,  # Initial box velocity [vx, vy, omega]
                 pusher0,  # Initial pusher position [x, y]
                 goal,  # Goal configuration [x, y, theta]
                 u_init=None,  # Initial robot position trajectory
                 # Cost weights
                 w_target: float = 20.0,
                 w_orient: float = 1.0,
                 w_v: float = 0.1,
                 w_ctrl: float = 0.01,
                 w_obs: float = 1.0,
                 # Optimizer settings
                 max_iters_inner: int = 30,
                 lr: float = 0.01,
                 lr_decay_step: int = 10,
                 lr_decay_gamma: float = 0.8,
                 obstacle_pos=None,
                 verbose: bool = True) -> Dict:
        """
        Optimize trajectory with ALM for velocity constraints.
        
        Returns:
            Dictionary with optimization results including:
            - loss: Task loss
            - q_final: Final configuration
            - u_seq: Optimized robot positions
            - constraint_violations: Velocity constraint violations
            - dual_variables: Lagrange multipliers
        """
        
        # Convert to tensors
        q0 = self._to_tensor(q0)
        v0 = self._to_tensor(v0)
        pr0 = self._to_tensor(pusher0)
        goal = self._to_tensor(goal)
        if obstacle_pos is not None:
            obstacle_pos = self._to_tensor(obstacle_pos)
        
        # Initialize decision variables (robot positions)
        if u_init is None:
            u_seq = torch.zeros(self.horizon, 2, dtype=torch.double,
                              device=self.device, requires_grad=True)
        else:
            u_seq = self._to_tensor(u_init, requires_grad=True)
        
        # Initialize Lagrange multipliers
        if self._mu_vel is None or self._mu_vel.shape[0] != self.horizon:
            self._mu_vel = torch.zeros(self.horizon, device=self.device)
        if self.enable_accel_constraint:
            if self._mu_accel is None or self._mu_accel.shape[0] != self.horizon - 1:
                self._mu_accel = torch.zeros(self.horizon - 1, device=self.device)
        
        # Initialize penalty parameter
        rho = self.alm_rho_init

        # Create optimizer
        if self.use_second_order:
            opt = torch.optim.LBFGS(
                [u_seq],
                lr=lr,
                max_iter=20,
                history_size=self.lbfgs_history,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-5,
                tolerance_change=1e-6
            )
            # State for LBFGS closure
            state = {
                'task_loss': None,
                'alm_loss': None,
                'constraint_viol': None,
                'q_final': None
            }
        else:
            opt = torch.optim.Adam([u_seq], lr=lr)
            total_steps = self.alm_outer_iters * max_iters_inner
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps, eta_min=lr*0.01
            )

        # ============================================
        # ALM Outer Loop
        # ============================================
        best_result = {
            'alm_loss': float('inf'),
            'task_loss': float('inf'),
            'u_seq': None,
            'violation': float('inf')
        }
        
        for outer_iter in range(self.alm_outer_iters):
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"[ALM Outer {outer_iter+1}/{self.alm_outer_iters}] rho={rho:.2e}")
                print(f"{'='*70}")
            
            # ============================================
            # Inner Optimization Loop
            # ============================================       
            
            # Inner loop iterations
            if self.use_second_order:
                # LBFGS
                for inner_iter in range(self.lbfgs_inner_steps):
                    def closure():
                        """Closure for LBFGS optimizer"""
                        opt.zero_grad()
                        
                        # Forward simulation via rollout
                        loss, q_final, lambdas, phis, qs, pusher_traj, \
                            goal_term, orient_term, ctrl_term, v_term, obs_term, pen_term = \
                            rollout(
                                u_seq, q0, v0, pr0,
                                self.horizon, self.dt,
                                self.m, self.Izz, self.half, self.mu, goal,
                                w_target=w_target, w_orient=w_orient,
                                w_v=w_v, w_ctrl=w_ctrl, w_obs=w_obs,
                                qp_solver=self.qp_solver,
                                dynamics_solver=self.dynamics_solver,
                                obstacle_pos=obstacle_pos,
                                device=self.device
                            )
                        
                        # Velocity constraints
                        v_seq, c_vel = self._compute_velocity_constraints(u_seq, pr0)
                        c_vel_violation = torch.relu(c_vel)  # max(0, c)
                        
                        # ALM term for velocity: mu^T * c^+ + (rho/2) * ||c^+||^2
                        alm_vel_term = (self._mu_vel * c_vel_violation).sum() + \
                                    (rho / 2) * (c_vel_violation ** 2).sum()
                        
                        # Acceleration constraints (optional)
                        alm_accel_term = 0.0
                        if self.enable_accel_constraint:
                            c_accel = self._compute_acceleration_constraints(v_seq)
                            c_accel_violation = torch.relu(c_accel)
                            alm_accel_term = (self._mu_accel * c_accel_violation).sum() + \
                                            (rho / 2) * (c_accel_violation ** 2).sum()
                        
                        # Total augmented Lagrangian
                        total_loss = loss + alm_vel_term + alm_accel_term
                        
                        # Backward pass
                        total_loss.backward()
                        
                        # Save state
                        with torch.no_grad():
                            state['task_loss'] = loss.item()
                            state['alm_loss'] = total_loss.item()
                            state['constraint_viol'] = c_vel_violation.max().item()
                            state['q_final'] = q_final.detach().clone()
                        
                        return total_loss
                    
                    opt.step(closure)
                    
                    if verbose and inner_iter % 5 == 0:
                        print(f"  [Inner LBFGS {inner_iter+1:2d}] "
                              f"Task={state['task_loss']:.4f}, "
                              f"ALM={state['alm_loss']:.4f}, "
                              f"MaxViol={state['constraint_viol']:.4e}")
            else:
                # Adam
                for inner_iter in range(max_iters_inner):
                    opt.zero_grad()
                    
                    # Forward simulation
                    loss, q_final, lambdas, phis, qs, pusher_traj, \
                        goal_term, orient_term, ctrl_term, v_term, obs_term, pen_term, solvedRatio = \
                        rollout(
                            u_seq, q0, v0, pr0,
                            self.horizon, self.dt,
                            self.m, self.Izz, self.half, self.mu, goal,
                            w_target=w_target, w_orient=w_orient,
                            w_v=w_v, w_ctrl=w_ctrl, w_obs=w_obs,
                            qp_solver=self.qp_solver,
                            dynamics_solver=self.dynamics_solver,
                            obstacle_pos=obstacle_pos,
                            device=self.device
                        )
                    
                    # Velocity constraints
                    v_seq, c_vel = self._compute_velocity_constraints(u_seq, pr0)
                    c_vel_violation = torch.relu(c_vel)
                    
                    # ALM term
                    alm_vel_term = (self._mu_vel * c_vel_violation).sum() + \
                                  (rho / 2) * (c_vel_violation ** 2).sum()
                    
                    # Acceleration constraints (optional)
                    alm_accel_term = 0.0
                    if self.enable_accel_constraint:
                        c_accel = self._compute_acceleration_constraints(v_seq)
                        c_accel_violation = torch.relu(c_accel)
                        alm_accel_term = (self._mu_accel * c_accel_violation).sum() + \
                                        (rho / 2) * (c_accel_violation ** 2).sum()
                    
                    total_loss = loss + alm_vel_term + alm_accel_term
                    
                    # Backward & step
                    total_loss.backward()
                    opt.step()
                    scheduler.step()
                    
                    if verbose and (inner_iter + 1) % 10 == 0:
                        print(f"  [Inner Adam {inner_iter+1:2d}] "
                              f"Task={loss.item():.4f}, "
                              f"ALM={total_loss.item():.4f}, "
                              f"MaxViol={c_vel_violation.max().item():.4e}, "
                              f"lr={scheduler.get_last_lr()[0]:.4e}, "
                              f"grad_norm={u_seq.grad.norm().item():.4e}, ")
                        print(f"  Terms: "
                              f"goal_term={goal_term:.4f} | "
                              f"orient_term={orient_term:.4f} | "
                              f"ctrl_term={ctrl_term:.4f} | "
                              f"v_term={v_term:.4f} | "
                              f"obs_term={obs_term:.4f} | "
                              f"pen_term={pen_term:.4f}"
                              )
                        print(f"Target Pos: {[f'{x:.3f}' for x in goal.tolist()]}, "
                                f"Final pos: {[f'{x:.3f}' for x in q_final.tolist()]}")
            
            # ============================================
            # Evaluate constraints & update dual variables
            # ============================================
            with torch.no_grad():
                # Final evaluation
                loss, q_final, lambdas, phis, qs, pusher_traj, \
                    goal_term, orient_term, ctrl_term, v_term, obs_term, pen_term, solvedRatio = \
                    rollout(
                        u_seq, q0, v0, pr0,
                        self.horizon, self.dt,
                        self.m, self.Izz, self.half, self.mu, goal,
                        w_target=w_target, w_orient=w_orient,
                        w_v=w_v, w_ctrl=w_ctrl, w_obs=w_obs,
                        qp_solver=self.qp_solver,
                        dynamics_solver=self.dynamics_solver,
                        obstacle_pos=obstacle_pos,
                        device=self.device
                    )
                
                # Constraint violations
                v_seq, c_vel = self._compute_velocity_constraints(u_seq, pr0)
                c_vel_violation = torch.relu(c_vel)
                
                max_vel_violation = c_vel_violation.max().item()
                mean_vel_violation = c_vel_violation.mean().item()
                num_violated = (c_vel_violation > 1e-6).sum().item()
                
                if self.enable_accel_constraint:
                    c_accel = self._compute_acceleration_constraints(v_seq)
                    c_accel_violation = torch.relu(c_accel)
                    max_accel_violation = c_accel_violation.max().item()
                    max_violation = max(max_vel_violation, max_accel_violation)
                else:
                    max_violation = max_vel_violation
                
                # Compute ALM loss
                alm_vel_term = (self._mu_vel * c_vel_violation).sum() + \
                              (rho / 2) * (c_vel_violation ** 2).sum()
                if self.enable_accel_constraint:
                    alm_accel_term = (self._mu_accel * c_accel_violation).sum() + \
                                    (rho / 2) * (c_accel_violation ** 2).sum()
                else:
                    alm_accel_term = 0.0
                
                alm_loss = loss.item() + alm_vel_term.item() + alm_accel_term
                
                # Print summary
                if verbose:
                    print(f"\n  [Summary]")
                    print(f"    Task Loss    : {loss.item():.6f}")
                    print(f"    ALM Loss     : {alm_loss:.6f}")
                    print(f"    Max Vel Viol : {max_vel_violation:.4e}")
                    print(f"    Mean Vel Viol: {mean_vel_violation:.4e}")
                    print(f"    # Violated   : {num_violated}/{self.horizon}")
                    if self.enable_accel_constraint:
                        print(f"    Max Acc Viol : {max_accel_violation:.4e}")
                    print(f"    mu_vel range : [{self._mu_vel.min().item():.3f}, "
                          f"{self._mu_vel.max().item():.3f}]")
                    print(f"    Final pos    : {[f'{x:.3f}' for x in q_final.tolist()]}")
                    print(f"    Goal pos     : {[f'{x:.3f}' for x in goal.tolist()]}")
                
                # Save best result
                if alm_loss < best_result['alm_loss']:
                    best_result.update({
                        'alm_loss': alm_loss,
                        'task_loss': loss.item(),
                        'u_seq': u_seq.detach().clone(),
                        'violation': max_violation,
                        'q_final': q_final.detach().clone(),
                        'lambdas': lambdas.detach().clone(),
                        'phis': phis.detach().clone(),
                        'qs': qs.detach().clone(),
                        'pusher_traj': pusher_traj.detach().clone()
                    })
                
                # ============================================
                # Dual variable update (Gradient ascent on dual)
                # ============================================
                # mu^{k+1} = max(0, mu^k + rho * c^+)
                self._mu_vel += rho * c_vel_violation
                self._mu_vel.clamp_(min=0.0, max=1e4)  # Prevent unbounded growth
                
                if self.enable_accel_constraint:
                    self._mu_accel += rho * c_accel_violation
                    self._mu_accel.clamp_(min=0.0, max=1e4)
                
                # ============================================
                # Penalty parameter update
                # ============================================
                if max_violation > self.alm_target_tol:
                    # Not converged, increase penalty
                    rho = min(rho * self.alm_eta, self.alm_rho_max)
                    if verbose:
                        print(f"    → Increasing rho to {rho:.2e}")
                
                # ============================================
                # Convergence check
                # ============================================
                if max_violation < self.alm_target_tol:
                    if verbose:
                        print(f"\n✓ Constraints satisfied! max_violation={max_violation:.4e} < {self.alm_target_tol:.4e}")
                    
        
        # ============================================
        # Return best solution
        # ============================================
        with torch.no_grad():
            # Compute velocity statistics for return
            v_seq, c_vel = self._compute_velocity_constraints(
                best_result['u_seq'], pr0
            )
            c_vel_violation = torch.relu(c_vel)
        
        return {
            'loss': best_result['task_loss'],
            'alm_loss': best_result['alm_loss'],
            'q_final': best_result['q_final'].cpu().numpy(),
            'u_seq': best_result['u_seq'].cpu().numpy(),
            'trajectory': best_result['qs'].cpu().numpy(),
            'pusher_trajectory': best_result['pusher_traj'].cpu().numpy(),
            'contact_forces': best_result['lambdas'].cpu().numpy(),
            'signed_distances': best_result['phis'].cpu().numpy(),
            'constraint_violations': c_vel_violation.cpu().numpy(),
            'dual_variables': self._mu_vel.cpu().numpy(),
            'velocities': v_seq.cpu().numpy()
        }
