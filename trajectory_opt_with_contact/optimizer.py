import time
import torch
import numpy as np

from .dynamics import rollout
from .qp_solver import ContactQPSolver
from .transcription import DirectTranscriptionOptimizer  # NEW
from .iLQR import ILQROptimizer


class TrajectoryOptimizer:
    """
    Backward-compatible: default = your original shooting/Adam pipeline.
    New flags:
        - use_transcription: switch to direct transcription (implicit Euler)
        - use_second_order: use LBFGS (2nd-order) instead of Adam
        - qp_backend: "cvxpy" (old) or "qpth" (interior-point)
        - ipm_eps, ipm_max_iter: only used when qp_backend="qpth"
    """
    def __init__(self, mass=1.0, side_length=0.2, mu=0.6,
                horizon=60, dt=0.05, device='cuda',
                TO_solver: str = 'shooting', # or 'transcription"
                dynamics_solver: str = 'IP',
                w_target = 20.0, w_v = 0.1, w_ctrl = 1e-3, w_obs = 1.0,
                use_second_order: bool = False, #TO solve with 1st order or 2nd order method
                # ALM knobs (forwarded to DirectTranscriptionOptimizer):
                alm_enabled: bool = True, alm_rho_init: float = 1e2, alm_rho_max: float = 1e8,
                alm_eta: float = 10.0, alm_target_tol: float = 1e-6,
                alm_outer_iters: int = 10, lbfgs_inner_steps: int = 10, lbfgs_history: int = 10):
        self.side = side_length
        self.half = side_length / 2
        self.mu = mu
        self.horizon = horizon
        self.dt = dt
        self.m = mass
        self.Izz = (1.0/6.0) * mass * (self.side**2 + self.side**2)

        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            device = 'cpu'
        self.device = torch.device(device)

        self.TO_solver = TO_solver
        self.dynamics_solver = dynamics_solver

        # ALM (Transcription) Option
        self.use_second_order = use_second_order
        self.qp_solver = None 
        if self.dynamics_solver == 'LCP':
            # Low-level QP, build it here so that it doesn't need to be rebuilt
            self.qp_solver = ContactQPSolver(mu=self.mu, n_contacts=1,
                                            backend='cvxpy')

        if self.TO_solver == 'transcription':
            self.trans = DirectTranscriptionOptimizer(
                mass=self.m, side_length=self.side, mu=self.mu,
                horizon=self.horizon, dt=self.dt, device=device,
                use_second_order=self.use_second_order,
                dynamics_solver=self.dynamics_solver,
                qp_solver = self.qp_solver, #only used for LCP dynamics solver
                alm_enabled=alm_enabled, alm_rho_init=alm_rho_init, alm_rho_max=alm_rho_max,
                alm_eta=alm_eta, alm_target_tol=alm_target_tol,
                alm_outer_iters=alm_outer_iters, lbfgs_inner_steps=lbfgs_inner_steps,
                lbfgs_history=lbfgs_history
            )
            
    def _to_tensor(self, x, requires_grad=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double, device=self.device)
        else:
            x = x.to(dtype=torch.double, device=self.device)
        x.requires_grad = requires_grad
        return x

    def optimize(self, q0, v0, pusher0, goal,
                w_target = 20.0, w_orient = 1.0, w_v = 0.1, w_ctrl = 1e-3, w_obs = 1.0,
                 u_init=None, max_iters=100, lr=0.01,
                 lr_decay_step=10, lr_decay_gamma=0.5,
                 obstacle_pos=None, verbose=True):
        if self.TO_solver == 'transcription':
            return self.trans.optimize(
                q0=q0, v0=v0, pusher0=pusher0, goal=goal,
                u_init=u_init, max_iters=max_iters, lr=lr,
                obstacle_pos=obstacle_pos, verbose=verbose
            )
        elif self.TO_solver == 'shooting':
            # ---- Original shooting pipeline (unchanged) ----
            q0 = self._to_tensor(q0, requires_grad=True)
            v0 = self._to_tensor(v0, requires_grad=True)
            pusher0 = self._to_tensor(pusher0, requires_grad=True)
            goal = self._to_tensor(goal, requires_grad=False)
            if obstacle_pos is not None:
                obstacle_pos = self._to_tensor(obstacle_pos, requires_grad=False)

            if u_init is None:
                u_seq = torch.zeros(self.horizon, 2, dtype=torch.double,
                                    device=self.device, requires_grad=True)
            else:
                u_seq = self._to_tensor(u_init, requires_grad=True)



           
            # opt = torch.optim.LBFGS([u_seq], max_iter=20, history_size=10,
            #                         lr=lr, line_search_fn="strong_wolfe")

            from .dynamics import rollout
            def to_scalar(x):
                return x.item() if torch.is_tensor(x) else float(x)
             # ============= Choose optimizer =============
            if self.use_second_order:
                # LBFGS for faster convergence / better stationarity
                opt = torch.optim.LBFGS(
                    [u_seq],
                    lr=1.0,                      # step size for line-search
                    max_iter=20,                 # max inner iters per .step()
                    history_size=20,             # you already have lbfgs_history if you want
                    line_search_fn="strong_wolfe", #"strong_wolfe" #
                    tolerance_grad=1e-4,
                    tolerance_change=1e-5,
                )
                state = {'grad_norm': None, 'loss_terms': None, 'q_final': None, 'lambdas': None, 'phis': None, 'qs': None, 'pusher_traj': None} 
                stallCounter = 0
                def closure():
                    opt.zero_grad()
                    loss, q_final, lambdas, phis, qs, pusher_traj, goal_term, orient_term, ctrl_term, v_term, obs_term, pen_term = rollout(
                            u_seq, q0, v0, pusher0,
                            self.horizon, self.dt,
                            self.m, self.Izz, self.half, self.mu, goal,
                            w_target=w_target, w_v=w_v,
                            w_ctrl=w_ctrl, w_obs=w_obs,
                            qp_solver=self.qp_solver,
                            dynamics_solver=self.dynamics_solver,
                            obstacle_pos=obstacle_pos,
                            device=self.device
                        )
                    loss.backward()
                    # Save grad norm and decomposition for logging
                    with torch.no_grad():
                        gnorm = u_seq.grad.norm().item()
                    state['grad_norm'] = gnorm
                    state['loss_terms'] = (loss.item(),
                                        to_scalar(goal_term),
                                        to_scalar(orient_term),
                                        to_scalar(ctrl_term),
                                        to_scalar(v_term),
                                        to_scalar(obs_term),
                                        to_scalar(pen_term))
                    state['q_final'] = q_final
                    state['lambdas'] = lambdas
                    state['phis'] = phis
                    state['qs'] = qs
                    state['pusher_traj'] = pusher_traj
                    return loss

                # Outer loop just to inspect progress
                for it in range(max_iters):
                    # Track previous iterate
                    prev_u = u_seq.detach().clone()

                    # One quasi-Newton step
                    loss = opt.step(closure)

                    # Step size in control space
                    with torch.no_grad():
                        du = (u_seq.detach() - prev_u).norm().item()

                    total, goal_t, orient_t, ctrl_t, v_t, obs_t, pen_t = state['loss_terms']
                    grad_norm = state['grad_norm']
                    q_final = state['q_final']
                    lambdas = state['lambdas']
                    phis = state['phis']
                    qs = state['qs']
                    pusher_traj = state['pusher_traj']

                    if du < 1e-12:
                        stallCounter += 1
                    else:
                        stallCounter = 0

                    if verbose:
                        print(
                            f"[Shooting-LBFGS] Iter {it+1:3d} | "
                            f"Total={total:.4f} | "
                            f"Goal={goal_t:.4f} | "
                            f"Orient={orient_t:.4f} | "
                            f"Ctrl={ctrl_t:.6f} | "
                            f"Vel={v_t:.4f} | "
                            f"Obs={obs_t:.4f} | "
                            f"Pen={pen_t:.4f} | "
                            f"||∇_u J||={grad_norm:.3e} | "
                            f"||Δu||={du:.3e}"
                        )
                        print(f"Target Pos: {[f'{x:.3f}' for x in goal.tolist()]}, "
                                f"Final pos: {[f'{x:.3f}' for x in q_final.tolist()]}")

                    # Reasonable stopping criteria
                    if grad_norm < 1e-4 and du < 1e-4 or stallCounter >= 5:                        
                        if verbose:
                            print(f"[Shooting-LBFGS] Early stop, "
                                f"grad_norm={grad_norm:.3e}, step={du:.3e}")
                            if stallCounter >= 5:
                                print(f"  Stopping due to step stalled for 5 consecutive iterations.")
                        break
                # print(f"LBFGS time taken: {time.time() - time_start:.2f} seconds")
                print("sensitivity of final loss to u0:", u_seq.grad[0].detach().cpu())
            else:
                opt = torch.optim.Adam([u_seq], lr=lr)
                use_fixed_schedule = True #True #
                best_loss = float('inf')
                patience = 100
                no_improve = 0
                time_start = time.time()
                if use_fixed_schedule:
                    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay_gamma)
                else:
                    lr_current = lr
                    min_lr, max_lr = 1e-5, lr              
                for it in range(max_iters):
                    opt.zero_grad()

                    loss, q_final, lambdas, phis, qs, pusher_traj, goal_term, orient_term, ctrl_term, v_term, obs_term, pen_term = rollout(
                        u_seq, q0, v0, pusher0, self.horizon, self.dt,
                        self.m, self.Izz, self.half, self.mu, goal,
                        w_target = w_target, w_orient = w_orient, w_v = w_v, w_ctrl = w_ctrl, w_obs = w_obs,
                        qp_solver = self.qp_solver,
                        dynamics_solver=self.dynamics_solver, obstacle_pos=obstacle_pos,
                        device=self.device
                    )

                    # (grad_u,) = torch.autograd.grad(loss, u_seq, create_graph=True, retain_graph=True)
                    # print('KKT_stationarity:', grad_u.norm().item())

                    loss.backward()
                    grad_norm = u_seq.grad.norm().item()
                    if verbose and (it+1) % 10 == 0:
                        print(f"||grad_u||={grad_norm:.4e}")

                    if not use_fixed_schedule:
                        # --- Adaptive LR based on grad norm ---
                        # If gradient is exploding, shrink LR
                        if grad_norm > 10.0:
                            lr_current = max(lr_current * 0.5, min_lr)
                        # If gradient is tiny but we haven't converged, you can gently shrink LR
                        elif grad_norm < 1e-3:
                            lr_current = max(lr_current * 0.8, min_lr)
                        # Optionally, if gradient is moderate and loss is improving, grow LR a bit
                        elif grad_norm < 1.0:
                            lr_current = min(lr_current * 1.05, max_lr)

                        for g in opt.param_groups:
                            g['lr'] = lr_current
                        # --- End adaptive LR ---
                    opt.step()
                    if use_fixed_schedule:
                        sched.step()
                        lr_current = sched.get_last_lr()[0]  # Get the current learning rate from the scheduler for logging

                    # Track loss improvement to avoid wasting time                    
                    if loss.item() + 1e-6 < best_loss:
                        best_loss = loss.item()
                        no_improve = 0
                    else:
                        no_improve += 1
                    if verbose and (it+1) % 10 == 0:
                        print(f"[Shooting] Iter {it+1:3d} | "
                            f"Total={to_scalar(loss):.4f} | "
                            f"Goal={to_scalar(goal_term):.4f} | "
                            f"Orient={to_scalar(orient_term):.4f} | "
                            f"Ctrl={to_scalar(ctrl_term):.6f} | "
                            f"Vel={to_scalar(v_term):.4f} | "
                            f"Obs={to_scalar(obs_term):.4f} | "
                            f"Pen={to_scalar(pen_term):.4f} | "
                            f"LR={lr_current:.3e} | "
                            f"||∇_u J||={grad_norm:.3e} | "
                            #f"||Δu||={du:.3e}"
                            )
                        print(f"Target Pos: {[f'{x:.3f}' for x in goal.tolist()]}, "
                                f"Final pos: {[f'{x:.3f}' for x in q_final.tolist()]}")

                    # Early stopping
                    if grad_norm < 1e-4 or no_improve >= patience:
                        if verbose:
                            print(f"[Shooting-Adam] Early stop at iter {it+1}, "
                                f"grad_norm={grad_norm:.3e}, "
                                f"no_improve={no_improve}")
                        break
                print(f"Adam time taken: {time.time() - time_start:.2f} seconds")
            return {
                'loss': loss.item(),
                'q_final': q_final.detach().cpu().numpy(),
                'u_seq': u_seq.detach().cpu().numpy(),
                'trajectory': qs.detach().cpu().numpy(),
                'pusher_trajectory': pusher_traj.detach().cpu().numpy(),
                'contact_forces': lambdas.detach().cpu().numpy(),
                'signed_distances': phis.detach().cpu().numpy()
            }
        elif self.TO_solver == 'iLQR':
            self.ilqr = ILQROptimizer(
                mass=self.m, side_length=self.side, mu=self.mu,
                w_target = w_target, w_v = w_v, w_ctrl = w_ctrl, w_obs = w_obs,
                horizon=self.horizon, dt=self.dt, device=self.device,
                dynamics_solver=self.dynamics_solver,
                qp_solver=self.qp_solver,
                obstacle_pos=obstacle_pos,
                max_iters=max_iters, regularization=1e-4
            )
            # iLQR uses its own iteration budget; we pass u_init if provided
            q0_t = self._to_tensor(q0, requires_grad=False)
            v0_t = self._to_tensor(v0, requires_grad=False)
            p0_t = self._to_tensor(pusher0, requires_grad=False)
            goal_t = self._to_tensor(goal, requires_grad=False)
            u0_t = None if u_init is None else self._to_tensor(u_init, requires_grad=False)
            out = self.ilqr.optimize(q0_t, v0_t, p0_t, goal_t, u_init=u0_t, verbose=verbose)
            # For API consistency with shooting, return a similar dict
            X = out["trajectory"]  # (T+1,8)
            return {
            'loss': out["loss"],
            'q_final': X[-1],                           # (3,)
            'u_seq': out["u_seq"],                      # (T,2)
            'trajectory': X,                            # (T+1,3)
            'pusher_trajectory': out["pusher_sequence"],# (T+1,2)
            'contact_forces': out["contact_forces"],    # list length T
            'signed_distances': out["signed_distances"] # list length T
        }

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
    

    def compute_geometric_initial_trajectory_position(
        self,
        robot_pos,      # [x, y] - initial robot position
        box_pos,        # [x, y] or [x, y, theta] - initial box position  
        goal_pos,       # [x, y] or [x, y, theta] - goal position
    ):
        """
        Create initial robot position trajectory based on geometric path: Robot -> Box -> Goal
        
        Uses the optimizer's horizon, dt, and half (box_half_size) automatically.
        
        Args:
            robot_pos: Initial robot position [x, y]
            box_pos: Initial box position [x, y] or [x, y, theta] (only x, y used)
            goal_pos: Goal position [x, y] or [x, y, theta] (only x, y used)
        
        Returns:
            robot_traj: Robot position trajectory as list [[x, y], ...] (horizon x 2)
                       Positions at t=1, ..., horizon (initial position at t=0 is not included)
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
            print(f"  Already at goal! Using stationary positions.")
            return [[float(robot_pos[0]), float(robot_pos[1])]] * self.horizon
        
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
        
        # Phase 1: Robot approaches box contact point
        dir_to_box = (box_pos - robot_pos) / (dist_robot_to_box + 1e-8)
        contact_offset = 0.0001  # self.half  # Slightly before contact
        target_contact_pos = box_pos - dir_to_box * contact_offset
        
        # Phase 2: Robot pushes box to goal (robot stays behind box)
        dir_to_goal = (goal_pos - box_pos) / (dist_box_to_goal + 1e-8)
        # Robot final position: slightly behind the goal
        target_final_pos = goal_pos - dir_to_goal * contact_offset
        
        # Create position trajectory
        robot_traj = []
        
        # Phase 1: Linear interpolation from robot_pos to target_contact_pos
        for i in range(steps_phase1):
            alpha = (i + 1) / steps_phase1  # alpha from 0 to 1
            pos = (1 - alpha) * robot_pos + alpha * target_contact_pos
            robot_traj.append([float(pos[0]), float(pos[1])])
        
        # Phase 2: Linear interpolation from target_contact_pos to target_final_pos
        for i in range(steps_phase2):
            alpha = (i + 1) / steps_phase2  # alpha from 0 to 1
            pos = (1 - alpha) * target_contact_pos + alpha * target_final_pos
            robot_traj.append([float(pos[0]), float(pos[1])])
        
        print(f"  Generated robot position trajectory: {len(robot_traj)} x 2")
        print(f"  Start position: [{robot_pos[0]:.4f}, {robot_pos[1]:.4f}]")
        print(f"  Contact position: [{target_contact_pos[0]:.4f}, {target_contact_pos[1]:.4f}]")
        print(f"  Final position: [{robot_traj[-1][0]:.4f}, {robot_traj[-1][1]:.4f}]")
        
        # Compute displacement statistics
        positions = [robot_pos] + [np.array(p) for p in robot_traj]
        displacements = [np.linalg.norm(positions[i+1] - positions[i]) 
                        for i in range(len(robot_traj))]
        print(f"  Per-step displacement range: [{min(displacements):.4f}, {max(displacements):.4f}] m")
        print(f"  Total displacement: {np.linalg.norm(np.array(robot_traj[-1]) - robot_pos):.4f} m")
        
        return robot_traj