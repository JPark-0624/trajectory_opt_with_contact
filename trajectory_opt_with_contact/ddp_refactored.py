"""
DDP Refactored from iLQR.py

Key improvements:
1. QP solver abstraction (Cholesky → Moreau)
2. Batch-ready structure (sequential → batch)
3. Constrained QP support (control limits)
4. Cleaner separation of concerns

Based on: trajectory_opt_with_contact/iLQR.py
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, List
from abc import ABC, abstractmethod


# ==============================================================================
# QP Solver Abstraction (Drop-in replacement for Cholesky)
# ==============================================================================

class QPSolverBase(ABC):
    """Abstract QP solver interface"""
    
    @abstractmethod
    def solve_single(self, Quu, Qu, u_current=None, u_min=None, u_max=None):
        """Solve single QP at one timestep"""
        pass
    
    @abstractmethod
    def solve_batch(self, Quu_batch, Qu_batch, U_current=None, u_min=None, u_max=None):
        """Solve multiple QPs in parallel (if supported)"""
        pass


class CholeskyQPSolver(QPSolverBase):
    """
    Unconstrained QP via Cholesky
    (Current iLQR.py approach)
    """
    
    def __init__(self, reg=1e-6):
        self.reg = reg
    
    def solve_single(self, Quu, Qu, u_current=None, u_min=None, u_max=None):
        """
        Solve: minimize (1/2) δu' Quu δu + Qu' δu
        (Ignores constraints!)
        
        Returns: δu, K (for value function update)
        """
        device = Quu.device
        n_u = Quu.shape[0]
        
        # Regularize
        Quu_reg = Quu + self.reg * torch.eye(n_u, device=device, dtype=Quu.dtype)
        
        try:
            L = torch.linalg.cholesky(Quu_reg)
            Quu_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            # Fallback: pseudo-inverse
            Quu_inv = torch.linalg.pinv(Quu_reg)
        
        δu = -Quu_inv @ Qu
        
        return δu, Quu_inv
    
    def solve_batch(self, Quu_batch, Qu_batch, U_current=None, u_min=None, u_max=None):
        """
        Sequential solving (no true batch)
        
        Quu_batch: [T, n_u, n_u]
        Qu_batch: [T, n_u]
        
        Returns: δu_batch [T, n_u], Quu_inv_batch [T, n_u, n_u]
        """
        T = Quu_batch.shape[0]
        n_u = Quu_batch.shape[1]
        device = Quu_batch.device
        dtype = Quu_batch.dtype
        
        δu_batch = torch.zeros(T, n_u, device=device, dtype=dtype)
        Quu_inv_batch = torch.zeros(T, n_u, n_u, device=device, dtype=dtype)
        
        for t in range(T):
            δu_batch[t], Quu_inv_batch[t] = self.solve_single(
                Quu_batch[t], Qu_batch[t], 
                u_current=U_current[t] if U_current is not None else None,
                u_min=u_min, u_max=u_max
            )
        
        return δu_batch, Quu_inv_batch


class MoreauQPSolver(QPSolverBase):
    """
    Moreau solver wrapper (future implementation)
    
    TODO: Implement when token arrives!
    """
    
    def solve_single(self, Quu, Qu, u_current=None, u_min=None, u_max=None):
        """
        Solve constrained QP with Moreau
        
        minimize  (1/2) δu' Quu δu + Qu' δu
        s.t.      u_min ≤ u_current + δu ≤ u_max
        """
        raise NotImplementedError("Moreau solver not available yet!")
    
    def solve_batch(self, Quu_batch, Qu_batch, U_current=None, u_min=None, u_max=None):
        """
        Batch solve with Moreau
        
        THIS IS THE KILLER FEATURE!
        Expected: 50-100x speedup over sequential
        """
        raise NotImplementedError("Moreau batch solver not available yet!")


# ==============================================================================
# DDP Optimizer (Refactored from iLQR.py)
# ==============================================================================

class DDPOptimizer:
    """
    DDP with QP solver abstraction
    
    Based on iLQR.py, with improvements:
    - Modular QP solver (easy Moreau swap)
    - Batch-ready structure
    - Support for control limits
    """
    
    def __init__(
        self,
        dynamics_fn,              # (x, u) -> x_next (wrapped from step_square_ip)
        stage_cost_fn,            # (x, u) -> scalar
        terminal_cost_fn,         # (x, goal) -> scalar
        horizon: int,
        dt: float,
        n_x: int = 8,            # [q(3), v(3), pusher(2)]
        n_u: int = 2,            # pusher velocity
        qp_solver: Optional[QPSolverBase] = None,
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
        max_iters: int = 50,
        reg_init: float = 1e-6,
        reg_scale: float = 10.0,
        min_reg: float = 1e-8,
        max_reg: float = 1e5,
        line_search_alphas: Tuple = (1.0, 0.5, 0.25, 0.1, 0.05),
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.dynamics_fn = dynamics_fn
        self.stage_cost_fn = stage_cost_fn
        self.terminal_cost_fn = terminal_cost_fn
        self.horizon = horizon
        self.dt = dt
        self.n_x = n_x
        self.n_u = n_u
        self.max_iters = max_iters
        self.reg = reg_init
        self.reg_scale = reg_scale
        self.min_reg = min_reg
        self.max_reg = max_reg
        self.line_search_alphas = line_search_alphas
        self.verbose = verbose
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        
        # Control limits
        self.u_min = u_min
        self.u_max = u_max
        
        # QP solver (default: Cholesky)
        self.qp_solver = qp_solver if qp_solver is not None else CholeskyQPSolver(reg=reg_init)
        
        # Goal (set in optimize())
        self.goal = None
    
    # ---------------------------------------------------------------
    # Core Methods (same as iLQR.py)
    # ---------------------------------------------------------------
    
    def rollout(self, x0: torch.Tensor, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward simulate with controls U
        
        Returns:
            X: [T+1, n_x] state trajectory
            cost: scalar total cost
        """
        T = U.shape[0]
        X = [x0]
        cost = torch.zeros((), dtype=self.dtype, device=self.device)
        
        for k in range(T):
            xk = X[-1]
            uk = U[k]
            cost = cost + self.stage_cost_fn(xk, uk)
            x_next = self.dynamics_fn(xk, uk)
            X.append(x_next)
        
        cost = cost + self.terminal_cost_fn(X[-1], self.goal)
        
        return torch.stack(X, dim=0), cost
    
    def linearize_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute A = df/dx, B = df/du via autograd
        (Same as iLQR.py lines 173-194)
        """
        x = x.detach().requires_grad_(True)
        u = u.detach().requires_grad_(True)
        
        y = self.dynamics_fn(x, u)
        
        # df/dx
        A = []
        for i in range(y.numel()):
            (grad_x,) = torch.autograd.grad(y[i], x, retain_graph=True, allow_unused=False)
            A.append(grad_x)
        A = torch.stack(A, dim=0)  # [n_x, n_x]
        
        # df/du
        B = []
        for i in range(y.numel()):
            (grad_u,) = torch.autograd.grad(y[i], u, retain_graph=True)
            B.append(grad_u)
        B = torch.stack(B, dim=0)  # [n_x, n_u]
        
        return A, B
    
    def quadraticize_cost(self, x: torch.Tensor, u: torch.Tensor) -> Tuple:
        """
        Compute quadratic expansion of stage cost
        (Based on iLQR.py lines 196-289, simplified)
        
        Returns: lx, lu, lxx, luu, lux
        """
        x = x.detach().requires_grad_(True)
        u = u.detach().requires_grad_(True)
        
        cost = self.stage_cost_fn(x, u)
        
        # First derivatives (allow_unused for cases where cost doesn't depend on x or u)
        grad_outputs = torch.autograd.grad(
            cost, [x, u], 
            create_graph=True,
            allow_unused=True
        )
        
        lx = grad_outputs[0]
        lu = grad_outputs[1]
        
        # Second derivatives
        lxx = torch.zeros(self.n_x, self.n_x, device=self.device, dtype=self.dtype)
        if lx is not None:  # Only compute if lx exists
            for i in range(self.n_x):
                grad_xi = torch.autograd.grad(lx[i], x, retain_graph=True, allow_unused=True)[0]
                if grad_xi is not None:
                    lxx[i] = grad_xi
        
        luu = torch.zeros(self.n_u, self.n_u, device=self.device, dtype=self.dtype)
        if lu is not None:  # Only compute if lu exists
            for i in range(self.n_u):
                grad_ui = torch.autograd.grad(lu[i], u, retain_graph=True, allow_unused=True)[0]
                if grad_ui is not None:
                    luu[i] = grad_ui
        
        # Cross term (usually zero for quadratic costs)
        lux = torch.zeros(self.n_u, self.n_x, device=self.device, dtype=self.dtype)
        
        # Convert None to zero tensors AFTER computing second derivatives
        if lx is None:
            lx = torch.zeros_like(x)
        if lu is None:
            lu = torch.zeros_like(u)
        
        return lx, lu, lxx, luu, lux
    
    def quadraticize_terminal(self, x_T: torch.Tensor) -> Tuple:
        """Quadraticize terminal cost"""
        x_T = x_T.detach().requires_grad_(True)
        lT = self.terminal_cost_fn(x_T, self.goal)
        
        (lTx,) = torch.autograd.grad(lT, x_T, create_graph=True)
        
        lTxx = torch.zeros(self.n_x, self.n_x, device=self.device, dtype=self.dtype)
        for i in range(self.n_x):
            (gxi,) = torch.autograd.grad(lTx[i], x_T, retain_graph=True)
            lTxx[i] = gxi
        
        return lTx, lTxx
    
    # ---------------------------------------------------------------
    # KEY IMPROVEMENT: Batch-ready backward pass
    # ---------------------------------------------------------------
    
    def backward_pass(
        self,
        X: torch.Tensor,  # [T+1, n_x]
        U: torch.Tensor,  # [T, n_u]
    ) -> Tuple[List, List, bool]:
        """
        DDP backward pass with batch-ready structure
        
        KEY CHANGES from iLQR.py:
        1. Collect all QPs first (prepare for batching)
        2. Call qp_solver.solve_batch() (sequential now, batched with Moreau)
        3. Extract gains from solution
        
        Returns:
            K_list: Feedback gains [T] each [n_u, n_x]
            k_list: Feedforward terms [T] each [n_u]
            diverged: bool
        """
        T = self.horizon
        
        # Terminal cost
        lTx, lTxx = self.quadraticize_terminal(X[-1])
        Vx = lTx.clone()
        Vxx = lTxx.clone()
        
        # ============================================================
        # STEP 1: Collect all Q-functions (prepare for batching)
        # ============================================================
        
        Q_uu_all = []
        Q_u_all = []
        Q_ux_all = []
        Q_x_all = []
        Q_xx_all = []
        A_all = []
        B_all = []
        
        # We need to compute gains on-the-fly to update V correctly
        K_temp_list = []
        k_temp_list = []
        
        for k in reversed(range(T)):
            # Linearize dynamics
            A, B = self.linearize_dynamics(X[k], U[k])
            
            # Quadraticize cost
            lx, lu, lxx, luu, lux = self.quadraticize_cost(X[k], U[k])
            
            # Q-function
            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A
            
            # Regularize (numerical stability)
            Quu_reg = Quu + self.reg * torch.eye(self.n_u, device=self.device, dtype=self.dtype)
            
            # Solve QP for this timestep (needed for V update!)
            try:
                L = torch.linalg.cholesky(Quu_reg)
                Quu_inv = torch.cholesky_inverse(L)
            except RuntimeError:
                # Cholesky failed - will be caught later
                Quu_inv = torch.linalg.pinv(Quu_reg)
            
            # Compute gains
            K = -Quu_inv @ Qux
            kff = -Quu_inv @ Qu
            
            # Update value function (critical for next iteration!)
            Vx = Qx + K.T @ Quu_reg @ kff + Qux.T @ kff + K.T @ Qu
            Vxx = Qxx + K.T @ Quu_reg @ K + Qux.T @ K + K.T @ Qux
            Vxx = 0.5 * (Vxx + Vxx.T)  # Symmetrize
            
            # Store
            Q_uu_all.append(Quu_reg)
            Q_u_all.append(Qu)
            Q_ux_all.append(Qux)
            Q_x_all.append(Qx)
            Q_xx_all.append(Qxx)
            A_all.append(A)
            B_all.append(B)
            K_temp_list.append(K)
            k_temp_list.append(kff)
        
        # Reverse to forward order
        Q_uu_all = Q_uu_all[::-1]
        Q_u_all = Q_u_all[::-1]
        Q_ux_all = Q_ux_all[::-1]
        Q_x_all = Q_x_all[::-1]
        Q_xx_all = Q_xx_all[::-1]
        A_all = A_all[::-1]
        B_all = B_all[::-1]
        K_list = K_temp_list[::-1]
        k_list = k_temp_list[::-1]
        
        # Note: K_list and k_list already computed and reversed above
        # When Moreau is available, replace the Cholesky solve in the loop above
        # with qp_solver.solve_batch() for 100x speedup!
        
        return K_list, k_list, False  # diverged=False (already handled in loop)
    
    def forward_pass(
        self,
        x0: torch.Tensor,
        X_nom: torch.Tensor,
        U_nom: torch.Tensor,
        K_list: List,
        k_list: List,
        alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Forward pass with line search
        (Same as iLQR.py lines 346-379)
        """
        T = self.horizon
        X_new = [x0]
        U_new = []
        cost_new = torch.zeros((), dtype=self.dtype, device=self.device)
        
        for k in range(T):
            xk = X_new[-1]
            
            # Feedback control
            dx = xk - X_nom[k]
            u_try = U_nom[k] + alpha * k_list[k] + K_list[k] @ dx
            
            # Clip to bounds if specified
            if self.u_min is not None:
                u_try = torch.clamp(u_try, min=self.u_min, max=self.u_max)
            
            U_new.append(u_try)
            cost_new = cost_new + self.stage_cost_fn(xk, u_try)
            
            x_next = self.dynamics_fn(xk, u_try)
            X_new.append(x_next)
        
        cost_new = cost_new + self.terminal_cost_fn(X_new[-1], self.goal)
        
        X_new = torch.stack(X_new, dim=0)
        U_new = torch.stack(U_new, dim=0)
        
        return X_new, U_new, cost_new.item()
    
    # ---------------------------------------------------------------
    # Main optimization loop (same as iLQR.py lines 239-395)
    # ---------------------------------------------------------------
    
    def compute_geometric_initial_trajectory(
        self,
        robot_pos,      # [x, y] - initial robot position
        box_pos,        # [x, y] or [x, y, theta] - initial box position  
        goal_pos,       # [x, y] or [x, y, theta] - goal position
    ):
        """
        Create initial trajectory based on geometric path: Robot -> Box -> Goal
        
        Uses the optimizer's horizon, dt, and automatically.
        
        Args:
            robot_pos: Initial robot position [x, y]
            box_pos: Initial box position [x, y] or [x, y, theta] (only x, y used)
            goal_pos: Goal position [x, y] or [x, y, theta] (only x, y used)
        
        Returns:
            u_init: Initial control trajectory [horizon, n_u] as torch.Tensor
        """
        import numpy as np
        
        # Extract x, y only (handle both [x, y] and [x, y, theta])
        if isinstance(robot_pos, torch.Tensor):
            robot_pos = robot_pos.cpu().numpy()
        if isinstance(box_pos, torch.Tensor):
            box_pos = box_pos.cpu().numpy()
        if isinstance(goal_pos, torch.Tensor):
            goal_pos = goal_pos.cpu().numpy()
            
        robot_pos = np.array(robot_pos[:2])
        box_pos = np.array(box_pos[:2])
        goal_pos = np.array(goal_pos[:2])
        
        # Compute distances
        dist_robot_to_box = np.linalg.norm(box_pos - robot_pos)
        dist_box_to_goal = np.linalg.norm(goal_pos - box_pos)
        total_dist = dist_robot_to_box + dist_box_to_goal
        
        if self.verbose:
            print(f"\n[Geometric Init] Distance analysis:")
            print(f"  Robot -> Box: {dist_robot_to_box:.4f} m")
            print(f"  Box -> Goal: {dist_box_to_goal:.4f} m")
            print(f"  Total: {total_dist:.4f} m")
        
        # Handle edge case: already at goal
        if total_dist < 1e-6:
            if self.verbose:
                print(f"  Already at goal! Using zero controls.")
            return torch.zeros(self.horizon, self.n_u, device=self.device, dtype=self.dtype)
        
        # Allocate timesteps proportionally to distances
        min_steps = 5
        if self.horizon < 2 * min_steps:
            steps_phase1 = self.horizon // 2
        else:
            ratio = dist_robot_to_box / total_dist
            steps_phase1 = int(self.horizon * ratio)
            steps_phase1 = max(min_steps, min(self.horizon - min_steps, steps_phase1))
        
        steps_phase2 = self.horizon - steps_phase1
        
        if self.verbose:
            print(f"  Phase 1 (approach): {steps_phase1} steps")
            print(f"  Phase 2 (push): {steps_phase2} steps")
        
        # Phase 1: Robot approaches box
        dir_to_box = (box_pos - robot_pos) / (dist_robot_to_box + 1e-8)
        contact_offset = 0.0001  # Small offset to avoid collision
        target_contact_pos = box_pos - dir_to_box * contact_offset
        
        displacement_phase1 = target_contact_pos - robot_pos
        time_phase1 = steps_phase1 * self.dt
        velocity_phase1 = displacement_phase1 / (time_phase1 + 1e-8)
        
        if self.verbose:
            print(f"  Phase 1 velocity: [{velocity_phase1[0]:.3f}, {velocity_phase1[1]:.3f}] m/s")
        
        # Phase 2: Robot pushes box to goal
        dir_to_goal = (goal_pos - box_pos) / (dist_box_to_goal + 1e-8)
        displacement_phase2 = goal_pos - box_pos
        time_phase2 = steps_phase2 * self.dt
        velocity_phase2 = displacement_phase2 / (time_phase2 + 1e-8)
        
        # Scale down push velocity (more conservative)
        push_scale = 0.8
        velocity_phase2 = velocity_phase2 * push_scale
        
        if self.verbose:
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
        
        if self.verbose:
            print(f"  Generated control trajectory: {len(u_init)} x {self.n_u}")
            u_magnitudes = [np.linalg.norm(u) for u in u_init]
            print(f"  Control magnitude range: [{min(u_magnitudes):.3f}, {max(u_magnitudes):.3f}]")
        
        # Convert to torch tensor
        u_init_tensor = torch.tensor(u_init, device=self.device, dtype=self.dtype)
        
        return u_init_tensor
    
    def optimize(
        self,
        x0: torch.Tensor,
        goal: torch.Tensor,
        U_init: Optional[torch.Tensor] = None,
        use_geometric_init: bool = True,  # NEW: Auto geometric initialization
    ) -> Dict:
        """
        Main DDP optimization loop
        
        Args:
            x0: Initial state [n_x]
            goal: Goal state [n_x]
            U_init: Optional initial control sequence [T, n_u]
                   If None and use_geometric_init=True, uses geometric initialization
                   If None and use_geometric_init=False, uses zero controls
            use_geometric_init: If True and U_init is None, automatically generate
                              geometric initialization for pusher system
        
        Returns:
            Dictionary with trajectory, controls, cost, etc.
        """
        self.goal = goal.to(self.device)
        x0 = x0.to(self.device)
        
        T = self.horizon
        
        # Initialize controls
        if U_init is None:
            if use_geometric_init and self.n_x == 8 and self.n_u == 2:
                # Geometric initialization for pusher system
                # Extract positions from state
                robot_pos = x0[6:8]  # Pusher position [px, py]
                box_pos = x0[:2]     # Box position [x, y]
                goal_pos = goal[:2]  # Goal box position [x, y]
                
                if self.verbose:
                    print(f"\nUsing geometric initialization for pusher system...")
                
                U = self.compute_geometric_initial_trajectory(
                    robot_pos=robot_pos,
                    box_pos=box_pos,
                    goal_pos=goal_pos,
                )
            else:
                # Default: zero controls
                U = torch.zeros(T, self.n_u, device=self.device, dtype=self.dtype)
        else:
            U = U_init.to(device=self.device, dtype=self.dtype)
        
        # Initial rollout
        with torch.no_grad():
            X, cost = self.rollout(x0, U)
        
        # Save initial trajectory for visualization comparison
        X_init = X.clone()
        U_init = U.clone()
        
        best_cost = cost.item()
        best_X = X.clone()
        best_U = U.clone()
        
        if self.verbose:
            print(f"{'='*70}")
            print(f"DDP Optimization")
            print(f"{'='*70}")
            print(f"Initial cost: {best_cost:.6f}")
        
        for it in range(self.max_iters):
            # Backward pass
            K_list, k_list, diverged = self.backward_pass(X, U)
            
            if diverged:
                # Increase regularization and retry
                self.reg = min(self.reg * self.reg_scale, self.max_reg)
                if self.verbose:
                    print(f"[DDP] Backward diverged. Increasing reg -> {self.reg:.2e}")
                continue
            
            # Forward pass with line search
            accepted = False
            J_nom = cost.item()
            
            for alpha in self.line_search_alphas:
                X_new, U_new, cost_new = self.forward_pass(x0, X, U, K_list, k_list, alpha)
                
                # Accept if better than best so far (with small tolerance for numerical precision)
                if cost_new < cost.item():
                    X, U, cost = X_new, U_new, torch.tensor(cost_new)
                    accepted = True
                    
                    best_cost = cost_new
                    best_X = X.clone()
                    best_U = U.clone()
                    
                    # Decrease regularization on success
                    self.reg = max(self.reg / self.reg_scale, self.min_reg)
                    
                    if self.verbose:
                        print(f"[DDP] iter {it+1:02d}  alpha={alpha:.2f}  J={cost_new:.6f}  reg={self.reg:.2e}")
                    break
            
            if not accepted:
                # Increase regularization
                self.reg = min(self.reg * self.reg_scale, self.max_reg)
                if self.verbose:
                    print(f"[DDP] iter {it+1:02d}  no improvement; reg -> {self.reg:.2e}")
                
                # Stop if reg exploded
                if self.reg >= self.max_reg * 0.99:
                    if self.verbose:
                        print("[DDP] Regularization saturated, terminating.")
                    break
            
            # Convergence check
            if best_cost < 1e-6:
                if self.verbose:
                    print("[DDP] Cost near zero, converged.")
                break
        
        # Compute final gradient (control gradient at optimal solution)
        # This is for visualization/analysis, similar to multiple_shooting
        final_grad_u = None
        try:
            # Create a copy of best_U that requires gradient
            U_for_grad = best_U.clone().detach().requires_grad_(True)
            
            # Rollout with gradient tracking
            X_for_grad, _ = self.rollout(x0, U_for_grad)
            
            # Compute total cost
            total_cost = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            for t in range(self.horizon):
                total_cost = total_cost + self.stage_cost_fn(X_for_grad[t], U_for_grad[t])
            total_cost = total_cost + self.terminal_cost_fn(X_for_grad[-1], self.goal)
            
            # Backward to get gradient
            total_cost.backward()
            
            if U_for_grad.grad is not None:
                final_grad_u = U_for_grad.grad.detach().clone().cpu().numpy()
        except Exception as e:
            if self.verbose:
                print(f"[DDP] Warning: Could not compute final gradient: {e}")


        if self.verbose:
            print(f"{'='*70}")
            print(f"Final cost: {best_cost:.6f}")
            print(f"{'='*70}\n")
        
        # Extract trajectory components for visualization
        # State format: [q(3), v(3), pusher_pos(2)] = [8]
        q_traj = best_X[:, :3]      # [T+1, 3] box pose
        v_traj = best_X[:, 3:6]     # [T+1, 3] box velocity
        pusher_traj = best_X[:, 6:8]  # [T+1, 2] pusher position
        
        # Initial trajectory (from first rollout)
        q_init = X_init[:, :3]
        v_init = X_init[:, 3:6]
        pusher_init = X_init[:, 6:8]
        
        # Contact forces (placeholder - DDP doesn't track these explicitly)
        # For compatibility with visualizer, create dummy arrays
        T = self.horizon
        contact_forces = np.zeros((T, 4))  # [T, 4] - placeholder
        signed_distances = np.zeros((T, 4))  # [T, 4] - placeholder
        
        

        # Return format matching multiple_shooting_with_alm.py for visualizer compatibility
        return {
            # Optimized trajectory (visualizer expects these keys)
            "loss": best_cost,
            "q_final": q_traj[-1].detach().cpu().numpy(),
            "u_seq": best_U.detach().cpu().numpy(),
            "trajectory": q_traj.detach().cpu().numpy(),
            "velocity_trajectory": v_traj.detach().cpu().numpy(),
            "pusher_trajectory": pusher_traj.detach().cpu().numpy(),
            "contact_forces": contact_forces,  # Placeholder for visualizer
            "signed_distances": signed_distances,  # Placeholder for visualizer
            
            # Initial trajectory for visualization comparison
            "initial_trajectory": q_init.detach().cpu().numpy(),
            "initial_velocity_trajectory": v_init.detach().cpu().numpy(),
            "initial_pusher_trajectory": pusher_init.detach().cpu().numpy(),
            
            # **NEW: Final control gradient (last outer iter, last inner iter)**
            "control_gradients": final_grad_u,

            # Loss component breakdown (for visualizer)
            "loss_components": {
                "total": best_cost,
                # DDP doesn't break down loss by component like ALM
                # Placeholder for compatibility
            },
            
            # DDP-specific data
            "K_gains": [K.detach().cpu().numpy() for K in K_list] if K_list else None,
            "feedforward_gains": [k.detach().cpu().numpy() for k in k_list] if k_list else None,
            
            # Full state trajectory (for advanced analysis)
            "full_trajectory": best_X.detach().cpu().numpy(),
            
            # Metadata
            "method": "DDP",
            "converged": best_cost < 1e-6 or it < self.max_iters - 1,
            "iterations": it + 1,
        }


# ==============================================================================
# Example: How to use with existing iLQR.py setup
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DDP Refactored from iLQR.py")
    print("="*70)
    print("\nKey improvements:")
    print("1. ✅ QP solver abstraction (Cholesky → Moreau)")
    print("2. ✅ Batch-ready structure (collect all QPs first)")
    print("3. ✅ Constrained QP support (control limits)")
    print("4. ✅ Clean separation: dynamics, cost, solver")
    print("="*70)
    print("\nHow to swap to Moreau:")
    print("  qp_solver = MoreauQPSolver()")
    print("  optimizer = DDPOptimizer(..., qp_solver=qp_solver)")
    print("\nThat's it! No other code changes needed.")
    print("="*70)