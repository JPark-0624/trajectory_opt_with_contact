"""
Efficient Control Matching IRL using Implicit Differentiation.

Instead of backpropagating through entire inner optimization loop,
we use the envelope theorem to compute ∂u*/∂w efficiently.

Key idea:
    At optimum: ∇_u L(u*; w) = 0
    Differentiate: ∇²_u L · ∂u*/∂w + ∇_u∇_w L = 0
    Solve: ∂u*/∂w = -[∇²_u L]^{-1} · ∇_u∇_w L
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .dynamics import rollout
from .optimizer import TrajectoryOptimizer


@dataclass
class ImplicitIRLOptions:
    """Options for implicit differentiation IRL."""
    max_outer_iters: int = 50
    max_inner_iters: int = 100
    lr_weights: float = 0.01
    lr_trajectory: float = 0.01
    
    # Convergence
    weight_tol: float = 1e-4
    control_tol: float = 1e-3
    inner_tol: float = 1e-4  # Inner optimization convergence
    
    # Hessian computation
    hessian_reg: float = 1e-4  # Regularization for Hessian inversion
    use_conjugate_gradient: bool = True  # Use CG instead of direct solve
    cg_max_iters: int = 50
    
    # Warm start
    warm_start_inner: bool = True
    
    verbose: bool = True


class ImplicitControlMatchingIRL:
    """
    Control matching IRL with implicit differentiation.
    
    More memory-efficient than unrolled differentiation,
    but requires computing Hessian-vector products.
    """
    
    def __init__(self, optimizer: TrajectoryOptimizer):
        self.optimizer = optimizer
        self.device = optimizer.device
        self.feature_names = ['w_target', 'w_orient', 'w_v', 'w_ctrl', 'w_obs']
        self.n_features = len(self.feature_names)
    
    def recover_weights(
        self,
        demo: Dict,
        w_init: Optional[torch.Tensor] = None,
        opts: Optional[ImplicitIRLOptions] = None
    ) -> Dict:
        """
        Recover weights using implicit differentiation.
        
        Args:
            demo: Demonstration dictionary
            w_init: Initial weight guess
            opts: Options
        
        Returns:
            result: Recovery results
        """
        if opts is None:
            opts = ImplicitIRLOptions()
        
        # Extract demo data
        q0 = self._to_tensor(demo['q0'])
        v0 = self._to_tensor(demo['v0'])
        pusher0 = self._to_tensor(demo['pusher0'])
        goal = self._to_tensor(demo['goal'])
        u_demo = self._to_tensor(demo['u_demo'])
        obstacle_pos = None if demo.get('obstacle_pos') is None else self._to_tensor(demo['obstacle_pos'])
        
        if opts.verbose:
            print("\n" + "="*80)
            print("IRL with Implicit Differentiation")
            print("="*80)
        
        # Initialize weights
        if w_init is None:
            w_raw = torch.zeros(self.n_features, dtype=torch.double,
                               device=self.device, requires_grad=True)
        else:
            w_raw = torch.logit(w_init.clone()).requires_grad_(True)
        
        optimizer_w = torch.optim.Adam([w_raw], lr=opts.lr_weights)
        
        loss_history = []
        control_error_history = []
        gradient_norm_history = []
        u_current = None
        
        for iter in range(opts.max_outer_iters):
            # Project to simplex
            w = F.softmax(w_raw, dim=0)
            
            # Solve inner optimization (detached from graph!)
            u_star, converged = self._solve_trajectory_converged(
                w.detach(), q0, v0, pusher0, goal, obstacle_pos,
                u_init=u_current if opts.warm_start_inner else None,
                max_iters=opts.max_inner_iters,
                lr=opts.lr_trajectory,
                tol=opts.inner_tol,
                verbose=False
            )
            
            u_current = u_star.detach()
            
            if not converged and opts.verbose:
                print(f"Warning: Inner optimization not fully converged at iter {iter}")
            
            # Compute gradient using implicit differentiation
            grad_w = self._implicit_gradient(
                u_star, u_demo, w, q0, v0, pusher0, goal, obstacle_pos, opts
            )
            
            # Control matching loss (for monitoring)
            with torch.no_grad():
                control_error = torch.norm(u_star - u_demo).item()
                loss = torch.sum((u_star - u_demo) ** 2).item()
            
            loss_history.append(loss)
            control_error_history.append(control_error)
            gradient_norm_history.append(torch.norm(grad_w).item())
            
            if opts.verbose and (iter % 10 == 0 or iter < 5):
                print(f"\nIter {iter:3d}:")
                print(f"  Control error: {control_error:.6f}")
                print(f"  Gradient norm: {gradient_norm_history[-1]:.6e}")
                self._print_weights(w)
            
            # Check convergence
            if iter > 0:
                weight_change = torch.norm(w - w_prev).item()
                if weight_change < opts.weight_tol and control_error < opts.control_tol:
                    if opts.verbose:
                        print(f"\nConverged at iteration {iter}")
                    break
            
            w_prev = w.detach().clone()
            
            # Manual gradient descent (we computed gradient ourselves)
            optimizer_w.zero_grad()
            w_raw.grad = self._transform_gradient_to_raw(grad_w, w_raw)
            optimizer_w.step()
        
        return {
            'w_recovered': F.softmax(w_raw, dim=0).detach(),
            'loss_history': loss_history,
            'control_error_history': control_error_history,
            'gradient_norm_history': gradient_norm_history,
            'converged': control_error < opts.control_tol,
            'num_iters': iter + 1,
            'u_final': u_current
        }
    
    def _solve_trajectory_converged(
        self,
        w: torch.Tensor,
        q0, v0, pusher0, goal, obstacle_pos,
        u_init: Optional[torch.Tensor],
        max_iters: int,
        lr: float,
        tol: float,
        verbose: bool
    ) -> Tuple[torch.Tensor, bool]:
        """
        Solve trajectory optimization until convergence.
        
        Returns:
            u_star: Optimal control
            converged: Whether optimization converged
        """
        if u_init is None:
            u_seq = torch.zeros(self.optimizer.horizon, 2,
                               dtype=torch.double, device=self.device,
                               requires_grad=True)
        else:
            u_seq = u_init.clone().detach().requires_grad_(True)
        
        opt = torch.optim.Adam([u_seq], lr=lr)
        
        prev_loss = float('inf')
        converged = False
        
        for iter in range(max_iters):
            opt.zero_grad()
            
            loss = self._weighted_rollout(u_seq, q0, v0, pusher0, goal, w, obstacle_pos)
            loss.backward()
            opt.step()
            
            # Check convergence
            loss_val = loss.item()
            if abs(loss_val - prev_loss) < tol:
                converged = True
                if verbose:
                    print(f"    Inner converged at iter {iter}")
                break
            prev_loss = loss_val
        
        return u_seq.detach(), converged
    
    def _implicit_gradient(
        self,
        u_star: torch.Tensor,
        u_demo: torch.Tensor,
        w: torch.Tensor,
        q0, v0, pusher0, goal, obstacle_pos,
        opts: ImplicitIRLOptions
    ) -> torch.Tensor:
        """
        Compute ∂loss/∂w using implicit differentiation.
        
        loss = ||u* - u_demo||²
        
        ∂loss/∂w = ∂loss/∂u* · ∂u*/∂w
                 = 2(u* - u_demo)ᵀ · ∂u*/∂w
        
        where ∂u*/∂w is computed via envelope theorem:
        ∂u*/∂w = -[∇²_u L]^{-1} · ∇_u∇_w L
        """
        u_star_req = u_star.detach().requires_grad_(True)
        w_req = w.detach().requires_grad_(True)
        
        # Compute inner loss at optimum
        L_star = self._weighted_rollout(u_star_req, q0, v0, pusher0, goal, w_req, obstacle_pos)
        
        # ∂loss/∂u* = 2(u* - u_demo)
        dloss_du = 2 * (u_star - u_demo)
        
        # Compute ∂u*/∂w using implicit function theorem
        # ∂u*/∂w = -[∇²_u L]^{-1} · ∇_u∇_w L
        
        if opts.use_conjugate_gradient:
            # More efficient: solve H · x = b using CG
            # where H = ∇²_u L, b = ∇_u∇_w L
            du_dw = self._compute_implicit_gradient_cg(
                L_star, u_star_req, w_req, opts
            )
        else:
            # Direct method: compute full Hessian
            du_dw = self._compute_implicit_gradient_direct(
                L_star, u_star_req, w_req, opts
            )
        
        # ∂loss/∂w = ∂loss/∂u* · ∂u*/∂w
        grad_w = torch.zeros(self.n_features, dtype=torch.double, device=self.device)
        
        for i in range(self.n_features):
            # du_dw[i] is (T, 2) - how u changes with w[i]
            grad_w[i] = torch.sum(dloss_du * du_dw[i])
        
        return grad_w
    
    def _compute_implicit_gradient_cg(
        self,
        L_star: torch.Tensor,
        u_star: torch.Tensor,
        w: torch.Tensor,
        opts: ImplicitIRLOptions
    ) -> torch.Tensor:
        """
        Compute ∂u*/∂w using conjugate gradient.
        
        For each w[i]:
            Solve: H · (∂u*/∂w[i]) = -∇_u∇_w[i] L
            where H = ∇²_u L (Hessian)
        """
        T, D = u_star.shape  # (horizon, 2)
        n_w = self.n_features
        
        du_dw = torch.zeros(n_w, T, D, dtype=torch.double, device=self.device)
        
        for i in range(n_w):
            # Compute ∇_u∇_w[i] L (mixed derivative)
            mixed_grad = self._compute_mixed_derivative(L_star, u_star, w, i)
            
            # Solve H · x = -mixed_grad using CG
            # H is implicit (Hessian-vector product)
            x = self._conjugate_gradient(
                lambda v: self._hessian_vector_product(L_star, u_star, v, opts.hessian_reg),
                -mixed_grad.reshape(-1),
                max_iters=opts.cg_max_iters
            )
            
            du_dw[i] = x.reshape(T, D)
        
        return du_dw
    
    def _compute_mixed_derivative(
        self,
        L: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        i: int
    ) -> torch.Tensor:
        """
        Compute ∇_u∇_w[i] L (mixed partial derivative).
        
        Returns: (T, D) tensor
        """
        # First: ∇_w[i] L
        dL_dw = torch.autograd.grad(L, w, create_graph=True)[0]
        dL_dwi = dL_dw[i]
        
        # Second: ∇_u (∇_w[i] L)
        mixed = torch.autograd.grad(dL_dwi, u, retain_graph=True)[0]
        
        return mixed
    
    def _hessian_vector_product(
        self,
        L: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        reg: float
    ) -> torch.Tensor:
        """
        Compute H·v where H = ∇²_u L + reg·I
        
        Using finite differences of gradient:
        H·v ≈ ∇_u L(u + εv) - ∇_u L(u) / ε
        """
        T, D = u.shape
        v = v.reshape(T, D)
        
        # Compute gradient at u
        grad_u = torch.autograd.grad(L, u, create_graph=True, retain_graph=True)[0]
        
        # Compute Hessian-vector product using double backprop
        Hv = torch.autograd.grad(grad_u, u, grad_outputs=v, retain_graph=True)[0]
        
        # Add regularization
        Hv = Hv + reg * v
        
        return Hv.reshape(-1)
    
    def _conjugate_gradient(
        self,
        A_func,  # Function that computes A·v
        b: torch.Tensor,
        max_iters: int,
        tol: float = 1e-6
    ) -> torch.Tensor:
        """
        Solve A·x = b using conjugate gradient.
        
        Args:
            A_func: Function that computes A·v for any vector v
            b: Right-hand side
            max_iters: Maximum iterations
            tol: Convergence tolerance
        
        Returns:
            x: Solution vector
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        
        rsold = torch.dot(r, r)
        
        for i in range(max_iters):
            Ap = A_func(p)
            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            
            rsnew = torch.dot(r, r)
            
            if torch.sqrt(rsnew) < tol:
                break
            
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        return x
    
    def _compute_implicit_gradient_direct(
        self,
        L_star: torch.Tensor,
        u_star: torch.Tensor,
        w: torch.Tensor,
        opts: ImplicitIRLOptions
    ) -> torch.Tensor:
        """
        Direct computation using full Hessian (memory intensive).
        """
        T, D = u_star.shape
        n_vars = T * D
        
        # Compute full Hessian
        H = torch.zeros(n_vars, n_vars, dtype=torch.double, device=self.device)
        
        grad_u = torch.autograd.grad(L_star, u_star, create_graph=True, retain_graph=True)[0]
        grad_flat = grad_u.reshape(-1)
        
        for i in range(n_vars):
            H[i] = torch.autograd.grad(grad_flat[i], u_star, retain_graph=True)[0].reshape(-1)
        
        # Add regularization
        H = H + opts.hessian_reg * torch.eye(n_vars, dtype=torch.double, device=self.device)
        
        # Compute ∂u*/∂w for each weight
        du_dw = torch.zeros(self.n_features, T, D, dtype=torch.double, device=self.device)
        
        for i in range(self.n_features):
            mixed_grad = self._compute_mixed_derivative(L_star, u_star, w, i)
            
            # Solve H · x = -mixed_grad
            x = torch.linalg.solve(H, -mixed_grad.reshape(-1))
            du_dw[i] = x.reshape(T, D)
        
        return du_dw
    
    def _weighted_rollout(self, u_seq, q0, v0, pusher0, goal, w, obstacle_pos):
        """Rollout with weighted cost."""
        loss, *_ = rollout(
            u_seq, q0, v0, pusher0,
            self.optimizer.horizon, self.optimizer.dt,
            self.optimizer.m, self.optimizer.Izz,
            self.optimizer.half, self.optimizer.mu, goal,
            w_target=w[0], w_orient=w[1], w_v=w[2],
            w_ctrl=w[3], w_obs=w[4],
            qp_solver=self.optimizer.qp_solver,
            dynamics_solver=self.optimizer.dynamics_solver,
            obstacle_pos=obstacle_pos,
            device=self.device
        )
        return loss
    
    def _transform_gradient_to_raw(self, grad_w, w_raw):
        """
        Transform gradient w.r.t. normalized weights to gradient w.r.t. raw weights.
        
        w = softmax(w_raw)
        ∂L/∂w_raw = ∂L/∂w · ∂w/∂w_raw
        """
        w = F.softmax(w_raw, dim=0)
        
        # Jacobian of softmax
        # ∂w[i]/∂w_raw[j] = w[i] * (δ[i,j] - w[j])
        n = len(w)
        J = torch.zeros(n, n, dtype=torch.double, device=self.device)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    J[i, j] = w[i] * (1 - w[j])
                else:
                    J[i, j] = -w[i] * w[j]
        
        grad_raw = J.T @ grad_w
        return grad_raw
    
    def _to_tensor(self, x, requires_grad=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double, device=self.device)
        else:
            x = x.to(dtype=torch.double, device=self.device)
        x.requires_grad = requires_grad
        return x
    
    def _print_weights(self, w: torch.Tensor):
        print("  Weights:")
        for name, val in zip(self.feature_names, w):
            print(f"    {name:12s}: {val.item():.6f}")


# Helper function to use with existing interface
def pack_demo(q0, v0, pusher0, goal, u_demo, obstacle_pos=None):
    """Pack demonstration data."""
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    return dict(
        q0=to_numpy(q0),
        v0=to_numpy(v0),
        pusher0=to_numpy(pusher0),
        goal=to_numpy(goal),
        u_demo=to_numpy(u_demo),
        obstacle_pos=None if obstacle_pos is None else to_numpy(obstacle_pos),
    )
