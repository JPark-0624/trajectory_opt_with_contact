import math
import torch
from torch import nn
from .dynamics import IPMOptions
import numpy as np

# NEW: iLQR optimizer ---------------------------------------------------------
class ILQROptimizer:
    """
    Iterative LQR on top of your contact dynamics.
    State: x = [q(3), v(3), pusher_pos(2)]  -> (8,)
    Control: u = u_push(2)

    Dynamics are pulled from your existing solvers:
      - LCP: step_square(...)
      - IP : step_square_pos_ip(...)

    Cost (default, stable & simple):
      stage:   l_k(x,u) = wu * ||u||^2 + wv * ||v||^2
      terminal l_T(x_T) = wpos * ||q_T[0:2] - goal[0:2]||^2 + wth * (theta_T - goal_th)^2

    You can extend stage cost (obs/penetration) later if desired.
    """
    def __init__(self, mass, side_length, mu, 
                w_target, w_v, w_ctrl, w_obs,
                horizon, dt, device,
                 dynamics_solver="LCP",
                 qp_solver=None,   # only used for LCP path
                 obstacle_pos=None,
                 # ilqr knobs
                 max_iters=50, regularization=1e-4, reg_scale=10.0,
                 min_reg=1e-8, max_reg=1e8, line_search_alphas=(1.0, 0.5, 0.25, 0.1, 0.05)):
        self.m = mass
        self.side = side_length
        self.half = side_length / 2.0
        self.mu = mu
        self.horizon = horizon
        self.dt = dt
        self.device = torch.device(device)
        self.dtype = torch.double
        self.w_target = w_target
        self.w_v = w_v
        self.w_ctrl = w_ctrl
        self.w_obs = w_obs
        self.obstacle_pos = obstacle_pos
        if self.obstacle_pos is not None:
            self.obstacle_pos=self._to_tensor(obstacle_pos, requires_grad=False)

        self.max_iters = max_iters
        self.reg = regularization
        self.reg_scale = reg_scale
        self.min_reg = min_reg
        self.max_reg = max_reg
        self.line_search_alphas = line_search_alphas

        # Moment of inertia of square plate about z (you already use this)
        self.Izz = (1.0/6.0) * self.m * (self.side**2 + self.side**2)

        # Hook the correct one-step dynamics
        self.dynamics_solver = dynamics_solver
        self.qp_solver = qp_solver
        try:
            # Import your one-step functions from .dynamics
            from .dynamics import step_square, step_square_pos_ip  # noqa: F401
            self._step_square = step_square
            self._step_square_ip = step_square_pos_ip
        except Exception:
            # If names differ in your project, fix here
            raise RuntimeError("Couldn't import one-step dynamics. "
                               "Make sure .dynamics exposes step_square(...) and step_square_pos_ip(...).")

    # ---------------------------------------------------------------
    # Packing / unpacking helpers for iLQR state
    # ---------------------------------------------------------------
    def _pack_x(self, q, v, pusher_pos):
        return torch.cat([q, v, pusher_pos], dim=-1)  # (8,)
    def _to_tensor(self, x, requires_grad=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double, device=self.device)
        else:
            x = x.to(dtype=torch.double, device=self.device)
        x.requires_grad = requires_grad
        return x
    def _unpack_x(self, x):
        q = x[..., 0:3]
        v = x[..., 3:6]
        p = x[..., 6:8]
        return q, v, p

    # ---------------------------------------------------------------
    # One-step dynamics wrapper: x_{k+1} = f(x_k, u_k)
    # ---------------------------------------------------------------
    def f(self, x, u):
        """
        Both x and u must be tensors with requires_grad possibly True.
        Shapes: x (8,), u (2,)
        Returns next state x_next (8,)
        """
        q, v, pusher = self._unpack_x(x)
        # advance pusher with control (point robot)
        pusher_next = pusher + self.dt * u

        if self.dynamics_solver == "LCP":
            q_next, v_next, lambdas, phi = self._step_square(
                q, v, pusher, u, self.dt,
                self.m, self.Izz, self.half, self.mu,
                qp_solver=self.qp_solver,
                device=self.device
            )
        else:
            # Interior-point (position-level) path
            q_next, v_next, pusher_pos_next, lambdas, phi = self._step_square_ip(
                q, v, pusher, u, self.dt,
                self.m, self.Izz, self.half, self.mu,
                ipm_opts=IPMOptions(target_mu=1e-4, max_newton=20, tol=1e-5, smooth_sdf=50.0, #smooth_sdf is unused
                    enable_viscous_ground_friction=True,
                    c_lin=8.0,          
                    c_ang=8.0 * self.half     
                    ),
                skip_solving_threshold=0.3)

        return self._pack_x(q_next, v_next, pusher_next)

    # ---------------------------------------------------------------
    # Costs
    # ---------------------------------------------------------------
    def stage_cost(self, x, u):
        """
        l_k = 1e-3 * ||u||^2  +  (w_obs/T) * 1/(||p - obs||^2 + 0.01)
        (obstacle part only if obstacle_pos is provided)
        """
        cost = self.w_ctrl * (u @ u)

        if self.obstacle_pos is not None:
            _, _, p = self._unpack_x(x)  # pusher position (2,)
            diff = p - self.obstacle_pos
            dist2 = (diff @ diff)
            cost = cost + (self.w_obs / float(self.horizon)) * (1.0 / (dist2 + 0.01))

        return cost

    def terminal_cost(self, x_T, goal):
        """
        l_T = 20.0 * ||q_T - goal||^2  +  0.1 * ||v_T||^2
        (goal is the same 3-vector you use in shooting: [x,y,theta])
        """
        q_T, v_T, _ = self._unpack_x(x_T)
        pos_th_err = q_T - goal  # match your rollout: use full q vs goal
        return self.w_target * (pos_th_err @ pos_th_err) + self.w_v * (v_T @ v_T)


    def rollout(self, x0, U):
        """Forward simulate with controls U: list/tensor (T,2). Returns X (T+1,8), total cost."""
        X = [x0]
        cost = torch.zeros((), dtype=self.dtype, device=self.device)
        for k in range(U.shape[0]):
            xk = X[-1]
            uk = U[k]
            cost = cost + self.stage_cost(xk, uk)
            xnext = self.f(xk, uk)
            X.append(xnext)
        cost = cost + self.terminal_cost(X[-1], self.goal)
        return torch.stack(X, dim=0), cost

    # ---------------------------------------------------------------
    # Autograd linearization and quadraticization
    # ---------------------------------------------------------------
    def linearize_dynamics(self, x, u):
        """Return A = df/dx (8x8), B = df/du (8x2) at (x,u)."""
        x = x.detach().requires_grad_(True)
        u = u.detach().requires_grad_(True)
        def fvec(z):
            x_, u_ = z
            return self.f(x_, u_)
        # Jacobians via autograd
        y = self.f(x, u)
        # df/dx
        A = []
        for i in range(y.numel()):
            (grad_x,) = torch.autograd.grad(y[i], x, retain_graph=True, allow_unused=False)
            A.append(grad_x)  # (8,)
        A = torch.stack(A, dim=0)  # (8,8)
        # df/du
        B = []
        for i in range(y.numel()):
            (grad_u,) = torch.autograd.grad(y[i], u, retain_graph=True)
            B.append(grad_u)  # (2,)
        B = torch.stack(B, dim=0)  # (8,2)
        return A, B
        
    def quad_cost_terms(self, x, u):
        """
        l_k = 1e-3*||u||^2 + (w_obs/T)/(||p-obs||^2 + 0.01)
        Return lx (8,), lu (2,), lxx (8x8), luu (2x2), lux (2x8)
        """
        device, dtype = self.device, self.dtype
        n_x = 8

        # Initialize
        lx  = torch.zeros(n_x, dtype=dtype, device=device)
        lu  = 2.0 * self.w_ctrl * u.clone()                 # dl/du = 2*w_u*u
        lxx = torch.zeros((n_x, n_x), dtype=dtype, device=device)
        luu = 2.0 * self.w_ctrl * torch.eye(2, dtype=dtype, device=device)  # d2l/du2
        lux = torch.zeros((2, n_x), dtype=dtype, device=device)

        # Obstacle term (acts only on pusher state x[6:8])
        if self.obstacle_pos is not None:
            _, _, p = self._unpack_x(x)  # (2,)
            d = p - self.obstacle_pos     # (2,)
            r = (d @ d)
            eps = 0.01
            c = (self.w_obs / float(self.horizon))

            # grad wrt p:  -2 d / (r+eps)^2
            gp = -2.0 * d / ((r + eps) ** 2)
            # Hessian wrt p:
            # H = -2 I /(r+eps)^2 + 8 d d^T /(r+eps)^3
            H = (-2.0 * torch.eye(2, dtype=dtype, device=device) / ((r + eps) ** 2)
                + 8.0 * torch.ger(d, d) / ((r + eps) ** 3))

            # scale by c
            gp = c * gp
            H  = c * H

            # place into lx / lxx at indices 6:8
            lx[6:8] = lx[6:8] + gp
            lxx[6:8, 6:8] = lxx[6:8, 6:8] + H

        return lx, lu, lxx, luu, lux
    # def quad_cost_terms(self, x, u):
    #     """
    #     Return (lx, lu, lxx, luu, lux) of stage cost at (x,u).
    #     """
    #     x = x.detach().requires_grad_(True)
    #     u = u.detach().requires_grad_(True)
    #     l = self.stage_cost(x, u)

    #     # First-order
    #     (lx,) = torch.autograd.grad(l, x, retain_graph=True, create_graph=True)
    #     (lu,) = torch.autograd.grad(l, u, retain_graph=True, create_graph=True)

    #     # Second-order (Hessians)
    #     lxx = torch.zeros((x.numel(), x.numel()), dtype=self.dtype, device=self.device)
    #     for i in range(x.numel()):
    #         (gxi,) = torch.autograd.grad(lx[i], x, retain_graph=True)
    #         lxx[i] = gxi

    #     luu = torch.zeros((u.numel(), u.numel()), dtype=self.dtype, device=self.device)
    #     for i in range(u.numel()):
    #         (gui,) = torch.autograd.grad(lu[i], u, retain_graph=True)
    #         luu[i] = gui

    #     lux = torch.zeros((u.numel(), x.numel()), dtype=self.dtype, device=self.device)
    #     for i in range(u.numel()):
    #         (guxi,) = torch.autograd.grad(lu[i], x, retain_graph=True)
    #         lux[i] = guxi

    #     return lx, lu, lxx, luu, lux

    def quad_terminal_terms(self, xT):
        """
        Return (lTx, lTxx) of terminal cost at xT.
        """
        xT = xT.detach().requires_grad_(True)
        lT = self.terminal_cost(xT, self.goal)

        (lTx,) = torch.autograd.grad(lT, xT, retain_graph=True, create_graph=True)
        lTxx = torch.zeros((xT.numel(), xT.numel()), dtype=self.dtype, device=self.device)
        for i in range(xT.numel()):
            (gxi,) = torch.autograd.grad(lTx[i], xT, retain_graph=True)
            lTxx[i] = gxi
        return lTx, lTxx

    # ---------------------------------------------------------------
    # Main iLQR loop
    # ---------------------------------------------------------------
    def optimize(self, q0, v0, pusher0, goal, u_init=None,
                 max_iters=None, verbose=True):
        self.goal = goal.detach() if isinstance(goal, torch.Tensor) else torch.tensor(goal, dtype=self.dtype, device=self.device)
        # initial state
        q0 = q0.to(dtype=self.dtype, device=self.device)
        v0 = v0.to(dtype=self.dtype, device=self.device)
        p0 = pusher0.to(dtype=self.dtype, device=self.device)
        x0 = self._pack_x(q0, v0, p0)

        T = self.horizon
        if u_init is None:
            U = torch.zeros(T, 2, dtype=self.dtype, device=self.device)
        else:
            U = u_init.to(dtype=self.dtype, device=self.device)
            if U.shape[0] != T:
                raise ValueError(f"u_init has horizon {U.shape[0]} but expected {T}")

        if max_iters is None:
            max_iters = self.max_iters

        # Nominal rollout
        X, J = self.rollout(x0, U)
        best_J = J.item()
        best = (X.clone(), U.clone(), best_J)

        for it in range(max_iters):
            # Linearize dynamics and quadraticize cost along nominal
            A_list, B_list = [], []
            lx_list, lu_list, lxx_list, luu_list, lux_list = [], [], [], [], []

            for k in range(T):
                Ak, Bk = self.linearize_dynamics(X[k], U[k])
                A_list.append(Ak)
                B_list.append(Bk)

                lx, lu, lxx, luu, lux = self.quad_cost_terms(X[k], U[k])
                lx_list.append(lx); lu_list.append(lu); lxx_list.append(lxx); luu_list.append(luu); lux_list.append(lux)

            lTx, lTxx = self.quad_terminal_terms(X[-1])

            # Backward pass (Riccati-like)
            Vx  = lTx.clone()
            Vxx = lTxx.clone()

            K_list = []
            k_list = []
            diverged = False

            for k in reversed(range(T)):
                Ak = A_list[k]; Bk = B_list[k]
                lx = lx_list[k]; lu = lu_list[k]
                lxx = lxx_list[k]; luu = luu_list[k]; lux = lux_list[k]

                # Q-function expansion
                Qx  = lx  + Ak.T @ Vx
                Qu  = lu  + Bk.T @ Vx
                Qxx = lxx + Ak.T @ Vxx @ Ak
                Quu = luu + Bk.T @ Vxx @ Bk
                Qux = lux + Bk.T @ Vxx @ Ak   # (2x8)

                # Regularize Quu to ensure PD
                Quu_reg = Quu + self.reg * torch.eye(Quu.shape[0], dtype=self.dtype, device=self.device)

                try:
                    L = torch.linalg.cholesky(Quu_reg)
                    Quu_inv = torch.cholesky_inverse(L)
                except RuntimeError:
                    diverged = True
                    break

                K = - Quu_inv @ Qux       # (2x8)
                kff = - Quu_inv @ Qu      # (2,)

                # Update value function
                Vx  = Qx  + K.T @ Quu @ kff + Qux.T @ kff + K.T @ Qu + Qx*0.0  # keep shape; algebraic clarity
                Vxx = Qxx + K.T @ Quu @ K  + Qux.T @ K + K.T @ Qux
                # (symmetrize to control numerical issues)
                Vxx = 0.5 * (Vxx + Vxx.T)

                K_list.append(K)
                k_list.append(kff)

            if diverged:
                # increase regularization and retry
                self.reg = min(self.reg * self.reg_scale, self.max_reg)
                if verbose:
                    print(f"[iLQR] Backward diverged. Increasing reg -> {self.reg:.2e}")
                continue

            # Forward line search with gains
            accepted = False
            for alpha in self.line_search_alphas:
                X_new = [x0]
                U_new = []
                cost_new = torch.zeros((), dtype=self.dtype, device=self.device)

                for k in range(T):
                    xk = X_new[-1]
                    Kn = K_list[T-1-k]   # reversed order in build
                    kn = k_list[T-1-k]
                    # feedback + feedforward
                    dx = xk - X[k]
                    u_try = U[k] + alpha * kn + Kn @ dx
                    U_new.append(u_try)
                    cost_new = cost_new + self.stage_cost(xk, u_try)
                    x_next = self.f(xk, u_try)
                    X_new.append(x_next)

                cost_new = cost_new + self.terminal_cost(X_new[-1], self.goal)
                J_new = cost_new.item()

                if J_new < best_J - 1e-9:
                    accepted = True
                    X = torch.stack(X_new, dim=0)
                    U = torch.stack(U_new, dim=0)
                    best_J = J_new
                    best = (X.clone(), U.clone(), best_J)
                    # decrease regularization on success
                    self.reg = max(self.reg / self.reg_scale, self.min_reg)
                    if verbose:
                        print(f"[iLQR] iter {it+1:02d}  alpha={alpha:.2f}  J={J_new:.6f}  reg={self.reg:.2e}")
                    break

            if not accepted:
                # increase regularization and try again
                self.reg = min(self.reg * self.reg_scale, self.max_reg)
                if verbose:
                    print(f"[iLQR] iter {it+1:02d}  no improvement; reg -> {self.reg:.2e}")
                # optional stop if reg exploded
                if self.reg >= self.max_reg * 0.99:
                    break

        Xb, Ub, Jb = best
        # Collect contacts/phis with a final pass
        qs, ps, lambdas, phis = self.rollout_with_contacts(Xb[0], Ub)

        # Convert contacts to CPU/numpy safely (shapes may vary by solver)
        def to_numpy_list(tensors):
            out = []
            for t in tensors:
                if isinstance(t, torch.Tensor):
                    out.append(t.detach().cpu().numpy())
                else:
                    # in case your solver returns tuples/lists
                    out.append(torch.as_tensor(t, dtype=self.dtype, device=self.device).detach().cpu().numpy())
            return np.array(out)
        return {
            "loss": Jb,
            "trajectory": qs.detach().cpu().numpy(),   # (T+1, 8)
            "u_seq": Ub.detach().cpu().numpy(),        # (T, 2)
            "pusher_sequence": ps.detach().cpu().numpy(),  # (T+1, 2)
            "contact_forces": to_numpy_list(lambdas),  # list length T
            "signed_distances": to_numpy_list(phis),   # list length T
        }

    def _step_with_contacts(self, q, v, pusher, u):
        if self.dynamics_solver == "LCP":
            q_next, v_next, lambdas, phi = self._step_square(
                q, v, pusher, u, self.dt,
                self.m, self.Izz, self.half, self.mu,
                qp_solver=self.qp_solver,
                device=self.device
            )
        else:
            # Interior-point (position-level) path
            q_next, v_next, pusher_pos_next, lambdas, phi = self._step_square_ip(
                q, v, pusher, u, self.dt,
                self.m, self.Izz, self.half, self.mu,
                ipm_opts=IPMOptions(target_mu=1e-4, max_newton=20, tol=1e-5, smooth_sdf=50.0, #smooth_sdf is unused
                    enable_viscous_ground_friction=True,
                    c_lin=8.0,          
                    c_ang=8.0 * self.half     
                    ),
                skip_solving_threshold=0.3)
        return q_next, v_next, lambdas, phi

    # --- final pass to collect contacts/phis/qs after convergence ---
    def rollout_with_contacts(self, x0, U):
        """
        Returns:
          qs:    (T+1, 3)   sequence of q (including q0 and q_T)
          ps:    (T+1, 2)   pusher positions
          lambdas: list length T, each a tensor (shape per your solver)
          phis:    list length T, each a tensor (usually scalar per contact)
        """
        T = U.shape[0]
        q_hist = []
        p_hist = []
        lambdas = []
        phis = []

        q, v, p = self._unpack_x(x0)
        q_hist.append(q)
        p_hist.append(p)

        for k in range(T):
            u = U[k]
            qn, vn, lamk, phik = self._step_with_contacts(q, v, p, u)
            p = p + self.dt * u  # point robot integrator
            q, v = qn, vn

            q_hist.append(q)
            p_hist.append(p)
            lambdas.append(lamk)
            phis.append(phik)

        qs = torch.stack(q_hist, dim=0)   # (T+1,3)
        ps = torch.stack(p_hist, dim=0)   # (T+1,2)
        return qs, ps, lambdas, phis