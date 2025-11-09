import torch
# from .dynamics import implicit_euler_defects
from .qp_solver import ContactQPSolver

class DirectTranscriptionOptimizer:
    """
    Direct transcription with implicit-Euler defects enforced via
    Augmented Lagrangian (ALM). Inner solver: LBFGS (2nd-order) or Adam.
    Low-level contact QP: cvxpy or qpth (interior-point).
    """

    def __init__(self, mass=1.0, side_length=0.2, mu=0.6,
                 horizon=60, dt=0.05, device='cuda',
                 use_second_order: bool = True,
                 qp_backend: str = "qpth",
                 ipm_eps: float = 1e-4,
                 ipm_max_iter: int = 50,
                 # ALM knobs:
                 alm_enabled: bool = True,
                 alm_rho_init: float = 1e2,
                 alm_rho_max: float = 1e8,
                 alm_eta: float = 10.0,           # rho multiplier when stuck
                 alm_target_tol: float = 1e-6,
                 alm_outer_iters: int = 10,
                 lbfgs_inner_steps: int = 10,
                 lbfgs_history: int = 10):
        self.m = mass
        self.side = side_length
        self.half = side_length / 2
        self.mu = mu
        self.T = horizon
        self.h = dt

        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            device = 'cpu'
        self.device = torch.device(device)
        self.Izz = (1.0/6.0) * self.m * (self.side**2 + self.side**2)

        # QP (contact) backend
        self.qp_solver = ContactQPSolver(mu=self.mu, n_contacts=1,
                                         backend=qp_backend,
                                         ipm_eps=ipm_eps,
                                         ipm_max_iter=ipm_max_iter)

        # Optimizer choice
        self.use_second_order = use_second_order

        # ALM params
        self.alm_enabled = alm_enabled
        self.alm_rho = alm_rho_init
        self.alm_rho_max = alm_rho_max
        self.alm_eta = alm_eta
        self.alm_target_tol = alm_target_tol
        self.alm_outer_iters = alm_outer_iters
        self.lbfgs_inner_steps = lbfgs_inner_steps
        self.lbfgs_history = lbfgs_history

        # Will be allocated after we know residual dimension
        self._mult_y = None   # Lagrange multipliers stacked for all residuals

    # ---------------- utils ----------------
    def _to(self, x, requires_grad=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double, device=self.device)
        else:
            x = x.to(dtype=torch.double, device=self.device)
        x.requires_grad = requires_grad
        return x

    def _pack(self, qs, vs, prs, us):
        return torch.cat([qs.flatten(), vs.flatten(), prs.flatten(), us.flatten()])

    def _unpack(self, z):
        ofs = 0
        qs = z[ofs:ofs+(self.T+1)*3].view(self.T+1, 3); ofs += (self.T+1)*3
        vs = z[ofs:ofs+(self.T+1)*3].view(self.T+1, 3); ofs += (self.T+1)*3
        prs = z[ofs:ofs+(self.T+1)*2].view(self.T+1, 2); ofs += (self.T+1)*2
        us = z[ofs:ofs+self.T*2].view(self.T, 2)
        return qs, vs, prs, us

    # -------------- residual builder --------------
    def _stack_defects(self, qs, vs, prs, us):
        """
        Build the big equality-constraint vector h(z) by stacking all
        implicit-Euler residuals [r_q, r_v, r_p] over k = 0..T-1.

        Returns:
            h_stack: (T*(3+3+2),) vector
            lam_hist: (T,2) contact forces
            phi_hist: (T,) signed distances
        """
        all_r = []
        lam_list = []
        phi_list = []
        for k in range(self.T):
            r, lam, phi = implicit_euler_defects(
                qs[k], vs[k], prs[k], us[k],
                qs[k+1], vs[k+1], prs[k+1],
                self.h, self.m, self.Izz, self.half, self.mu,
                self.qp_solver, device=self.device
            )
            all_r.append(r)
            lam_list.append(lam)
            phi_list.append(phi)

        h_stack = torch.cat(all_r, dim=0)                           # shape (T*8,)
        lam_hist = torch.stack(lam_list, dim=0)                      # (T,2)
        phi_hist = torch.stack(phi_list, dim=0) if isinstance(phi_list[0], torch.Tensor) \
                   else torch.tensor(phi_list, dtype=torch.double, device=self.device)
        return h_stack, lam_hist, phi_hist

    # -------------- objective pieces --------------
    def _task_cost(self, qs, vs, prs, us, goal, obstacle_pos=None):
        goal_term = 20.0 * (qs[-1] - goal).pow(2).sum()
        ctrl_term = 1e-3 * (us.pow(2).sum())
        v_term = 0.1 * (vs[-1].pow(2).sum())

        obs_term = 0.0
        if obstacle_pos is not None:
            for k in range(self.T):
                obs_term = obs_term + 1.0 / ((prs[k] - obstacle_pos).pow(2).sum() + 0.01)
            obs_term = obs_term / self.T

        return goal_term + ctrl_term + v_term + obs_term

    def _ic_cost(self, qs, vs, prs, q0, v0, pr0, w=1e6):
        # Hard-ish initial conditions via a very large quadratic (or eliminate variables)
        return w * ((qs[0] - q0).pow(2).sum() + (vs[0] - v0).pow(2).sum() + (prs[0] - pr0).pow(2).sum())

    # -------------- main optimize --------------
    def optimize(self, q0, v0, pusher0, goal,
                 u_init=None, max_iters=100, lr=0.5,
                 obstacle_pos=None, verbose=True):
        q0 = self._to(q0)
        v0 = self._to(v0)
        pr0 = self._to(pusher0)
        goal = self._to(goal)
        if obstacle_pos is not None:
            obstacle_pos = self._to(obstacle_pos)

        # Decision variables
        qs = torch.zeros(self.T+1, 3, dtype=torch.double, device=self.device)
        vs = torch.zeros(self.T+1, 3, dtype=torch.double, device=self.device)
        prs = torch.zeros(self.T+1, 2, dtype=torch.double, device=self.device)
        us = torch.zeros(self.T, 2, dtype=torch.double, device=self.device)

        # Init (simple kinematic seed)
        qs[0] = q0; vs[0] = v0; prs[0] = pr0
        if u_init is not None:
            us = self._to(u_init)
        for k in range(self.T):
            vs[k+1] = vs[k]
            qs[k+1] = torch.stack([qs[k,0] + self.h*vs[k,0],
                                   qs[k,1] + self.h*vs[k,1],
                                   qs[k,2] + self.h*vs[k,2]])
            prs[k+1] = prs[k] + self.h*us[k]

        z = self._pack(qs, vs, prs, us).detach().clone().requires_grad_(True)

        # Prepare optimizer
        if self.use_second_order:
            opt = torch.optim.LBFGS([z], max_iter=20, history_size=self.lbfgs_history,
                                    lr=lr, line_search_fn="strong_wolfe")
        else:
            opt = torch.optim.Adam([z], lr=lr)

        # Allocate multipliers after first residual build
        with torch.no_grad():
            qs0, vs0, prs0, us0 = self._unpack(z)
            h0, _, _ = self._stack_defects(qs0, vs0, prs0, us0)
            if self._mult_y is None or self._mult_y.numel() != h0.numel():
                self._mult_y = torch.zeros_like(h0)

        # ALM outer loop
        best = {'ALMloss': float('inf'), 'z': None, 'qs': None, 'prs': None,
                'lam': None, 'phi': None}

        for outer in range(self.alm_outer_iters if self.alm_enabled else 1):
            # Inner loop (LBFGS or Adam)
            if self.use_second_order:
                # LBFGS needs a closure
                for _ in range(self.lbfgs_inner_steps):
                    def closure():
                        opt.zero_grad()
                        qs_, vs_, prs_, us_ = self._unpack(z)
                        # task + initial condition
                        L_task = self._task_cost(qs_, vs_, prs_, us_, goal, obstacle_pos) \
                               + self._ic_cost(qs_, vs_, prs_, q0, v0, pr0)
                        # constraints
                        h_stack, lam_hist, phi_hist = self._stack_defects(qs_, vs_, prs_, us_)
                        if self.alm_enabled:
                            L_aug = L_task + (self._mult_y @ h_stack) + 0.5 * self.alm_rho * (h_stack @ h_stack)
                        else:
                            # Fallback: plain penalty if ALM disabled
                            L_aug = L_task + 1e3 * (h_stack @ h_stack)

                        L_aug.backward()
                        return L_aug
                    opt.step(closure)
            else:
                for it in range(max_iters):
                    opt.zero_grad()
                    qs_, vs_, prs_, us_ = self._unpack(z)
                    L_task = self._task_cost(qs_, vs_, prs_, us_, goal, obstacle_pos) \
                           + self._ic_cost(qs_, vs_, prs_, q0, v0, pr0)
                    h_stack, lam_hist, phi_hist = self._stack_defects(qs_, vs_, prs_, us_)
                    if self.alm_enabled:
                        L_aug = L_task + (self._mult_y @ h_stack) + 0.5 * self.alm_rho * (h_stack @ h_stack)
                    else:
                        L_aug = L_task + 1e3 * (h_stack @ h_stack)
                    L_aug.backward()
                    opt.step()
                    if verbose and (it+1) % 1 == 0:
                        print(f"[Adam inner] it={it+1} | L_aug={L_aug.item():.6e}")

            # Evaluate residuals & update multipliers
            with torch.no_grad():
                qs_, vs_, prs_, us_ = self._unpack(z)
                L_task = self._task_cost(qs_, vs_, prs_, us_, goal, obstacle_pos) \
                       + self._ic_cost(qs_, vs_, prs_, q0, v0, pr0)
                h_stack, lam_hist, phi_hist = self._stack_defects(qs_, vs_, prs_, us_)
                res_norm = h_stack.norm().item()
                L_aug_val = (L_task + (self._mult_y @ h_stack)
                             + 0.5 * self.alm_rho * (h_stack @ h_stack)).item()

                if verbose:
                    print(f"[ALM] outer={outer+1}/{self.alm_outer_iters} | "
                          f"rho={self.alm_rho:.3e} | ||h||={res_norm:.3e} | L_aug={L_aug_val:.6e}")

                # Save best
                if L_aug_val < best['ALMloss']:
                    best.update({
                        'ALMloss': L_aug_val,
                        'z': z.detach().clone(),
                        'qs': qs_.detach().clone(),
                        'prs': prs_.detach().clone(),
                        'lam': lam_hist.detach().clone(),
                        'phi': phi_hist.detach().clone()
                    })

                if not self.alm_enabled:
                    # No ALM, exit after one inner solve
                    break

                # Multiplier update
                self._mult_y = self._mult_y + self.alm_rho * h_stack

                # Penalty increase if needed
                if res_norm > self.alm_target_tol * 10:
                    self.alm_rho = min(self.alm_rho * self.alm_eta, self.alm_rho_max)

                # Converged constraints?
                if res_norm < self.alm_target_tol:
                    if verbose:
                        print(f"[ALM] Converged constraints: ||h||={res_norm:.3e} < {self.alm_target_tol:.1e}")
                    break

        # Unpack best solution for return (preserves your API)
        with torch.no_grad():
            if best['z'] is None:
                best['z'] = z
                qs_out, vs_out, prs_out, us_out = self._unpack(z)
                _, lam_hist, phi_hist = self._stack_defects(qs_out, vs_out, prs_out, us_out)
            else:
                qs_out, vs_out, prs_out, us_out = self._unpack(best['z'])
                lam_hist = best['lam']
                phi_hist = best['phi']
            
            task_loss_t = self._task_cost(qs_out, vs_out, prs_out, us_out, goal, obstacle_pos)

        return {
            'loss':task_loss_t,
            'ALMloss': float(best['ALMloss']),
            'q_final': qs_out[-1].detach().cpu().numpy(),
            'u_seq': us_out.detach().cpu().numpy(),
            'trajectory': qs_out.detach().cpu().numpy(),
            'pusher_trajectory': prs_out.detach().cpu().numpy(),
            'contact_forces': lam_hist.detach().cpu().numpy(),
            'signed_distances': phi_hist.detach().cpu().numpy()
        }