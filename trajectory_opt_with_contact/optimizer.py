import torch
from .dynamics import rollout
from .qp_solver import ContactQPSolver
from .transcription import DirectTranscriptionOptimizer  # NEW

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
                dynamics_solver: str = 'LCP', #'IP' is interior point
                # ALM knobs (forwarded to DirectTranscriptionOptimizer):
                use_second_order: bool = False, #TO solve with 1st order or 2nd order method
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
                 u_init=None, max_iters=100, lr=0.01,
                 lr_decay_step=10, lr_decay_gamma=0.5,
                 obstacle_pos=None, verbose=True):
        if self.TO_solver == 'transcription':
            return self.trans.optimize(
                q0=q0, v0=v0, pusher0=pusher0, goal=goal,
                u_init=u_init, max_iters=max_iters, lr=lr,
                obstacle_pos=obstacle_pos, verbose=verbose
            )

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

        opt = torch.optim.Adam([u_seq], lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay_gamma)

        from .dynamics import rollout
        for it in range(max_iters):
            opt.zero_grad()
            print('-------------------------------------')
            loss, q_final, lambdas, phis, qs, pusher_traj, goal_term, ctrl_term, v_term, obs_term, pen_term = rollout(
                u_seq, q0, v0, pusher0, self.horizon, self.dt,
                self.m, self.Izz, self.half, self.mu, goal,
                qp_solver = self.qp_solver,
                dynamics_solver=self.dynamics_solver, obstacle_pos=obstacle_pos,
                device=self.device
            )
            loss.backward()
            opt.step()
            sched.step()
            def to_scalar(x):
                return x.item() if torch.is_tensor(x) else float(x)
            if verbose and (it+1) % 1 == 0:
                print(f"[Shooting] Iter {it+1:3d} | "
                    f"Total={to_scalar(loss):.4f} | "
                    f"Goal={to_scalar(goal_term):.4f} | "
                    f"Ctrl={to_scalar(ctrl_term):.6f} | "
                    f"Vel={to_scalar(v_term):.4f} | "
                    f"Obs={to_scalar(obs_term):.4f} | "
                    f"Pen={to_scalar(pen_term):.4f} | "
                    f"Final x={to_scalar(q_final[0]):.3f}")

        return {
            'loss': loss.item(),
            'q_final': q_final.detach().cpu().numpy(),
            'u_seq': u_seq.detach().cpu().numpy(),
            'trajectory': qs.detach().cpu().numpy(),
            'pusher_trajectory': pusher_traj.detach().cpu().numpy(),
            'contact_forces': lambdas.detach().cpu().numpy(),
            'signed_distances': phis.detach().cpu().numpy()
        }