# ioc.py
import torch
from typing import Dict, Optional, Tuple
from .utils import save_demo_npz, load_demo_npz, save_demo_pt, load_demo_pt, pack_demo
from icecream import ic

Tensor = torch.Tensor

def _to_double(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.double, device=device)
    return torch.tensor(x, dtype=torch.double, device=device)

class IOCFitter:
    """
    Inverse Optimal Control on a differentiable single-shooting optimal control problem.

    Mode 1: KKT stationarity (recommended)
        Given a demonstrated (locally optimal) control sequence u_demo and problem data,
        find non-negative weights w that minimize || dJ_w/du (u_demo) ||^2.

    Mode 2: (optional) Unrolled bilevel trajectory matching
        Unroll K steps of gradient descent on u starting from a fresh init for each w,
        then match the resulting state/trajectory to the demonstrated one.

    Assumptions:
        - You use the 'shooting' pipeline (controls-only decision variable).
        - rollout(...) returns a scalar total loss and individual terms, and accepts
          weights via w_target, w_v, w_ctrl, w_obs (and optionally others).
        - Everything is differentiable (torch.double).

    """
    def __init__(self,
                 traj_opt,   # instance of TrajectoryOptimizer
                 demo: Dict, # dict carrying demo data (see fit_* docstrings)
                 learnable_terms: Tuple[str, ...] = ("w_target","w_v","w_ctrl","w_obs"),
                 normalize_weights: bool = True,
                 device: Optional[torch.device] = None):

        self.traj_opt = traj_opt
        self.device = traj_opt.device if device is None else device

        # Store demonstration
        # Required: q0, v0, pusher0, goal, u_demo (T,2)
        # Optional: obstacle_pos
        self.q0 = _to_double(demo["q0"], self.device)
        self.q0.requires_grad = True
        self.v0 = _to_double(demo["v0"], self.device)
        self.v0.requires_grad = True
        self.pusher0 = _to_double(demo["pusher0"], self.device)
        self.pusher0.requires_grad = True
        self.goal = _to_double(demo["goal"], self.device)
        self.obstacle_pos = None
        if "obstacle_pos" in demo and demo["obstacle_pos"] is not None:
            self.obstacle_pos = _to_double(demo["obstacle_pos"], self.device)

        u_demo_np_or_t = demo["u_demo"]
        self.u_demo = _to_double(u_demo_np_or_t, self.device).detach()
        assert self.u_demo.shape == (self.traj_opt.horizon, 2), "u_demo must be (T,2)"

        # Which cost terms are learnable
        self.learnable_terms = learnable_terms
        self.normalize_weights = normalize_weights

        # Build a minimal parameter vector theta -> positive weights via softplus
        self.theta = torch.nn.Parameter(torch.rand(len(self.learnable_terms), dtype=torch.double, device=self.device))

        # Names -> indices helper
        self._name_to_idx = {name: i for i, name in enumerate(self.learnable_terms)}

    def _weights_from_theta(self) -> Dict[str, Tensor]:
        # positive via softplus
        raw = torch.nn.functional.softplus(self.theta) #self.theta #  # non-negative
        if self.normalize_weights:
            s = raw.sum().clamp_min(1e-12)
            raw = raw / s
        # Fill dict with defaults (use your current defaults for the non-learned)
        w = {
            "w_target": torch.tensor(20.0, dtype=torch.double, device=self.device),
            "w_v":      torch.tensor(0.1,  dtype=torch.double, device=self.device),
            "w_ctrl":   torch.tensor(10, dtype=torch.double, device=self.device),
            "w_obs":    torch.tensor(1.0,  dtype=torch.double, device=self.device),
        }
        for name, idx in self._name_to_idx.items():
            w[name] = raw[idx]
        return w

    @torch.no_grad()
    def current_weights(self) -> Dict[str, float]:
        w = self._weights_from_theta()
        return {k: float(v.item()) for k,v in w.items()}

    def _rollout_with_weights(self, u_seq: Tensor, weights: Dict[str, Tensor]):
        """
        Calls your rollout(...) exactly like TrajectoryOptimizer.optimize() does,
        but with explicit weights 'weights'.
        Returns (loss, q_final, lambdas, phis, qs, pusher_traj, goal_term, ctrl_term, v_term, obs_term, pen_term)
        """
        from .dynamics import rollout

        return rollout(
            u_seq, self.q0, self.v0, self.pusher0, self.traj_opt.horizon, self.traj_opt.dt,
            self.traj_opt.m, self.traj_opt.Izz, self.traj_opt.half, self.traj_opt.mu, self.goal,
            w_target = weights["w_target"], w_v = weights["w_v"], w_ctrl = weights["w_ctrl"], w_obs = weights["w_obs"],
            qp_solver = self.traj_opt.qp_solver,
            dynamics_solver = self.traj_opt.dynamics_solver,
            obstacle_pos = self.obstacle_pos,
            device = self.device
        )

    # -----------------------------
    # 1) KKT stationarity objective
    # -----------------------------
    def fit_weights_kkt(self,
                        max_outer_iters: int = 300,
                        lr: float = 0.05,
                        weight_sum_target: Optional[float] = 1.0,
                        weight_sum_reg: float = 1e-3,
                        verbose: bool = True):
        """
        Learn weights by minimizing || dJ/du (u_demo) ||^2 subject to non-negativity (via softplus)
        and optional (sum - target)^2 regularization for identifiability.

        Inputs:
         - u_demo: from demo dict (already stored)
        """
        opt = torch.optim.Adam([self.theta], lr=lr)

        for it in range(1, max_outer_iters+1):
            opt.zero_grad()

            # Create a leaf that requires grad to take grad wrt u_demo
            u_demo_var = self.u_demo.clone().detach().requires_grad_(True)

            weights = self._weights_from_theta()
            ic(weights)
            # Compute scalar objective at demo controls
            total, *_terms = self._rollout_with_weights(u_demo_var, weights)

            # ic(total.requires_grad, u_demo_var.requires_grad)
            # gradient wrt controls at u_demo (stationarity)
            (grad_u,) = torch.autograd.grad(total, u_demo_var, create_graph=True, retain_graph=True)

            kkt_stationarity = (grad_u**2).sum()
            ic(kkt_stationarity)
            exit()
            # Optional L2 regularization pulling sum(weights) to a target (usually 1.0)
            reg = torch.tensor(0.0, dtype=torch.double, device=self.device)
            if self.normalize_weights is False and weight_sum_target is not None:
                w_vec = torch.stack([weights[name] for name in self.learnable_terms], dim=0)
                reg = weight_sum_reg * (w_vec.sum() - weight_sum_target).pow(2)

            loss_outer = kkt_stationarity + reg
            loss_outer.backward()
            opt.step()

            if verbose and (it % 1 == 0 or it == 1 or it == max_outer_iters):
                ws = self.current_weights()
                print(f"[IOC-KKT] iter {it:4d} | obj={loss_outer.item():.6e} | "
                      + " ".join([f"{k}={ws[k]:.4f}" for k in self.learnable_terms]))

        return self.current_weights()

    # -----------------------------------------------------
    # 2) (Optional) Unrolled bilevel trajectory matching
    # -----------------------------------------------------
    def fit_weights_unrolled(self,
                             u_init: Optional[Tensor] = None,
                             match: str = "trajectory",   # "trajectory" or "terminal"
                             inner_steps: int = 30,
                             inner_lr: float = 0.05,
                             max_outer_iters: int = 200,
                             outer_lr: float = 0.05,
                             verbose: bool = True):
        """
        Bilevel (unrolled) alternative: For each outer iteration (weights),
        unroll inner gradient steps on u to minimize J_w(u); then match
        the resulting (q, optionally entire trajectory) to the demo.

        This is more compute-heavy but sometimes more robust in practice.

        Args:
            u_init: optional starting control sequence (T,2). If None, zeros.
            match:  "trajectory" uses MSE over all states; "terminal" matches q_T only.
        """
        # Build demo trajectory once (using the provided demo controls)
        with torch.no_grad():
            tmp_w = self._weights_from_theta()  # any weights (they don't matter to simulate with demo u)
            _, qf_d, _, _, qs_demo, pusher_demo, *_ = self._rollout_with_weights(self.u_demo, tmp_w)

        qs_demo = qs_demo.detach()  # (T+1,3)

        if u_init is None:
            u0 = torch.zeros_like(self.u_demo)
        else:
            u0 = _to_double(u_init, self.device)
            assert u0.shape == self.u_demo.shape

        opt_outer = torch.optim.Adam([self.theta], lr=outer_lr)

        for it in range(1, max_outer_iters+1):
            opt_outer.zero_grad()

            # Inner variable to unroll
            u_var = u0.clone().detach().requires_grad_(True)
            inner_opt = torch.optim.SGD([u_var], lr=inner_lr)

            weights = self._weights_from_theta()

            # Unroll inner solver K steps
            for _ in range(inner_steps):
                inner_opt.zero_grad()
                loss, *_ = self._rollout_with_weights(u_var, weights)
                loss.backward()
                inner_opt.step()

            # After unrolled steps, compare rollouts to demo
            _, qf_hat, _, _, qs_hat, _, *_ = self._rollout_with_weights(u_var, weights)

            if match == "terminal":
                outer_obj = torch.nn.functional.mse_loss(qf_hat, qs_demo[-1])
            else:
                outer_obj = torch.nn.functional.mse_loss(qs_hat, qs_demo)

            outer_obj.backward()
            opt_outer.step()

            if verbose and (it % 10 == 0 or it == 1 or it == max_outer_iters):
                ws = self.current_weights()
                print(f"[IOC-Unrolled] iter {it:4d} | obj={outer_obj.item():.6e} | "
                      + " ".join([f"{k}={ws[k]:.4f}" for k in self.learnable_terms]))

        return self.current_weights()
