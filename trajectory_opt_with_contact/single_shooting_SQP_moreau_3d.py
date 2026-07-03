"""
Single-shooting SQP for 3D velocity-commanded sphere robots.

This is the 3D counterpart of single_shooting_SQP_moreau.py, built around
dynamics_3d_velocity_robot.step_cube_sphere_velocity_robot_ip.

Decision variable:
    U[t, robot, dim] with shape (T, 2, 3)

Rollout state:
    cube:   p, q, v
    robots: p1, p2

Objective residuals:
    control effort
    control smoothness
    terminal object position
    terminal object orientation (optional weight)
    terminal object velocity
    contact maintenance via phi > 0

No symmetry or problem-specific grasp-shape term is included.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import time

import moreau
import moreau.torch as moreau_torch
import numpy as np
import torch

from trajectory_opt_with_contact.dynamics_3d import IPMOptions3D
from trajectory_opt_with_contact.dynamics_3d_velocity_robot import (
    step_cube_sphere_velocity_robot_ground_ip,
    step_cube_sphere_velocity_robot_ip,
)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / (torch.linalg.norm(q) + 1e-12)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_orientation_residual(q: torch.Tensor, q_goal: torch.Tensor) -> torch.Tensor:
    """Small-angle orientation residual as a 3-vector."""
    q = quat_normalize(q)
    q_goal = quat_normalize(q_goal)
    q_err = quat_multiply(q_goal, quat_conjugate(q))
    q_err = torch.where(q_err[0] < 0.0, -q_err, q_err)
    return 2.0 * q_err[1:]


@dataclass
class SQPConfig3D:
    maxIters: int = 20
    cost_tol: float = 1e-6
    proj_grad_tol: float = 1e-4

    use_line_search: bool = True
    line_search_mode: str = "monotone_cost"
    line_search_max_iters: int = 10
    line_search_beta: float = 0.5
    line_search_ftol: float = 1e-8
    reject_line_search_fail: bool = False
    line_search_log: bool = False
    filter_pos_tol: float = 1e-4
    filter_orient_tol: float = 1e-4
    filter_vel_tol: float = 1e-4
    filter_contact_abs_tol: float = 1e-3
    filter_contact_growth: float = 2.0
    filter_cost_abs_tol: float = 1e-2
    filter_cost_growth: float = 0.05
    filter_pos_regress_tol: float = 2e-2
    filter_orient_regress_tol: float = 5e-2
    filter_vel_regress_tol: float = 5e-1

    u_min: float = -1.5
    u_max: float = 1.5

    use_trust_region: bool = True
    trust_region_iters: int = 3
    trust_region_size: float = 1.0
    trust_region_min: float = 1e-3
    trust_region_max: float = 5.0
    trust_region_shrink: float = 0.5
    trust_region_expand: float = 1.2
    trust_region_expand_alpha: float = 0.5
    trust_region_shrink_alpha: float = 0.05

    use_gauss_newton: bool = True
    hessian_regularization: float = 1e-6
    adaptive_hessian_regularization: bool = True
    hessian_regularization_min: float = 1e-8
    hessian_regularization_max: float = 1e-2
    hessian_regularization_shrink: float = 0.5
    hessian_regularization_grow: float = 10.0
    adaptive_target_mu: bool = True
    target_mu_min: float = 0.0
    target_mu_max: float = 1e-2
    target_mu_shrink: float = 0.5
    target_mu_grow: float = 10.0
    target_mu_grow_alpha: float = 0.125
    target_mu_shrink_alpha: float = 0.5
    target_mu_escape_failures: int = 2
    final_target_mu_min_iters: int = 10
    return_best_iterate: bool = False
    reevaluate_final_at_target_mu_min: bool = False
    vectorize_jacobian: bool = True
    profile_timing: bool = False
    compute_final_stationarity: bool = True

    u_init_mode: str = "zero"
    verbose: bool = False


@dataclass
class CostWeights3D:
    wControl: object = 1e-2
    wControlSmooth: object = 1e-1
    wObjVel: object = 1.0
    wTargetPos: object = 100.0
    wTargetOrient: object = 0.0
    wContact: object = 10.0


class SingleShootingSQP3D:
    """Single-shooting SQP/Gauss-Newton optimizer for the 3D velocity model."""

    def __init__(
        self,
        mass: float,
        side_length: float,
        mu: float,
        horizon: int,
        dt: float,
        sdf_grid,
        r_sphere: float,
        m_robot: float = 1.0,
        I_body: Optional[torch.Tensor] = None,
        device: str = "cpu",
        ipmOpts: Optional[IPMOptions3D] = None,
        use_ground_contact: bool = False,
        ground_z: float = 0.0,
        ground_sharpness: float = 80.0,
    ):
        self.mass = mass
        self.side = side_length
        self.half = side_length / 2.0
        self.mu = mu
        self.horizon = horizon
        self.dt = dt
        self.sdf_grid = sdf_grid
        self.r_sphere = r_sphere
        self.m_robot = m_robot
        self.device = torch.device(device)
        self.ipmOpts = ipmOpts if ipmOpts is not None else IPMOptions3D()
        self.use_ground_contact = use_ground_contact
        self.ground_z = ground_z
        self.ground_sharpness = ground_sharpness

        if I_body is None:
            I = (1.0 / 6.0) * mass * (side_length ** 2)
            I_body = torch.tensor([I, I, I], dtype=torch.float64, device=self.device)
        self.I_body = self._to_tensor(I_body)

        self._init_moreau_torch_solver(device)

    def _init_moreau_torch_solver(self, device: str):
        T = self.horizon
        n = T * 6
        m = n * 4

        P_row_offsets = torch.arange(0, (n + 1) * n, n, dtype=torch.int32)
        P_col_indices = torch.arange(n, dtype=torch.int32).repeat(n)

        A_row_offsets = torch.arange(0, m + 1, dtype=torch.int32)
        A_col_indices = torch.zeros(m, dtype=torch.int32)
        for idx in range(n):
            base = idx * 4
            A_col_indices[base + 0] = idx
            A_col_indices[base + 1] = idx
            A_col_indices[base + 2] = idx
            A_col_indices[base + 3] = idx

        cones = moreau.Cones(num_zero_cones=0, num_nonneg_cones=m)
        moreau_device = (
            "cuda"
            if torch.device(device).type == "cuda" and moreau.device_available("cuda")
            else "cpu"
        )
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

    @staticmethod
    def _line_search_metrics(result: Dict) -> Dict[str, float]:
        return {
            "cost": result["loss"].item(),
            "pos": result["terminalPosNorm"].item(),
            "orient": result["terminalOrientNorm"].item(),
            "vel": math.sqrt(max(result["terminalVelEnergy"].item(), 0.0)),
            "contact": result["contactCost"].item(),
        }

    def _accept_line_search_trial(
        self,
        trial_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        cfg: SQPConfig3D,
    ) -> Tuple[bool, str]:
        cost_ok = trial_metrics["cost"] <= current_metrics["cost"] + cfg.line_search_ftol
        if cfg.line_search_mode == "monotone_cost":
            return cost_ok, "cost" if cost_ok else "reject"

        if cfg.line_search_mode not in ("progress_filter", "guarded_progress_filter"):
            raise ValueError(f"unknown line_search_mode: {cfg.line_search_mode}")

        pos_ok = trial_metrics["pos"] <= current_metrics["pos"] - cfg.filter_pos_tol
        orient_ok = trial_metrics["orient"] <= current_metrics["orient"] - cfg.filter_orient_tol
        vel_ok = trial_metrics["vel"] <= current_metrics["vel"] - cfg.filter_vel_tol
        contact_limit = max(
            current_metrics["contact"] * cfg.filter_contact_growth,
            current_metrics["contact"] + cfg.filter_contact_abs_tol,
        )
        contact_ok = trial_metrics["contact"] <= contact_limit

        if not contact_ok:
            return False, "contact_guard"
        if cost_ok:
            return True, "cost"

        if cfg.line_search_mode == "guarded_progress_filter":
            cost_limit = current_metrics["cost"] + max(
                cfg.filter_cost_abs_tol,
                abs(current_metrics["cost"]) * cfg.filter_cost_growth,
            )
            cost_guard_ok = trial_metrics["cost"] <= cost_limit
            pos_guard_ok = trial_metrics["pos"] <= current_metrics["pos"] + cfg.filter_pos_regress_tol
            orient_guard_ok = (
                trial_metrics["orient"]
                <= current_metrics["orient"] + cfg.filter_orient_regress_tol
            )
            vel_guard_ok = trial_metrics["vel"] <= current_metrics["vel"] + cfg.filter_vel_regress_tol
            if not cost_guard_ok:
                return False, "cost_guard"
            if not (pos_guard_ok and orient_guard_ok and vel_guard_ok):
                return False, "metric_guard"

        if pos_ok:
            return True, "pos"
        if vel_ok:
            return True, "vel"
        if orient_ok:
            return True, "orient"
        return False, "reject"

    def forward_simulate(
        self,
        p0: torch.Tensor,
        q0: torch.Tensor,
        v0: torch.Tensor,
        p1_0: torch.Tensor,
        p2_0: torch.Tensor,
        U: torch.Tensor,
        store_contact_data: bool = False,
    ) -> Tuple:
        T = self.horizon
        ps = [p0]
        qs = [q0]
        vs = [v0]
        p1s = [p1_0]
        p2s = [p2_0]
        vs1s = []
        vs2s = []

        if store_contact_data:
            contact_forces = []
            signed_distances = []
            ground_forces = []
            ground_distances = []

        p, q, v = p0, q0, v0
        p1, p2 = p1_0, p2_0
        z_prev = None
        for t in range(T):
            if self.use_ground_contact:
                result = step_cube_sphere_velocity_robot_ground_ip(
                    p, q, v, p1, p2, U[t, 0], U[t, 1],
                    self.dt, self.mass, self.I_body, self.m_robot, self.mu,
                    self.sdf_grid, self.r_sphere, self.half,
                    ground_z=self.ground_z,
                    ground_sharpness=self.ground_sharpness,
                    ipm_opts=self.ipmOpts,
                    z_prev=z_prev,
                )
                (
                    p_next, q_next, v_next,
                    p1_next, p2_next, vs1_next, vs2_next,
                    lam1, lam2, phi1, phi2, _lamG, _phiG, z_prev,
                ) = result
            else:
                result = step_cube_sphere_velocity_robot_ip(
                    p, q, v, p1, p2, U[t, 0], U[t, 1],
                    self.dt, self.mass, self.I_body, self.m_robot, self.mu,
                    self.sdf_grid, self.r_sphere,
                    ipm_opts=self.ipmOpts,
                    z_prev=z_prev,
                )
                (
                    p_next, q_next, v_next,
                    p1_next, p2_next, vs1_next, vs2_next,
                    lam1, lam2, phi1, phi2, z_prev,
                ) = result
                _lamG = None
                _phiG = None

            ps.append(p_next)
            qs.append(q_next)
            vs.append(v_next)
            p1s.append(p1_next)
            p2s.append(p2_next)
            vs1s.append(vs1_next)
            vs2s.append(vs2_next)

            if store_contact_data:
                contact_forces.append(torch.stack([lam1, lam2]))
                signed_distances.append(torch.stack([phi1, phi2]))
                if self.use_ground_contact:
                    ground_forces.append(_lamG)
                    ground_distances.append(_phiG)

            p, q, v = p_next, q_next, v_next
            p1, p2 = p1_next, p2_next

        ps = torch.stack(ps)
        qs = torch.stack(qs)
        vs = torch.stack(vs)
        p1s = torch.stack(p1s)
        p2s = torch.stack(p2s)
        vs1s = torch.stack(vs1s)
        vs2s = torch.stack(vs2s)
        terminal_vel_energy = (vs[-1] ** 2).sum()

        if store_contact_data:
            contact_forces = torch.stack(contact_forces)
            signed_distances = torch.stack(signed_distances)
            if self.use_ground_contact:
                ground_forces = torch.stack(ground_forces)
                ground_distances = torch.stack(ground_distances)
            else:
                ground_forces = None
                ground_distances = None
            return (
                ps, qs, vs, p1s, p2s, vs1s, vs2s,
                terminal_vel_energy, contact_forces, signed_distances,
                ground_forces, ground_distances,
            )
        return ps, qs, vs, p1s, p2s, vs1s, vs2s, terminal_vel_energy, None, None, None, None

    def compute_residuals(
        self,
        U: torch.Tensor,
        p0: torch.Tensor,
        q0: torch.Tensor,
        v0: torch.Tensor,
        p1_0: torch.Tensor,
        p2_0: torch.Tensor,
        goal_p: torch.Tensor,
        w: CostWeights3D,
        goal_q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            ps, qs, vs, _, _, _, _,
            _, _, phis, _, _,
        ) = self.forward_simulate(p0, q0, v0, p1_0, p2_0, U, store_contact_data=True)

        def _as_tensor(w_val):
            if isinstance(w_val, torch.Tensor):
                return w_val.to(dtype=U.dtype, device=self.device)
            return torch.tensor(w_val, dtype=U.dtype, device=self.device)

        residuals = []
        residuals.append(torch.sqrt(_as_tensor(w.wControl)) * U.flatten())

        if self.horizon > 1:
            residuals.append(
                torch.sqrt(_as_tensor(w.wControlSmooth))
                * (U[1:] - U[:-1]).flatten()
            )

        residuals.append(torch.sqrt(_as_tensor(w.wTargetPos)) * (ps[-1] - goal_p))

        if goal_q is None:
            goal_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=U.dtype, device=self.device)
        residuals.append(
            torch.sqrt(_as_tensor(w.wTargetOrient))
            * quat_orientation_residual(qs[-1], goal_q)
        )

        residuals.append(torch.sqrt(_as_tensor(w.wObjVel)) * vs[-1])
        residuals.append(torch.sqrt(_as_tensor(w.wContact)) * phis.clamp(min=0.0).flatten())

        return torch.cat(residuals)

    def compute_gauss_newton_hessian(
        self,
        U: torch.Tensor,
        p0: torch.Tensor,
        q0: torch.Tensor,
        v0: torch.Tensor,
        p1_0: torch.Tensor,
        p2_0: torch.Tensor,
        goal_p: torch.Tensor,
        w: CostWeights3D,
        goal_q: Optional[torch.Tensor] = None,
        regularization: float = 1e-6,
        vectorize_jacobian: bool = False,
    ):
        timing = {}
        U_leaf = U.detach().requires_grad_(True)

        with torch.enable_grad():
            t0 = time.time()
            r = self.compute_residuals(U_leaf, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            timing["residual_eval"] = time.time() - t0

            t0 = time.time()
            J = torch.autograd.functional.jacobian(
                lambda U_flat: self.compute_residuals(
                    U_flat.reshape(self.horizon, 2, 3),
                    p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q,
                ),
                U_leaf.flatten(),
                create_graph=False,
                # Fast reverse-mode Jacobian. The 3D velocity model backward is
                # written in VJP style, matching the 2D dynamics path, so this
                # can use PyTorch's vectorized batched VJP.
                vectorize=vectorize_jacobian,
                strategy="reverse-mode",
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            timing["jacobian_computation"] = time.time() - t0

        t0 = time.time()
        g = J.T @ r
        P = J.T @ J
        if regularization > 0:
            P = P + regularization * torch.eye(P.shape[0], dtype=P.dtype, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        timing["hessian_assembly"] = time.time() - t0

        return P, g, timing, J.detach(), r.detach()

    def build_qp_matrices(
        self,
        U_curr: torch.Tensor,
        p0: torch.Tensor,
        q0: torch.Tensor,
        v0: torch.Tensor,
        p1_0: torch.Tensor,
        p2_0: torch.Tensor,
        goal_p: torch.Tensor,
        w: CostWeights3D,
        cfg: SQPConfig3D,
        goal_q: Optional[torch.Tensor] = None,
        regularization: Optional[float] = None,
        trust_radius: Optional[float] = None,
    ):
        n = self.horizon * 6
        reg = cfg.hessian_regularization if regularization is None else float(regularization)

        if cfg.use_gauss_newton:
            P_torch, g_torch, timing, _, _ = self.compute_gauss_newton_hessian(
                U_curr, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q,
                regularization=reg,
                vectorize_jacobian=cfg.vectorize_jacobian,
            )
        else:
            U_ad = U_curr.clone().detach().requires_grad_(True)
            r = self.compute_residuals(U_ad, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q)
            loss = 0.5 * r.dot(r)
            loss.backward()
            g_torch = U_ad.grad.flatten()
            P_torch = torch.eye(n, dtype=torch.float64, device=self.device)
            timing = {"identity_hessian": 0.0}

        U_flat = U_curr.detach().flatten()
        A_values = torch.tensor(
            [-1.0, 1.0, -1.0, 1.0], dtype=torch.float64, device=self.device
        ).repeat(n)
        b_lower = U_flat - cfg.u_min
        b_upper = cfg.u_max - U_flat
        if cfg.use_trust_region and trust_radius is not None:
            trust_bound = float(trust_radius)
        else:
            trust_bound = 1e12
        b_trust = torch.full_like(U_flat, trust_bound)
        b_vec = torch.stack([b_lower, b_upper, b_trust, b_trust], dim=1).flatten()

        t0 = time.time()
        solution = self._moreau_torch_solver.solve(
            P_torch.flatten(), A_values, g_torch, b_vec
        )
        timing["qp_solve"] = time.time() - t0
        delta_U = solution.x.reshape(self.horizon, 2, 3)
        return delta_U, g_torch, timing

    def evaluate_cost(
        self,
        U: torch.Tensor,
        p0: torch.Tensor,
        q0: torch.Tensor,
        v0: torch.Tensor,
        p1_0: torch.Tensor,
        p2_0: torch.Tensor,
        goal_p: torch.Tensor,
        w: CostWeights3D,
        goal_q: Optional[torch.Tensor] = None,
    ) -> Dict:
        (
            ps, qs, vs, p1s, p2s, vs1s, vs2s,
            terminal_vel_energy, contact_forces, phis, ground_forces, ground_phis,
        ) = self.forward_simulate(p0, q0, v0, p1_0, p2_0, U, store_contact_data=True)

        control_energy = (U ** 2).sum()
        control_smooth = ((U[1:] - U[:-1]) ** 2).sum() if len(U) > 1 else U.new_zeros(())
        target_pos_cost = ((ps[-1] - goal_p) ** 2).sum()
        if goal_q is None:
            goal_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=U.dtype, device=self.device)
        orient_res = quat_orientation_residual(qs[-1], goal_q)
        target_orient_cost = (orient_res ** 2).sum()
        contact_cost = (phis.clamp(min=0.0) ** 2).sum()

        def _w(val):
            return float(val) if not isinstance(val, torch.Tensor) else val.item()

        loss = (
            _w(w.wControl) * control_energy
            + _w(w.wControlSmooth) * control_smooth
            + _w(w.wObjVel) * terminal_vel_energy
            + _w(w.wTargetPos) * target_pos_cost
            + _w(w.wTargetOrient) * target_orient_cost
            + _w(w.wContact) * contact_cost
        )

        return {
            "loss": loss,
            "ps": ps,
            "qs": qs,
            "vs": vs,
            "p1s": p1s,
            "p2s": p2s,
            "vs1s": vs1s,
            "vs2s": vs2s,
            "contact_forces": contact_forces,
            "phis": phis,
            "ground_forces": ground_forces,
            "ground_phis": ground_phis,
            "controlEnergy": control_energy,
            "controlSmooth": control_smooth,
            "terminalPosNormSq": target_pos_cost,
            "terminalOrientNormSq": target_orient_cost,
            "terminalOrientNorm": torch.linalg.norm(orient_res),
            "terminalVelEnergy": terminal_vel_energy,
            "terminalPosNorm": torch.linalg.norm(ps[-1] - goal_p),
            "contactCost": contact_cost,
            "contactSeparationRatio": (phis > 0.01).float().mean(),
            "meanPhi": phis.mean(),
        }

    def optimize(
        self,
        p0,
        q0,
        v0,
        p1_0,
        p2_0,
        goal_p,
        goal_q=None,
        U_init=None,
        cfg: Optional[SQPConfig3D] = None,
        w: Optional[CostWeights3D] = None,
        verbose: bool = True,
    ) -> Dict:
        p0 = self._to_tensor(p0)
        q0 = self._to_tensor(q0)
        v0 = self._to_tensor(v0)
        p1_0 = self._to_tensor(p1_0)
        p2_0 = self._to_tensor(p2_0)
        goal_p = self._to_tensor(goal_p)
        goal_q = self._to_tensor(goal_q) if goal_q is not None else None

        cfg = cfg if cfg is not None else SQPConfig3D()
        w = w if w is not None else CostWeights3D()

        if U_init is None:
            U_curr = torch.zeros(self.horizon, 2, 3, dtype=torch.float64, device=self.device)
            U_init_eval = U_curr
        else:
            U_curr = self._to_tensor(U_init).reshape(self.horizon, 2, 3)
            U_init_eval = U_curr

        if verbose:
            print("\n" + "=" * 72)
            print("Single Shooting SQP 3D")
            print("=" * 72)
            print(f"Horizon: {self.horizon}, dt: {self.dt}")
            print(f"Control shape: {tuple(U_curr.shape)}")
            print(f"Control bounds: [{cfg.u_min}, {cfg.u_max}]")
            print(f"PyTorch device: {self.device}")
            print("=" * 72)

        cost_curr = None
        start_time = time.time()
        history = {
            "loss": [],
            "grad_norm": [],
            "proj_grad_norm": [],
            "step_norm": [],
            "cost_change": [],
            "alpha": [],
            "time_hessian": [],
            "time_residual_eval": [],
            "time_jacobian": [],
            "time_hessian_assembly": [],
            "time_qp": [],
            "time_line_search": [],
            "time_total_iter": [],
            "line_search_evals": [],
            "line_search_reason": [],
            "line_search_delta_cost": [],
            "line_search_delta_pos": [],
            "line_search_delta_orient": [],
            "line_search_delta_vel": [],
            "line_search_delta_contact": [],
            "trust_radius": [],
            "hessian_regularization": [],
            "target_mu": [],
            "target_mu_escape": [],
            "target_mu_hard_refine": [],
            "best_loss": [],
        }
        converged = False
        final_cost_change = float("inf")
        accepted_result = None
        trust_radius = float(cfg.trust_region_size)
        regularization_curr = float(cfg.hessian_regularization)
        original_target_mu = float(self.ipmOpts.target_mu)
        target_mu_min = (
            float(cfg.target_mu_min)
            if cfg.adaptive_target_mu and float(cfg.target_mu_min) > 0.0
            else original_target_mu
        )
        target_mu_curr = max(target_mu_min, min(float(cfg.target_mu_max), original_target_mu))
        target_mu_max = float(cfg.target_mu_max)
        target_mu_at_max_failures = 0
        last_eval_target_mu = None
        self.ipmOpts.target_mu = target_mu_curr
        best_eval = self.evaluate_cost(
            U_curr, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q
        )
        best_cost = best_eval["loss"].item()
        best_U = U_curr.detach().clone()
        best_iteration = 0

        n = self.horizon * 6
        for iteration in range(cfg.maxIters):
            iter_start = time.time()
            hard_refine = (
                cfg.adaptive_target_mu
                and cfg.final_target_mu_min_iters > 0
                and iteration >= max(cfg.maxIters - cfg.final_target_mu_min_iters, 0)
            )
            if hard_refine and target_mu_curr != target_mu_min:
                target_mu_curr = target_mu_min
                target_mu_at_max_failures = 0
            self.ipmOpts.target_mu = target_mu_curr
            if last_eval_target_mu != target_mu_curr:
                accepted_result = self.evaluate_cost(
                    U_curr, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q
                )
                cost_curr = accepted_result["loss"]
                last_eval_target_mu = target_mu_curr

            delta_U, g_vec, qp_timing = self.build_qp_matrices(
                U_curr, p0, q0, v0, p1_0, p2_0, goal_p, w, cfg, goal_q,
                regularization=regularization_curr,
                trust_radius=trust_radius,
            )
            hessian_time = time.time() - iter_start
            qp_time = qp_timing.get("qp_solve", 0.0)
            residual_time = qp_timing.get("residual_eval", 0.0)
            jacobian_time = qp_timing.get("jacobian_computation", 0.0)
            hessian_assembly_time = qp_timing.get("hessian_assembly", 0.0)

            grad_norm = g_vec.detach().norm().item()
            with torch.no_grad():
                g_flat = g_vec.detach().flatten()
                U_flat = U_curr.detach().flatten()
                proj_g = g_flat.clone()
                proj_g[(U_flat >= cfg.u_max - 1e-6) & (g_flat <= 0)] = 0.0
                proj_g[(U_flat <= cfg.u_min + 1e-6) & (g_flat >= 0)] = 0.0
                proj_grad_norm = proj_g.norm().item()

            step_norm = torch.linalg.norm(delta_U).item()

            alpha_accepted = 1.0
            line_search_evals = 0
            accept_reason = "first"
            current_metrics = (
                self._line_search_metrics(accepted_result)
                if accepted_result is not None
                else None
            )
            accepted_metrics = None
            line_start = time.time()
            if cfg.use_line_search and cost_curr is not None:
                alpha = 1.0
                for ls_iter in range(cfg.line_search_max_iters):
                    U_trial = U_curr + alpha * delta_U
                    result = self.evaluate_cost(
                        U_trial, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q
                    )
                    line_search_evals += 1
                    cost_trial = result["loss"]
                    trial_metrics = self._line_search_metrics(result)
                    accept_trial, accept_reason = self._accept_line_search_trial(
                        trial_metrics, current_metrics, cfg
                    )
                    if accept_trial:
                        U_new = U_trial
                        cost_new = cost_trial
                        alpha_accepted = alpha
                        accepted_metrics = trial_metrics
                        break
                    alpha *= cfg.line_search_beta
                else:
                    if cfg.reject_line_search_fail:
                        U_new = U_curr
                        result = self.evaluate_cost(
                            U_new, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q
                        )
                        line_search_evals += 1
                        cost_new = result["loss"]
                        alpha_accepted = 0.0
                        accept_reason = "failed_reject"
                        accepted_metrics = self._line_search_metrics(result)
                    else:
                        U_new = U_trial
                        cost_new = cost_trial
                        alpha_accepted = alpha
                        accept_reason = "failed_use_last"
                        accepted_metrics = self._line_search_metrics(result)
            else:
                U_new = U_curr + delta_U
                result = self.evaluate_cost(
                    U_new, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q
                )
                line_search_evals += 1
                cost_new = result["loss"]
                accepted_metrics = self._line_search_metrics(result)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            line_time = time.time() - line_start

            if cfg.return_best_iterate and cost_new.item() < best_cost:
                best_cost = cost_new.item()
                best_U = U_new.detach().clone()
                best_iteration = iteration + 1

            if current_metrics is None:
                delta_metrics = {k: float("nan") for k in ("cost", "pos", "orient", "vel", "contact")}
            else:
                delta_metrics = {
                    k: accepted_metrics[k] - current_metrics[k]
                    for k in ("cost", "pos", "orient", "vel", "contact")
                }

            history["loss"].append(cost_new.item())
            history["grad_norm"].append(grad_norm)
            history["proj_grad_norm"].append(proj_grad_norm)
            history["step_norm"].append(step_norm)
            history["alpha"].append(alpha_accepted)
            history["time_hessian"].append(hessian_time)
            history["time_residual_eval"].append(residual_time)
            history["time_jacobian"].append(jacobian_time)
            history["time_hessian_assembly"].append(hessian_assembly_time)
            history["time_qp"].append(qp_time)
            history["time_line_search"].append(line_time)
            history["time_total_iter"].append(time.time() - iter_start)
            history["line_search_evals"].append(line_search_evals)
            history["line_search_reason"].append(accept_reason)
            history["line_search_delta_cost"].append(delta_metrics["cost"])
            history["line_search_delta_pos"].append(delta_metrics["pos"])
            history["line_search_delta_orient"].append(delta_metrics["orient"])
            history["line_search_delta_vel"].append(delta_metrics["vel"])
            history["line_search_delta_contact"].append(delta_metrics["contact"])
            history["trust_radius"].append(trust_radius)
            history["hessian_regularization"].append(regularization_curr)
            history["target_mu"].append(target_mu_curr)
            target_mu_escape = False
            history["target_mu_hard_refine"].append(hard_refine)
            history["best_loss"].append(best_cost if cfg.return_best_iterate else float("nan"))

            if verbose:
                print(f"\nSQP iter {iteration + 1}/{cfg.maxIters}")
                print(f"  cost={cost_new.item():.6e} grad={grad_norm:.3e} "
                      f"proj={proj_grad_norm:.3e} step={step_norm:.3e} alpha={alpha_accepted:.3f} "
                      f"trust={trust_radius:.3e} reg={regularization_curr:.1e} "
                      f"target_mu={target_mu_curr:.1e}")
                if cfg.line_search_log:
                    print(
                        "  line-search: "
                        f"mode={cfg.line_search_mode} reason={accept_reason} "
                        f"evals={line_search_evals} "
                        f"dJ={delta_metrics['cost']:+.3e} "
                        f"dpos={delta_metrics['pos']:+.3e} "
                        f"dorient={delta_metrics['orient']:+.3e} "
                        f"dvel={delta_metrics['vel']:+.3e} "
                        f"dcontact={delta_metrics['contact']:+.3e}"
                    )
                print(f"  terminal_pos={result['ps'][-1].detach().cpu().numpy().tolist()} "
                      f"target={goal_p.detach().cpu().numpy().tolist()}")
                print(f"  terminal_q={result['qs'][-1].detach().cpu().numpy().tolist()} "
                      f"orient_err={result['terminalOrientNorm'].item():.3e}")
                print(f"  costs: target={float(w.wTargetPos) * result['terminalPosNormSq'].item():.3e} "
                      f"orient={float(w.wTargetOrient) * result['terminalOrientNormSq'].item():.3e} "
                      f"vel={float(w.wObjVel) * result['terminalVelEnergy'].item():.3e} "
                      f"control={float(w.wControl) * result['controlEnergy'].item():.3e} "
                      f"smooth={float(w.wControlSmooth) * result['controlSmooth'].item():.3e} "
                      f"contact={float(w.wContact) * result['contactCost'].item():.3e}")
                if cfg.profile_timing:
                    print(
                        "  timing: "
                        f"build={hessian_time:.3f}s "
                        f"residual={residual_time:.3f}s "
                        f"jacobian={jacobian_time:.3f}s "
                        f"assembly={hessian_assembly_time:.3f}s "
                        f"qp={qp_time:.3f}s "
                        f"line={line_time:.3f}s "
                        f"ls_evals={line_search_evals}"
                    )

            line_search_failed = accept_reason in ("failed_reject", "failed_use_last")
            target_mu_was_at_max = target_mu_curr >= target_mu_max * (1.0 - 1e-12)
            if target_mu_was_at_max and alpha_accepted <= 0.0:
                target_mu_at_max_failures += 1
            else:
                target_mu_at_max_failures = 0
            target_mu_escape = (
                cfg.adaptive_target_mu
                and not hard_refine
                and cfg.target_mu_escape_failures > 0
                and target_mu_at_max_failures >= cfg.target_mu_escape_failures
            )
            if cfg.use_trust_region:
                if target_mu_escape:
                    trust_radius = min(cfg.trust_region_max, float(cfg.trust_region_size))
                elif alpha_accepted <= 0.0:
                    trust_radius = max(
                        cfg.trust_region_min,
                        trust_radius * cfg.trust_region_shrink,
                    )
                elif alpha_accepted >= cfg.trust_region_expand_alpha:
                    trust_radius = min(
                        cfg.trust_region_max,
                        trust_radius * cfg.trust_region_expand,
                    )
            if cfg.adaptive_hessian_regularization:
                if target_mu_escape:
                    regularization_curr = float(cfg.hessian_regularization)
                elif line_search_failed:
                    regularization_curr = min(
                        cfg.hessian_regularization_max,
                        regularization_curr * cfg.hessian_regularization_grow,
                    )
                elif alpha_accepted >= cfg.trust_region_expand_alpha:
                    regularization_curr = max(
                        cfg.hessian_regularization_min,
                        regularization_curr * cfg.hessian_regularization_shrink,
                    )
            if cfg.adaptive_target_mu and not hard_refine:
                if target_mu_escape:
                    target_mu_curr = target_mu_min
                    target_mu_at_max_failures = 0
                elif alpha_accepted < cfg.target_mu_grow_alpha:
                    target_mu_curr = min(
                        target_mu_max,
                        target_mu_curr * float(cfg.target_mu_grow),
                    )
                elif alpha_accepted >= cfg.target_mu_shrink_alpha:
                    target_mu_curr = max(
                        target_mu_min,
                        target_mu_curr * float(cfg.target_mu_shrink),
                    )
            history["target_mu_escape"].append(target_mu_escape)

            prev_cost = cost_curr
            U_curr = U_new
            cost_curr = cost_new
            accepted_result = result
            last_eval_target_mu = self.ipmOpts.target_mu

            if prev_cost is not None:
                cost_change = abs(cost_new.item() - prev_cost.item())
                final_cost_change = cost_change
                history["cost_change"].append(cost_change)
                if cost_change < cfg.cost_tol and proj_grad_norm < cfg.proj_grad_tol:
                    converged = True
                    break
            else:
                history["cost_change"].append(0.0)

        solve_time = time.time() - start_time
        U_return = best_U if cfg.return_best_iterate else U_curr

        final_target_mu = (
            target_mu_min
            if cfg.reevaluate_final_at_target_mu_min or cfg.return_best_iterate
            else float(last_eval_target_mu if last_eval_target_mu is not None else target_mu_curr)
        )
        self.ipmOpts.target_mu = final_target_mu
        if (
            not cfg.return_best_iterate
            and not cfg.reevaluate_final_at_target_mu_min
            and accepted_result is not None
        ):
            final_eval = accepted_result
        else:
            final_eval = self.evaluate_cost(
                U_return, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q
            )
        if cfg.compute_final_stationarity:
            _, g_final, _, J_r_final, r_final = self.compute_gauss_newton_hessian(
                U_return, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q,
                regularization=regularization_curr,
            )
            P_final = J_r_final.T @ J_r_final
            final_grad_norm = g_final.norm().item()

            with torch.no_grad():
                g_flat = g_final.detach().flatten()
                U_flat = U_return.detach().flatten()
                proj_g = g_flat.clone()
                proj_g[(U_flat >= cfg.u_max - 1e-6) & (g_flat <= 0)] = 0.0
                proj_g[(U_flat <= cfg.u_min + 1e-6) & (g_flat >= 0)] = 0.0
                final_proj_grad_norm = proj_g.norm().item()
            ift_data = {
                "P": P_final.detach(),
                "J_r": J_r_final.detach(),
                "r": r_final.detach(),
            }
        else:
            final_grad_norm = float("nan")
            final_proj_grad_norm = float("nan")
            g_final = None
            ift_data = None

        self.ipmOpts.target_mu = final_target_mu
        init_eval = self.evaluate_cost(
            U_init_eval, p0, q0, v0, p1_0, p2_0, goal_p, w, goal_q
        )

        return {
            "loss": final_eval["loss"].detach().cpu(),
            "p_final": final_eval["ps"][-1].detach().cpu().numpy(),
            "q_final": final_eval["qs"][-1].detach().cpu().numpy(),
            "v_final": final_eval["vs"][-1].detach().cpu().numpy(),
            "u_seq": U_return.detach().cpu().numpy(),
            "trajectory": final_eval["ps"].detach().cpu().numpy(),
            "orientation_trajectory": final_eval["qs"].detach().cpu().numpy(),
            "velocity_trajectory": final_eval["vs"].detach().cpu().numpy(),
            "robot1_trajectory": final_eval["p1s"].detach().cpu().numpy(),
            "robot2_trajectory": final_eval["p2s"].detach().cpu().numpy(),
            "robot1_velocity_trajectory": final_eval["vs1s"].detach().cpu().numpy(),
            "robot2_velocity_trajectory": final_eval["vs2s"].detach().cpu().numpy(),
            "contact_forces": final_eval["contact_forces"].detach().cpu().numpy(),
            "signed_distances": final_eval["phis"].detach().cpu().numpy(),
            "ground_forces": (
                None
                if final_eval["ground_forces"] is None
                else final_eval["ground_forces"].detach().cpu().numpy()
            ),
            "ground_signed_distances": (
                None
                if final_eval["ground_phis"] is None
                else final_eval["ground_phis"].detach().cpu().numpy()
            ),
            "initial_trajectory": init_eval["ps"].detach().cpu().numpy(),
            "initial_robot1_trajectory": init_eval["p1s"].detach().cpu().numpy(),
            "initial_robot2_trajectory": init_eval["p2s"].detach().cpu().numpy(),
            "loss_components": {
                "total": final_eval["loss"].item(),
                "control_energy": final_eval["controlEnergy"].item(),
                "control_smooth": final_eval["controlSmooth"].item(),
                "obj_vel": final_eval["terminalVelEnergy"].item(),
                "target_pos": final_eval["terminalPosNormSq"].item(),
                "target_orient": final_eval["terminalOrientNormSq"].item(),
                "contact": final_eval["contactCost"].item(),
                "w_control": w.wControl,
                "w_smooth": w.wControlSmooth,
                "w_objvel": w.wObjVel,
                "w_targetpos": w.wTargetPos,
                "w_orient": w.wTargetOrient,
                "w_contact": w.wContact,
            },
            "control_gradients": (
                None
                if g_final is None
                else g_final.detach().cpu().numpy().reshape(self.horizon, 2, 3)
            ),
            "ift_data": ift_data,
            "history": {k: np.array(v) for k, v in history.items()},
            "stationarityInfo": {
                "final_grad_norm": final_grad_norm,
                "final_proj_grad_norm": final_proj_grad_norm,
                "final_cost_change": final_cost_change,
                "cost_tol": cfg.cost_tol,
                "proj_grad_tol": cfg.proj_grad_tol,
                "exact_final_stationarity": cfg.compute_final_stationarity,
                "final_terminal_error": final_eval["terminalPosNorm"].item(),
                "final_orientation_error": final_eval["terminalOrientNorm"].item(),
                "final_target_mu": final_target_mu,
                "reevaluated_final_at_target_mu_min": cfg.reevaluate_final_at_target_mu_min,
                "best_iteration": best_iteration if cfg.return_best_iterate else None,
                "best_loss": best_cost if cfg.return_best_iterate else float("nan"),
                "returned_best_iterate": cfg.return_best_iterate,
                "converged": (
                    cfg.compute_final_stationarity
                    and (
                        converged
                        or (
                            final_cost_change < cfg.cost_tol
                            and final_proj_grad_norm < cfg.proj_grad_tol
                        )
                    )
                ),
            },
            "solve_time": solve_time,
        }

    def _to_tensor(self, x):
        if x is None:
            return None
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, dtype=torch.float64, device=self.device)
        return x.to(dtype=torch.float64, device=self.device)
