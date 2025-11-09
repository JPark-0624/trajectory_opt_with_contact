"""
Dynamics simulation with contact-implicit integration.

Implements time-stepping for rigid body dynamics with frictional contact
using differentiable QP solvers.
"""

import torch
from .geometry import contact_frame_and_J, contact_jacobians, perp, _linearize_obb_gap_at
from .qp_solver import ContactQPSolver
from dataclasses import dataclass
from icecream import ic

def step_square(q, v, pusher_pos, u_push, h, m, Izz, half, mu, 
                qp_solver=None, alpha_stab=0.1, eps_H=1e-4, device=None):
    """
    Single time-step integration with contact-implicit dynamics.
    
    Args:
        q: Configuration (x, y, theta)
        v: Velocity (vx, vy, omega)
        pusher_pos: Pusher position (px, py)
        u_push: Pusher control input (ux, uy)
        h: Time step
        m: Mass of the square
        Izz: Moment of inertia
        half: Half side length of the square
        mu: Coefficient of friction
        qp_solver: ContactQPSolver instance (creates new if None)
        alpha_stab: Baumgarte stabilization parameter
        eps_H: Regularization for contact matrix
        device: Torch device (auto-detected if None)
    
    Returns:
        q_next: Next configuration (3,)
        v_next: Next velocity (3,)
        pusher_pos_next: Next pusher position (2,)
        lam_star: Contact forces [λ_n, λ_t] (2,)
        phi: Signed distance
    """
    if device is None:
        device = q.device
    
    if qp_solver is None:
        qp_solver = ContactQPSolver(mu=mu, n_contacts=1)
    
    # Mass matrix inverse
    M_inv = torch.diag(torch.tensor([1.0/m, 1.0/m, 1.0/Izz], 
                                     dtype=q.dtype, device=device))
    
    # Semi-implicit Euler prediction (no external forces)
    v_pred = v
    
    # Build contact frame and Jacobian
    J, n, t, c_world, phi = contact_frame_and_J(q, v_pred, pusher_pos, half, mu)
    
    # Check for contact
    if phi > 0.001:
        # No contact → free motion with damping
        v_next = v_pred - 0.3 * v_pred * h
        q_next = torch.stack([
            q[0] + h * v_next[0],
            q[1] + h * v_next[1],
            q[2] + h * v_next[2],
        ])
        pusher_pos_next = pusher_pos + h * u_push
        lam_star = torch.zeros(2, dtype=q.dtype, device=q.device)
        return q_next, v_next, pusher_pos_next, lam_star, phi
    
    # Contact detected → solve for contact forces
    # Delassus operator: H = J M^{-1} J^T + regularization
    H = J @ M_inv @ J.T + eps_H * torch.eye(J.shape[0], dtype=q.dtype, device=device)
    
    # Relative velocity term
    v_push_proj = torch.stack([torch.dot(n, u_push), torch.dot(t, u_push)])
    
    # Linear term: b = J v_pred - v_pusher
    b = (J @ v_pred) - v_push_proj
    
    # Baumgarte stabilization for penetration
    if phi < 0.0:
        b = b.clone()
        b[0] = b[0] + alpha_stab * (phi / h)
    
    # Cholesky factorization for numerical stability
    L = torch.linalg.cholesky(H)
    
    # Solve QP for contact forces
    lam_star = qp_solver.solve(L, b)
    
    # Apply contact impulse
    v_next = v_pred + M_inv @ (J.T @ lam_star)
    
    # Damping (stabilization)
    v_next = v_next - 0.3 * v_next * h
    
    # Integrate positions
    q_next = torch.stack([
        q[0] + h * v_next[0],
        q[1] + h * v_next[1],
        q[2] + h * v_next[2],
    ])
    pusher_pos_next = pusher_pos + h * u_push
    return q_next, v_next, pusher_pos_next, lam_star, phi

def rollout(u_seq, q0, v0, pr0, horizon, h, m, Izz, half, mu, goal_xy, 
            qp_solver=None, obstacle_pos=None, device=None):
    """
    Rollout a trajectory given control sequence.
    
    Args:
        u_seq: Control sequence (T, 2) pusher velocities
        q0: Initial configuration (3,)
        v0: Initial velocity (3,)
        pr0: Initial pusher position (2,)
        horizon: Number of time steps
        h: Time step
        m: Mass
        Izz: Moment of inertia
        half: Half side length
        mu: Friction coefficient
        goal_xy: Goal configuration (3,)
        qp_solver: ContactQPSolver instance (creates new if None)
        obstacle_pos: Optional obstacle position for cost (2,)
        device: Torch device (auto-detected if None)
    
    Returns:
        loss: Total trajectory cost
        q: Final configuration
        lambdas: Contact forces history (T, 2)
        phis: Signed distances history (T,)
        qs: Configuration history (T+1, 3)
        qrobot_hist: Pusher position history (T+1, 2)
    """
    if device is None:
        device = q0.device
    
    if qp_solver is None:
        qp_solver = ContactQPSolver(mu=mu, n_contacts=1)
    
    if obstacle_pos is None:
        obstacle_pos = torch.tensor([0.2, -0.2], device=device)
    
    q = q0
    v = v0
    pr = pr0
    lambdas = []
    phis = []
    qs = [q0]
    qrobot_hist = [pr0]
    obs_term = 0.0
    
    # Simulate forward
    for k in range(horizon):
        q, v, pr, lam_k, phik = step_square(
            q, v, pr, u_seq[k], h, m, Izz, half, mu, 
            qp_solver=qp_solver, alpha_stab=0.1, device=device
        )
        lambdas.append(lam_k)
        phis.append(phik)
        qs.append(q)
        qrobot_hist.append(pr)
        
        # Obstacle avoidance term
        obs_term += 1.0 / (torch.sum((pr - obstacle_pos) ** 2) + 0.01)
    
    obs_term /= horizon
    
    # Cost function
    goal_term = 20.0 * torch.sum((q - goal_xy) ** 2)      # Goal reaching
    ctrl_term = 1e-3 * torch.sum(u_seq ** 2)               # Control effort
    v_term = 0.1 * torch.sum(v ** 2)                       # Terminal velocity
    pen_term = 0.0  # Penetration penalty (disabled)
    
    loss = goal_term + ctrl_term + pen_term + v_term + obs_term
    
    return loss, q, torch.stack(lambdas), torch.stack(phis), torch.stack(qs), torch.stack(qrobot_hist)


# %%%%%%% Interior Point %%%%%%%%%%%%%%%%%%

from .geometry import obb_contact

@dataclass
class IPMOptions:
    target_mu: float = 1e-4       # prescribed duality gap for each complementarity pair
    max_newton: int = 20
    tol: float = 1e-6
    ls_beta: float = 0.5          # backtracking factor
    frac_to_boundary: float = 0.99
    smooth_sdf: float = 0.0       # set >0 (e.g., 50.0) for a smoother SDF/normal near edges

    # ----- Option A: viscous ground friction (surrogate for Coulomb support) -----
    enable_viscous_ground_friction: bool = False
    c_lin: float = 0.0   # linear drag coefficient for vx, vy (N·s/m)
    c_ang: float = 0.0   # angular drag coefficient for omega (N·m·s/rad)


@torch.no_grad()
def fraction_to_boundary_step(x: torch.Tensor, dx: torch.Tensor, frac: float) -> float:
    """Return max step in (0,1] so that x + step*dx stays strictly positive."""
    mask = dx < 0
    if mask.any():
        step = torch.min(-x[mask] / dx[mask]) * frac
        return float(torch.clamp(step, max=1.0).item())
    return 1.0


def step_square_pos_ip(
    qk: torch.Tensor,
    vk: torch.Tensor,
    pusher_pos: torch.Tensor,
    u_push: torch.Tensor,
    h: float,
    m: float,
    Izz: float,
    half: float,
    mu: float,
    skip_solving_threshold: float,
    ipm_opts: IPMOptions = IPMOptions(),
    device=None,
    ):
    """
    One-step implicit integration with *position-level* contact complementarity solved by interior point.

    Unknowns: q_{k+1}, v_{k+1}, \lambda_N, beta (2 facets), r (slack for |t|-cone),
              y (gap slack), s (cone slack), w (2 facet slips)

    Complementarity pairs forced to prescribed duality gap mu*: 
      y * lambda_N = mu*,   r * s = mu*,   beta_i * w_i = mu*.

    Normal gap uses position-level constraint y = phi(q_{k+1}, pusher_{k+1}).
    Tangential uses Anitescu linearized cone with two facets (\pm t).

    Returns: q_{k+1}, v_{k+1}, pusher_pos_{k+1}, lam_vec([lambda_N, lambda_t_total]), phi
    """
    if device is None:
        device = qk.device
    dtype = qk.dtype

    hT = torch.tensor(h, dtype=dtype, device=device)
    muT = torch.tensor(mu, dtype=dtype, device=device)

    # Mass inverse
    M_inv = torch.diag(torch.tensor([1.0 / m, 1.0 / m, 1.0 / Izz], dtype=dtype, device=device))

    # Kinematic pusher (known)
    pusher_pos_next = pusher_pos + hT * u_push

    # Quick free prediction to skip IP solve when well separated
    q_free = torch.stack([qk[0] + hT * vk[0], qk[1] + hT * vk[1], qk[2] + hT * vk[2]])
    cnt_free = obb_contact(q_free, pusher_pos_next, half, smooth=ipm_opts.smooth_sdf)
    if cnt_free.phi > skip_solving_threshold:
        # no contact; plain semi-implicit Euler with mild damping
        v_next = vk - 0.3 * vk * hT
        q_next = torch.stack([qk[0] + hT * v_next[0], qk[1] + hT * v_next[1], qk[2] + hT * v_next[2]])
        lam = torch.zeros(2, dtype=dtype, device=device)
        return q_next, v_next, pusher_pos_next, lam, cnt_free.phi

    # Unknowns initialization
    q = q_free.clone().requires_grad_(True)
    v = vk.clone().requires_grad_(True)

    lamN = torch.tensor(1e-3, dtype=dtype, device=device, requires_grad=True)
    beta = torch.full((2,), 1e-3, dtype=dtype, device=device, requires_grad=True)  # two facets (+t,-t)
    r = torch.tensor(1e-3, dtype=dtype, device=device, requires_grad=True)

    y = torch.tensor(max(ipm_opts.target_mu, 1e-4), dtype=dtype, device=device, requires_grad=True)
    s = torch.tensor(max(ipm_opts.target_mu, 1e-4), dtype=dtype, device=device, requires_grad=True)
    w = torch.full((2,), max(ipm_opts.target_mu, 1e-4), dtype=dtype, device=device, requires_grad=True)

    def pack(_q, _v, _lamN, _beta, _r, _y, _s, _w):
        return torch.cat([_q, _v, _lamN.view(1), _beta, _r.view(1), _y.view(1), _s.view(1), _w])

    def unpack(z):
        _q = z[0:3]
        _v = z[3:6]
        _lamN = z[6]
        _beta = z[7:9]
        _r = z[9]
        _y = z[10]
        _s = z[11]
        _w = z[12:14]
        return _q, _v, _lamN, _beta, _r, _y, _s, _w

    def residual(z: torch.Tensor) -> torch.Tensor:
        _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)

        cnt = obb_contact(_q, pusher_pos_next, half, smooth=ipm_opts.smooth_sdf)
        n, t, r_cp = cnt.normal, cnt.tangent, cnt.r_cp
        Jn, Jt = contact_jacobians(n, t, r_cp)

        # Tangential relative velocity at cp (scalar along t)
        v_cp = _v[0:2] + _v[2] * perp(r_cp)
        v_rel_t = torch.dot(t, v_cp - u_push)

        # Two facet slip velocities: [ +v_t, -v_t ]
        v_facets = torch.stack([v_rel_t, -v_rel_t])

        # Dynamics (implicit Euler on velocities) and kinematics
        lamT_total = _beta[0] - _beta[1]
        impulse = Jn * _lamN + Jt * lamT_total

        # Option A: viscous ground friction as an external force ~ -C * v
        Fg = torch.zeros(3, dtype=dtype, device=device)
        if ipm_opts.enable_viscous_ground_friction and (ipm_opts.c_lin > 0.0 or ipm_opts.c_ang > 0.0):
            c_linT = torch.tensor(ipm_opts.c_lin, dtype=dtype, device=device)
            c_angT = torch.tensor(ipm_opts.c_ang, dtype=dtype, device=device)
            Fg = torch.stack([
                -c_linT * _v[0],  # Fx
                -c_linT * _v[1],  # Fy
                -c_angT * _v[2],  # Tau
            ])

        # Forces must be multiplied by h to get an impulse; contact impulses already are
        dv = hT * (M_inv @ (impulse + hT * Fg))
        r_dyn = _v - vk - dv                     # (3,)
        r_kin = _q - qk - hT * _v                # (3,)

        # Equalities tying slacks to physical quantities
        r_gap = _y - cnt.phi                      # (1,)
        r_cone = _s - (muT * _lamN - torch.sum(_beta))  # (1,)
        r_slip = _w - (v_facets + _r)            # (2,)

        mu_star = torch.tensor(ipm_opts.target_mu, dtype=dtype, device=device)
        # Central-path complementarity (softened)
        r_c1 = _y * _lamN - mu_star              # (1,)
        r_c2 = _r * _s - mu_star                 # (1,)
        r_c3 = _beta * _w - mu_star              # (2,)

        return torch.cat([r_dyn, r_kin, r_gap.view(1), r_cone.view(1), r_slip, r_c1.view(1), r_c2.view(1), r_c3])

    # Newton iterations on R(z)=0
    z = pack(q, v, lamN, beta, r, y, s, w)

    for it in range(ipm_opts.max_newton):
        z = z.clone().detach().requires_grad_(True)
        R = residual(z)
        res_norm = float(torch.linalg.norm(R).item())
        if res_norm < ipm_opts.tol:
            break
        # Dense Jacobian via autograd
        J = []
        for i in range(R.numel()):
            (grad_i,) = torch.autograd.grad(R[i], z, retain_graph=True, create_graph=False, allow_unused=False)
            J.append(grad_i.view(1, -1))
        J = torch.cat(J, dim=0)  # (N,N)

        # Levenberg-style regularization for robustness
        reg = 1e-8 * torch.eye(J.shape[0], dtype=dtype, device=device)
        try:
            dz = torch.linalg.solve(J + reg, -R)
        except RuntimeError:
            # fallback to least-squares if singular
            dz, *_ = torch.linalg.lstsq(J + reg, -R)

        # Fraction-to-boundary for positivity variables
        _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)
        alpha_pos = 1.0
        pos_vars = [(_lamN.view(1), dz[6].view(1)), (_beta, dz[7:9]), (_r.view(1), dz[9].view(1)),
                    (_y.view(1), dz[10].view(1)), (_s.view(1), dz[11].view(1)), (_w, dz[12:14])]
        for x, dx in pos_vars:
            alpha_pos = min(alpha_pos, fraction_to_boundary_step(x, dx, ipm_opts.frac_to_boundary))
        alpha = alpha_pos

        # Backtracking to reduce residual
        newton_decrease = False
        for _ in range(15):
            z_trial = z + alpha * dz
            # enforce tiny floors to stay >0
            z_trial[6] = torch.clamp(z_trial[6], min=1e-12)         # lamN
            z_trial[7:9] = torch.clamp(z_trial[7:9], min=1e-12)     # beta
            z_trial[9] = torch.clamp(z_trial[9], min=1e-12)         # r
            z_trial[10] = torch.clamp(z_trial[10], min=1e-12)       # y
            z_trial[11] = torch.clamp(z_trial[11], min=1e-12)       # s
            z_trial[12:14] = torch.clamp(z_trial[12:14], min=1e-12) # w
            Rt = residual(z_trial)
            if torch.linalg.norm(Rt) <= (1.0 - 1e-4 * alpha) * torch.linalg.norm(R):
                z = z_trial.detach()
                newton_decrease = True
                break
            alpha *= ipm_opts.ls_beta
        if not newton_decrease:
            # take the (clipped) fraction-to-boundary step even if residual didn't shrink enough
            z = z_trial.detach()

    # Unpack and assemble outputs
    q_next, v_next, lamN, beta, r, y, s, w = unpack(z)
    lam_t = beta[0] - beta[1]
    lam_vec = torch.stack([lamN, lam_t])
    phi = y  # y equals the gap at solution (softened)

    return q_next, v_next, pusher_pos_next, lam_vec, phi




# ========= Linearized-gap, position-level IP solver ========= #



def step_square_pos_ip_lin(
    qk: torch.Tensor,
    vk: torch.Tensor,
    pusher_pos: torch.Tensor,
    u_push: torch.Tensor,
    h: float,
    m: float,
    Izz: float,
    half: float,
    mu: float,
    skip_solving_threshold: float,
    ipm_opts: IPMOptions = IPMOptions(),
    device=None,
    relinearize_each_newton: bool = True,
):
    """
    Position-level interior-point step with a *linearized* gap function.
    Geometry (n, t, r_cp) and phi linearization are updated at the current iterate each Newton step.
    """
    if device is None:
        device = qk.device
    dtype = qk.dtype

    hT  = torch.tensor(h,  dtype=dtype, device=device)
    muT = torch.tensor(mu, dtype=dtype, device=device)

    Minv = torch.diag(torch.tensor([1.0/m, 1.0/m, 1.0/Izz], dtype=dtype, device=device))

    # kinematic pusher
    pusher_pos_next = pusher_pos + hT * u_push

    # fast early-exit if obviously separate
    q_free = torch.stack([qk[0] + hT*vk[0], qk[1] + hT*vk[1], qk[2] + hT*vk[2]])
    cnt_free = obb_contact(q_free, pusher_pos_next, half, smooth=ipm_opts.smooth_sdf)
    if cnt_free.phi > skip_solving_threshold:
        v_next = vk - 0.3 * vk * hT
        q_next = torch.stack([qk[0] + hT * v_next[0],
                              qk[1] + hT * v_next[1],
                              qk[2] + hT * v_next[2]])
        lam = torch.zeros(2, dtype=dtype, device=device)
        return q_next, v_next, pusher_pos_next, lam, cnt_free.phi

    # Unknowns init
    q = q_free.clone().requires_grad_(True)
    v = vk.clone().requires_grad_(True)

    lamN = torch.tensor(1e-3, dtype=dtype, device=device, requires_grad=True)
    beta = torch.full((2,), 1e-3, dtype=dtype, device=device, requires_grad=True)   # two-facet tangential
    r    = torch.tensor(1e-3, dtype=dtype, device=device, requires_grad=True)

    floor = max(ipm_opts.target_mu, 1e-4)
    y = torch.tensor(floor, dtype=dtype, device=device, requires_grad=True)
    s = torch.tensor(floor, dtype=dtype, device=device, requires_grad=True)
    w = torch.full((2,), floor, dtype=dtype, device=device, requires_grad=True)

    def pack(_q, _v, _lamN, _beta, _r, _y, _s, _w):
        return torch.cat([_q, _v, _lamN.view(1), _beta, _r.view(1), _y.view(1), _s.view(1), _w])

    def unpack(z):
        _q    = z[0:3]
        _v    = z[3:6]
        _lamN = z[6]
        _beta = z[7:9]
        _r    = z[9]
        _y    = z[10]
        _s    = z[11]
        _w    = z[12:14]
        return _q, _v, _lamN, _beta, _r, _y, _s, _w

    # container for current linearization (mutated each Newton iter if enabled)
    lin = _linearize_obb_gap_at(q.detach(), pusher_pos_next, half, ipm_opts.smooth_sdf)

    def residual(z: torch.Tensor) -> torch.Tensor:
        _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)

        # --- use *frozen* geometry from lin ---
        n  = lin['n']
        t  = lin['t']
        rc = lin['r_cp_ref']  # world vector from center to cp
        Jn, Jt = contact_jacobians(n, t, rc)

        # tangential slip velocity along frozen t
        v_cp   = _v[0:2] + _v[2] * perp(rc)
        v_rel_t = torch.dot(t, v_cp - u_push)
        v_facets = torch.stack([v_rel_t, -v_rel_t])  # two facets

        # implicit Euler on velocities
        lamT_total = _beta[0] - _beta[1]
        impulse = Jn * _lamN + Jt * lamT_total

        # optional viscous ground friction
        Fg = torch.zeros(3, dtype=dtype, device=device)
        if ipm_opts.enable_viscous_ground_friction and (ipm_opts.c_lin > 0.0 or ipm_opts.c_ang > 0.0):
            Fg = torch.stack([
                -torch.tensor(ipm_opts.c_lin, dtype=dtype, device=device) * _v[0],
                -torch.tensor(ipm_opts.c_lin, dtype=dtype, device=device) * _v[1],
                -torch.tensor(ipm_opts.c_ang, dtype=dtype, device=device) * _v[2],
            ])

        dv   = hT * (Minv @ (impulse + hT * Fg))
        r_dyn = _v - vk - dv
        r_kin = _q - qk - hT * _v

        # --- linearized gap & cone relations ---
        phi_lin = lin['phi0'] + torch.dot(lin['Jphi'], (_q - lin['q_ref']))
        r_gap   = _y - phi_lin

        r_cone = _s - (muT * _lamN - torch.sum(_beta))
        r_slip = _w - (v_facets + _r)

        mu_star = torch.tensor(ipm_opts.target_mu, dtype=dtype, device=device)
        r_c1 = _y * _lamN - mu_star
        r_c2 = _r * _s     - mu_star
        r_c3 = _beta * _w  - mu_star  # elementwise

        return torch.cat([r_dyn, r_kin, r_gap.view(1), r_cone.view(1), r_slip,
                          r_c1.view(1), r_c2.view(1), r_c3])

    # Newton on R(z)=0
    z = pack(q, v, lamN, beta, r, y, s, w)

    for it in range(ipm_opts.max_newton):
        # (optional) relinearize geometry at current q before forming residual/Jac
        if relinearize_each_newton:
            q_cur, *_ = unpack(z.detach())
            lin = _linearize_obb_gap_at(q_cur, pusher_pos_next, half, ipm_opts.smooth_sdf)

        z = z.clone().detach().requires_grad_(True)
        R = residual(z)
        if float(torch.linalg.norm(R)) < ipm_opts.tol:
            break

        # dense Jacobian via autograd (cheap now because phi is affine)
        J_rows = []
        for i in range(R.numel()):
            (g_i,) = torch.autograd.grad(R[i], z, retain_graph=True, create_graph=False)
            J_rows.append(g_i.view(1, -1))
        J = torch.cat(J_rows, dim=0)

        reg = 1e-8 * torch.eye(J.shape[0], dtype=dtype, device=device)
        try:
            dz = torch.linalg.solve(J + reg, -R)
        except RuntimeError:
            dz, *_ = torch.linalg.lstsq(J + reg, -R)

        # fraction-to-boundary for positive variables
        _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)
        alpha_pos = 1.0
        pos_vars = [(_lamN.view(1), dz[6].view(1)),
                    (_beta,          dz[7:9]),
                    (_r.view(1),     dz[9].view(1)),
                    (_y.view(1),     dz[10].view(1)),
                    (_s.view(1),     dz[11].view(1)),
                    (_w,             dz[12:14])]
        for x, dx in pos_vars:
            alpha_pos = min(alpha_pos, fraction_to_boundary_step(x, dx, ipm_opts.frac_to_boundary))

        alpha = alpha_pos
        # simple backtracking on residual
        ok = False
        for _ in range(12):
            z_trial = z + alpha * dz
            # keep positivity
            z_trial[6]      = torch.clamp(z_trial[6],      min=1e-12)
            z_trial[7:9]    = torch.clamp(z_trial[7:9],    min=1e-12)
            z_trial[9]      = torch.clamp(z_trial[9],      min=1e-12)
            z_trial[10]     = torch.clamp(z_trial[10],     min=1e-12)
            z_trial[11]     = torch.clamp(z_trial[11],     min=1e-12)
            z_trial[12:14]  = torch.clamp(z_trial[12:14],  min=1e-12)

            Rt = residual(z_trial)
            if torch.linalg.norm(Rt) <= (1.0 - 1e-4 * alpha) * torch.linalg.norm(R):
                z = z_trial.detach()
                ok = True
                break
            alpha *= ipm_opts.ls_beta

        if not ok:
            z = z_trial.detach()

    q_next, v_next, lamN, beta, r, y, s, w = unpack(z)
    lam_t = beta[0] - beta[1]
    lam_vec = torch.stack([lamN, lam_t])

    # y is the (softened) gap; with linearization, it equals phi_lin at convergence
    phi = y
    return q_next, v_next, pusher_pos_next, lam_vec, phi
