"""
Dynamics simulation with contact-implicit integration.

Implements time-stepping for rigid body dynamics with frictional contact
using differentiable QP solvers.
"""

import torch
from .geometry import contact_frame_and_J
from .qp_solver import ContactQPSolver


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

def implicit_euler_defects(qk, vk, prk, uk,
                           qk1, vk1, prk1,
                           h, m, Izz, half, mu,
                           qp_solver: ContactQPSolver,
                           alpha_stab=0.1, eps_H=1e-4, device=None):
    """
    Compute dynamics residuals for implicit Euler transcription at a single knot.

    Implicit scheme:
      q_{k+1} = q_k + h * v_{k+1}
      v_{k+1} = v_k + M^{-1} J(q_{k+1})^T λ_{k}  - 0.3*h*v_{k+1}      (damping)
      pr_{k+1} = pr_k + h * u_k
      λ_k solves contact QP at (q_{k+1}, v_{k+1}, pr_k, u_k)

    Returns residual vector r concatenating [r_q, r_v, r_p] (shape (3+3+2,))
    along with the λ_k and signed distance φ_k for logging.
    """
    if device is None:
        device = qk.device

    # Mass inverse
    M_inv = torch.diag(torch.tensor([1.0/m, 1.0/m, 1.0/Izz], dtype=qk.dtype, device=device))

    # Contact Jacobian and signed distance evaluated at the implicit state
    J, n, t, c_world, phi = contact_frame_and_J(qk1, vk1, prk, half, mu)

    # Delassus operator H = J M^-1 J^T + eps I
    H = J @ M_inv @ J.T + eps_H * torch.eye(J.shape[0], dtype=qk.dtype, device=device)
    L = torch.linalg.cholesky(H)

    # Relative velocity in contact frame (implicit depends on v_{k+1})
    v_push_proj = torch.stack([torch.dot(n, uk), torch.dot(t, uk)])

    # b = J v_{k+1} - v_push_proj + Baumgarte(phi/h) on normal
    b = (J @ vk1) - v_push_proj
    if phi < 0.0:
        b = b.clone()
        b[0] = b[0] + alpha_stab * (phi / h)

    lam = qp_solver.solve(L, b)

    # Implicit velocity update residual r_v = 0
    # vk1 ?= vk + M_inv J^T lam - 0.3*h*vk1
    rhs_v = vk + (M_inv @ (J.T @ lam)) - 0.3 * h * vk1
    r_v = vk1 - rhs_v

    # Position residual r_q = 0: qk1 ?= qk + h*vk1
    q_pred = torch.stack([qk[0] + h * vk1[0],
                          qk[1] + h * vk1[1],
                          qk[2] + h * vk1[2]])
    r_q = qk1 - q_pred

    # Pusher residual r_p = 0: prk1 ?= prk + h*uk
    r_p = prk1 - (prk + h * uk)

    r = torch.cat([r_q, r_v, r_p], dim=0)
    return r, lam, phi