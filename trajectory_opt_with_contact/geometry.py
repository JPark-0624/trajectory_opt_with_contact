"""
Geometry utilities for contact detection and Jacobian computation.

Provides functions for:
- 2D rotation matrices
- Signed distance and closest points for squares
- Contact Jacobians for rigid bodies
"""

import torch
from dataclasses import dataclass

def rot(theta):
    """
    Create a 2D rotation matrix.
    
    Args:
        theta: Rotation angle (radians)
    
    Returns:
        R: 2x2 rotation matrix
    """
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    return R


def closest_point_on_square_and_normal(p_local, half):
    """
    Find the closest point on a square boundary and its outward normal.
    
    Args:
        p_local: Query point in body frame (2,)
        half: Half side length of the square
    
    Returns:
        q_local: Closest point on boundary in body frame (2,)
        n_local: Outward unit normal at boundary point (2,)
        phi: Signed distance (positive outside, negative inside)
    """
    # Signed distance for rectangle (square)
    ax = torch.abs(p_local[0]) - half
    ay = torch.abs(p_local[1]) - half
    d = torch.stack([ax, ay])
    outside = torch.clamp(d, min=0.0)
    outside_len = torch.linalg.norm(outside)
    inside = torch.clamp(torch.max(d), max=0.0)
    phi = outside_len + inside
    
    # Closest point - use functional operations instead of in-place
    q_local_x = torch.clamp(p_local[0], -half, half)
    q_local_y = torch.clamp(p_local[1], -half, half)
    q_local = torch.stack([q_local_x, q_local_y])
    
    # Determine outward normal
    eps = 1e-9
    if outside_len > eps:
        n_local = (p_local - q_local) / (outside_len + 1e-12)
    else:
        dx = half - torch.abs(p_local[0])
        dy = half - torch.abs(p_local[1])
        if dx < dy:
            n_local = torch.tensor([torch.sign(p_local[0]), 0.0], 
                                  dtype=p_local.dtype, device=p_local.device)
        else:
            n_local = torch.tensor([0.0, torch.sign(p_local[1])], 
                                  dtype=p_local.dtype, device=p_local.device)
    
    # Re-project q_local to boundary when inside
    if (phi < 0.0):
        if torch.abs(n_local[0]) > 0.5:
            q_local = torch.stack([half * torch.sign(p_local[0]), q_local[1]])
        else:
            q_local = torch.stack([q_local[0], half * torch.sign(p_local[1])])
    
    # Normalize n_local
    n_local = n_local / (torch.linalg.norm(n_local) + 1e-12)
    return q_local, n_local, phi


def contact_frame_and_J(q_body, v_body, pusher_pos, half, mu):
    """
    Compute contact frame and Jacobian for a point pusher and square.
    
    Args:
        q_body: Body configuration (x, y, theta)
        v_body: Body velocity (vx, vy, omega)
        pusher_pos: Pusher position in world frame (px, py)
        half: Half side length of the square
        mu: Coefficient of friction (unused, kept for API compatibility)
    
    Returns:
        J: Contact Jacobian (2x3) mapping body velocity to contact velocity
        n: Contact normal in world frame (2,)
        t: Contact tangent in world frame (2,)
        c_world: Contact point in world frame (2,)
        phi: Signed distance (positive = separated, negative = penetrating)
    """
    x, y, th = q_body
    R = rot(th)
    center = torch.stack([x, y])
    
    # Transform pusher position to body frame
    p_local = R.T @ (pusher_pos - center)
    
    # Find closest point and normal on square
    q_local, n_local, phi = closest_point_on_square_and_normal(p_local, half)
    
    # Transform back to world frame
    c_world = center + R @ q_local
    n = R @ n_local
    n = -n  # Flip normal to point into the body
    n = n / (torch.linalg.norm(n) + 1e-12)
    t = torch.stack([-n[1], n[0]])  # Tangent (perpendicular to normal)
    
    # Compute Jacobian: maps body velocity [vx, vy, omega] to contact velocity [v_n, v_t]
    r = c_world - center  # Vector from COM to contact point
    
    # Cross product terms for angular velocity contribution
    s_n_omega = -n[0] * r[1] + n[1] * r[0]
    s_t_omega = -t[0] * r[1] + t[1] * r[0]
    
    J = torch.stack([
        torch.stack([n[0], n[1], s_n_omega]),  # Normal direction
        torch.stack([t[0], t[1], s_t_omega]),  # Tangent direction
    ])
    
    return J, n, t, c_world, phi



## Differentiable Contact Detection
# ========================= Geometry / Kinematics ========================= #
def rot2(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

@torch.jit.script_if_tracing
def perp(v: torch.Tensor) -> torch.Tensor:
    # 2D +90deg rotation
    return torch.stack([-v[1], v[0]])

@dataclass
class OBBContact:
    phi: torch.Tensor            # signed distance (>0 no contact)
    cp_world: torch.Tensor       # closest point on box in world frame (2,)
    normal: torch.Tensor         # outward unit normal pointing from box to point (2,)
    tangent: torch.Tensor        # unit tangent (2,)
    r_cp: torch.Tensor           # vector from box center to contact point (2,)


def obb_contact(q_xytheta: torch.Tensor,
                p_world: torch.Tensor,
                half: float,
                smooth: float = 0.0) -> OBBContact:
    """
    Differentiable closest-point and signed distance from world point to a square OBB.
    If smooth > 0, uses a softmax-based smooth max near edges/corners for better gradients.
    """
    x, y, th = q_xytheta[0], q_xytheta[1], q_xytheta[2]
    c = torch.stack([x, y])
    R = rot2(th)

    # point in box local frame
    p_local = R.T @ (p_world - c)
    # clamp to box (closest point in local frame)
    cp_local = torch.clamp(p_local, min=-half, max=half)
    cp_world = c + R @ cp_local

    # vector from cp to point (world)
    v = p_world - cp_world
    dist = torch.linalg.norm(v) + 1e-12

    # inside/outside test via local overflow beyond box halfs
    overflow = torch.abs(p_local) - half
    if smooth <= 0.0:
        outside = torch.clamp(overflow, min=0.0)
        outside_norm = torch.linalg.norm(outside)
        inside = torch.clamp(torch.max(overflow), max=0.0)
        phi = outside_norm + inside  # standard rectangle SDF
    else:
        # smooth maximums (log-sum-exp) to avoid kinks
        alpha = torch.tensor(smooth, dtype=q_xytheta.dtype, device=q_xytheta.device)
        zero = torch.tensor(0.0, dtype=q_xytheta.dtype, device=q_xytheta.device)
        # softplus-like for vector overflow
        outside = torch.maximum(overflow, zero)
        outside_norm = torch.sqrt((outside**2).sum() + 1e-12)
        # smooth max(overflow_x, overflow_y)
        m = torch.log(torch.exp(alpha * overflow[0]) + torch.exp(alpha * overflow[1])) / alpha
        inside = torch.minimum(m, zero)
        phi = outside_norm + inside

    # normal: from box to point
    if dist > 1e-10:
        n = v / dist
    else:
        # degenerate; pick any outward direction in box normal
        # approximate by local outward sign
        axis = torch.argmax(torch.abs(overflow))
        sign = torch.sign(p_local[axis]) if p_local[axis].abs() > 1e-6 else torch.tensor(1.0, device=q_xytheta.device, dtype=q_xytheta.dtype)
        n_local = torch.tensor([0.0, 0.0], device=q_xytheta.device, dtype=q_xytheta.dtype)
        n_local[axis] = sign
        n = (R @ n_local)
        n = n / (torch.linalg.norm(n) + 1e-12)

    t = perp(n)
    t = t / (torch.linalg.norm(t) + 1e-12)

    r_cp = cp_world - c

    return OBBContact(phi=phi, cp_world=cp_world, normal=n, tangent=t, r_cp=r_cp)


def contact_jacobians(n: torch.Tensor, t: torch.Tensor, r_cp: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Rows Jn, Jt mapping body generalized velocity [vx,vy,omega] to normal/tangent relative velocity at cp."""
    # v_cp = v_xy + omega * perp(r_cp)
    rxn = r_cp[0] * n[1] - r_cp[1] * n[0]  # z-component of r x n
    rxt = r_cp[0] * t[1] - r_cp[1] * t[0]
    Jn = torch.stack([n[0], n[1], rxn])
    Jt = torch.stack([t[0], t[1], rxt])
    return Jn, Jt


def _linearize_obb_gap_at(q_ref: torch.Tensor,
                          p_world: torch.Tensor,
                          half: float,
                          smooth_sdf: float = 0.0):
    """
    Build an affine approximation of the signed gap at q ≈ q_ref:
        phi(q) ≈ phi0 + Jphi @ (q - q_ref)

    Returns:
      dict with keys:
        phi0, Jphi(3,), n(2,), t(2,), r_cp_ref(2,), R_ref(2,2), cp_local_ref(2,), q_ref(3,)
    """
    # geometry at reference (uses your existing differentiable contact util)
    cnt_ref = obb_contact(q_ref, p_world, half, smooth=smooth_sdf)

    x_ref, y_ref, th_ref = q_ref[0], q_ref[1], q_ref[2]
    c_ref = torch.stack([x_ref, y_ref])
    R_ref = rot2(th_ref)

    # recover local witness point used at reference
    # cp_world = c_ref + R_ref @ cp_local  => cp_local = R_ref^T (cp_world - c_ref)
    cp_local_ref = R_ref.T @ (cnt_ref.cp_world - c_ref)

    n = cnt_ref.normal
    t = cnt_ref.tangent
    r_cp_ref = R_ref @ cp_local_ref  # world vector from center to contact point

    # affine phi: n^T (p - (c + R cp_local_ref))
    # value at reference:
    phi0 = torch.dot(n, p_world - (c_ref + R_ref @ cp_local_ref))

    # Jacobian wrt q = [x, y, theta]
    # d/dx: -n_x, d/dy: -n_y
    # d/dθ: - n^T (dR/dθ cp_local) = - n^T (R_ref @ perp(cp_local_ref))
    dphi_dtheta = - torch.dot(n, R_ref @ perp(cp_local_ref))
    Jphi = torch.stack([-n[0], -n[1], dphi_dtheta])

    return {
        'phi0': phi0.detach(),
        'Jphi': Jphi.detach(),
        'n': n.detach(),
        't': t.detach(),
        'r_cp_ref': r_cp_ref.detach(),
        'R_ref': R_ref.detach(),
        'cp_local_ref': cp_local_ref.detach(),
        'q_ref': q_ref.detach(),
    }
