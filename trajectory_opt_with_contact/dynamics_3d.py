"""
dynamics_3d.py

3D contact-implicit dynamics for a cube + 2 sphere system.

Design principle (mirrors dynamics.py 2D exactly):
  - Newton solves z where dim(z) == dim(R) == 33
  - q_rot is NOT in z — updated explicitly after Newton solve
    (unit constraint prevents including it in Newton system)
  - Sphere positions p1, p2 are NOT in z — determined directly
    by control: p_next = p_k + h*u (same as 2D pusher)

Analogy with 2D:
  2D: q=[x,y,θ]  v=[vx,vy,ω]
      z = [q(3), v(3), contact(8)] = 14D   R = 14D

  3D: p=[x,y,z]  q_rot=[qw,qx,qy,qz]  v=[vx,vy,vz,wx,wy,wz]
      z = [p(3), v(6), contact1(12), contact2(12)] = 33D   R = 33D

      After Newton:
        q_rot_{k+1} = normalize(qk + h/2 · Ω(ω*) · qk)

State vector z (33D):
    [ 0: 3]  p       cube position
    [ 3: 9]  v       cube twist [vx,vy,vz,wx,wy,wz]
    --- contact 1 ---
    [ 9]     lamN1
    [10:14]  beta1   [bt1+, bt1-, bt2+, bt2-]
    [14]     r1
    [15]     y1
    [16]     s1      (scalar, friction cone slack)
    [17:21]  w1      [wt1+, wt1-, wt2+, wt2-]
    --- contact 2 ---
    [21]     lamN2
    [22:26]  beta2
    [26]     r2
    [27]     y2
    [28]     s2
    [29:33]  w2

Residual R (33D):
    [ 0: 6]  r_dyn
    [ 6: 9]  r_kin_p
    [ 9:21]  contact1 residual (12D)
    [21:33]  contact2 residual (12D)

Per-contact (12D each):
    z: lamN(1)+beta(4)+r(1)+y(1)+s(1)+w(4) = 12
    R: r_gap(1)+r_cone(1)+r_slip(4)+comp_ylam(1)+comp_rs(1)+comp_bw(4) = 12  ✓
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from trajectory_opt_with_contact.geometry_3d import (
    quat_to_rot,
    cube_sphere_contact,
    contact_jacobians_3d,
    SDFGrid,
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

_GRAVITY = torch.tensor([0., 0., -9.81])

# Set to True externally to print Newton iteration details
NEWTON_DEBUG = False

NZ = 33
NR = 33

# z index layout
_P     = slice(0, 3)
_V     = slice(3, 9)
_VL    = slice(3, 6)   # linear velocity
_VA    = slice(6, 9)   # angular velocity

_LAMN1 = 9
_BETA1 = slice(10, 14)
_R1    = 14
_Y1    = 15
_S1    = 16
_W1    = slice(17, 21)

_LAMN2 = 21
_BETA2 = slice(22, 26)
_R2    = 26
_Y2    = 27
_S2    = 28
_W2    = slice(29, 33)


# ──────────────────────────────────────────────
# IPMOptions
# ──────────────────────────────────────────────

@dataclass
class IPMOptions3D:
    target_mu:        float = 1e-4
    max_newton:       int   = 30
    tol:              float = 1e-6
    ls_beta:          float = 0.5
    frac_to_boundary: float = 0.99


# ──────────────────────────────────────────────
# Quaternion utilities
# ──────────────────────────────────────────────

def quat_omega_matrix(omega: torch.Tensor) -> torch.Tensor:
    """Ω(ω): (4,4) s.t. dq/dt = (1/2) Ω(ω) q"""
    wx, wy, wz = omega[0], omega[1], omega[2]
    O = omega.new_zeros(1).squeeze()
    return torch.stack([
        torch.stack([ O,  -wx, -wy, -wz]),
        torch.stack([wx,    O,  wz, -wy]),
        torch.stack([wy,  -wz,   O,  wx]),
        torch.stack([wz,   wy, -wx,   O]),
    ])


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / (q.norm() + 1e-12)


def quat_integrate(qk: torch.Tensor, omega: torch.Tensor, h: float) -> torch.Tensor:
    """q_{k+1} = normalize(qk + h/2 · Ω(ω) · qk)"""
    return quat_normalize(qk + (h / 2.0) * (quat_omega_matrix(omega) @ qk))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _pos_slices():
    """All positive-variable index locations in z."""
    return [
        _LAMN1, _BETA1, _R1, _Y1, _S1, _W1,
        _LAMN2, _BETA2, _R2, _Y2, _S2, _W2,
    ]


def clamp_positive(z: torch.Tensor) -> torch.Tensor:
    z = z.clone()
    for idx in _pos_slices():
        z[idx] = z[idx].clamp(min=1e-12)
    return z


@torch.no_grad()
def fraction_to_boundary_step(x, dx, frac):
    mask = dx < 0
    if mask.any():
        return float(torch.clamp(torch.min(-x[mask] / dx[mask]) * frac, max=1.0).item())
    return 1.0


def backtracking_line_search(res_fn, z, dz, alpha_init, ls_beta,
                              max_steps=15, c1=1e-4):
    with torch.no_grad():
        n0    = torch.linalg.norm(res_fn(z))
        alpha = float(alpha_init)
        z_last = None
        for _ in range(max_steps):
            z_t = clamp_positive(z + alpha * dz)
            z_last = z_t
            if torch.linalg.norm(res_fn(z_t)) <= (1.0 - c1 * alpha) * n0:
                return z_t.detach(), alpha, True
            alpha *= ls_beta
        return z_last.detach(), alpha, False


# ──────────────────────────────────────────────
# Default initialization
# ──────────────────────────────────────────────

def default_z(pk, vk, dtype, device):
    z = torch.zeros(NZ, dtype=dtype, device=device)
    z[_P] = pk.detach()
    z[_V] = vk.detach()
    c = 1e-2
    for idx in _pos_slices():
        z[idx] = c
    return z


# ──────────────────────────────────────────────
# Per-contact residual (12D → 12D)
# ──────────────────────────────────────────────

def contact_residual_block(lamN, beta, r, y, s, w, cnt, v, sphere_vel, muT, mu_st):
    """
    12D residual for one contact.
    v: (6,) cube twist [vlin(3), vang(3)]
    """
    # gap
    r_gap  = (y - cnt.phi).unsqueeze(0)

    # friction cone: s = mu*lamN - sum(beta)
    r_cone = (s - (muT * lamN - beta.sum())).unsqueeze(0)

    # slip velocity at contact point
    v_cp     = v[:3] + torch.linalg.cross(v[3:], cnt.r_cp)
    v_rel    = sphere_vel - v_cp
    v_t1     = torch.dot(cnt.tangent1, v_rel)
    v_t2     = torch.dot(cnt.tangent2, v_rel)
    v_facets = torch.stack([v_t1, -v_t1, v_t2, -v_t2])
    r_slip   = w - (v_facets + r)

    # complementarity
    comp_ylam = (y * lamN   - mu_st).unsqueeze(0)
    comp_rs   = (r * s      - mu_st).unsqueeze(0)
    comp_bw   = beta * w    - mu_st

    return torch.cat([r_gap, r_cone, r_slip, comp_ylam, comp_rs, comp_bw])  # (12,)


# ──────────────────────────────────────────────
# Full residual (33D)
# ──────────────────────────────────────────────

def residual_3d(z, pk, qk, vk, p1_next, p2_next,
                h, m, I_body, mu, target_mu,
                sdf_grid, r_sphere, device, dtype,
                u1=None, u2=None):
    """R(z) = 0, shape (33,)."""

    p     = z[_P]
    v     = z[_V]
    lamN1 = z[_LAMN1]; beta1 = z[_BETA1]; r1 = z[_R1]
    y1    = z[_Y1];    s1    = z[_S1];    w1 = z[_W1]
    lamN2 = z[_LAMN2]; beta2 = z[_BETA2]; r2 = z[_R2]
    y2    = z[_Y2];    s2    = z[_S2];    w2 = z[_W2]

    hT   = z.new_tensor(h)
    mT   = z.new_tensor(m)
    muT  = z.new_tensor(mu)
    mu_st= z.new_tensor(target_mu)
    grav = _GRAVITY.to(dtype=dtype, device=device)
    u1 = z.new_zeros(3) if u1 is None else u1.to(dtype=dtype, device=device)
    u2 = z.new_zeros(3) if u2 is None else u2.to(dtype=dtype, device=device)

    # Quaternion at k+1 (use detached ω to avoid differentiating through
    # quat_integrate inside Newton — geometry is evaluated at this q_next)
    omega  = v[3:].detach()
    q_next = quat_integrate(qk, omega, h)
    q_pose = torch.cat([p, q_next])

    # Contact geometry
    cnt1 = cube_sphere_contact(q_pose, p1_next, sdf_grid, r_sphere)
    cnt2 = cube_sphere_contact(q_pose, p2_next, sdf_grid, r_sphere)
    Jn1, Jt1_1, Jt2_1 = contact_jacobians_3d(
        cnt1.normal, cnt1.tangent1, cnt1.tangent2, cnt1.r_cp)
    Jn2, Jt1_2, Jt2_2 = contact_jacobians_3d(
        cnt2.normal, cnt2.tangent1, cnt2.tangent2, cnt2.r_cp)

    # M_inv
    M_inv = torch.cat([mT.new_ones(3) / mT, 1.0 / I_body])

    # Contact generalized force.  Unlike the 2D legacy dynamics, the 3D
    # tests interpret lamN/beta as forces, so the timestep h converts them
    # to impulse in the momentum balance below.
    contact_force = (Jn1 * lamN1
                     + Jt1_1 * (beta1[1] - beta1[0])
                     + Jt2_1 * (beta1[3] - beta1[2])
                     + Jn2 * lamN2
                     + Jt1_2 * (beta2[1] - beta2[0])
                     + Jt2_2 * (beta2[3] - beta2[2]))
    contact_impulse = hT * contact_force

    grav_impulse = torch.cat([hT * mT * grav, grav.new_zeros(3)])

    # r_dyn (6)
    r_dyn   = v - vk - M_inv * (contact_impulse + grav_impulse)

    # r_kin_p (3)
    r_kin_p = p - pk - hT * v[:3]

    # Contact residuals
    R_c1 = contact_residual_block(lamN1, beta1, r1, y1, s1, w1, cnt1, v, u1, muT, mu_st)
    R_c2 = contact_residual_block(lamN2, beta2, r2, y2, s2, w2, cnt2, v, u2, muT, mu_st)

    return torch.cat([r_dyn, r_kin_p, R_c1, R_c2])   # (33,)


# ──────────────────────────────────────────────
# autograd.Function
# ──────────────────────────────────────────────

class StepCubeSphereIPFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pk, qk, vk, p1k, p2k, u1, u2,
                h_t, m_t, I_body, mu_t,
                target_mu, tol, max_newton, frac_to_boundary, ls_beta,
                sdf_grid, r_sphere, z_prev):

        device = pk.device
        dtype  = pk.dtype
        h, m, mu = h_t.item(), m_t.item(), mu_t.item()

        p1_next = (p1k + h_t * u1).detach()
        p2_next = (p2k + h_t * u2).detach()

        def res_fn(z):
            return residual_3d(z, pk.detach(), qk.detach(), vk.detach(),
                               p1_next, p2_next,
                               h, m, I_body.detach(), mu, target_mu,
                               sdf_grid, r_sphere, device, dtype,
                               u1=u1.detach(), u2=u2.detach())

        # Init
        z = (z_prev.clone().detach() if z_prev is not None
             else default_z(pk.detach(), vk.detach(), dtype, device))
        z = clamp_positive(z)

        # Newton loop
        J_star = None
        converged = False
        debug = NEWTON_DEBUG
        for newton_iter in range(max_newton):
            R = res_fn(z)
            norm_R = torch.linalg.norm(R)

            if debug:
                nan_z = torch.isnan(z).any().item()
                nan_R = torch.isnan(R).any().item()
                print(f"    [Newton {newton_iter:2d}] ||R||={norm_R.item():.3e}"
                      f"  nan_z={nan_z}  nan_R={nan_R}"
                      f"  p={z[_P].tolist()}  v_lin={z[_VL].tolist()}")

            if torch.isnan(norm_R):
                if debug:
                    print(f"    [Newton] nan detected — aborting")
                break

            if norm_R < tol:
                converged = True
                break

            J = torch.autograd.functional.jacobian(
                res_fn, z, vectorize=True
            ).detach()   # (33, 33)  reverse-mode (more stable on CUDA)

            if debug and torch.isnan(J).any():
                print(f"    [Newton {newton_iter}] Jacobian contains nan!")
                nan_rows = torch.isnan(J).any(dim=1).nonzero().squeeze()
                print(f"      nan rows: {nan_rows.tolist()}")

            try:
                dz = torch.linalg.solve(J, -R.detach())
            except torch.linalg.LinAlgError:
                dz = torch.linalg.lstsq(J, -R.detach()).solution

            # Fraction-to-boundary
            z_pos  = torch.cat([z[_LAMN1:_LAMN1+1], z[_BETA1],
                                 z[_R1:_R1+1], z[_Y1:_Y1+1], z[_S1:_S1+1], z[_W1],
                                 z[_LAMN2:_LAMN2+1], z[_BETA2],
                                 z[_R2:_R2+1], z[_Y2:_Y2+1], z[_S2:_S2+1], z[_W2]])
            dz_pos = torch.cat([dz[_LAMN1:_LAMN1+1], dz[_BETA1],
                                 dz[_R1:_R1+1], dz[_Y1:_Y1+1], dz[_S1:_S1+1], dz[_W1],
                                 dz[_LAMN2:_LAMN2+1], dz[_BETA2],
                                 dz[_R2:_R2+1], dz[_Y2:_Y2+1], dz[_S2:_S2+1], dz[_W2]])
            alpha = fraction_to_boundary_step(z_pos, dz_pos, frac_to_boundary)
            z, _, _ = backtracking_line_search(res_fn, z, dz, alpha, ls_beta)
            J_star = J

        # IFT backward must use the Jacobian at the final converged solution.
        # During Newton, J_star may still point to the pre-update iterate when
        # the next loop iteration satisfies the tolerance and breaks.
        J_star = torch.autograd.functional.jacobian(
            res_fn, z, vectorize=True
        ).detach()

        # Explicit quaternion update
        q_next = quat_integrate(qk.detach(), z[_VA].detach(), h)

        ctx.save_for_backward(z, J_star,
                              pk.detach(), qk.detach(), vk.detach(),
                              p1_next.detach(), p2_next.detach(),
                              u1.detach(), u2.detach())
        ctx.h          = h
        ctx.m          = m
        ctx.mu         = mu
        ctx.target_mu  = target_mu
        ctx.sdf_grid   = sdf_grid
        ctx.r_sphere   = r_sphere
        ctx.I_body     = I_body.detach()

        # Contact info for output
        q_pose = torch.cat([z[_P], q_next])
        cnt1   = cube_sphere_contact(q_pose, p1_next, sdf_grid, r_sphere)
        cnt2   = cube_sphere_contact(q_pose, p2_next, sdf_grid, r_sphere)

        return (z[_P].clone(), q_next, z[_V].clone(),
                p1_next, p2_next,
                z[_LAMN1], z[_LAMN2],
                cnt1.phi, cnt2.phi,
                z.detach())

    @staticmethod
    def backward(ctx, grad_p, grad_q, grad_v,
                 grad_p1, grad_p2,
                 grad_lam1, grad_lam2,
                 grad_phi1, grad_phi2, grad_z_out):
        """
        IFT backward — same formula as 2D dynamics.py:
          w    = J^{-T} · (∂g/∂z)^T · λ_out
          ∂L/∂θ = (∂g/∂θ)^T · λ_out  -  (∂R/∂θ)^T · w

        θ = (pk, vk, p1_next, p2_next)
        ∂R/∂θ computed via autograd through residual_3d.
        """
        z, J_star, pk, qk, vk, p1_next, p2_next, u1, u2 = ctx.saved_tensors
        device, dtype = z.device, z.dtype
        h, m, mu     = ctx.h, ctx.m, ctx.mu

        if NEWTON_DEBUG:
            print(f"  [IFT backward] grad_p={grad_p}  grad_v={grad_v}")
        target_mu    = ctx.target_mu
        sdf_grid     = ctx.sdf_grid
        r_sphere     = ctx.r_sphere
        I_body       = ctx.I_body

        # ── Step 1: (∂g/∂z)^T λ_out ──
        # g outputs: p_next=z[_P], v_next=z[_V], lamN1=z[_LAMN1], lamN2=z[_LAMN2]
        dLdz = torch.zeros(NZ, dtype=dtype, device=device)
        if grad_p    is not None: dLdz[_P]     += grad_p
        if grad_v    is not None: dLdz[_V]     += grad_v
        if grad_lam1 is not None: dLdz[_LAMN1] += grad_lam1.squeeze()
        if grad_lam2 is not None: dLdz[_LAMN2] += grad_lam2.squeeze()

        # ── Step 2: solve J^T w = dLdz ──
        try:
            w = torch.linalg.solve(J_star.T, dLdz)
        except torch.linalg.LinAlgError:
            w = torch.linalg.lstsq(J_star.T, dLdz).solution

        # ── Step 3: ∂R/∂θ via autograd ──
        # θ variables that appear in R:
        #   pk, vk  → r_dyn, r_kin_p
        #   p1_next, p2_next → contact geometry (phi, normal, Jacobians)
        # p1_next = p1k + h*u1, p2_next = p2k + h*u2
        # so ∂p1_next/∂u1 = h*I, ∂p1_next/∂p1k = I  (chain rule)

        pk_r      = pk.requires_grad_(True)
        vk_r      = vk.requires_grad_(True)
        p1_next_r = p1_next.requires_grad_(True)
        p2_next_r = p2_next.requires_grad_(True)
        u1_r      = u1.requires_grad_(True)
        u2_r      = u2.requires_grad_(True)

        with torch.enable_grad():
            R_θ = residual_3d(
                z.detach(), pk_r, qk, vk_r,
                p1_next_r, p2_next_r,
                h, m, I_body, mu, target_mu,
                sdf_grid, r_sphere, device, dtype,
                u1=u1_r, u2=u2_r,
            )
            # ∂L/∂θ += -(∂R/∂θ)^T w  via vjp
            R_θ.backward(-w)

        grad_pk = pk_r.grad      if pk_r.grad      is not None else torch.zeros(3, dtype=dtype, device=device)
        grad_vk = vk_r.grad      if vk_r.grad      is not None else torch.zeros(6, dtype=dtype, device=device)
        g_p1    = p1_next_r.grad if p1_next_r.grad is not None else torch.zeros(3, dtype=dtype, device=device)
        g_p2    = p2_next_r.grad if p2_next_r.grad is not None else torch.zeros(3, dtype=dtype, device=device)
        g_u1    = u1_r.grad      if u1_r.grad      is not None else torch.zeros(3, dtype=dtype, device=device)
        g_u2    = u2_r.grad      if u2_r.grad      is not None else torch.zeros(3, dtype=dtype, device=device)

        # p1_next = p1k + h*u1  →  grad_u1 = h * g_p1,  grad_p1k = g_p1
        h_t   = torch.tensor(h, dtype=dtype, device=device)
        grad_u1  = h_t * g_p1 + g_u1
        grad_u2  = h_t * g_p2 + g_u2
        grad_p1k = g_p1
        grad_p2k = g_p2

        return (
            grad_pk, None, grad_vk,        # pk, qk, vk
            grad_p1k, grad_p2k,            # p1k, p2k
            grad_u1,  grad_u2,             # u1, u2
            None, None, None, None,        # h, m, I_body, mu
            None, None, None, None, None,  # IPM params
            None, None, None,              # sdf_grid, r_sphere, z_prev
        )


# ──────────────────────────────────────────────
# Public wrapper
# ──────────────────────────────────────────────

def step_cube_sphere_ip(
    pk, qk, vk, p1k, p2k, u1, u2,
    h, m, I_body, mu,
    sdf_grid, r_sphere,
    ipm_opts: IPMOptions3D = None,
    z_prev: Optional[torch.Tensor] = None,
):
    """
    One timestep of 3D contact-implicit dynamics.

    Returns:
        p_next, q_next, v_next  : cube state at k+1
        p1_next, p2_next        : sphere positions at k+1
        lamN1, lamN2            : normal forces
        phi1, phi2              : signed distances
        z_star (33,)            : Newton solution (for warm-starting)
    """
    if ipm_opts is None:
        ipm_opts = IPMOptions3D()

    device, dtype = pk.device, pk.dtype
    h_t  = torch.tensor(h,  dtype=dtype, device=device)
    m_t  = torch.tensor(m,  dtype=dtype, device=device)
    mu_t = torch.tensor(mu, dtype=dtype, device=device)

    return StepCubeSphereIPFn.apply(
        pk, qk, vk, p1k, p2k, u1, u2,
        h_t, m_t, I_body, mu_t,
        ipm_opts.target_mu,
        ipm_opts.tol,
        ipm_opts.max_newton,
        ipm_opts.frac_to_boundary,
        ipm_opts.ls_beta,
        sdf_grid, r_sphere,
        z_prev,
    )
