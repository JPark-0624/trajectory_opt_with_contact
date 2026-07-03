"""
Lightweight analytic ground-contact dynamics for 3D sanity tests.

The ground is an infinite plane z=ground_z.  It does not use an SDF; cube
ground contact is a softmin over the cube vertices and sphere ground contact is
analytic.  This keeps the cube SDF reserved for cube-sphere contact.
"""

from typing import Optional

import torch

from trajectory_opt_with_contact.dynamics_3d import (
    IPMOptions3D,
    contact_residual_block,
    fraction_to_boundary_step,
    quat_integrate,
)
from trajectory_opt_with_contact.geometry_3d import (
    contact_jacobians_3d,
    cube_ground_contact,
    sphere_ground_contact,
)


_GRAVITY = torch.tensor([0.0, 0.0, -9.81])


def _clamp_positive(z, slices):
    z = z.clone()
    for idx in slices:
        z[idx] = z[idx].clamp(min=1e-12)
    return z


def _positive_vector(z, slices):
    return torch.cat([
        z[s:s + 1] if isinstance(s, int) else z[s]
        for s in slices
    ])


def _line_search(res_fn, z, dz, alpha_init, ls_beta, slices, max_steps=15, c1=1e-4):
    with torch.no_grad():
        n0 = torch.linalg.norm(res_fn(z))
        alpha = float(alpha_init)
        for _ in range(max_steps):
            z_t = _clamp_positive(z + alpha * dz, slices)
            if torch.linalg.norm(res_fn(z_t)) <= (1.0 - c1 * alpha) * n0:
                return z_t.detach(), alpha, True
            alpha *= ls_beta
        return z.detach(), alpha, False


def _contact_force(cnt, beta, lamN):
    lam_t1 = beta[1] - beta[0]
    lam_t2 = beta[3] - beta[2]
    linear = cnt.normal * lamN + cnt.tangent1 * lam_t1 + cnt.tangent2 * lam_t2
    Jn, Jt1, Jt2 = contact_jacobians_3d(cnt.normal, cnt.tangent1, cnt.tangent2, cnt.r_cp)
    generalized = Jn * lamN + Jt1 * lam_t1 + Jt2 * lam_t2
    return linear, generalized


def _init_contact(z, cnt, body_vel6, lam_idx, beta_sl, r_idx, y_idx, s_idx, w_sl,
                  mass, h, mu, target_mu):
    mu_st = z.new_tensor(target_mu)
    ground_vel = z.new_zeros(3)
    v_cp = body_vel6[:3] + torch.linalg.cross(body_vel6[3:], cnt.r_cp)
    v_rel = ground_vel - v_cp
    v_n = torch.dot(cnt.normal, v_rel)
    active = cnt.phi.detach() < 5e-3
    closing = torch.clamp(v_n, min=0.0)
    lam_active = torch.clamp(z.new_tensor(mass / h) * closing, min=1e-1)
    y_far = torch.clamp(cnt.phi.detach(), min=1e-3)
    lam = torch.where(active, lam_active, torch.clamp(mu_st / y_far, min=1e-8))
    y = torch.clamp(mu_st / lam, min=1e-8)

    v_t1 = torch.dot(cnt.tangent1, v_rel)
    v_t2 = torch.dot(cnt.tangent2, v_rel)
    facets = torch.stack([v_t1, -v_t1, v_t2, -v_t2])
    r = torch.clamp(-facets.min() + 1e-3, min=1e-3)
    w = torch.clamp(facets + r + 1e-3, min=1e-3)
    beta = torch.clamp(mu_st / w, min=1e-8, max=1e-2)
    s = torch.clamp(z.new_tensor(mu) * lam - beta.sum(), min=1e-3)

    z[lam_idx] = lam
    z[beta_sl] = beta
    z[r_idx] = r
    z[y_idx] = y
    z[s_idx] = s
    z[w_sl] = w


def _solve_newton(res_fn, z, pos_slices, opts):
    z = _clamp_positive(z, pos_slices)
    for _ in range(opts.max_newton):
        R = res_fn(z)
        if torch.linalg.norm(R) < opts.tol:
            break
        J = torch.autograd.functional.jacobian(res_fn, z, vectorize=True).detach()
        try:
            dz = torch.linalg.solve(J, -R.detach())
        except torch.linalg.LinAlgError:
            dz = torch.linalg.lstsq(J, -R.detach()).solution
        alpha = fraction_to_boundary_step(_positive_vector(z, pos_slices),
                                          _positive_vector(dz, pos_slices),
                                          opts.frac_to_boundary)
        z_next, _, ok = _line_search(res_fn, z, dz, alpha, opts.ls_beta, pos_slices)
        if not ok:
            break
        z = z_next
    return z


def step_cube_ground_ip(pk, qk, vk, h, m, I_body, half, mu=0.5,
                        ground_z=0.0, ground_sharpness=80.0,
                        ipm_opts: Optional[IPMOptions3D] = None,
                        z_prev: Optional[torch.Tensor] = None):
    """One cube timestep with analytic ground contact."""
    if ipm_opts is None:
        ipm_opts = IPMOptions3D()
    device, dtype = pk.device, pk.dtype
    pos_slices = [9, slice(10, 14), 14, 15, 16, slice(17, 21)]

    def res_fn(z):
        p, v = z[:3], z[3:9]
        lamN, beta, r, y, s, w = z[9], z[10:14], z[14], z[15], z[16], z[17:21]
        hT = z.new_tensor(h)
        mT = z.new_tensor(m)
        muT = z.new_tensor(mu)
        mu_st = z.new_tensor(ipm_opts.target_mu)
        grav = _GRAVITY.to(dtype=dtype, device=device)
        q_next = quat_integrate(qk, v[3:].detach(), h)
        cnt = cube_ground_contact(torch.cat([p, q_next]), half, ground_z, ground_sharpness)
        _, force_gen = _contact_force(cnt, beta, lamN)
        M_inv = torch.cat([mT.new_ones(3) / mT, 1.0 / I_body])
        grav_impulse = torch.cat([hT * mT * grav, grav.new_zeros(3)])
        r_dyn = v - vk - M_inv * (hT * force_gen + grav_impulse)
        r_kin = p - pk - hT * v[:3]
        R_c = contact_residual_block(lamN, beta, r, y, s, w, cnt, v, z.new_zeros(3), muT, mu_st)
        return torch.cat([r_dyn, r_kin, R_c])

    if z_prev is not None:
        z0 = z_prev.detach().clone()
    else:
        z0 = torch.zeros(21, dtype=dtype, device=device)
        grav = _GRAVITY.to(dtype=dtype, device=device)
        v_free = vk.detach().clone()
        v_free[:3] = v_free[:3] + z0.new_tensor(h) * grav
        z0[:3] = pk + z0.new_tensor(h) * v_free[:3]
        z0[3:9] = v_free
        for idx in pos_slices:
            z0[idx] = 1e-2
        with torch.no_grad():
            q_next = quat_integrate(qk.detach(), z0[6:9], h)
            cnt = cube_ground_contact(torch.cat([z0[:3], q_next]), half, ground_z, ground_sharpness)
            _init_contact(z0, cnt, z0[3:9], 9, slice(10, 14), 14, 15, 16, slice(17, 21),
                          m, h, mu, ipm_opts.target_mu)

    z = _solve_newton(res_fn, z0, pos_slices, ipm_opts)
    q_next = quat_integrate(qk.detach(), z[6:9].detach(), h)
    cnt = cube_ground_contact(torch.cat([z[:3], q_next]), half, ground_z, ground_sharpness)
    return z[:3].clone(), q_next, z[3:9].clone(), z[9], cnt.phi, z.detach()


def step_sphere_ground_ip(pk, vk, h, m, r_sphere, mu=0.5, ground_z=0.0,
                          ipm_opts: Optional[IPMOptions3D] = None,
                          z_prev: Optional[torch.Tensor] = None):
    """One translational sphere timestep with analytic ground contact."""
    if ipm_opts is None:
        ipm_opts = IPMOptions3D()
    device, dtype = pk.device, pk.dtype
    pos_slices = [6, slice(7, 11), 11, 12, 13, slice(14, 18)]

    def res_fn(z):
        p, v = z[:3], z[3:6]
        lamN, beta, r, y, s, w = z[6], z[7:11], z[11], z[12], z[13], z[14:18]
        hT = z.new_tensor(h)
        mT = z.new_tensor(m)
        muT = z.new_tensor(mu)
        mu_st = z.new_tensor(ipm_opts.target_mu)
        grav = _GRAVITY.to(dtype=dtype, device=device)
        cnt = sphere_ground_contact(p, r_sphere, ground_z)
        force_lin, _ = _contact_force(cnt, beta, lamN)
        r_dyn = v - vk - (hT / mT) * (force_lin + mT * grav)
        r_kin = p - pk - hT * v
        v6 = torch.cat([v, z.new_zeros(3)])
        R_c = contact_residual_block(lamN, beta, r, y, s, w, cnt, v6, z.new_zeros(3), muT, mu_st)
        return torch.cat([r_dyn, r_kin, R_c])

    if z_prev is not None:
        z0 = z_prev.detach().clone()
    else:
        z0 = torch.zeros(18, dtype=dtype, device=device)
        grav = _GRAVITY.to(dtype=dtype, device=device)
        v_free = vk.detach() + z0.new_tensor(h) * grav
        z0[:3] = pk + z0.new_tensor(h) * v_free
        z0[3:6] = v_free
        for idx in pos_slices:
            z0[idx] = 1e-2
        with torch.no_grad():
            cnt = sphere_ground_contact(z0[:3], r_sphere, ground_z)
            _init_contact(z0, cnt, torch.cat([z0[3:6], z0.new_zeros(3)]),
                          6, slice(7, 11), 11, 12, 13, slice(14, 18),
                          m, h, mu, ipm_opts.target_mu)

    z = _solve_newton(res_fn, z0, pos_slices, ipm_opts)
    cnt = sphere_ground_contact(z[:3], r_sphere, ground_z)
    return z[:3].clone(), z[3:6].clone(), z[6], cnt.phi, z.detach()
