"""
dynamics_3d_velocity_robot.py

3D contact-implicit dynamics for a cube + two velocity-commanded sphere robots.

This variant keeps the user/control meaning of u1, u2 as commanded/free sphere
velocities, but the realized sphere velocities are solved together with contact.
Contact impulses affect both the cube and the sphere robots by action-reaction.

State vector z (45D):
    [ 0: 3]  cube p
    [ 3: 9]  cube v = [vx, vy, vz, wx, wy, wz]
    [ 9:12]  sphere1 p
    [12:15]  sphere1 realized velocity
    [15:18]  sphere2 p
    [18:21]  sphere2 realized velocity
    [21:33]  contact1 block
    [33:45]  contact2 block
"""

from dataclasses import dataclass
from typing import Optional

import torch

from trajectory_opt_with_contact.dynamics_3d import (
    IPMOptions3D,
    contact_residual_block,
    fraction_to_boundary_step,
    quat_integrate,
)
from trajectory_opt_with_contact.geometry_3d import (
    SDFGrid,
    contact_jacobians_3d,
    cube_ground_contact,
    cube_sphere_contact,
)


_GRAVITY = torch.tensor([0.0, 0.0, -9.81])
NEWTON_DEBUG = False

NZ = 45
NR = 45

_P = slice(0, 3)
_V = slice(3, 9)
_VL = slice(3, 6)
_VA = slice(6, 9)
_P1 = slice(9, 12)
_VS1 = slice(12, 15)
_P2 = slice(15, 18)
_VS2 = slice(18, 21)

_LAMN1 = 21
_BETA1 = slice(22, 26)
_R1 = 26
_Y1 = 27
_S1 = 28
_W1 = slice(29, 33)

_LAMN2 = 33
_BETA2 = slice(34, 38)
_R2 = 38
_Y2 = 39
_S2 = 40
_W2 = slice(41, 45)

NZ_GROUND = 57
NR_GROUND = 57

_LAMNG = 45
_BETAG = slice(46, 50)
_RG = 50
_YG = 51
_SG = 52
_WG = slice(53, 57)


def _pos_slices():
    return [
        _LAMN1, _BETA1, _R1, _Y1, _S1, _W1,
        _LAMN2, _BETA2, _R2, _Y2, _S2, _W2,
    ]


def _pos_slices_ground():
    return _pos_slices() + [_LAMNG, _BETAG, _RG, _YG, _SG, _WG]


def clamp_positive(z: torch.Tensor) -> torch.Tensor:
    z = z.clone()
    for idx in _pos_slices():
        z[idx] = z[idx].clamp(min=1e-12)
    return z


def backtracking_line_search_velocity(res_fn, z, dz, alpha_init, ls_beta,
                                      max_steps=15, c1=1e-4):
    with torch.no_grad():
        n0 = torch.linalg.norm(res_fn(z))
        alpha = float(alpha_init)
        z_last = z
        for _ in range(max_steps):
            z_t = clamp_positive(z + alpha * dz)
            z_last = z_t
            if torch.linalg.norm(res_fn(z_t)) <= (1.0 - c1 * alpha) * n0:
                return z_t.detach(), alpha, True
            alpha *= ls_beta
        return z_last.detach(), alpha, False


def backtracking_line_search_velocity_ground(res_fn, z, dz, alpha_init, ls_beta,
                                             max_steps=15, c1=1e-4):
    with torch.no_grad():
        n0 = torch.linalg.norm(res_fn(z))
        alpha = float(alpha_init)
        z_last = z
        for _ in range(max_steps):
            z_t = clamp_positive_ground(z + alpha * dz)
            z_last = z_t
            if torch.linalg.norm(res_fn(z_t)) <= (1.0 - c1 * alpha) * n0:
                return z_t.detach(), alpha, True
            alpha *= ls_beta
        return z_last.detach(), alpha, False


def _positive_vector(z: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        z[_LAMN1:_LAMN1 + 1], z[_BETA1], z[_R1:_R1 + 1],
        z[_Y1:_Y1 + 1], z[_S1:_S1 + 1], z[_W1],
        z[_LAMN2:_LAMN2 + 1], z[_BETA2], z[_R2:_R2 + 1],
        z[_Y2:_Y2 + 1], z[_S2:_S2 + 1], z[_W2],
    ])


def clamp_positive_ground(z: torch.Tensor) -> torch.Tensor:
    z = z.clone()
    for idx in _pos_slices_ground():
        z[idx] = z[idx].clamp(min=1e-12)
    return z


def _positive_vector_ground(z: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        z[_LAMN1:_LAMN1 + 1], z[_BETA1], z[_R1:_R1 + 1],
        z[_Y1:_Y1 + 1], z[_S1:_S1 + 1], z[_W1],
        z[_LAMN2:_LAMN2 + 1], z[_BETA2], z[_R2:_R2 + 1],
        z[_Y2:_Y2 + 1], z[_S2:_S2 + 1], z[_W2],
        z[_LAMNG:_LAMNG + 1], z[_BETAG], z[_RG:_RG + 1],
        z[_YG:_YG + 1], z[_SG:_SG + 1], z[_WG],
    ])


def default_z(pk, qk, vk, p1k, p2k, u1, u2, h, m_robot, mu, target_mu,
              sdf_grid, r_sphere, dtype, device):
    z = torch.zeros(NZ, dtype=dtype, device=device)
    grav = _GRAVITY.to(dtype=dtype, device=device)
    v_free = vk.detach().clone()
    v_free[:3] = v_free[:3] + z.new_tensor(h) * grav
    z[_V] = v_free
    z[_P] = (pk + z.new_tensor(h) * v_free[:3]).detach()
    z[_P1] = (p1k + z.new_tensor(h) * u1).detach()
    z[_VS1] = u1.detach()
    z[_P2] = (p2k + z.new_tensor(h) * u2).detach()
    z[_VS2] = u2.detach()
    c = 1e-2
    for idx in _pos_slices():
        z[idx] = c

    with torch.no_grad():
        q_next = quat_integrate(qk.detach(), z[_VA], h)
        q_pose = torch.cat([z[_P], q_next])
        cnt1 = cube_sphere_contact(q_pose, z[_P1], sdf_grid, r_sphere)
        cnt2 = cube_sphere_contact(q_pose, z[_P2], sdf_grid, r_sphere)
        _init_contact_block(z, cnt1, z[_V], z[_VS1],
                            _LAMN1, _BETA1, _R1, _Y1, _S1, _W1,
                            h, m_robot, mu, target_mu)
        _init_contact_block(z, cnt2, z[_V], z[_VS2],
                            _LAMN2, _BETA2, _R2, _Y2, _S2, _W2,
                            h, m_robot, mu, target_mu)
        f1 = (cnt1.normal * z[_LAMN1]
              + cnt1.tangent1 * (z[_BETA1][1] - z[_BETA1][0])
              + cnt1.tangent2 * (z[_BETA1][3] - z[_BETA1][2]))
        f2 = (cnt2.normal * z[_LAMN2]
              + cnt2.tangent1 * (z[_BETA2][1] - z[_BETA2][0])
              + cnt2.tangent2 * (z[_BETA2][3] - z[_BETA2][2]))
        z[_VS1] = (u1 - z.new_tensor(h / m_robot) * f1).detach()
        z[_VS2] = (u2 - z.new_tensor(h / m_robot) * f2).detach()
        z[_P1] = (p1k + z.new_tensor(h) * z[_VS1]).detach()
        z[_P2] = (p2k + z.new_tensor(h) * z[_VS2]).detach()
    return z


def _init_contact_block(z, cnt, cube_v, sphere_v,
                        lam_idx, beta_idx, r_idx, y_idx, s_idx, w_idx,
                        h, m_robot, mu, target_mu):
    mu_st = z.new_tensor(target_mu)

    v_cp = cube_v[:3] + torch.linalg.cross(cube_v[3:], cnt.r_cp)
    v_rel = sphere_v - v_cp
    v_n = torch.dot(cnt.normal, v_rel)

    y_far = torch.clamp(cnt.phi.detach(), min=1e-3)
    active = cnt.phi.detach() < 5e-3
    closing = torch.clamp(v_n, min=0.0)
    lam_active = torch.clamp(z.new_tensor(m_robot / h) * closing, min=1e-1)
    lam = torch.where(active, lam_active, torch.clamp(mu_st / y_far, min=1e-8))
    y = torch.clamp(mu_st / lam, min=1e-8)

    v_t1 = torch.dot(cnt.tangent1, v_rel)
    v_t2 = torch.dot(cnt.tangent2, v_rel)
    facets = torch.stack([v_t1, -v_t1, v_t2, -v_t2])

    r = torch.clamp(-facets.min() + 1e-3, min=1e-3)
    w = torch.clamp(facets + r + 1e-3, min=1e-3)
    beta = torch.clamp(mu_st / w, min=1e-8, max=1e-2)
    s = torch.clamp(z.new_tensor(mu) * lam - beta.sum(), min=1e-3)

    z[y_idx] = y
    z[lam_idx] = lam
    z[r_idx] = r
    z[w_idx] = w
    z[beta_idx] = beta
    z[s_idx] = s


def _contact_forces(cnt, Jn, Jt1, Jt2, lamN, beta):
    lam_t1 = beta[1] - beta[0]
    lam_t2 = beta[3] - beta[2]
    generalized = Jn * lamN + Jt1 * lam_t1 + Jt2 * lam_t2
    linear = cnt.normal * lamN + cnt.tangent1 * lam_t1 + cnt.tangent2 * lam_t2
    return generalized, linear


def residual_3d_velocity_robot(z, pk, qk, vk, p1k, p2k, u1, u2,
                               h, m_cube, I_body, m_robot, mu, target_mu,
                               sdf_grid, r_sphere, device, dtype):
    """R(z)=0 for the velocity-commanded dynamic sphere model."""
    p = z[_P]
    v = z[_V]
    p1 = z[_P1]
    vs1 = z[_VS1]
    p2 = z[_P2]
    vs2 = z[_VS2]

    lamN1 = z[_LAMN1]; beta1 = z[_BETA1]; r1 = z[_R1]
    y1 = z[_Y1]; s1 = z[_S1]; w1 = z[_W1]
    lamN2 = z[_LAMN2]; beta2 = z[_BETA2]; r2 = z[_R2]
    y2 = z[_Y2]; s2 = z[_S2]; w2 = z[_W2]

    hT = z.new_tensor(h)
    mT = z.new_tensor(m_cube)
    mrT = z.new_tensor(m_robot)
    muT = z.new_tensor(mu)
    mu_st = z.new_tensor(target_mu)
    grav = _GRAVITY.to(dtype=dtype, device=device)
    u1 = u1.to(dtype=dtype, device=device)
    u2 = u2.to(dtype=dtype, device=device)

    q_next = quat_integrate(qk, v[3:].detach(), h)
    q_pose = torch.cat([p, q_next])

    cnt1 = cube_sphere_contact(q_pose, p1, sdf_grid, r_sphere)
    cnt2 = cube_sphere_contact(q_pose, p2, sdf_grid, r_sphere)
    Jn1, Jt1_1, Jt2_1 = contact_jacobians_3d(
        cnt1.normal, cnt1.tangent1, cnt1.tangent2, cnt1.r_cp)
    Jn2, Jt1_2, Jt2_2 = contact_jacobians_3d(
        cnt2.normal, cnt2.tangent1, cnt2.tangent2, cnt2.r_cp)

    force1_gen, force1_lin = _contact_forces(cnt1, Jn1, Jt1_1, Jt2_1, lamN1, beta1)
    force2_gen, force2_lin = _contact_forces(cnt2, Jn2, Jt1_2, Jt2_2, lamN2, beta2)

    cube_force = force1_gen + force2_gen
    cube_impulse = hT * cube_force
    grav_impulse = torch.cat([hT * mT * grav, grav.new_zeros(3)])
    M_inv_cube = torch.cat([mT.new_ones(3) / mT, 1.0 / I_body])

    r_cube_dyn = v - vk - M_inv_cube * (cube_impulse + grav_impulse)
    r_cube_kin = p - pk - hT * v[:3]

    # u is the free/commanded velocity. Contact reaction changes the realized
    # velocity by the opposite linear contact impulse.
    r_s1_dyn = vs1 - u1 + (hT / mrT) * force1_lin
    r_s2_dyn = vs2 - u2 + (hT / mrT) * force2_lin
    r_s1_kin = p1 - p1k - hT * vs1
    r_s2_kin = p2 - p2k - hT * vs2

    R_c1 = contact_residual_block(lamN1, beta1, r1, y1, s1, w1,
                                  cnt1, v, vs1, muT, mu_st)
    R_c2 = contact_residual_block(lamN2, beta2, r2, y2, s2, w2,
                                  cnt2, v, vs2, muT, mu_st)

    return torch.cat([
        r_cube_dyn,
        r_cube_kin,
        r_s1_dyn,
        r_s1_kin,
        r_s2_dyn,
        r_s2_kin,
        R_c1,
        R_c2,
    ])


def default_z_ground(pk, qk, vk, p1k, p2k, u1, u2, h, m_robot, mu, target_mu,
                     sdf_grid, r_sphere, half, ground_z, ground_sharpness,
                     dtype, device):
    z = torch.zeros(NZ_GROUND, dtype=dtype, device=device)
    z[:NZ] = default_z(
        pk, qk, vk, p1k, p2k, u1, u2,
        h, m_robot, mu, target_mu, sdf_grid, r_sphere, dtype, device,
    )
    for idx in [_LAMNG, _BETAG, _RG, _YG, _SG, _WG]:
        z[idx] = 1e-2
    with torch.no_grad():
        q_next = quat_integrate(qk.detach(), z[_VA], h)
        cnt_g = cube_ground_contact(
            torch.cat([z[_P], q_next]), half, ground_z, ground_sharpness)
        _init_contact_block(
            z, cnt_g, z[_V], z.new_zeros(3),
            _LAMNG, _BETAG, _RG, _YG, _SG, _WG,
            h, m_robot, mu, target_mu,
        )
    return z


def residual_3d_velocity_robot_ground(z, pk, qk, vk, p1k, p2k, u1, u2,
                                      h, m_cube, I_body, m_robot, mu,
                                      target_mu, sdf_grid, r_sphere, half,
                                      ground_z, ground_sharpness, device, dtype):
    """R(z)=0 for velocity-commanded robots with cube-ground contact."""
    p = z[_P]
    v = z[_V]
    p1 = z[_P1]
    vs1 = z[_VS1]
    p2 = z[_P2]
    vs2 = z[_VS2]

    lamN1 = z[_LAMN1]; beta1 = z[_BETA1]; r1 = z[_R1]
    y1 = z[_Y1]; s1 = z[_S1]; w1 = z[_W1]
    lamN2 = z[_LAMN2]; beta2 = z[_BETA2]; r2 = z[_R2]
    y2 = z[_Y2]; s2 = z[_S2]; w2 = z[_W2]
    lamNG = z[_LAMNG]; betaG = z[_BETAG]; rG = z[_RG]
    yG = z[_YG]; sG = z[_SG]; wG = z[_WG]

    hT = z.new_tensor(h)
    mT = z.new_tensor(m_cube)
    mrT = z.new_tensor(m_robot)
    muT = z.new_tensor(mu)
    mu_st = z.new_tensor(target_mu)
    grav = _GRAVITY.to(dtype=dtype, device=device)
    u1 = u1.to(dtype=dtype, device=device)
    u2 = u2.to(dtype=dtype, device=device)

    q_next = quat_integrate(qk, v[3:].detach(), h)
    q_pose = torch.cat([p, q_next])

    cnt1 = cube_sphere_contact(q_pose, p1, sdf_grid, r_sphere)
    cnt2 = cube_sphere_contact(q_pose, p2, sdf_grid, r_sphere)
    cntG = cube_ground_contact(q_pose, half, ground_z, ground_sharpness)
    Jn1, Jt1_1, Jt2_1 = contact_jacobians_3d(
        cnt1.normal, cnt1.tangent1, cnt1.tangent2, cnt1.r_cp)
    Jn2, Jt1_2, Jt2_2 = contact_jacobians_3d(
        cnt2.normal, cnt2.tangent1, cnt2.tangent2, cnt2.r_cp)
    JnG, Jt1_G, Jt2_G = contact_jacobians_3d(
        cntG.normal, cntG.tangent1, cntG.tangent2, cntG.r_cp)

    force1_gen, force1_lin = _contact_forces(cnt1, Jn1, Jt1_1, Jt2_1, lamN1, beta1)
    force2_gen, force2_lin = _contact_forces(cnt2, Jn2, Jt1_2, Jt2_2, lamN2, beta2)
    forceG_gen, _ = _contact_forces(cntG, JnG, Jt1_G, Jt2_G, lamNG, betaG)

    cube_force = force1_gen + force2_gen + forceG_gen
    cube_impulse = hT * cube_force
    grav_impulse = torch.cat([hT * mT * grav, grav.new_zeros(3)])
    M_inv_cube = torch.cat([mT.new_ones(3) / mT, 1.0 / I_body])

    r_cube_dyn = v - vk - M_inv_cube * (cube_impulse + grav_impulse)
    r_cube_kin = p - pk - hT * v[:3]

    r_s1_dyn = vs1 - u1 + (hT / mrT) * force1_lin
    r_s2_dyn = vs2 - u2 + (hT / mrT) * force2_lin
    r_s1_kin = p1 - p1k - hT * vs1
    r_s2_kin = p2 - p2k - hT * vs2

    R_c1 = contact_residual_block(lamN1, beta1, r1, y1, s1, w1,
                                  cnt1, v, vs1, muT, mu_st)
    R_c2 = contact_residual_block(lamN2, beta2, r2, y2, s2, w2,
                                  cnt2, v, vs2, muT, mu_st)
    R_cG = contact_residual_block(lamNG, betaG, rG, yG, sG, wG,
                                  cntG, v, z.new_zeros(3), muT, mu_st)

    return torch.cat([
        r_cube_dyn,
        r_cube_kin,
        r_s1_dyn,
        r_s1_kin,
        r_s2_dyn,
        r_s2_kin,
        R_c1,
        R_c2,
        R_cG,
    ])


class StepCubeSphereVelocityRobotIPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pk, qk, vk, p1k, p2k, u1, u2,
                h_t, m_cube_t, I_body, m_robot_t, mu_t,
                target_mu, tol, max_newton, frac_to_boundary, ls_beta,
                sdf_grid, r_sphere, z_prev):
        device = pk.device
        dtype = pk.dtype
        h = h_t.item()
        m_cube = m_cube_t.item()
        m_robot = m_robot_t.item()
        mu = mu_t.item()

        def res_fn(z):
            return residual_3d_velocity_robot(
                z, pk.detach(), qk.detach(), vk.detach(),
                p1k.detach(), p2k.detach(), u1.detach(), u2.detach(),
                h, m_cube, I_body.detach(), m_robot, mu, target_mu,
                sdf_grid, r_sphere, device, dtype,
            )

        z = (z_prev.clone().detach() if z_prev is not None
             else default_z(pk.detach(), qk.detach(), vk.detach(),
                            p1k.detach(), p2k.detach(), u1.detach(), u2.detach(),
                            h, m_robot, mu, target_mu, sdf_grid, r_sphere, dtype, device))
        z = clamp_positive(z)

        for newton_iter in range(max_newton):
            R = res_fn(z)
            norm_R = torch.linalg.norm(R)
            if NEWTON_DEBUG:
                print(f"    [VelocityRobot Newton {newton_iter:2d}] "
                      f"||R||={norm_R.item():.3e} p={z[_P].tolist()}")
            if torch.isnan(norm_R):
                break
            if norm_R < tol:
                break

            J = torch.autograd.functional.jacobian(res_fn, z, vectorize=True).detach()
            try:
                dz = torch.linalg.solve(J, -R.detach())
            except torch.linalg.LinAlgError:
                dz = torch.linalg.lstsq(J, -R.detach()).solution

            alpha = fraction_to_boundary_step(_positive_vector(z), _positive_vector(dz), frac_to_boundary)
            z_next, _, ls_ok = backtracking_line_search_velocity(res_fn, z, dz, alpha, ls_beta)
            if not ls_ok:
                if NEWTON_DEBUG:
                    print("      line search failed; keeping previous Newton iterate")
                break
            z = z_next

        J_star = torch.autograd.functional.jacobian(res_fn, z, vectorize=True).detach()
        q_next = quat_integrate(qk.detach(), z[_VA].detach(), h)

        ctx.save_for_backward(
            z, J_star,
            pk.detach(), qk.detach(), vk.detach(),
            p1k.detach(), p2k.detach(), u1.detach(), u2.detach(),
        )
        ctx.h = h
        ctx.m_cube = m_cube
        ctx.m_robot = m_robot
        ctx.mu = mu
        ctx.target_mu = target_mu
        ctx.sdf_grid = sdf_grid
        ctx.r_sphere = r_sphere
        ctx.I_body = I_body.detach()

        q_pose = torch.cat([z[_P], q_next])
        cnt1 = cube_sphere_contact(q_pose, z[_P1], sdf_grid, r_sphere)
        cnt2 = cube_sphere_contact(q_pose, z[_P2], sdf_grid, r_sphere)

        return (
            z[_P].clone(), q_next, z[_V].clone(),
            z[_P1].clone(), z[_P2].clone(),
            z[_VS1].clone(), z[_VS2].clone(),
            z[_LAMN1], z[_LAMN2],
            cnt1.phi, cnt2.phi,
            z.detach(),
        )

    @staticmethod
    def backward(ctx, grad_p, grad_q, grad_v,
                 grad_p1, grad_p2, grad_vs1, grad_vs2,
                 grad_lam1, grad_lam2,
                 grad_phi1, grad_phi2, grad_z_out):
        z, J_star, pk, qk, vk, p1k, p2k, u1, u2 = ctx.saved_tensors
        device, dtype = z.device, z.dtype

        zero3 = torch.zeros(3, dtype=dtype, device=device)
        zero4 = torch.zeros(4, dtype=dtype, device=device)
        zero6 = torch.zeros(6, dtype=dtype, device=device)
        zero_scalar = torch.zeros((), dtype=dtype, device=device)

        with torch.enable_grad():
            z_out = z.detach().clone().requires_grad_(True)
            qk_out = qk.detach().clone().requires_grad_(True)
            q_next_out = quat_integrate(qk_out, z_out[_VA], ctx.h)
            q_pose_out = torch.cat([z_out[_P], q_next_out])
            cnt1_out = cube_sphere_contact(
                q_pose_out, z_out[_P1], ctx.sdf_grid, ctx.r_sphere)
            cnt2_out = cube_sphere_contact(
                q_pose_out, z_out[_P2], ctx.sdf_grid, ctx.r_sphere)

            output_grad_z, output_grad_qk = torch.autograd.grad(
                outputs=(
                    z_out[_P],
                    q_next_out,
                    z_out[_V],
                    z_out[_P1],
                    z_out[_P2],
                    z_out[_VS1],
                    z_out[_VS2],
                    z_out[_LAMN1],
                    z_out[_LAMN2],
                    cnt1_out.phi,
                    cnt2_out.phi,
                ),
                inputs=(z_out, qk_out),
                grad_outputs=(
                    zero3 if grad_p is None else grad_p,
                    zero4 if grad_q is None else grad_q,
                    zero6 if grad_v is None else grad_v,
                    zero3 if grad_p1 is None else grad_p1,
                    zero3 if grad_p2 is None else grad_p2,
                    zero3 if grad_vs1 is None else grad_vs1,
                    zero3 if grad_vs2 is None else grad_vs2,
                    zero_scalar if grad_lam1 is None else grad_lam1,
                    zero_scalar if grad_lam2 is None else grad_lam2,
                    zero_scalar if grad_phi1 is None else grad_phi1,
                    zero_scalar if grad_phi2 is None else grad_phi2,
                ),
                retain_graph=False,
                allow_unused=True,
            )

        dLdz = torch.zeros(NZ, dtype=dtype, device=device) if output_grad_z is None else output_grad_z
        direct_grad_qk = zero4 if output_grad_qk is None else output_grad_qk

        try:
            w = torch.linalg.solve(J_star.T, dLdz)
        except torch.linalg.LinAlgError:
            w = torch.linalg.lstsq(J_star.T, dLdz).solution

        pk_r = pk.detach().clone().requires_grad_(True)
        qk_r = qk.detach().clone().requires_grad_(True)
        vk_r = vk.detach().clone().requires_grad_(True)
        p1k_r = p1k.detach().clone().requires_grad_(True)
        p2k_r = p2k.detach().clone().requires_grad_(True)
        u1_r = u1.detach().clone().requires_grad_(True)
        u2_r = u2.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            R_theta = residual_3d_velocity_robot(
                z.detach(), pk_r, qk_r, vk_r,
                p1k_r, p2k_r, u1_r, u2_r,
                ctx.h, ctx.m_cube, ctx.I_body, ctx.m_robot,
                ctx.mu, ctx.target_mu,
                ctx.sdf_grid, ctx.r_sphere, device, dtype,
            )
            R_theta_grads = torch.autograd.grad(
                outputs=R_theta,
                inputs=(pk_r, qk_r, vk_r, p1k_r, p2k_r, u1_r, u2_r),
                grad_outputs=-w,
                retain_graph=False,
                allow_unused=True,
            )

        def _safe(g, ref_zero):
            return ref_zero if g is None else g

        grad_pk = _safe(R_theta_grads[0], zero3)
        grad_qk = direct_grad_qk + _safe(R_theta_grads[1], zero4)
        grad_vk = _safe(R_theta_grads[2], zero6)
        grad_p1k = _safe(R_theta_grads[3], zero3)
        grad_p2k = _safe(R_theta_grads[4], zero3)
        grad_u1 = _safe(R_theta_grads[5], zero3)
        grad_u2 = _safe(R_theta_grads[6], zero3)

        return (
            grad_pk, grad_qk, grad_vk,
            grad_p1k, grad_p2k, grad_u1, grad_u2,
            None, None, None, None, None,
            None, None, None, None, None,
            None, None, None,
        )


def step_cube_sphere_velocity_robot_ip(
    pk, qk, vk, p1k, p2k, u1, u2,
    h, m_cube, I_body, m_robot, mu,
    sdf_grid, r_sphere,
    ipm_opts: Optional[IPMOptions3D] = None,
    z_prev: Optional[torch.Tensor] = None,
):
    """
    One timestep with velocity-commanded dynamic sphere robots.

    Returns:
        p_next, q_next, v_next,
        p1_next, p2_next, vs1_next, vs2_next,
        lamN1, lamN2, phi1, phi2, z_star
    """
    if ipm_opts is None:
        ipm_opts = IPMOptions3D()

    device, dtype = pk.device, pk.dtype
    h_t = torch.tensor(h, dtype=dtype, device=device)
    m_cube_t = torch.tensor(m_cube, dtype=dtype, device=device)
    m_robot_t = torch.tensor(m_robot, dtype=dtype, device=device)
    mu_t = torch.tensor(mu, dtype=dtype, device=device)

    return StepCubeSphereVelocityRobotIPFn.apply(
        pk, qk, vk, p1k, p2k, u1, u2,
        h_t, m_cube_t, I_body, m_robot_t, mu_t,
        ipm_opts.target_mu,
        ipm_opts.tol,
        ipm_opts.max_newton,
        ipm_opts.frac_to_boundary,
        ipm_opts.ls_beta,
        sdf_grid, r_sphere,
        z_prev,
    )


class StepCubeSphereVelocityRobotGroundIPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pk, qk, vk, p1k, p2k, u1, u2,
                h_t, m_cube_t, I_body, m_robot_t, mu_t,
                half_t, ground_z_t, ground_sharpness_t,
                target_mu, tol, max_newton, frac_to_boundary, ls_beta,
                sdf_grid, r_sphere, z_prev):
        device = pk.device
        dtype = pk.dtype
        h = h_t.item()
        m_cube = m_cube_t.item()
        m_robot = m_robot_t.item()
        mu = mu_t.item()
        half = half_t.item()
        ground_z = ground_z_t.item()
        ground_sharpness = ground_sharpness_t.item()

        def res_fn(z):
            return residual_3d_velocity_robot_ground(
                z, pk.detach(), qk.detach(), vk.detach(),
                p1k.detach(), p2k.detach(), u1.detach(), u2.detach(),
                h, m_cube, I_body.detach(), m_robot, mu, target_mu,
                sdf_grid, r_sphere, half, ground_z, ground_sharpness,
                device, dtype,
            )

        z = (z_prev.clone().detach() if z_prev is not None
             else default_z_ground(pk.detach(), qk.detach(), vk.detach(),
                                   p1k.detach(), p2k.detach(), u1.detach(), u2.detach(),
                                   h, m_robot, mu, target_mu, sdf_grid, r_sphere,
                                   half, ground_z, ground_sharpness, dtype, device))
        z = clamp_positive_ground(z)

        for newton_iter in range(max_newton):
            R = res_fn(z)
            norm_R = torch.linalg.norm(R)
            if NEWTON_DEBUG:
                print(f"    [VelocityRobotGround Newton {newton_iter:2d}] "
                      f"||R||={norm_R.item():.3e} p={z[_P].tolist()}")
            if torch.isnan(norm_R):
                break
            if norm_R < tol:
                break

            J = torch.autograd.functional.jacobian(res_fn, z, vectorize=True).detach()
            try:
                dz = torch.linalg.solve(J, -R.detach())
            except torch.linalg.LinAlgError:
                dz = torch.linalg.lstsq(J, -R.detach()).solution

            alpha = fraction_to_boundary_step(
                _positive_vector_ground(z), _positive_vector_ground(dz),
                frac_to_boundary)
            z_next, _, ls_ok = backtracking_line_search_velocity_ground(res_fn, z, dz, alpha, ls_beta)
            if not ls_ok:
                if NEWTON_DEBUG:
                    print("      line search failed; keeping previous Newton iterate")
                break
            z = clamp_positive_ground(z_next)

        J_star = torch.autograd.functional.jacobian(res_fn, z, vectorize=True).detach()
        q_next = quat_integrate(qk.detach(), z[_VA].detach(), h)

        ctx.save_for_backward(
            z, J_star,
            pk.detach(), qk.detach(), vk.detach(),
            p1k.detach(), p2k.detach(), u1.detach(), u2.detach(),
        )
        ctx.h = h
        ctx.m_cube = m_cube
        ctx.m_robot = m_robot
        ctx.mu = mu
        ctx.half = half
        ctx.ground_z = ground_z
        ctx.ground_sharpness = ground_sharpness
        ctx.target_mu = target_mu
        ctx.sdf_grid = sdf_grid
        ctx.r_sphere = r_sphere
        ctx.I_body = I_body.detach()

        q_pose = torch.cat([z[_P], q_next])
        cnt1 = cube_sphere_contact(q_pose, z[_P1], sdf_grid, r_sphere)
        cnt2 = cube_sphere_contact(q_pose, z[_P2], sdf_grid, r_sphere)
        cntG = cube_ground_contact(q_pose, half, ground_z, ground_sharpness)

        return (
            z[_P].clone(), q_next, z[_V].clone(),
            z[_P1].clone(), z[_P2].clone(),
            z[_VS1].clone(), z[_VS2].clone(),
            z[_LAMN1], z[_LAMN2],
            cnt1.phi, cnt2.phi,
            z[_LAMNG], cntG.phi,
            z.detach(),
        )

    @staticmethod
    def backward(ctx, grad_p, grad_q, grad_v,
                 grad_p1, grad_p2, grad_vs1, grad_vs2,
                 grad_lam1, grad_lam2,
                 grad_phi1, grad_phi2,
                 grad_lamG, grad_phiG, grad_z_out):
        z, J_star, pk, qk, vk, p1k, p2k, u1, u2 = ctx.saved_tensors
        device, dtype = z.device, z.dtype

        zero3 = torch.zeros(3, dtype=dtype, device=device)
        zero4 = torch.zeros(4, dtype=dtype, device=device)
        zero6 = torch.zeros(6, dtype=dtype, device=device)
        zero_scalar = torch.zeros((), dtype=dtype, device=device)

        with torch.enable_grad():
            z_out = z.detach().clone().requires_grad_(True)
            qk_out = qk.detach().clone().requires_grad_(True)
            q_next_out = quat_integrate(qk_out, z_out[_VA], ctx.h)
            q_pose_out = torch.cat([z_out[_P], q_next_out])
            cnt1_out = cube_sphere_contact(
                q_pose_out, z_out[_P1], ctx.sdf_grid, ctx.r_sphere)
            cnt2_out = cube_sphere_contact(
                q_pose_out, z_out[_P2], ctx.sdf_grid, ctx.r_sphere)
            cntG_out = cube_ground_contact(
                q_pose_out, ctx.half, ctx.ground_z, ctx.ground_sharpness)

            output_grad_z, output_grad_qk = torch.autograd.grad(
                outputs=(
                    z_out[_P],
                    q_next_out,
                    z_out[_V],
                    z_out[_P1],
                    z_out[_P2],
                    z_out[_VS1],
                    z_out[_VS2],
                    z_out[_LAMN1],
                    z_out[_LAMN2],
                    cnt1_out.phi,
                    cnt2_out.phi,
                    z_out[_LAMNG],
                    cntG_out.phi,
                ),
                inputs=(z_out, qk_out),
                grad_outputs=(
                    zero3 if grad_p is None else grad_p,
                    zero4 if grad_q is None else grad_q,
                    zero6 if grad_v is None else grad_v,
                    zero3 if grad_p1 is None else grad_p1,
                    zero3 if grad_p2 is None else grad_p2,
                    zero3 if grad_vs1 is None else grad_vs1,
                    zero3 if grad_vs2 is None else grad_vs2,
                    zero_scalar if grad_lam1 is None else grad_lam1,
                    zero_scalar if grad_lam2 is None else grad_lam2,
                    zero_scalar if grad_phi1 is None else grad_phi1,
                    zero_scalar if grad_phi2 is None else grad_phi2,
                    zero_scalar if grad_lamG is None else grad_lamG,
                    zero_scalar if grad_phiG is None else grad_phiG,
                ),
                retain_graph=False,
                allow_unused=True,
            )

        dLdz = torch.zeros(NZ_GROUND, dtype=dtype, device=device) if output_grad_z is None else output_grad_z
        direct_grad_qk = zero4 if output_grad_qk is None else output_grad_qk

        try:
            w = torch.linalg.solve(J_star.T, dLdz)
        except torch.linalg.LinAlgError:
            w = torch.linalg.lstsq(J_star.T, dLdz).solution

        pk_r = pk.detach().clone().requires_grad_(True)
        qk_r = qk.detach().clone().requires_grad_(True)
        vk_r = vk.detach().clone().requires_grad_(True)
        p1k_r = p1k.detach().clone().requires_grad_(True)
        p2k_r = p2k.detach().clone().requires_grad_(True)
        u1_r = u1.detach().clone().requires_grad_(True)
        u2_r = u2.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            R_theta = residual_3d_velocity_robot_ground(
                z.detach(), pk_r, qk_r, vk_r,
                p1k_r, p2k_r, u1_r, u2_r,
                ctx.h, ctx.m_cube, ctx.I_body, ctx.m_robot,
                ctx.mu, ctx.target_mu,
                ctx.sdf_grid, ctx.r_sphere,
                ctx.half, ctx.ground_z, ctx.ground_sharpness,
                device, dtype,
            )
            R_theta_grads = torch.autograd.grad(
                outputs=R_theta,
                inputs=(pk_r, qk_r, vk_r, p1k_r, p2k_r, u1_r, u2_r),
                grad_outputs=-w,
                retain_graph=False,
                allow_unused=True,
            )

        def _safe(g, ref_zero):
            return ref_zero if g is None else g

        grad_pk = _safe(R_theta_grads[0], zero3)
        grad_qk = direct_grad_qk + _safe(R_theta_grads[1], zero4)
        grad_vk = _safe(R_theta_grads[2], zero6)
        grad_p1k = _safe(R_theta_grads[3], zero3)
        grad_p2k = _safe(R_theta_grads[4], zero3)
        grad_u1 = _safe(R_theta_grads[5], zero3)
        grad_u2 = _safe(R_theta_grads[6], zero3)

        return (
            grad_pk, grad_qk, grad_vk,
            grad_p1k, grad_p2k, grad_u1, grad_u2,
            None, None, None, None, None,
            None, None, None,
            None, None, None, None, None,
            None, None, None,
        )


def step_cube_sphere_velocity_robot_ground_ip(
    pk, qk, vk, p1k, p2k, u1, u2,
    h, m_cube, I_body, m_robot, mu,
    sdf_grid, r_sphere, half,
    ground_z=0.0, ground_sharpness=80.0,
    ipm_opts: Optional[IPMOptions3D] = None,
    z_prev: Optional[torch.Tensor] = None,
):
    """
    One timestep with velocity-commanded robots plus analytic cube-ground contact.

    Returns:
        p_next, q_next, v_next,
        p1_next, p2_next, vs1_next, vs2_next,
        lamN1, lamN2, phi1, phi2, lamNG, phiG, z_star
    """
    if ipm_opts is None:
        ipm_opts = IPMOptions3D()

    device, dtype = pk.device, pk.dtype
    h_t = torch.tensor(h, dtype=dtype, device=device)
    m_cube_t = torch.tensor(m_cube, dtype=dtype, device=device)
    m_robot_t = torch.tensor(m_robot, dtype=dtype, device=device)
    mu_t = torch.tensor(mu, dtype=dtype, device=device)
    half_t = torch.tensor(half, dtype=dtype, device=device)
    ground_z_t = torch.tensor(ground_z, dtype=dtype, device=device)
    ground_sharpness_t = torch.tensor(ground_sharpness, dtype=dtype, device=device)

    return StepCubeSphereVelocityRobotGroundIPFn.apply(
        pk, qk, vk, p1k, p2k, u1, u2,
        h_t, m_cube_t, I_body, m_robot_t, mu_t,
        half_t, ground_z_t, ground_sharpness_t,
        ipm_opts.target_mu,
        ipm_opts.tol,
        ipm_opts.max_newton,
        ipm_opts.frac_to_boundary,
        ipm_opts.ls_beta,
        sdf_grid, r_sphere,
        z_prev,
    )
