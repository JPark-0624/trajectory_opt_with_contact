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
        lam_star: Contact impulses
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
    
    # Solve QP for contact impulses
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


# %%%%%%% Interior Point %%%%%%%%%%%%%%%%%%

from .geometry import obb_contact, obb_contact_blend2

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

class StepSquarePosIPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                qk, vk, pusher_pos, u_push,
                h, m, Izz, half, mu,
                target_mu, smooth_sdf, tol, max_newton,
                frac_to_boundary, ls_beta,
                enable_viscous_ground_friction, c_lin, c_ang,
                skip_solving_threshold):
        """
        All differentiable arguments must be tensors.
        Scalars like m, Izz, half, mu can be tensors (dtype/shape-compatible).
        You can choose which ones require grad.
        """
        device = qk.device
        dtype = qk.dtype

        # ---- Local copies (avoid in-graph ops for solver path) ----
        hT   = torch.tensor(h, dtype=dtype, device=device, requires_grad=True)
        muT  = torch.tensor(mu, dtype=dtype, device=device, requires_grad=True)
        mT   =  torch.tensor(m, dtype=dtype, device=device, requires_grad=True)
        IzzT =  torch.tensor(Izz, dtype=dtype, device=device, requires_grad=True)
        halfT=  torch.tensor(half, dtype=dtype, device=device, requires_grad=True)

        # Mass inverse
        M_inv = torch.diag(torch.stack([1.0/mT, 1.0/mT, 1.0/IzzT]))

        # Kinematic pusher update (explicit)
        pusher_pos_next = pusher_pos + hT * u_push

        # Free prediction & cheap skip
        q_free = qk + hT * vk
        cnt_free = obb_contact_blend2(q_free, pusher_pos_next, halfT)   # user-provided
        if cnt_free.phi > skip_solving_threshold:
            # no-contact branch: semi-implicit Euler with mild damping
            v_next = vk - 0.3 * vk * hT
            q_next = qk + hT * v_next
            lam_vec = torch.zeros(2, dtype=dtype, device=device)
            phi = cnt_free.phi

            # Save for backward: needed to propagate grads (no implicit solve here)
            ctx.save_for_backward(
                None, None, None, None,  # placeholders when no solve
                qk, vk, pusher_pos, u_push, hT, mT, IzzT, halfT, muT,
            )
            ctx.constants = dict(
                target_mu=target_mu, smooth_sdf=smooth_sdf, tol=tol, max_newton=max_newton,
                frac_to_boundary=frac_to_boundary, ls_beta=ls_beta,
                enable_viscous_ground_friction=enable_viscous_ground_friction,
                c_lin=c_lin, c_ang=c_ang, skip_solving_threshold=skip_solving_threshold,
                solved=False
            )
            return q_next, v_next, pusher_pos_next, lam_vec, phi

        # ============ Define residual R(z) ============
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

        def residual(z):
            _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)

            cnt = obb_contact_blend2(_q, pusher_pos_next, halfT)  # user-provided, differentiable
            n, t, r_cp = cnt.normal, cnt.tangent, cnt.r_cp
            Jn, Jt = contact_jacobians(n, t, r_cp)               # user-provided

            # Tangential rel velocity at cp
            v_cp = _v[0:2] + _v[2] * perp(r_cp)
            v_rel_t = torch.dot(t, v_cp - u_push)

            # Two facets +/-
            v_facets = torch.stack([v_rel_t, -v_rel_t])

            lamT_total = _beta[0] - _beta[1]
            impulse = Jn * _lamN + Jt * lamT_total

            # Option A: viscous ground friction
            Fg = torch.zeros(3, dtype=dtype, device=device)
            if enable_viscous_ground_friction and (c_lin > 0.0 or c_ang > 0.0):
                Fg = torch.stack([
                    torch.tensor(-c_lin, dtype=dtype, device=device) * _v[0],
                    torch.tensor(-c_lin, dtype=dtype, device=device) * _v[1],
                    torch.tensor(-c_ang, dtype=dtype, device=device) * _v[2],
                ])

            dv = M_inv @ (impulse + hT * Fg)

            r_dyn = _v - vk - dv                              # (3,)
            r_kin = _q - qk - hT * _v                         # (3,)
            r_gap = _y - cnt.phi                              # (1,)
            r_cone = _s - (muT * _lamN - torch.sum(_beta))    # (1,)
            r_slip = _w - (v_facets + _r)                     # (2,)

            mu_star = torch.tensor(target_mu, dtype=dtype, device=device)
            r_c1 = _y * _lamN - mu_star
            r_c2 = _r * _s - mu_star
            r_c3 = _beta * _w - mu_star                       # (2,)

            return torch.cat([r_dyn, r_kin, r_gap.view(1), r_cone.view(1), r_slip,
                              r_c1.view(1), r_c2.view(1), r_c3])

        # Initialize z
        q = q_free.clone().detach().requires_grad_(True)
        v = vk.clone().detach().requires_grad_(True)
        lamN = torch.tensor(1e-3, dtype=dtype, device=device, requires_grad=True)
        beta = torch.full((2,), 1e-3, dtype=dtype, device=device, requires_grad=True)
        r = torch.tensor(1e-3, dtype=dtype, device=device, requires_grad=True)
        y = torch.tensor(max(float(target_mu), 1e-4), dtype=dtype, device=device, requires_grad=True)
        s = torch.tensor(max(float(target_mu), 1e-4), dtype=dtype, device=device, requires_grad=True)
        w = torch.full((2,), max(float(target_mu), 1e-4), dtype=dtype, device=device, requires_grad=True)

        z = torch.cat([q, v, lamN.view(1), beta, r.view(1), y.view(1), s.view(1), w])

        # Newton on R(z)=0 (same as your current code, compact)
        for _ in range(int(max_newton)):
            z = z.detach().requires_grad_(True)
            R = residual(z)
            if float(torch.linalg.norm(R)) < float(tol):
                break

            J = torch.autograd.functional.jacobian(residual, z, strict=False, create_graph=False)
            J = J.reshape(R.numel(), z.numel())

            reg = 1e-8 * torch.eye(J.shape[0], dtype=dtype, device=device)
            try:
                dz = torch.linalg.solve(J + reg, -R)
            except RuntimeError:
                dz, *_ = torch.linalg.lstsq(J + reg, -R)

            # fraction-to-boundary for positive vars
            _q,_v,_lamN,_beta,_r,_y,_s,_w = (
                z[0:3], z[3:6], z[6], z[7:9], z[9], z[10], z[11], z[12:14]
            )
            alpha_pos = 1.0
            def frac_to_bd(x, dx, tau):
                # find maximum alpha in (0,1] s.t. x + alpha*dx >= (1-tau)*x
                with torch.no_grad():
                    mask = dx < 0
                    if torch.any(mask):
                        al = ((1.0 - tau) * x[mask] - x[mask]) / dx[mask]
                        a = torch.min(al).item()
                        return max(min(0.99*a, 1.0), 1e-6)
                    return 1.0
            tau = float(frac_to_boundary)
            for x, dx in [(_lamN.view(1), dz[6].view(1)),
                          (_beta, dz[7:9]), (_r.view(1), dz[9].view(1)),
                          (_y.view(1), dz[10].view(1)), (_s.view(1), dz[11].view(1)),
                          (_w, dz[12:14])]:
                alpha_pos = min(alpha_pos, frac_to_bd(x, dx, tau))
            alpha = alpha_pos

            # backtracking
            good = False
            for _ in range(15):
                z_trial = z + alpha * dz
                # clamp positivity
                z_trial[6] = torch.clamp(z_trial[6], min=1e-12)
                z_trial[7:9] = torch.clamp(z_trial[7:9], min=1e-12)
                z_trial[9] = torch.clamp(z_trial[9], min=1e-12)
                z_trial[10] = torch.clamp(z_trial[10], min=1e-12)
                z_trial[11] = torch.clamp(z_trial[11], min=1e-12)
                z_trial[12:14] = torch.clamp(z_trial[12:14], min=1e-12)

                Rt = residual(z_trial)
                if torch.linalg.norm(Rt) <= (1.0 - 1e-4 * alpha) * torch.linalg.norm(R):
                    z = z_trial.detach()
                    good = True
                    break
                alpha *= ls_beta
            if not good:
                z = z_trial.detach()

        # Unpack solution
        q_next = z[0:3]
        v_next = z[3:6]
        lamN = z[6]
        beta = z[7:9]
        y = z[10]
        lam_t = beta[0] - beta[1]
        lam_vec = torch.stack([lamN, lam_t])
        phi = y

        # Save for backward: we store just enough to rebuild J and do IFT
        ctx.save_for_backward(
            z.detach(),                           # z*
            pusher_pos_next.detach(),             # pusher at k+1
            q_next.detach(), v_next.detach(),     # outputs (z-part)
            qk, vk, pusher_pos, u_push, hT, mT, IzzT, halfT, muT
        )
        ctx.constants = dict(
            target_mu=float(target_mu), smooth_sdf=float(smooth_sdf), tol=float(tol), max_newton=int(max_newton),
            frac_to_boundary=float(frac_to_boundary), ls_beta=float(ls_beta),
            enable_viscous_ground_friction=bool(enable_viscous_ground_friction),
            c_lin=float(c_lin), c_ang=float(c_ang), skip_solving_threshold=float(skip_solving_threshold),
            solved=True
        )
        return q_next, v_next, pusher_pos_next, lam_vec, phi

    @staticmethod
    def backward(ctx, grad_q_next, grad_v_next, grad_pusher_pos_next, grad_lam_vec, grad_phi):
        # Retrieve
        (z_star, pusher_pos_next, q_next, v_next,
         qk, vk, pusher_pos, u_push, hT, mT, IzzT, halfT, muT) = ctx.saved_tensors
        C = ctx.constants

        # If we skipped solve, gradients flow through explicit formulas only
        if not C["solved"]:
            # Upstream grads may be None if those outputs weren't used; make them zeros
            gz_q  = torch.zeros_like(qk)   if grad_q_next is None else grad_q_next
            gz_v  = torch.zeros_like(vk)   if grad_v_next is None else grad_v_next
            gz_pn = torch.zeros_like(pusher_pos) if grad_pusher_pos_next is None else grad_pusher_pos_next
            gz_l  = torch.zeros(2, dtype=qk.dtype, device=qk.device) if grad_lam_vec is None else grad_lam_vec
            gz_phi= torch.zeros((), dtype=qk.dtype, device=qk.device) if grad_phi is None else grad_phi

            # Skip-branch forward definitions:
            # v_next = (1 - 0.3*hT) * vk
            # q_next = qk + hT * v_next = qk + (hT - 0.3*hT^2) * vk
            # pusher_pos_next = pusher_pos + hT * u_push

            one = torch.tensor(1.0, dtype=qk.dtype, device=qk.device)
            coeff = (one - 0.3 * hT)              # scalar tensor
            dv_dvk = coeff                        # ∂v_next/∂vk
            dq_dvk = hT * coeff                   # ∂q_next/∂vk
            dq_dh  = (one - 0.6 * hT) * vk        # ∂q_next/∂h
            dv_dh  = (-0.3) * vk                  # ∂v_next/∂h

            # Gradients
            g_qk = gz_q                           # ∂q_next/∂qk = I
            g_vk = gz_v * dv_dvk + gz_q * dq_dvk  # chain rule through q_next
            g_pusher = gz_pn                      # ∂pusher_next/∂pusher_pos = I
            g_u = gz_pn * hT                      # ∂pusher_next/∂u = h
            g_h = (gz_pn @ u_push) + (gz_q @ dq_dh) + (gz_v @ dv_dh)

            # No grads w.r.t. m, Izz, half, mu, and all hyper-params in skip branch
            return (
                g_qk, g_vk, g_pusher, g_u,
                None, None, None, None, None,
                None, None, None, None, None, None, None,  # up to c_lin
                None,  # c_lin
                None,  # c_ang
                None,  # skip_solving_threshold
            )

        device = qk.device
        dtype  = qk.dtype

        # ---- Rebuild residual at (z*, params) with graph disabled for z ----
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

        def residual(z, qk_, vk_, pusher_pos_, u_push_, h_, m_, Izz_, half_, mu_):
            # Same as in forward (keep params explicit)
            M_inv = torch.diag(torch.stack([1.0/m_, 1.0/m_, 1.0/Izz_]))
            pusher_next = pusher_pos_ + h_ * u_push_

            _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)

            cnt = obb_contact_blend2(_q, pusher_next, half_)  # differentiable
            n, t, r_cp = cnt.normal, cnt.tangent, cnt.r_cp
            Jn, Jt = contact_jacobians(n, t, r_cp)

            v_cp = _v[0:2] + _v[2] * perp(r_cp)
            v_rel_t = torch.dot(t, v_cp - u_push_)
            v_facets = torch.stack([v_rel_t, -v_rel_t])

            lamT_total = _beta[0] - _beta[1]
            impulse = Jn * _lamN + Jt * lamT_total

            Fg = torch.zeros(3, dtype=dtype, device=device)
            if C["enable_viscous_ground_friction"] and (C["c_lin"] > 0.0 or C["c_ang"] > 0.0):
                Fg = torch.stack([
                    torch.tensor(-C["c_lin"], dtype=dtype, device=device) * _v[0],
                    torch.tensor(-C["c_lin"], dtype=dtype, device=device) * _v[1],
                    torch.tensor(-C["c_ang"], dtype=dtype, device=device) * _v[2],
                ])

            dv = M_inv @ (impulse + h_ * Fg)

            r_dyn = _v - vk_ - dv
            r_kin = _q - qk_ - h_ * _v
            r_gap = _y - cnt.phi
            r_cone = _s - (mu_ * _lamN - torch.sum(_beta))
            r_slip = _w - (v_facets + _r)

            mu_star = torch.tensor(C["target_mu"], dtype=dtype, device=device)
            r_c1 = _y * _lamN - mu_star
            r_c2 = _r * _s - mu_star
            r_c3 = _beta * _w - mu_star

            return torch.cat([r_dyn, r_kin, r_gap.view(1), r_cone.view(1), r_slip,
                              r_c1.view(1), r_c2.view(1), r_c3])

        # Outputs g(z, params): (q_next, v_next, pusher_pos_next, lam_vec, phi)
        def outputs_from(z, qk_, vk_, pusher_pos_, u_push_, h_):
            q_next_ = z[0:3]
            v_next_ = z[3:6]
            lamN_   = z[6]
            beta_   = z[7:9]
            y_      = z[10]
            #print("grad_enabled:", torch.is_grad_enabled())
            lam_t_  = beta_[0] - beta_[1]
            lam_vec_ = torch.stack([lamN_, lam_t_])
            phi_    = y_
            pusher_next_ = pusher_pos_ + h_ * u_push_
            #ic(pusher_pos_.requires_grad,h_.requires_grad, u_push_.requires_grad)
            return q_next_, v_next_, pusher_next_, lam_vec_, phi_

        # Build J = dR/dz at solution
        
        z_star_req = z_star.detach().requires_grad_(True)
        with torch.enable_grad():
            R_star = residual(z_star_req, qk.detach(), vk.detach(), pusher_pos.detach(), u_push.detach(),
                          hT.detach(), mT.detach(), IzzT.detach(), halfT.detach(), muT.detach())
        J = torch.autograd.functional.jacobian(
            lambda zz: residual(zz, qk.detach(), vk.detach(), pusher_pos.detach(), u_push.detach(),
                                hT.detach(), mT.detach(), IzzT.detach(), halfT.detach(), muT.detach()),
            z_star_req, strict=False, create_graph=False).reshape(R_star.numel(), z_star_req.numel())
        reg = 1e-8 * torch.eye(J.shape[0], dtype=dtype, device=device)
        # g_z^T * lambda  (lambda = upstream adjoints on outputs except pusher which is handled separately)
        with torch.enable_grad():
            qn, vn, pn, lamv, ph = outputs_from(z_star_req, qk.detach(), vk.detach(),
                                            pusher_pos.detach(), u_push.detach(), hT.detach())
        g_z_T_lambda = torch.autograd.grad(
            outputs= (qn, vn, lamv, ph),
            inputs = z_star_req,
            grad_outputs=(grad_q_next, grad_v_next, grad_lam_vec, grad_phi),
            retain_graph=False, allow_unused=False
            )[0]  # (nz,)

        # Solve (J^T) w = g_z^T lambda
        # Small dense system -> direct solve
        w = torch.linalg.solve(J.T + reg, g_z_T_lambda)

        # g_theta^T * lambda (includes pusher_pos_next term for u_push & h)
        z_star_req2 = z_star.detach().requires_grad_(True)
        with torch.enable_grad():
            qn, vn, pn, lamv, ph = outputs_from(z_star_req2, qk.detach(), vk.detach(),
                                                pusher_pos, u_push, hT.detach())
        # Params we support grads for:
        params = [qk, vk, pusher_pos, u_push] #, hT, mT, IzzT, halfT, muT]

        # ic(qk.requires_grad, vk.requires_grad, pusher_pos.requires_grad, u_push.requires_grad) #, hT.requires_grad, mT.requires_grad, IzzT.requires_grad, halfT.requires_grad, muT.requires_grad)
        # ic(qn.requires_grad, vn.requires_grad, pn.requires_grad, lamv.requires_grad, ph.requires_grad)
        

        g_theta_T_lambda = torch.autograd.grad(
            outputs=(qn, vn, pn, lamv, ph),
            inputs=params,
            grad_outputs=(grad_q_next, grad_v_next, grad_pusher_pos_next, grad_lam_vec, grad_phi),
            retain_graph=False, allow_unused=True
        )

        z_star_req_no = z_star.detach().requires_grad_(False)  # z is fixed at the solution here
        with torch.enable_grad():
            R_star_w = residual(
                z_star_req_no,            # treat z* as constant when differentiating wrt params
                qk, vk, pusher_pos, u_push,   # <-- NOT detached
                hT, mT, IzzT, halfT, muT      # <-- NOT detached
            )

        # R_theta^T * w
        R_theta_T_w = torch.autograd.grad(
            outputs=R_star_w,
            inputs=params,
            grad_outputs=w,
            retain_graph=False, allow_unused=True
        )

        # Final param grads
        grads = []
        for gt, rt in zip(g_theta_T_lambda, R_theta_T_w):
            if gt is None and rt is None:
                grads.append(None)
            else:
                gt = torch.zeros_like(rt) if gt is None else gt
                rt = torch.zeros_like(gt) if rt is None else rt
                grads.append(gt - rt)

        # Return grads matching forward inputs:
        g_qk, g_vk, g_pusher, g_u = grads #, g_h, g_m, g_Izz, g_half, g_mu

        # The rest (non-tensor hyperparameters) have no gradients
        return (g_qk, g_vk, g_pusher, g_u,
                None, None, None, None, None, #g_h, g_m, g_Izz, g_half, g_mu,
                None, None, None, None, None, None, None, None, None, None)

def step_square_pos_ip(
    qk: torch.Tensor,
    vk: torch.Tensor,
    pusher_pos: torch.Tensor,
    u_push: torch.Tensor,
    h: torch.Tensor,           # pass as tensor for gradient through time-step if desired
    m: torch.Tensor,           # you can pass torch.tensor(m, dtype=..., device=...)
    Izz: torch.Tensor,
    half: torch.Tensor,
    mu: torch.Tensor,
    ipm_opts: IPMOptions,
    skip_solving_threshold: float,
    ):
    """
    Wrapper with the same outputs as your original step() but with
    implicit (IFT) gradients instead of backprop through iterations.
    """
    return StepSquarePosIPFn.apply(
        qk, vk, pusher_pos, u_push,
        h, m, Izz, half, mu,
        torch.tensor(ipm_opts.target_mu, device=qk.device, dtype=qk.dtype),
        torch.tensor(getattr(ipm_opts, "smooth_sdf", 0.0), device=qk.device, dtype=qk.dtype),
        torch.tensor(ipm_opts.tol, device=qk.device, dtype=qk.dtype),
        torch.tensor(ipm_opts.max_newton, device=qk.device, dtype=qk.dtype),
        torch.tensor(ipm_opts.frac_to_boundary, device=qk.device, dtype=qk.dtype),
        torch.tensor(ipm_opts.ls_beta, device=qk.device, dtype=qk.dtype),
        torch.tensor(getattr(ipm_opts, "enable_viscous_ground_friction", False), device=qk.device, dtype=torch.bool),
        torch.tensor(getattr(ipm_opts, "c_lin", 0.0), device=qk.device, dtype=qk.dtype),
        torch.tensor(getattr(ipm_opts, "c_ang", 0.0), device=qk.device, dtype=qk.dtype),
        torch.tensor(skip_solving_threshold, device=qk.device, dtype=qk.dtype),
    )


# def step_square_pos_ip(
#     qk: torch.Tensor,
#     vk: torch.Tensor,
#     pusher_pos: torch.Tensor,
#     u_push: torch.Tensor,
#     h: float,
#     m: float,
#     Izz: float,
#     half: float,
#     mu: float,
#     skip_solving_threshold: float,
#     ipm_opts: IPMOptions = IPMOptions(),
#     device=None,
#     ):
#     """
#     One-step implicit integration with *position-level* contact complementarity solved by interior point.

#     Unknowns: q_{k+1}, v_{k+1}, \lambda_N, beta (2 facets), r (slack for |t|-cone),
#               y (gap slack), s (cone slack), w (2 facet slips)

#     Complementarity pairs forced to prescribed duality gap mu*: 
#       y * lambda_N = mu*,   r * s = mu*,   beta_i * w_i = mu*.

#     Normal gap uses position-level constraint y = phi(q_{k+1}, pusher_{k+1}).
#     Tangential uses Anitescu linearized cone with two facets (\pm t).

#     Returns: q_{k+1}, v_{k+1}, pusher_pos_{k+1}, lam_vec([lambda_N, lambda_t_total]), phi
#     """
#     if device is None:
#         device = qk.device
#     dtype = qk.dtype

#     hT = torch.tensor(h, dtype=dtype, device=device)
#     muT = torch.tensor(mu, dtype=dtype, device=device)

#     # Mass inverse
#     M_inv = torch.diag(torch.tensor([1.0 / m, 1.0 / m, 1.0 / Izz], dtype=dtype, device=device))

#     # Kinematic pusher (known)
#     pusher_pos_next = pusher_pos + hT * u_push

#     # Quick free prediction to skip IP solve when well separated
#     q_free = torch.stack([qk[0] + hT * vk[0], qk[1] + hT * vk[1], qk[2] + hT * vk[2]])
#     # cnt_free = obb_contact(q_free, pusher_pos_next, half, smooth=ipm_opts.smooth_sdf)
#     cnt_free = obb_contact_blend2(q_free, pusher_pos_next, half)
#     if cnt_free.phi > skip_solving_threshold:
#         # no contact; plain semi-implicit Euler with mild damping
#         v_next = vk - 0.3 * vk * hT
#         q_next = torch.stack([qk[0] + hT * v_next[0], qk[1] + hT * v_next[1], qk[2] + hT * v_next[2]])
#         lam = torch.zeros(2, dtype=dtype, device=device)
#         return q_next, v_next, pusher_pos_next, lam, cnt_free.phi

#     # Unknowns initialization
#     q = q_free.clone().requires_grad_(True)
#     v = vk.clone().requires_grad_(True)

#     lamN = torch.tensor(1e-3, dtype=dtype, device=device, requires_grad=True)
#     beta = torch.full((2,), 1e-3, dtype=dtype, device=device, requires_grad=True)  # two facets (+t,-t)
#     r = torch.tensor(1e-3, dtype=dtype, device=device, requires_grad=True)

#     y = torch.tensor(max(ipm_opts.target_mu, 1e-4), dtype=dtype, device=device, requires_grad=True)
#     s = torch.tensor(max(ipm_opts.target_mu, 1e-4), dtype=dtype, device=device, requires_grad=True)
#     w = torch.full((2,), max(ipm_opts.target_mu, 1e-4), dtype=dtype, device=device, requires_grad=True)

#     def pack(_q, _v, _lamN, _beta, _r, _y, _s, _w):
#         return torch.cat([_q, _v, _lamN.view(1), _beta, _r.view(1), _y.view(1), _s.view(1), _w])

#     def unpack(z):
#         _q = z[0:3]
#         _v = z[3:6]
#         _lamN = z[6]
#         _beta = z[7:9]
#         _r = z[9]
#         _y = z[10]
#         _s = z[11]
#         _w = z[12:14]
#         return _q, _v, _lamN, _beta, _r, _y, _s, _w

#     def residual(z: torch.Tensor) -> torch.Tensor:
#         _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)

#         # cnt = obb_contact(_q, pusher_pos_next, half, smooth=ipm_opts.smooth_sdf)
#         cnt = obb_contact_blend2(_q, pusher_pos_next, half)
#         n, t, r_cp = cnt.normal, cnt.tangent, cnt.r_cp
#         Jn, Jt = contact_jacobians(n, t, r_cp)

#         # Tangential relative velocity at cp (scalar along t)
#         v_cp = _v[0:2] + _v[2] * perp(r_cp)
#         v_rel_t = torch.dot(t, v_cp - u_push)

#         # Two facet slip velocities: [ +v_t, -v_t ]
#         v_facets = torch.stack([v_rel_t, -v_rel_t])

#         # Dynamics (implicit Euler on velocities) and kinematics
#         lamT_total = _beta[0] - _beta[1]
#         impulse = Jn * _lamN + Jt * lamT_total

#         # Option A: viscous ground friction as an external force ~ -C * v
#         Fg = torch.zeros(3, dtype=dtype, device=device)
#         if ipm_opts.enable_viscous_ground_friction and (ipm_opts.c_lin > 0.0 or ipm_opts.c_ang > 0.0):
#             c_linT = torch.tensor(ipm_opts.c_lin, dtype=dtype, device=device)
#             c_angT = torch.tensor(ipm_opts.c_ang, dtype=dtype, device=device)
#             Fg = torch.stack([
#                 -c_linT * _v[0],  # Fx
#                 -c_linT * _v[1],  # Fy
#                 -c_angT * _v[2],  # Tau
#             ])

#         # Forces must be multiplied by h to get an impulse; contact impulses already are
#         dv = M_inv @ (impulse + hT * Fg)
#         r_dyn = _v - vk - dv                     # (3,)
#         r_kin = _q - qk - hT * _v                # (3,)

#         # Equalities tying slacks to physical quantities
#         r_gap = _y - cnt.phi                      # (1,)
#         r_cone = _s - (muT * _lamN - torch.sum(_beta))  # (1,)
#         r_slip = _w - (v_facets + _r)            # (2,)

#         mu_star = torch.tensor(ipm_opts.target_mu, dtype=dtype, device=device)
#         # Central-path complementarity (softened)
#         r_c1 = _y * _lamN - mu_star              # (1,)
#         r_c2 = _r * _s - mu_star                 # (1,)
#         r_c3 = _beta * _w - mu_star              # (2,)

#         return torch.cat([r_dyn, r_kin, r_gap.view(1), r_cone.view(1), r_slip, r_c1.view(1), r_c2.view(1), r_c3])

#     # Newton iterations on R(z)=0
#     z = pack(q, v, lamN, beta, r, y, s, w)

#     for it in range(ipm_opts.max_newton):
#         z = z.clone().detach().requires_grad_(True)
#         R = residual(z) #.requires_grad_(True)
#         res_norm = float(torch.linalg.norm(R).item())
#         if res_norm < ipm_opts.tol:
#             break
#         # Dense Jacobian via autograd
#         # J = []
#         # for i in range(R.numel()):
#         #     ic(R[i].requires_grad, z.requires_grad)
#         #     (grad_i,) = torch.autograd.grad(R[i], z, retain_graph=True, create_graph=False, allow_unused=False)
#         #     J.append(grad_i.view(1, -1))
#         # J = torch.cat(J, dim=0)  # (N,N)

#         J = torch.autograd.functional.jacobian(residual, z, strict=False, create_graph=False)
#         J = J.reshape(R.numel(), z.numel())

#         # Levenberg-style regularization for robustness
#         reg = 1e-8 * torch.eye(J.shape[0], dtype=dtype, device=device)
#         try:
#             dz = torch.linalg.solve(J + reg, -R)
#         except RuntimeError:
#             # fallback to least-squares if singular
#             dz, *_ = torch.linalg.lstsq(J + reg, -R)

#         # Fraction-to-boundary for positivity variables
#         _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)
#         alpha_pos = 1.0
#         pos_vars = [(_lamN.view(1), dz[6].view(1)), (_beta, dz[7:9]), (_r.view(1), dz[9].view(1)),
#                     (_y.view(1), dz[10].view(1)), (_s.view(1), dz[11].view(1)), (_w, dz[12:14])]
#         for x, dx in pos_vars:
#             alpha_pos = min(alpha_pos, fraction_to_boundary_step(x, dx, ipm_opts.frac_to_boundary))
#         alpha = alpha_pos

#         # Backtracking to reduce residual
#         newton_decrease = False
#         for _ in range(15):
#             z_trial = z + alpha * dz
#             # enforce tiny floors to stay >0
#             z_trial[6] = torch.clamp(z_trial[6], min=1e-12)         # lamN
#             z_trial[7:9] = torch.clamp(z_trial[7:9], min=1e-12)     # beta
#             z_trial[9] = torch.clamp(z_trial[9], min=1e-12)         # r
#             z_trial[10] = torch.clamp(z_trial[10], min=1e-12)       # y
#             z_trial[11] = torch.clamp(z_trial[11], min=1e-12)       # s
#             z_trial[12:14] = torch.clamp(z_trial[12:14], min=1e-12) # w
#             Rt = residual(z_trial)
#             if torch.linalg.norm(Rt) <= (1.0 - 1e-4 * alpha) * torch.linalg.norm(R):
#                 z = z_trial.detach()
#                 newton_decrease = True
#                 break
#             alpha *= ipm_opts.ls_beta
#         if not newton_decrease:
#             # take the (clipped) fraction-to-boundary step even if residual didn't shrink enough
#             z = z_trial.detach()

#     # Unpack and assemble outputs
#     q_next, v_next, lamN, beta, r, y, s, w = unpack(z)
#     lam_t = beta[0] - beta[1]
#     lam_vec = torch.stack([lamN, lam_t])
#     phi = y  # y equals the gap at solution (softened)

#     return q_next, v_next, pusher_pos_next, lam_vec, phi

def implicit_euler_defects(qk, vk, prk, uk,
                           qk1, vk1, prk1,
                           h, m, Izz, half, mu,
                           dynamics_solver, qp_solver):
    if dynamics_solver == 'LCP':
         q_next, v_next, pr_next, lam, phi = step_square(
            qk, vk, prk, uk, h, m, Izz, half, mu, 
            qp_solver=qp_solver, alpha_stab=0.1, device=device
        )
    elif dynamics_solver == 'IP':
        q_next, v_next, pr_next, lam, phi = step_square_pos_ip(
        qk, vk, prk, uk, h=h, m=m, Izz=Izz, half=half, mu=mu,
        skip_solving_threshold = 0.3,
            ipm_opts=IPMOptions(target_mu=1e-4, max_newton=20, tol=1e-4, smooth_sdf=50.0,
            enable_viscous_ground_friction=True,
            c_lin=8.0,          
            c_ang=8.0 * half     
            ))

    r_v = vk1 - v_next
    r_q = qk1 - q_next
    r_p = prk1 - pr_next
    r = torch.cat([r_q, r_v, r_p], dim=0)
    return r, lam, phi

def rollout(u_seq, q0, v0, pr0, horizon, h, m, Izz, half, mu, goal_xy, 
            w_target = 20.0, w_v = 0.1, w_ctrl = 1e-3, w_obs = 1.0,
            qp_solver = None, dynamics_solver=None, obstacle_pos=None, device=None):
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
    
    if qp_solver is None and dynamics_solver == 'LCP':
        qp_solver = ContactQPSolver(mu=mu, n_contacts=1)
    
    # if obstacle_pos is None:
    #     obstacle_pos = torch.tensor([0.2, -0.2], device=device)
    
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
        if dynamics_solver == 'LCP':
            q, v, pr, lamk, phik = step_square(
                q, v, pr, u_seq[k], h, m, Izz, half, mu, 
                qp_solver=qp_solver, alpha_stab=0.1, device=device
            )
        elif dynamics_solver == 'IP':
            q, v, pr, lamk, phik = step_square_pos_ip(
                q, v, pr, u_seq[k], h=h, m=m, Izz=Izz, half=half, mu=mu,
                skip_solving_threshold = 0.3,
                ipm_opts=IPMOptions(target_mu=1e-4, max_newton=20, tol=1e-5, smooth_sdf=50.0, #smooth_sdf is unused
                    enable_viscous_ground_friction=True,
                    c_lin=8.0,          
                    c_ang=8.0 * half     
                    ))

        lambdas.append(lamk)
        phis.append(phik)
        qs.append(q)
        qrobot_hist.append(pr)
        
        # Obstacle avoidance term
        if obstacle_pos is not None:
            obs_term += w_obs / (torch.sum((pr - obstacle_pos) ** 2) + 0.01)
    
    obs_term /= horizon
    
    # Cost function
    goal_term = w_target * torch.sum((q - goal_xy) ** 2)      # Goal reaching
    ctrl_term = w_ctrl * torch.sum(u_seq ** 2)               # Control effort
    v_term = w_v * torch.sum(v ** 2)                       # Terminal velocity
    pen_term = 0.0  # Penetration penalty (disabled)
    
    loss = goal_term + ctrl_term + pen_term + v_term + obs_term
    
    return loss, q, torch.stack(lambdas), torch.stack(phis), torch.stack(qs), torch.stack(qrobot_hist),\
        goal_term, ctrl_term, v_term, obs_term, pen_term

