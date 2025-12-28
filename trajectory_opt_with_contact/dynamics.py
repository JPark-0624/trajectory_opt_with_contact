"""
Dynamics simulation with contact-implicit integration.

Implements time-stepping for rigid body dynamics with frictional contact
using differentiable QP solvers.
"""

import string
import torch
from .geometry import contact_frame_and_J, contact_jacobians, perp, _linearize_obb_gap_at
from .qp_solver import ContactQPSolver
from .analytical_jacobian import AnalyticalJacobian, HybridJacobian
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

def safeSolve(A, b, reg=1e-8):
    """Try to solve (A+regI)x=b and fallback to lstsq on failure."""
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    Ar = A + reg * I
    try:
        return torch.linalg.solve(Ar, b)
    except RuntimeError:
        x, *_ = torch.linalg.lstsq(Ar, b)
        return x

def buildAutogradJacobian(residualFn, z, vectorize=True):
    """Convert R(z) autograd jacobian to (m,n) Matrix."""
    zReq = z.detach().requires_grad_(True)
    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(
            residualFn,
            zReq,
            strict=False,
            create_graph=False,
            vectorize=vectorize,
        )
    R = residualFn(z)  # shape 확인용
    return J.reshape(R.numel(), zReq.numel()).detach()

def fracToBoundaryAlpha(x, dx, tau):
    """
    return maximum alpha who satisfies x + alpha*dx >= (1-tau)*x
    """
    with torch.no_grad():
        mask = dx < 0
        if torch.any(mask):
            # (1-tau)x - x = -tau*x
            al = (-tau * x[mask]) / dx[mask]
            a = torch.min(al).item()
            return max(min(0.99 * a, 1.0), 1e-6)
        return 1.0

def computePositivityAlpha(z, dz, tau):
    """alpha_pos calculation for positivity of lamN, beta, r, y, s, w."""

    with torch.no_grad():
        alphaPos = 1.0
        lamN = z[6].view(1);     dLamN = dz[6].view(1)
        beta = z[7:9];           dBeta = dz[7:9]
        r = z[9].view(1);        dR = dz[9].view(1)
        y = z[10].view(1);       dY = dz[10].view(1)
        s = z[11].view(1);       dS = dz[11].view(1)
        w = z[12:14];            dW = dz[12:14]

        for x, dx in [(lamN, dLamN), (beta, dBeta), (r, dR), (y, dY), (s, dS), (w, dW)]:
            alphaPos = min(alphaPos, fracToBoundaryAlpha(x, dx, tau))
        return float(alphaPos)

def clampPositive(z):
    """solver clamp."""
    z = z.clone()
    z[6] = torch.clamp(z[6], min=1e-12)
    z[7:9] = torch.clamp(z[7:9], min=1e-12)
    z[9] = torch.clamp(z[9], min=1e-12)
    z[10] = torch.clamp(z[10], min=1e-12)
    z[11] = torch.clamp(z[11], min=1e-12)
    z[12:14] = torch.clamp(z[12:14], min=1e-12)
    return z

def backtrackingLineSearch(residualFn, z, dz, alphaInit, lsBeta, maxSteps=15, c1=1e-4):
    """Extracted backtracking logic from forward."""
    with torch.no_grad():
        R0 = residualFn(z)
        n0 = torch.linalg.norm(R0)
        alpha = float(alphaInit)
        zTrialLast = None

        for _ in range(maxSteps):
            zTrial = clampPositive(z + alpha * dz)
            zTrialLast = zTrial
            Rt = residualFn(zTrial)
            if torch.linalg.norm(Rt) <= (1.0 - c1 * alpha) * n0:
                return zTrial.detach(), alpha, True
            alpha *= float(lsBeta)

        return zTrialLast.detach(), alpha, False

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

def asTensor(x, *, dtype, device, requires_grad=False):
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        if requires_grad:
            t = t.detach().clone().requires_grad_(True)
        return t
    return torch.tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)


class StepSquarePosIPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                qk, vk, pusher_pos, u_push,
                h, m, Izz, half, mu,
                target_mu, smooth_sdf, tol, max_newton,
                frac_to_boundary, ls_beta,
                enable_viscous_ground_friction, c_lin, c_ang,
                skip_solving_threshold,
                z_prev,
                jacobian_type,
                debugOut=None,
                modeAConfig=None):  
        """
        All differentiable arguments must be tensors.
        Scalars like m, Izz, half, mu can be tensors (dtype/shape-compatible).
        You can choose which ones require grad.
        
        Args:
            jacobian_type: "analytical" for analytical Jacobian computation,
                        "hybrid" for hybrid Jacobian computation (analytical + autograd),
                        "autograd" for PyTorch autograd (default).
        """
        device = qk.device
        dtype = qk.dtype

        # ----------------
        # Debug flags (Mode A)
        # ----------------
       
        modeAEnabled = bool(modeAConfig and modeAConfig.get("enabled", False))
        collectJacobians = bool(modeAConfig and modeAConfig.get("collect", "per_iter") == "per_iter")
        steplog = None
        if debugOut is not None and modeAEnabled:
            steplog = {
                "newton": [],
                "jacobians": [] if collectJacobians else None,
                "residuals": [] if collectJacobians else None, 
                "z_stars": [] if collectJacobians else None,
            }
            print(f"\n{'='*60}")
            print(f"Timestep {len(debugOut)}")
            print(f"{'='*60}")
            
            if z_prev is not None:
                print(f"✓ True warm start:")
                print(f"  lambda_prev = {z_prev[6]:.6f}")
                print(f"  beta_prev = [{z_prev[7]:.6f}, {z_prev[8]:.6f}]")
            else:
                print(f"✗ Cold start (z_prev is None)")
                print(f"  lambda_init = 1e-3")


        # patch : Juneil Park
        # Convert scalar inputs to tensors at once
        # to avoid multiple tensor declarations.
        # all constants do not require grad.
        hT   = asTensor(h, dtype=dtype, device=device)
        muT  = asTensor(mu, dtype=dtype, device=device)
        mT   = asTensor(m, dtype=dtype, device=device)
        IzzT = asTensor(Izz, dtype=dtype, device=device)
        halfT= asTensor(half, dtype=dtype, device=device)

        target_muT = asTensor(target_mu, dtype=dtype, device=device)
        # smooth_sdf, tol, max_newton, frac_to_boundary, ls_beta, enable_viscous_ground_friction does not need to be tensor
        c_linT = asTensor(c_lin, dtype=dtype, device=device)
        c_angT = asTensor(c_ang, dtype=dtype, device=device)

        # Initialize analytical Jacobian computer if requested
        
        analytical_jacobian = None
        hybrid_jacobian = None

        if jacobian_type == "analytical":
            analytical_jacobian = AnalyticalJacobian(
                mass = m,
                Izz = Izz,
                half_length = half,
                mu=mu,
                h=h,
                device=device,
                dtype=dtype,
                enable_viscous_ground_friction=enable_viscous_ground_friction,
                c_lin=c_lin,
                c_ang=c_ang
            )
        elif jacobian_type == "hybrid":
            hybrid_jacobian = HybridJacobian(
                mass = m,
                Izz = Izz,
                half_length = half,
                mu=mu,
                h=h,
                device=device,
                dtype=dtype,
                enable_viscous_ground_friction=enable_viscous_ground_friction,
                c_lin=c_lin,
                c_ang=c_ang
            )
        elif jacobian_type == "autograd":
            pass

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
                None, None, None,  # placeholders when no solve
                qk, vk, pusher_pos, u_push, hT, mT, IzzT, halfT, muT,
            )
            # store constants - all python scalars
            ctx.constants = dict(
                target_mu=float(target_mu), smooth_sdf=float(smooth_sdf), tol=float(tol),
                max_newton=int(max_newton),
                frac_to_boundary=float(frac_to_boundary), ls_beta=float(ls_beta),
                enable_viscous_ground_friction=bool(enable_viscous_ground_friction),
                c_lin=float(c_lin), c_ang=float(c_ang),
                skip_solving_threshold=float(skip_solving_threshold),
                solved=False,
            )
            # print("Skipping implicit solve (no contact).")
            debugOut.append(steplog) if debugOut is not None and modeAEnabled else None
            return q_next, v_next, pusher_pos_next, lam_vec, phi, z_prev

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
            Jn, Jt = contact_jacobians(n, t, r_cp)      # user-provided

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
                    -c_linT* _v[0],
                    -c_linT* _v[1],
                    -c_angT* _v[2],
                ])

            dv = M_inv @ (impulse + hT * Fg)

            r_dyn = _v - vk - dv                              # (3,)
            r_kin = _q - qk - hT * _v                         # (3,)
            r_gap = _y - cnt.phi                              # (1,)
            r_cone = _s - (muT * _lamN - torch.sum(_beta))    # (1,)
            r_slip = _w - (v_facets + _r)                     # (2,)

            r_c1 = _y * _lamN - target_muT
            r_c2 = _r * _s - target_muT
            r_c3 = _beta * _w - target_muT                       # (2,)

            return torch.cat([r_dyn, r_kin, r_gap.view(1), r_cone.view(1), r_slip,
                              r_c1.view(1), r_c2.view(1), r_c3])

        def compute_geom_grads(q, v, lamN, beta, pusher_pos, u_push, halfT):
            """
                Computes all geometry terms + their derivatives,
                consistent with dynamics.py residual.
                Uses autograd for all geometry derivatives.
            """

            # --- 1. Compute geometry (same as forward residual) ---
            cnt = obb_contact_blend2(q, pusher_pos, halfT)
            n  = cnt.normal
            t  = cnt.tangent
            r_cp = cnt.r_cp

            # contact jacobians
            Jn, Jt = contact_jacobians(n, t, r_cp)

            # --- 2. phi(q) gradient ---
            phi = cnt.phi
            dphi_dq = torch.autograd.grad(phi, q, retain_graph=True)[0]

            # --- 3. tangent relative velocity ---

            perp_rcp = torch.stack([-r_cp[1], r_cp[0]])
            v_cp = v[0:2] + v[2] * perp_rcp
            v_rel = v_cp - u_push  # or v_push depending on your code
            v_rel_t = torch.dot(t, v_rel)

            # derivatives wrt q and v
            dvfac_dq = torch.autograd.grad(v_rel_t, q, retain_graph=True)[0]
            dvfac_dv = torch.autograd.grad(v_rel_t, v, retain_graph=True)[0]

            # --- 4. contact impulse derivative wrt q ---
                    
            # force(q) = Jn(q)*lamN + Jt(q)*(beta0 - beta1)

            lam = lamN
            slip = beta[0] - beta[1]

            def force_fn(q_local: torch.Tensor) -> torch.Tensor:
                cnt_loc = obb_contact_blend2(q_local, pusher_pos, halfT)
                n_loc, t_loc, r_cp_loc = cnt_loc.normal, cnt_loc.tangent, cnt_loc.r_cp
                Jn_loc, Jt_loc = contact_jacobians(n_loc, t_loc, r_cp_loc)
                force_loc = Jn_loc * lam + Jt_loc * slip   # (3,)
                return force_loc

            # dforce_dq: (3,3), row i: ∂force_i/∂q_j
            dforce_dq = torch.autograd.functional.jacobian(
                force_fn,
                q,
                vectorize=True,
                create_graph=False
            )                           # (3,3)

            geom_grad = {
                'cnt': cnt,
                'Jn': Jn,
                'Jt': Jt,
                'dphi_dq': dphi_dq,
                'dvfac_dq': torch.stack([dvfac_dq, -dvfac_dq]),  # ← 부호 반대!
                'dvfac_dv': torch.stack([dvfac_dv, -dvfac_dv]),  # ← 부호 반대!
                'dforce_dq': dforce_dq   # hybrid J will combine them with λ, β
            }
            return cnt, geom_grad

        # Initialize z
        q = q_free.clone().detach().requires_grad_(True)
        v = vk.clone().detach().requires_grad_(True)

        if z_prev is not None:
            lamN = z_prev[6].clone().detach().requires_grad_(True)
            beta = z_prev[7:9].clone().detach().requires_grad_(True)
            r = z_prev[9].clone().detach().requires_grad_(True)
            y = z_prev[10].clone().detach().requires_grad_(True)
            s = z_prev[11].clone().detach().requires_grad_(True)
            w = z_prev[12:14].clone().detach().requires_grad_(True)
        else:
            lamN = torch.tensor(1e-3, dtype=dtype, device=device, requires_grad=True)
            beta = torch.full((2,), 1e-3, dtype=dtype, device=device, requires_grad=True)
            r = torch.tensor(1e-3, dtype=dtype, device=device, requires_grad=True)
            y = torch.tensor(max(target_mu, 1e-4), dtype=dtype, device=device, requires_grad=True)
            s = torch.tensor(max(target_mu, 1e-4), dtype=dtype, device=device, requires_grad=True)
            w = torch.full((2,), max(target_mu, 1e-4), dtype=dtype, device=device, requires_grad=True)

        z = torch.cat([q, v, lamN.view(1), beta, r.view(1), y.view(1), s.view(1), w])

        J_last = None
        newton_iters = 0
        with torch.no_grad():
            # Newton on R(z)=0 (same as your current code, compact)
            for _ in range(int(max_newton)):
                newton_iters += 1
                R = residual(z)  # pure numeric, no graph
                Rn = float(torch.linalg.norm(R))
                ### <- expected position for compute_geom_grads
                if Rn < float(tol):
                    break
                
                # Build Jacobian: analytical or autograd
                if jacobian_type == "analytical":
                    # Use analytical Jacobian computation
                    _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)
                    J = analytical_jacobian.compute_jacobian(z, R,
                        pusher_pos=pusher_pos_next
                    )
                    J_last = J.detach()
                    
                elif jacobian_type == "hybrid":
                    # Use hybrid Jacobian computation
                    z_req = z.detach().requires_grad_(True)
                    _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z_req)
                    with torch.enable_grad():
                        q_req = _q.detach().clone().requires_grad_(True)
                        v_req = _v.detach().clone().requires_grad_(True)
                        lamN_req = _lamN.detach().clone().requires_grad_(False)
                        beta_req = _beta.detach().clone().requires_grad_(False)

                        cnt, geom_grads = compute_geom_grads(
                            q_req, v_req, lamN_req, beta_req, pusher_pos_next, u_push, halfT
                        )

                    # Now compute hybrid Jacobian using this geometry info
                    J = hybrid_jacobian.compute_jacobian(
                        z_req,
                        qk=qk,
                        vk=vk,
                        u=u_push,
                        cnt=cnt,
                        geom_grads=geom_grads,
                    )
                    J_last = J.detach()
                else:
                    # Use PyTorch autograd (original method)
                    z_req = z.detach().requires_grad_(True)
                    with torch.enable_grad():
                        J = torch.autograd.functional.jacobian(
                            residual,
                            z_req,
                            strict=False,
                            create_graph=False,
                            vectorize=True,  # <-- important
                        )
                    J = J.reshape(R.numel(), z_req.numel())
                    J_last = J.detach()

                if steplog is not None and collectJacobians:
                    steplog["jacobians"].append(J_last.clone().cpu())
                    steplog["residuals"].append(R.detach().clone().cpu())
                    steplog["z_stars"].append(z.detach().clone().cpu())

                dzFresh = safeSolve(J_last, -R)


                tau = float(frac_to_boundary)
                alpha_pos = computePositivityAlpha(z, dzFresh, tau)

                # backtracking
                zNew, alpha, ok = backtrackingLineSearch(
                    residualFn=residual,
                    z=z,
                    dz=dzFresh,
                    alphaInit=alpha_pos,
                    lsBeta=ls_beta,
                    maxSteps=15,
                    c1=1e-4
                )
                z = zNew.detach()

        if J_last is None:
            # Compute final Jacobian: analytical or autograd
            _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)
            if jacobian_type == "analytical":
                
                J = analytical_jacobian.compute_jacobian(z, R,
                        pusher_pos=pusher_pos_next
                    )
                J_last = J.detach()
            elif jacobian_type == "hybrid":
                with torch.enable_grad():
                        q_req = _q.detach().clone().requires_grad_(True)
                        v_req = _v.detach().clone().requires_grad_(True)
                        lamN_req = _lamN.detach().clone().requires_grad_(False)
                        beta_req = _beta.detach().clone().requires_grad_(False)

                        cnt, geom_grads = compute_geom_grads(
                            q_req, v_req, lamN_req, beta_req, pusher_pos_next, u_push, halfT
                        )

                # Now compute hybrid Jacobian using this geometry info
                J = hybrid_jacobian.compute_jacobian(
                    z,
                    qk=qk,
                    vk=vk,
                    u=u_push,
                    cnt=cnt,
                    geom_grads=geom_grads,
                )
                J_last = J.detach()
            else:
                # Use PyTorch autograd (original method)
                z_req = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    R = residual(z_req)
                    J = torch.autograd.functional.jacobian(
                        residual,
                        z_req,
                        strict=False,
                        create_graph=False,
                        vectorize=True,
                    )
                J_last = J.reshape(R.numel(), z_req.numel()).detach()

        # at end of forward, after you have z and J_last
        q_next = z[0:3]
        v_next = z[3:6]
        lamN   = z[6]
        beta   = z[7:9]
        y      = z[10]
        lam_t  = beta[0] - beta[1]
        lam_vec = torch.stack([lamN, lam_t])
        phi    = y

        # store constants - all python scalars
        ctx.constants = dict(
            target_mu=float(target_mu), smooth_sdf=float(smooth_sdf), tol=float(tol),
            max_newton=int(max_newton),
            frac_to_boundary=float(frac_to_boundary), ls_beta=float(ls_beta),
            enable_viscous_ground_friction=bool(enable_viscous_ground_friction),
            c_lin=float(c_lin), c_ang=float(c_ang),
            skip_solving_threshold=float(skip_solving_threshold),
            solved=True,
        )

        ctx.z_star = z.detach()
        ctx.J = J_last  # (14, 14) dense tensor
        ctx.save_for_backward(
            pusher_pos_next.detach(),
            q_next.detach(), v_next.detach(),
            qk, vk, pusher_pos, u_push, hT, mT, IzzT, halfT, muT
        )
        
        # print(f"  IPM converged in {newton_iters} iterations.")
        debugOut.append(steplog) if debugOut is not None and modeAEnabled else None

        return q_next, v_next, pusher_pos_next, lam_vec, phi, z.detach()


    @staticmethod
    def backward(ctx, grad_q_next, grad_v_next, grad_pusher_pos_next, grad_lam_vec, grad_phi, grad_z_star=None):
        # Retrieve
        (pusher_pos_next, q_next, v_next,
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

        # declare tensor variables outside of residual to prevent multiple computation
        target_muT = asTensor(C["target_mu"], dtype=dtype, device=device)
        M_inv = torch.diag(torch.stack([1.0/mT, 1.0/mT, 1.0/IzzT])) 
        c_linT = asTensor(C["c_lin"], dtype=dtype, device=device)
        c_angT = asTensor(C["c_ang"], dtype=dtype, device=device)

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
                    -c_linT * _v[0],
                    -c_linT * _v[1],
                    -c_angT * _v[2],
                ])

            dv = M_inv @ (impulse + h_ * Fg)

            r_dyn = _v - vk_ - dv
            r_kin = _q - qk_ - h_ * _v
            r_gap = _y - cnt.phi
            r_cone = _s - (mu_ * _lamN - torch.sum(_beta))
            r_slip = _w - (v_facets + _r)


            r_c1 = _y * _lamN - target_muT
            r_c2 = _r * _s - target_muT
            r_c3 = _beta * _w - target_muT

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
        z_star = ctx.z_star
        J = ctx.J  # cached from forward, shape (nz, nz)
        reg = 1e-8 * torch.eye(J.shape[0], dtype=dtype, device=device)

        # we still need a grad-enabled copy of z_star for outputs_from:
        z_star_req = z_star.detach().requires_grad_(True)
        with torch.enable_grad():
            qn, vn, pn, lamv, ph = outputs_from(
                z_star_req, qk.detach(), vk.detach(),
                pusher_pos.detach(), u_push.detach(), hT.detach()
            )
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
                None, None, None, None, None, None, None, None, None, None, None, None) 

def step_square_pos_ip(
    qk: torch.Tensor,
    vk: torch.Tensor,
    pusher_pos: torch.Tensor,
    u_push: torch.Tensor,
    h: float,           # non tensor datatype. make tensor in forward at once.
    m: float,           # non tensor datatype. make tensor in forward at once.
    Izz: float,        # non tensor datatype. make tensor in forward at once.
    half: float,       # non tensor datatype. make tensor in forward at once.
    mu: float,        # non tensor datatype. make tensor in forward at once.
    ipm_opts: IPMOptions,
    skip_solving_threshold: float,
    z_prev = None,
    jacobian_type: string = "autograd",  # NEW PARAMETER
    debugOut = None,
    modeAConfig=None
    ):
    """
    Wrapper with the same outputs as your original step() but with
    implicit (IFT) gradients instead of backprop through iterations.
    
    Args:
        jacobian_type: 
            "analytical" : Use analytical Jacobian computation.
            "hybrid" : Use hybrid Jacobian computation (analytical + autograd).
            "autograd" : Use PyTorch autograd (default).
    """
    return StepSquarePosIPFn.apply(
        qk, vk, pusher_pos, u_push,
        h, m, Izz, half, mu,
        ipm_opts.target_mu,
        getattr(ipm_opts, "smooth_sdf", 0.0),
        ipm_opts.tol,
        ipm_opts.max_newton,
        ipm_opts.frac_to_boundary,
        ipm_opts.ls_beta,
        getattr(ipm_opts, "enable_viscous_ground_friction", False),
        getattr(ipm_opts, "c_lin", 0.0),
        getattr(ipm_opts, "c_ang", 0.0),
        skip_solving_threshold,
        z_prev,
        jacobian_type,
        debugOut,
        modeAConfig
    )



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
                ipm_opts=IPMOptions(target_mu=1e-4, max_newton=20, tol=1e-3, smooth_sdf=50.0, #smooth_sdf is unused
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
    ctrl_term = w_ctrl * torch.sum((u_seq*h) ** 2)               # Control effort
    v_term = w_v * torch.sum(v ** 2)                       # Terminal velocity
    pen_term = 0.0  # Penetration penalty (disabled)
    
    loss = goal_term + ctrl_term + pen_term + v_term + obs_term
    
    return loss, q, torch.stack(lambdas), torch.stack(phis), torch.stack(qs), torch.stack(qrobot_hist),\
        goal_term, ctrl_term, v_term, obs_term, pen_term