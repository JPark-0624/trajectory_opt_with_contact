from trajectory_opt_with_contact import IPMOptions, step_square_pos_ip
import torch

def check_grad_wrt_u(device="cpu", dtype=torch.float64, seed=0, contact=True, verbose=True):
    """
    Finite-difference check for ∂L/∂u_push only.
    Set contact=True to place the pusher near the box (solver path has contact influence).
    Set contact=False to test no-contact dynamics (still solver path; u only affects pusher_next).
    """
    torch.manual_seed(seed)
    torch.set_default_dtype(dtype)

    # Problem parameters (tensors, as your wrapper expects)
    half_val = 0.5
    qk   = torch.tensor([0.00, 0.00, 0.0], device=device, dtype=dtype, requires_grad=True)
    vk   = torch.tensor([0.00, 0.00, 0.0], device=device, dtype=dtype, requires_grad=True)

    if contact:
        # Slightly outside the right face, moving inward → contact influences residual
        pusher_pos = torch.tensor([half_val + 0.02, 0.00], device=device, dtype=dtype, requires_grad=True)
        u_push     = torch.tensor([-0.30, 0.00],      device=device, dtype=dtype, requires_grad=True)
    else:
        # Far away → no actual penetration, but we still force solver path
        pusher_pos = torch.tensor([half_val + 0.1, 0.0], device=device, dtype=dtype, requires_grad=True)
        u_push     = torch.tensor([0.10, -0.15],     device=device, dtype=dtype, requires_grad=True)

    h    = torch.tensor(0.02, device=device, dtype=dtype, requires_grad=False)
    m    = torch.tensor(2.00, device=device, dtype=dtype, requires_grad=False)
    Izz  = torch.tensor(0.20, device=device, dtype=dtype, requires_grad=False)
    half = torch.tensor(half_val, device=device, dtype=dtype, requires_grad=False)
    mu   = torch.tensor(0.60, device=device, dtype=dtype, requires_grad=False)

    ipm_opts = IPMOptions(
        target_mu=1e-4, smooth_sdf=0.0,
        tol=1e-10, max_newton=80,
        frac_to_boundary=0.1, ls_beta=0.5,
        enable_viscous_ground_friction=False, c_lin=0.0, c_ang=0.0
    )

    # Force the solver path: skip happens when phi > threshold, so set threshold VERY NEGATIVE
    skip_solving_threshold = torch.tensor(10, device=device, dtype=dtype)

    # Random linear weights to build a scalar loss (ensures gradient signal flows to u)
    gq   = torch.randn(3,  device=device, dtype=dtype)
    gv   = torch.randn(3,  device=device, dtype=dtype)
    gp   = torch.randn(2,  device=device, dtype=dtype)
    glam = torch.randn(2,  device=device, dtype=dtype)
    gphi = torch.randn((), device=device, dtype=dtype)

    def loss_from_outputs(q_next, v_next, pusher_next, lam_vec, phi):
        return (q_next * gq).sum() + (v_next * gv).sum() + (pusher_next * gp).sum() + (lam_vec * glam).sum() + phi * gphi

    # ---- Autograd (your IFT backward) gradient wrt u ----
    qn, vn, pn, lamv, phi = step_square_pos_ip(
        qk, vk, pusher_pos, u_push, h, m, Izz, half, mu,
        ipm_opts=ipm_opts, skip_solving_threshold=float(skip_solving_threshold.item())
    )
    L = loss_from_outputs(qn, vn, pn, lamv, phi)
    (grad_u_auto,) = torch.autograd.grad(L, (u_push,), retain_graph=False, allow_unused=False)

    # ---- Finite-difference (central difference) wrt u ----
    eps = 1e-6 if dtype == torch.float64 else 5e-4
    grad_u_fd = torch.zeros_like(u_push, dtype=torch.float64)

    for i in range(u_push.numel()):
        e = torch.zeros_like(u_push)
        e[i] = eps

        # +eps
        up = (u_push.detach() + e).requires_grad_(False)
        qn_p, vn_p, pn_p, lamv_p, phi_p = step_square_pos_ip(
            qk, vk, pusher_pos, up, h, m, Izz, half, mu,
            ipm_opts=ipm_opts, skip_solving_threshold=float(skip_solving_threshold.item())
        )
        Lp = loss_from_outputs(qn_p, vn_p, pn_p, lamv_p, phi_p).item()

        # -eps
        um = (u_push.detach() - e).requires_grad_(False)
        qn_m, vn_m, pn_m, lamv_m, phi_m = step_square_pos_ip(
            qk, vk, pusher_pos, um, h, m, Izz, half, mu,
            ipm_opts=ipm_opts, skip_solving_threshold=float(skip_solving_threshold.item())
        )
        Lm = loss_from_outputs(qn_m, vn_m, pn_m, lamv_m, phi_m).item()

        grad_u_fd[i] = (Lp - Lm) / (2.0 * eps)

    # ---- Report ----
    grad_u_auto64 = grad_u_auto.detach().to(torch.float64)
    abs_err = (grad_u_auto64 - grad_u_fd).abs()
    rel_err = abs_err / (grad_u_fd.abs() + 1e-12)
    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()

    if verbose:
        regime = "CONTACT" if contact else "NO-CONTACT"
        print(f"[u_push | {regime}]")
        print(" autograd :", grad_u_auto64.cpu().numpy())
        print(" finite   :", grad_u_fd.cpu().numpy())
        print(" abs err  :", abs_err.cpu().numpy())
        print(" rel err  :", rel_err.cpu().numpy())
        print(f" max |Δ|={max_abs:.3e}, max rel err={max_rel:.3e}")

    return {"max_abs": max_abs, "max_rel": max_rel,
            "grad_u_auto": grad_u_auto64, "grad_u_fd": grad_u_fd}

# ---- Run both regimes if you want ----
summary_contact    = check_grad_wrt_u(contact=True)
summary_nocontact  = check_grad_wrt_u(contact=False)
print("\nSummary:", {"contact": summary_contact, "no_contact": summary_nocontact})