"""
test_geometry_3d.py

Tests for geometry_3d.py.

Run:
    python test_geometry_3d.py
    python test_geometry_3d.py -v   # verbose

Test groups:
  1. quat_to_rot      - rotation matrix properties
  2. tangent_frame    - orthonormality
  3. SDFGrid          - SDF values, gradient direction, autograd
  4. cube_sphere_contact - phi sign, normal direction, contact point
  5. contact_jacobians_3d - Jn @ v = normal velocity (numerical check)
"""

import torch
import sys
import math


from trajectory_opt_with_contact.geometry_3d import (
    quat_to_rot, tangent_frame,
    SDFGrid, CubeContact3D,
    cube_sphere_contact, contact_jacobians_3d,
)

DTYPE  = torch.float64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS    = 1e-6   # tolerance for most checks
EPS_GRAD = 1e-4 # tolerance for gradient checks (grid interpolation is C0)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def identity_pose(dtype=DTYPE, device=DEVICE) -> torch.Tensor:
    """Cube at origin, no rotation."""
    return torch.tensor([0., 0., 0., 1., 0., 0., 0.], dtype=dtype, device=device)


def pose_from_pos(px, py, pz, dtype=DTYPE, device=DEVICE) -> torch.Tensor:
    """Cube at given position, no rotation."""
    return torch.tensor([px, py, pz, 1., 0., 0., 0.], dtype=dtype, device=device)


def rot_z_pose(angle_deg, dtype=DTYPE, device=DEVICE) -> torch.Tensor:
    """Cube at origin, rotated around Z by angle_deg."""
    a = math.radians(angle_deg) / 2
    return torch.tensor([0., 0., 0., math.cos(a), 0., 0., math.sin(a)],
                        dtype=dtype, device=device)


def make_grid(half=0.1, resolution=64) -> SDFGrid:
    return SDFGrid.from_cube(half=half, resolution=resolution,
                             device=DEVICE, dtype=DTYPE)


passed = 0
failed = 0

def check(name: str, condition: bool, detail: str = ''):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1


# ══════════════════════════════════════════════
# 1. quat_to_rot
# ══════════════════════════════════════════════

def test_quat_to_rot():
    print("\n── 1. quat_to_rot ──")

    # Identity quaternion -> identity matrix
    q_id = torch.tensor([1., 0., 0., 0.], dtype=DTYPE, device=DEVICE)
    R = quat_to_rot(q_id)
    err = (R - torch.eye(3, dtype=DTYPE, device=DEVICE)).abs().max().item()
    check("identity quat -> I", err < EPS, f"max_err={err:.2e}")

    # R is orthogonal: R^T R = I
    q = torch.tensor([math.cos(0.3), math.sin(0.3)*0.6,
                      math.sin(0.3)*0.8, 0.], dtype=DTYPE, device=DEVICE)
    q = q / q.norm()
    R = quat_to_rot(q)
    err = (R.T @ R - torch.eye(3, dtype=DTYPE, device=DEVICE)).abs().max().item()
    check("R^T R = I", err < EPS, f"max_err={err:.2e}")

    # det(R) = +1
    det = torch.linalg.det(R).item()
    check("det(R) = 1", abs(det - 1.0) < EPS, f"det={det:.6f}")

    # 90-degree rotation around Z: x-axis -> y-axis
    a = math.pi / 4   # half-angle for 90 deg rotation
    q_z90 = torch.tensor([math.cos(a), 0., 0., math.sin(a)], dtype=DTYPE, device=DEVICE)
    R_z90 = quat_to_rot(q_z90)
    x_rotated = R_z90 @ torch.tensor([1., 0., 0.], dtype=DTYPE, device=DEVICE)
    err = (x_rotated - torch.tensor([0., 1., 0.], dtype=DTYPE, device=DEVICE)).norm().item()
    check("90deg Z: x->y", err < EPS, f"err={err:.2e}")


# ══════════════════════════════════════════════
# 2. tangent_frame
# ══════════════════════════════════════════════

def test_tangent_frame():
    print("\n── 2. tangent_frame ──")

    for name, n_vec in [
        ("z-axis",  [0., 0., 1.]),
        ("-z-axis", [0., 0., -1.]),
        ("x-axis",  [1., 0., 0.]),
        ("oblique", [1/3**0.5, 1/3**0.5, 1/3**0.5]),
    ]:
        n = torch.tensor(n_vec, dtype=DTYPE, device=DEVICE)
        n = n / n.norm()
        t1, t2 = tangent_frame(n)

        # Unit vectors
        check(f"[{name}] |t1|=1", abs(t1.norm().item() - 1.0) < EPS)
        check(f"[{name}] |t2|=1", abs(t2.norm().item() - 1.0) < EPS)

        # Orthogonality
        check(f"[{name}] n.t1=0",  abs(torch.dot(n, t1).item())  < EPS)
        check(f"[{name}] n.t2=0",  abs(torch.dot(n, t2).item())  < EPS)
        check(f"[{name}] t1.t2=0", abs(torch.dot(t1, t2).item()) < EPS)

        # Right-handed: t1 x t2 = n
        cross = torch.linalg.cross(t1, t2)
        err = (cross - n).norm().item()
        check(f"[{name}] t1×t2=n", err < EPS, f"err={err:.2e}")


# ══════════════════════════════════════════════
# 3. SDFGrid
# ══════════════════════════════════════════════

def test_sdf_grid():
    print("\n── 3. SDFGrid ──")

    half = 0.1
    grid = make_grid(half=half, resolution=128)

    # --- 3a. SDF sign correctness ---
    # Point clearly outside: phi > 0
    p_out = torch.tensor([0.2, 0., 0.], dtype=DTYPE, device=DEVICE)
    phi_out, _ = grid.query(p_out)
    check("outside: phi > 0", phi_out.item() > 0,
          f"phi={phi_out.item():.4f}")

    # Point clearly inside: phi < 0
    p_in = torch.tensor([0.0, 0., 0.], dtype=DTYPE, device=DEVICE)
    phi_in, _ = grid.query(p_in)
    check("inside (center): phi < 0", phi_in.item() < 0,
          f"phi={phi_in.item():.4f}")

    # Point on face: phi ~ 0
    p_face = torch.tensor([half, 0., 0.], dtype=DTYPE, device=DEVICE)
    phi_face, _ = grid.query(p_face)
    check("on face: phi ~ 0", abs(phi_face.item()) < 0.01,
          f"phi={phi_face.item():.4f}")

    # --- 3b. SDF value accuracy vs analytical ---
    # Analytical cube SDF
    def analytical_sdf(p, h):
        d = p.abs() - h
        return d.clamp(min=0.).norm() + d.max().clamp(max=0.)

    test_points = [
        torch.tensor([0.15, 0.,   0.  ], dtype=DTYPE, device=DEVICE),
        torch.tensor([0.15, 0.15, 0.  ], dtype=DTYPE, device=DEVICE),
        torch.tensor([0.15, 0.15, 0.15], dtype=DTYPE, device=DEVICE),
        torch.tensor([0.05, 0.,   0.  ], dtype=DTYPE, device=DEVICE),
    ]
    for i, p in enumerate(test_points):
        phi_grid_val, _ = grid.query(p)
        phi_analytical  = analytical_sdf(p, half)
        err = abs(phi_grid_val.item() - phi_analytical.item())
        check(f"SDF accuracy point {i+1}", err < 0.005,
              f"grid={phi_grid_val.item():.4f}, analytical={phi_analytical.item():.4f}, err={err:.4f}")

    # --- 3c. Gradient direction (outward) ---
    # Outside the cube, gradient should point roughly away from cube
    p_right = torch.tensor([0.15, 0., 0.], dtype=DTYPE, device=DEVICE)
    _, grad = grid.query(p_right)
    gx = grad[0].item()
    check("gradient points outward (+x face)", gx > 0.5,
          f"grad={grad.tolist()}")

    p_top = torch.tensor([0., 0., 0.15], dtype=DTYPE, device=DEVICE)
    _, grad_top = grid.query(p_top)
    gz = grad_top[2].item()
    check("gradient points outward (+z face)", gz > 0.5,
          f"grad={grad_top.tolist()}")

    # --- 3d. Differentiability (autograd through query_differentiable) ---
    # grid_sample runs in float32 internally (CUDA limitation).
    # Gradient flows back through float32 path — check nonzero and direction.
    # query_differentiable now accepts float64 directly (explicit trilinear).
    p_diff = torch.tensor([0.15, 0.02, 0.01], dtype=DTYPE,
                          device=DEVICE, requires_grad=True)
    phi_diff = grid.query_differentiable(p_diff)
    phi_diff.backward()
    grad_auto = p_diff.grad
    grad_norm = grad_auto.norm().item() if grad_auto is not None else 0.0
    check("autograd through query_differentiable",
          grad_norm > 0,
          f"grad_norm={grad_norm:.4f}, grad={grad_auto}")

    # Direction agreement: autograd vs central-diff query()
    p0 = torch.tensor([0.15, 0.02, 0.01], dtype=DTYPE, device=DEVICE)
    _, cd_grad = grid.query(p0)
    cosine = torch.dot(grad_auto, cd_grad) / (grad_auto.norm() * cd_grad.norm() + 1e-12)
    check("autograd vs central-diff direction", cosine.item() > 0.99,
          f"cosine={cosine.item():.4f}")


# ══════════════════════════════════════════════
# 4. cube_sphere_contact
# ══════════════════════════════════════════════

def test_cube_sphere_contact():
    print("\n── 4. cube_sphere_contact ──")

    half     = 0.1
    r_sphere = 0.02
    grid     = make_grid(half=half, resolution=128)

    # --- 4a. Sphere clearly separated ---
    q_pose   = identity_pose()
    p_sphere = torch.tensor([0.2, 0., 0.], dtype=DTYPE, device=DEVICE)
    cnt = cube_sphere_contact(q_pose, p_sphere, grid, r_sphere)
    check("separated: phi > 0", cnt.phi.item() > 0,
          f"phi={cnt.phi.item():.4f}")

    # --- 4b. Sphere touching (+x face) ---
    # Sphere center at x = half + r_sphere -> phi = 0
    p_touch = torch.tensor([half + r_sphere, 0., 0.], dtype=DTYPE, device=DEVICE)
    cnt_touch = cube_sphere_contact(q_pose, p_touch, grid, r_sphere)
    check("touching: phi ~ 0", abs(cnt_touch.phi.item()) < 0.01,
          f"phi={cnt_touch.phi.item():.4f}")

    # Normal should point in +x direction (toward sphere)
    nx = cnt_touch.normal[0].item()
    check("touching +x: normal points in +x", nx > 0.8,
          f"normal={cnt_touch.normal.tolist()}")

    # --- 4c. Sphere penetrating ---
    p_pen = torch.tensor([half, 0., 0.], dtype=DTYPE, device=DEVICE)   # center on surface
    cnt_pen = cube_sphere_contact(q_pose, p_pen, grid, r_sphere)
    check("penetrating: phi < 0", cnt_pen.phi.item() < 0,
          f"phi={cnt_pen.phi.item():.4f}")

    # --- 4d. Sphere above (+z face) ---
    p_top = torch.tensor([0., 0., half + r_sphere + 0.01], dtype=DTYPE, device=DEVICE)
    cnt_top = cube_sphere_contact(q_pose, p_top, grid, r_sphere)
    check("above +z: phi > 0", cnt_top.phi.item() > 0)
    nz = cnt_top.normal[2].item()
    check("above +z: normal points in +z", nz > 0.8,
          f"normal={cnt_top.normal.tolist()}")

    # --- 4e. Cube translated: phi should be same as cube at origin ---
    offset = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE, device=DEVICE)
    q_shifted = torch.cat([offset, torch.tensor([1., 0., 0., 0.], dtype=DTYPE, device=DEVICE)])
    p_shifted = p_touch + offset
    cnt_shifted = cube_sphere_contact(q_shifted, p_shifted, grid, r_sphere)
    err = abs(cnt_shifted.phi.item() - cnt_touch.phi.item())
    check("translation invariance", err < 0.01,
          f"phi_orig={cnt_touch.phi.item():.4f}, phi_shifted={cnt_shifted.phi.item():.4f}")

    # --- 4f. Cube rotated 90 deg around Z: sphere on +y face ---
    q_rot90 = rot_z_pose(90)
    # Sphere that was at +x face of unrotated cube should now be at +y face
    p_rot = torch.tensor([0., half + r_sphere, 0.], dtype=DTYPE, device=DEVICE)
    cnt_rot = cube_sphere_contact(q_rot90, p_rot, grid, r_sphere)
    check("rotated cube: phi ~ 0", abs(cnt_rot.phi.item()) < 0.02,
          f"phi={cnt_rot.phi.item():.4f}")

    # --- 4g. Normal unit length ---
    check("normal is unit", abs(cnt_touch.normal.norm().item() - 1.0) < EPS)
    check("tangent1 is unit", abs(cnt_touch.tangent1.norm().item() - 1.0) < EPS)
    check("tangent2 is unit", abs(cnt_touch.tangent2.norm().item() - 1.0) < EPS)

    # --- 4h. Tangents orthogonal to normal ---
    dot_n_t1 = torch.dot(cnt_touch.normal, cnt_touch.tangent1).abs().item()
    dot_n_t2 = torch.dot(cnt_touch.normal, cnt_touch.tangent2).abs().item()
    check("n ⊥ t1", dot_n_t1 < EPS, f"dot={dot_n_t1:.2e}")
    check("n ⊥ t2", dot_n_t2 < EPS, f"dot={dot_n_t2:.2e}")


# ══════════════════════════════════════════════
# 5. contact_jacobians_3d
# ══════════════════════════════════════════════

def test_contact_jacobians_3d():
    print("\n── 5. contact_jacobians_3d ──")

    half     = 0.1
    r_sphere = 0.02
    grid     = make_grid(half=half, resolution=128)

    q_pose   = identity_pose()
    p_sphere = torch.tensor([half + r_sphere + 0.01, 0., 0.], dtype=DTYPE, device=DEVICE)
    cnt      = cube_sphere_contact(q_pose, p_sphere, grid, r_sphere)

    Jn, Jt1, Jt2 = contact_jacobians_3d(cnt.normal, cnt.tangent1, cnt.tangent2, cnt.r_cp)

    # Shape check
    check("Jn shape (6,)",  Jn.shape  == (6,))
    check("Jt1 shape (6,)", Jt1.shape == (6,))
    check("Jt2 shape (6,)", Jt2.shape == (6,))

    # --- 5a. Pure linear velocity in normal direction ---
    # twist = [n, 0, 0, 0, 0, 0] -> v_n = 1, v_t1 = 0, v_t2 = 0
    n = cnt.normal
    twist_n = torch.cat([n, torch.zeros(3, dtype=DTYPE, device=DEVICE)])
    v_n  = (Jn  @ twist_n).item()
    v_t1 = (Jt1 @ twist_n).item()
    v_t2 = (Jt2 @ twist_n).item()
    check("pure normal vel: Jn@twist=1",   abs(v_n  - 1.0) < EPS, f"v_n={v_n:.4f}")
    check("pure normal vel: Jt1@twist=0",  abs(v_t1)       < EPS, f"v_t1={v_t1:.4f}")
    check("pure normal vel: Jt2@twist=0",  abs(v_t2)       < EPS, f"v_t2={v_t2:.4f}")

    # --- 5b. Pure linear velocity in t1 direction ---
    t1 = cnt.tangent1
    twist_t1 = torch.cat([t1, torch.zeros(3, dtype=DTYPE, device=DEVICE)])
    v_n2  = (Jn  @ twist_t1).item()
    v_t12 = (Jt1 @ twist_t1).item()
    check("pure t1 vel: Jn@twist=0",   abs(v_n2)        < EPS, f"v_n={v_n2:.4f}")
    check("pure t1 vel: Jt1@twist=1",  abs(v_t12 - 1.0) < EPS, f"v_t1={v_t12:.4f}")

    # --- 5c. Pure angular velocity about normal axis -> zero contact velocity ---
    # Rotation about n: angular velocity w = omega * n
    # Contact point velocity = omega x r_cp
    # Normal component = n . (omega*n x r_cp) = omega * n . (n x r_cp) = 0
    omega = 1.0
    twist_rot_n = torch.cat([torch.zeros(3, dtype=DTYPE, device=DEVICE), n * omega])
    v_n_rot = (Jn @ twist_rot_n).item()
    check("spin about normal: Jn@twist=0", abs(v_n_rot) < EPS, f"v_n={v_n_rot:.4f}")

    # --- 5d. Numerical verification: Jn @ twist == n . (v + w x r_cp) ---
    # Random twist
    torch.manual_seed(42)
    twist = torch.randn(6, dtype=DTYPE, device=DEVICE)
    v_lin = twist[:3]
    w_ang = twist[3:]
    r_cp  = cnt.r_cp

    # Manual computation
    v_cp = v_lin + torch.linalg.cross(w_ang, r_cp)
    v_n_manual  = torch.dot(n,           v_cp).item()
    v_t1_manual = torch.dot(cnt.tangent1, v_cp).item()
    v_t2_manual = torch.dot(cnt.tangent2, v_cp).item()

    v_n_jac  = (Jn  @ twist).item()
    v_t1_jac = (Jt1 @ twist).item()
    v_t2_jac = (Jt2 @ twist).item()

    check("Jn  @ twist == n.(v+w×r)",  abs(v_n_jac  - v_n_manual)  < EPS, f"jac={v_n_jac:.6f}, manual={v_n_manual:.6f}")
    check("Jt1 @ twist == t1.(v+w×r)", abs(v_t1_jac - v_t1_manual) < EPS, f"jac={v_t1_jac:.6f}, manual={v_t1_manual:.6f}")
    check("Jt2 @ twist == t2.(v+w×r)", abs(v_t2_jac - v_t2_manual) < EPS, f"jac={v_t2_jac:.6f}, manual={v_t2_manual:.6f}")

    # --- 5e. Sphere 2 (different position) ---
    p_sphere2 = torch.tensor([0., half + r_sphere + 0.01, 0.], dtype=DTYPE, device=DEVICE)
    cnt2 = cube_sphere_contact(q_pose, p_sphere2, grid, r_sphere)
    Jn2, Jt12, Jt22 = contact_jacobians_3d(cnt2.normal, cnt2.tangent1, cnt2.tangent2, cnt2.r_cp)

    twist2 = torch.cat([cnt2.normal, torch.zeros(3, dtype=DTYPE, device=DEVICE)])
    v_n2_check = (Jn2 @ twist2).item()
    check("sphere2: pure normal vel Jn@twist=1", abs(v_n2_check - 1.0) < EPS, f"v_n={v_n2_check:.4f}")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

if __name__ == '__main__':
    verbose = '-v' in sys.argv

    print("=" * 55)
    print("  geometry_3d.py  —  test suite")
    print("=" * 55)
    print(f"Device: {DEVICE}")

    test_quat_to_rot()
    test_tangent_frame()
    test_sdf_grid()
    test_cube_sphere_contact()
    test_contact_jacobians_3d()

    print("\n" + "=" * 55)
    total = passed + failed
    print(f"  Result: {passed}/{total} passed", end="")
    if failed > 0:
        print(f"  ({failed} FAILED)")
    else:
        print("  — all passed")
    print("=" * 55)

    sys.exit(0 if failed == 0 else 1)