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


# Default workspace covers [-1, 1]^3 (task-agnostic).
# r_sphere sets precision: cell_size < r_sphere.
WORKSPACE  = 2.0    # [m] physical half-extent of reachable space
R_SPHERE   = 0.02   # [m] sphere radius (precision target)

def make_grid(half=0.1, workspace=WORKSPACE, r_sphere=R_SPHERE,
              resolution=None) -> SDFGrid:
    return SDFGrid.from_cube(half=half, r_sphere=r_sphere,
                             workspace=workspace, resolution=resolution,
                             device=DEVICE, dtype=DTYPE)


passed = 0
failed = 0

def check(name: str, condition: bool, detail: str = ''):
    global passed, failed
    if condition:
        print(f"  [PASS] {name} {detail}")
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

def analytical_sdf(p: torch.Tensor, half: float) -> float:
    """Exact cube SDF for reference."""
    d = p.abs() - half
    return (d.clamp(min=0.).norm() + d.max().clamp(max=0.)).item()


def test_sdf_grid():
    print("\n── 3. SDFGrid ──")

    half = 0.1
    grid = make_grid(half=half)   # workspace=1.0, r_sphere=0.02 → auto N

    cell_size = 2.0 * grid.bounds / (grid.resolution - 1)
    print(f"  Grid info: bounds={grid.bounds:.3f}m  N={grid.resolution}"
          f"  cell={cell_size*1000:.2f}mm  r_sphere={R_SPHERE*1000:.1f}mm")
    print(f"  cell/r_sphere = {cell_size/R_SPHERE:.3f}  (must be < 1.0)")

    # --- 3a. SDF sign correctness ---
    print("\n  [3a] SDF sign")
    cases = [
        (torch.tensor([0.2,  0.,  0.], dtype=DTYPE, device=DEVICE), "outside  [0.2, 0, 0]",   ">0"),
        (torch.tensor([0.0,  0.,  0.], dtype=DTYPE, device=DEVICE), "inside   [0,   0, 0]",   "<0"),
        (torch.tensor([half, 0.,  0.], dtype=DTYPE, device=DEVICE), "on face  [0.1, 0, 0]",   "~0"),
        (torch.tensor([0.0,  0., -0.05], dtype=DTYPE, device=DEVICE), "inside  [0,0,-0.05]", "<0"),
    ]
    for p, label, expect in cases:
        phi, _ = grid.query(p)
        v = phi.item()
        a = analytical_sdf(p, half)
        if expect == ">0":
            ok = v > 0
        elif expect == "<0":
            ok = v < 0
        else:
            ok = abs(v) < 0.01
        check(f"sign {label}", ok,
              f"phi={v:.5f}  analytical={a:.5f}  diff={v-a:.2e}")

    # --- 3b. SDF accuracy vs analytical ---
    print("\n  [3b] SDF accuracy vs analytical")
    tol = cell_size   # error should be within one cell
    test_pts = [
        (torch.tensor([0.15,  0.,    0.  ], dtype=DTYPE, device=DEVICE), "face  +x  [0.15,0,0]"),
        (torch.tensor([0.15,  0.15,  0.  ], dtype=DTYPE, device=DEVICE), "edge  xy  [0.15,0.15,0]"),
        (torch.tensor([0.15,  0.15,  0.15], dtype=DTYPE, device=DEVICE), "corner    [0.15,0.15,0.15]"),
        (torch.tensor([0.05,  0.,    0.  ], dtype=DTYPE, device=DEVICE), "inside    [0.05,0,0]"),
        (torch.tensor([-0.15, 0.,    0.  ], dtype=DTYPE, device=DEVICE), "face  -x  [-0.15,0,0]"),
        (torch.tensor([0.,    0.15,  0.  ], dtype=DTYPE, device=DEVICE), "face  +y  [0,0.15,0]"),
        (torch.tensor([0.,    0.,    0.15], dtype=DTYPE, device=DEVICE), "face  +z  [0,0,0.15]"),
        (torch.tensor([0.5,   0.,    0.  ], dtype=DTYPE, device=DEVICE), "far   +x  [0.5,0,0]"),
        (torch.tensor([0.99,  0.,    0.  ], dtype=DTYPE, device=DEVICE), "far   +x  [0.99,0,0]"),
    ]
    for p, label in test_pts:
        phi_g, _ = grid.query(p)
        phi_a    = analytical_sdf(p, half)
        err      = abs(phi_g.item() - phi_a)
        check(f"accuracy {label}", err < tol,
              f"grid={phi_g.item():.5f}  analytical={phi_a:.5f}"
              f"  err={err*1000:.2f}mm  tol={tol*1000:.2f}mm")

    # --- 3c. Gradient direction ---
    print("\n  [3c] Gradient direction (outward)")
    grad_cases = [
        (torch.tensor([0.15, 0.,  0. ], dtype=DTYPE, device=DEVICE),
         torch.tensor([1.,   0.,  0. ], dtype=DTYPE, device=DEVICE), "+x face"),
        (torch.tensor([-0.15,0.,  0. ], dtype=DTYPE, device=DEVICE),
         torch.tensor([-1.,  0.,  0. ], dtype=DTYPE, device=DEVICE), "-x face"),
        (torch.tensor([0.,  0.15, 0. ], dtype=DTYPE, device=DEVICE),
         torch.tensor([0.,  1.,   0. ], dtype=DTYPE, device=DEVICE), "+y face"),
        (torch.tensor([0.,  0.,  0.15], dtype=DTYPE, device=DEVICE),
         torch.tensor([0.,  0.,   1. ], dtype=DTYPE, device=DEVICE), "+z face"),
        (torch.tensor([0.,  0., -0.15], dtype=DTYPE, device=DEVICE),
         torch.tensor([0.,  0.,  -1. ], dtype=DTYPE, device=DEVICE), "-z face"),
    ]
    for p, expected_n, label in grad_cases:
        _, grad = grid.query(p)
        grad_norm = grad.norm().item()
        grad_unit = grad / (grad_norm + 1e-12)
        cosine = torch.dot(grad_unit, expected_n).item()
        check(f"grad direction {label}", cosine > 0.9,
              f"grad={[f'{g:.4f}' for g in grad.tolist()]}  |grad|={grad_norm:.4f}"
              f"  cos(expected)={cosine:.4f}")

    # --- 3d. Gradient magnitude ---
    print("\n  [3d] Gradient magnitude (SDF property: |∇φ| = 1 outside)")
    # For points well outside the cube, |∇φ| should be close to 1
    magnitude_pts = [
        torch.tensor([0.15, 0.,  0. ], dtype=DTYPE, device=DEVICE),
        torch.tensor([0.,  0.15, 0. ], dtype=DTYPE, device=DEVICE),
        torch.tensor([0.,  0.,  0.15], dtype=DTYPE, device=DEVICE),
        torch.tensor([0.5, 0.,  0.  ], dtype=DTYPE, device=DEVICE),
    ]
    for p in magnitude_pts:
        _, grad = grid.query(p)
        mag = grad.norm().item()
        check(f"  |∇φ|≈1 at {p.tolist()}", abs(mag - 1.0) < 0.1,
              f"|∇φ|={mag:.4f}")

    # --- 3e. Autograd ---
    print("\n  [3e] Autograd through query_differentiable")
    test_auto_pts = [
        (torch.tensor([0.15, 0.02, 0.01], dtype=DTYPE, device=DEVICE), "outside near +x"),
        (torch.tensor([0.5,  0.,   0.  ], dtype=DTYPE, device=DEVICE), "far outside  +x"),
        (torch.tensor([0.05, 0.,   0.  ], dtype=DTYPE, device=DEVICE), "inside"),
    ]
    for p0, label in test_auto_pts:
        p_diff = p0.clone().requires_grad_(True)
        phi_d  = grid.query_differentiable(p_diff)
        phi_d.backward()
        g_auto = p_diff.grad
        g_norm = g_auto.norm().item()

        # Compare direction with central-diff
        _, cd_grad = grid.query(p0)
        cosine = torch.dot(g_auto, cd_grad) / (g_auto.norm() * cd_grad.norm() + 1e-12)

        check(f"autograd nonzero [{label}]", g_norm > 0,
              f"|grad|={g_norm:.4f}  grad={[f'{g:.4f}' for g in g_auto.tolist()]}")
        check(f"autograd direction [{label}]", cosine.item() > 0.99,
              f"cosine={cosine.item():.4f}"
              f"  auto={[f'{g:.4f}' for g in g_auto.tolist()]}"
              f"  cd={[f'{g:.4f}' for g in cd_grad.tolist()]}")

    # --- 3f. Resolution auto-compute check ---
    print("\n  [3f] Auto resolution")
    for ws, rs in [(0.5, 0.02), (1.0, 0.02), (1.0, 0.05), (2.0, 0.02)]:
        g = SDFGrid.from_cube(half=0.1, r_sphere=rs, workspace=ws,
                              device=DEVICE, dtype=DTYPE)
        cs = 2.0 * g.bounds / (g.resolution - 1)
        check(f"cell<r_sphere  ws={ws} rs={rs}", cs < rs,
              f"N={g.resolution}  cell={cs*1000:.2f}mm  r={rs*1000:.1f}mm")


# ══════════════════════════════════════════════
# 3g. Gradient continuity across corners and edges
# ══════════════════════════════════════════════

def test_gradient_continuity():
    """
    Walk paths that cross face/edge/corner transitions and check
    that the gradient does not jump discontinuously.

    A jump is defined as consecutive gradient angle change > threshold.
    The SDF of a convex shape has a continuous gradient everywhere
    outside the shape — corners produce a kink in phi but gradient
    direction should transition smoothly through the grid interpolation.
    """
    print("── 3g. Gradient continuity across corners/edges ──")

    half = 0.1
    grid = make_grid(half=half)
    N_steps = 200
    MAX_JUMP_DEG = 10.0  # degrees — generous threshold for grid interpolation

    def grad_along_path(pts):
        """Return list of (phi, grad_unit) along path."""
        results = []
        for p in pts:
            phi, grad = grid.query(p)
            gnorm = grad.norm().item()
            grad_unit = (grad / (gnorm + 1e-12))
            results.append((phi.item(), grad_unit))
        return results

    def angle_between(g0, g1):
        """Angle between two unit gradient vectors in degrees (0~180, no abs)."""
        cos_a = torch.dot(g0, g1).clamp(-1.0, 1.0).item()
        return math.degrees(math.acos(cos_a))   # no abs: 180deg flip counts

    def max_angle_jump(results):
        """Max consecutive gradient direction change in degrees."""
        max_jump = 0.0
        for i in range(1, len(results)):
            g0 = results[i-1][1]
            g1 = results[i][1]
            max_jump = max(max_jump, angle_between(g0, g1))
        return max_jump

    # --- Path 1: arc around +x/+y edge (xy plane, r=0.18m) ---
    # Sweeps from +x face region to +y face region, crossing the edge
    angles = torch.linspace(0, math.pi/2, N_steps, dtype=DTYPE, device=DEVICE)
    r = half + 0.08   # 0.18m from origin, clearly outside
    path1 = [torch.stack([r*torch.cos(a), r*torch.sin(a),
                          torch.zeros(1, dtype=DTYPE, device=DEVICE).squeeze()])
             for a in angles]
    res1 = grad_along_path(path1)
    jump1 = max_angle_jump(res1)
    check("arc xy: max gradient jump < threshold",
          jump1 < MAX_JUMP_DEG,
          f"max_jump={jump1:.2f}deg  threshold={MAX_JUMP_DEG:.1f}deg")

    # --- Path 2: arc around xyz corner (r=0.22m, diagonal sweep) ---
    # Approaches the (1,1,1) corner from different directions
    ts = torch.linspace(0, 1, N_steps, dtype=DTYPE, device=DEVICE)
    # Sweep from [1,0,0] direction to [1,1,1]/sqrt(3) direction
    p_start = torch.tensor([1., 0., 0.], dtype=DTYPE, device=DEVICE)
    p_end   = torch.tensor([1., 1., 1.], dtype=DTYPE, device=DEVICE)
    p_end   = p_end / p_end.norm()
    r_corner = half + 0.08
    path2 = []
    for t in ts:
        d = (1-t) * p_start + t * p_end
        d = d / d.norm()
        path2.append(d * r_corner)
    res2 = grad_along_path(path2)
    jump2 = max_angle_jump(res2)
    check("corner xyz: max gradient jump < threshold",
          jump2 < MAX_JUMP_DEG,
          f"max_jump={jump2:.2f}deg  threshold={MAX_JUMP_DEG:.1f}deg")

    # --- Path 3: linear sweep across +x face → +y face (through edge) ---
    # p moves along y at fixed x=0.18, z=0
    # Starts in +x face region, crosses edge, enters +y face region
    ys = torch.linspace(-0.05, 0.22, N_steps, dtype=DTYPE, device=DEVICE)
    path3 = [torch.stack([torch.tensor(half+0.08, dtype=DTYPE, device=DEVICE),
                          y,
                          torch.tensor(0., dtype=DTYPE, device=DEVICE)])
             for y in ys]
    res3 = grad_along_path(path3)
    jump3 = max_angle_jump(res3)
    check("linear +x→+y face: max gradient jump < threshold",
          jump3 < MAX_JUMP_DEG,
          f"max_jump={jump3:.2f}deg  threshold={MAX_JUMP_DEG:.1f}deg")

    # --- Path 4: close to surface along edge ---
    # p moves along z just outside the +x/+y edge (tight path near surface)
    zs = torch.linspace(-0.15, 0.15, N_steps, dtype=DTYPE, device=DEVICE)
    r_edge = half + 0.02   # just 2cm outside edge
    path4 = [torch.stack([torch.tensor(r_edge/math.sqrt(2), dtype=DTYPE, device=DEVICE),
                          torch.tensor(r_edge/math.sqrt(2), dtype=DTYPE, device=DEVICE),
                          z])
             for z in zs]
    res4 = grad_along_path(path4)
    jump4 = max_angle_jump(res4)
    check("along +x/+y edge: max gradient jump < threshold",
          jump4 < MAX_JUMP_DEG,
          f"max_jump={jump4:.2f}deg  threshold={MAX_JUMP_DEG:.1f}deg")

    # --- Detailed printout: all 4 paths, only points with jump > 1 deg ---
    def print_path_profile(name, results, param_vals, param_label):
        print(f"\n  Profile — {name}:")
        print(f"  {param_label:>12}  {'phi':>8}  {'gx':>7}  {'gy':>7}  {'gz':>7}  {'jump_deg':>9}")
        prev_g = None
        for i, (phi_v, g_unit) in enumerate(results):
            jump = angle_between(prev_g, g_unit) if prev_g is not None else 0.0
            if i == 0 or i == len(results)-1 or jump > 1.0:
                pval = param_vals[i].item() if hasattr(param_vals[i], 'item') else param_vals[i]
                print(f"  {pval:12.4f}  {phi_v:8.5f}"
                      f"  {g_unit[0].item():7.4f}  {g_unit[1].item():7.4f}"
                      f"  {g_unit[2].item():7.4f}  {jump:9.2f}")
            prev_g = g_unit

    print_path_profile('arc xy (face->edge->face)',
                       res1, [math.degrees(a.item()) for a in angles], 'angle_deg')
    print_path_profile('corner xyz sweep',
                       res2, ts, 't')
    print_path_profile('linear +x->+y face',
                       res3, ys, 'y [m]')
    print_path_profile('along +x/+y edge (z sweep)',
                       res4, zs, 'z [m]')


# ══════════════════════════════════════════════
# 4. cube_sphere_contact
# ══════════════════════════════════════════════

def test_cube_sphere_contact():
    print("\n── 4. cube_sphere_contact ──")

    half     = 0.1
    r_sphere = 0.02
    grid     = make_grid(half=half)

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
    grid     = make_grid(half=half)

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
# 7. SDF Smoothing comparison
# ══════════════════════════════════════════════

def test_sdf_smoothing():
    """
    Compare sharp vs smoothed SDF:
      - phi values should be close (smoothing doesn't change SDF much)
      - gradient continuity should improve at edges/corners
      - gradient direction should still point outward

    Also compares contact-heavy gradient FD stability before/after smoothing.
    """
    print("\n── 7. SDF Smoothing comparison ──")

    half = 0.1
    sharpness = 50.0

    grid_sharp  = make_grid(half=half)
    print()
    grid_smooth = SDFGrid.from_cube(
        half=half, r_sphere=R_SPHERE, workspace=WORKSPACE,
        smooth=True, sharpness=sharpness,
        device=DEVICE, dtype=DTYPE,
    )

    # --- 7a. phi accuracy: smooth vs sharp vs analytical ---
    print("\n  [7a] phi: smooth vs sharp vs analytical")
    test_pts = [
        (torch.tensor([0.15, 0.,   0.  ], dtype=DTYPE, device=DEVICE), "face  +x"),
        (torch.tensor([0.15, 0.15, 0.  ], dtype=DTYPE, device=DEVICE), "edge  xy"),
        (torch.tensor([0.15, 0.15, 0.15], dtype=DTYPE, device=DEVICE), "corner"),
        (torch.tensor([0.05, 0.,   0.  ], dtype=DTYPE, device=DEVICE), "inside"),
    ]
    for p, label in test_pts:
        phi_sh,  _ = grid_sharp.query(p)
        phi_sm,  _ = grid_smooth.query(p)
        phi_an     = analytical_sdf(p, half)
        diff = abs(phi_sm.item() - phi_sh.item())
        print(f"  {label:10s}  sharp={phi_sh.item():.5f}  smooth={phi_sm.item():.5f}"
              f"  analytical={phi_an:.5f}  |smooth-sharp|={diff:.5f}")
        check(f"phi close: {label}", diff < 0.01,
              f"|smooth-sharp|={diff:.5f}")

    # --- 7b. gradient continuity: sharp vs smooth at edge ---
    print("\n  [7b] Gradient continuity: arc xy (face→edge→face)")
    import math as _math
    N_steps = 200
    r_arc   = half + 0.08
    angles  = torch.linspace(0, _math.pi/2, N_steps, dtype=DTYPE, device=DEVICE)

    def max_jump(grid_obj):
        prev_g = None
        max_j  = 0.0
        for a in angles:
            p = torch.stack([r_arc * a.cos(),
                             r_arc * a.sin(),
                             torch.zeros(1, dtype=DTYPE, device=DEVICE).squeeze()])
            _, grad = grid_obj.query(p)
            gn = grad / (grad.norm() + 1e-12)
            if prev_g is not None:
                cos_a = torch.dot(prev_g, gn).clamp(-1., 1.).item()
                max_j = max(max_j, _math.degrees(_math.acos(cos_a)))
            prev_g = gn
        return max_j

    jump_sharp  = max_jump(grid_sharp)
    jump_smooth = max_jump(grid_smooth)
    print(f"  sharp:  max_jump={jump_sharp:.2f}deg")
    print(f"  smooth: max_jump={jump_smooth:.2f}deg")
    check("smooth reduces edge gradient jump", jump_smooth < jump_sharp,
          f"smooth={jump_smooth:.2f}deg  sharp={jump_sharp:.2f}deg")
    check("smooth gradient jump < 5deg", jump_smooth < 5.0,
          f"max_jump={jump_smooth:.2f}deg")

    # --- 7c. gradient direction still outward after smoothing ---
    print("\n  [7c] Gradient direction: smooth grid")
    grad_cases = [
        (torch.tensor([0.15, 0.,  0. ], dtype=DTYPE, device=DEVICE),
         torch.tensor([1.,   0.,  0. ], dtype=DTYPE, device=DEVICE), "+x face"),
        (torch.tensor([0.,  0.15, 0. ], dtype=DTYPE, device=DEVICE),
         torch.tensor([0.,  1.,   0. ], dtype=DTYPE, device=DEVICE), "+y face"),
        (torch.tensor([0.,  0.,  0.15], dtype=DTYPE, device=DEVICE),
         torch.tensor([0.,  0.,   1. ], dtype=DTYPE, device=DEVICE), "+z face"),
    ]
    for p, expected_n, label in grad_cases:
        _, grad = grid_smooth.query(p)
        cosine = torch.dot(grad / (grad.norm() + 1e-12), expected_n).item()
        check(f"smooth grad direction {label}", cosine > 0.9,
              f"cos={cosine:.4f}")

    # --- 7d. autograd through smooth grid ---
    print("\n  [7d] Autograd: smooth grid")
    p_diff = torch.tensor([0.15, 0.02, 0.01], dtype=DTYPE, device=DEVICE,
                          requires_grad=True)
    phi_d = grid_smooth.query_differentiable(p_diff)
    phi_d.backward()
    _, cd_grad = grid_smooth.query(p_diff.detach())
    g_auto = p_diff.grad
    cosine = torch.dot(g_auto, cd_grad) / (g_auto.norm() * cd_grad.norm() + 1e-12)
    check("smooth autograd nonzero", g_auto.norm().item() > 0,
          f"|grad|={g_auto.norm().item():.4f}")
    check("smooth autograd vs central-diff direction", cosine.item() > 0.99,
          f"cosine={cosine.item():.4f}")

    # --- 7e. Contact-heavy FD stability: sharp vs smooth ---
    print("\n  [7e] FD stability comparison (contact heavy)")
    print("  (uses cube_sphere_contact, not full dynamics)")

    q_pose   = torch.tensor([0., 0., 0.5, 1., 0., 0., 0.], dtype=DTYPE, device=DEVICE)
    gap      = 0.002
    p_sphere = torch.tensor([half+R_SPHERE+gap, 0., 0.5], dtype=DTYPE, device=DEVICE)
    eps_vals = [1e-2, 1e-3, 1e-4, 1e-5]

    for label_g, grid_obj in [("sharp ", grid_sharp), ("smooth", grid_smooth)]:
        print(f"  [{label_g}] d(phi)/d(p_sphere_x):")
        for eps_try in eps_vals:
            p_plus  = p_sphere.clone(); p_plus[0]  += eps_try
            p_minus = p_sphere.clone(); p_minus[0] -= eps_try
            cnt_p = cube_sphere_contact(q_pose, p_plus,  grid_obj, R_SPHERE)
            cnt_m = cube_sphere_contact(q_pose, p_minus, grid_obj, R_SPHERE)
            fd = (cnt_p.phi.item() - cnt_m.phi.item()) / (2 * eps_try)
            print(f"    eps={eps_try:.0e}  FD={fd:10.5f}")

    # Final: gradient improvement summary
    improved = jump_smooth < jump_sharp
    check("smoothing improves edge gradient continuity overall", improved,
          f"sharp={jump_sharp:.2f}deg → smooth={jump_smooth:.2f}deg")


# ══════════════════════════════════════════════
# Main (updated)
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
    test_gradient_continuity()
    test_cube_sphere_contact()
    test_contact_jacobians_3d()
    test_sdf_smoothing()

    print("\n" + "=" * 55)
    total = passed + failed
    print(f"  Result: {passed}/{total} passed", end="")
    if failed > 0:
        print(f"  ({failed} FAILED)")
    else:
        print("  — all passed")
    print("=" * 55)

    sys.exit(0 if failed == 0 else 1)