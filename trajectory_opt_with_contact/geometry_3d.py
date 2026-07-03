"""
geometry_3d.py

3D contact geometry for cube-sphere contact.

Design:
  - SDFGrid: precomputed SDF in cube body frame (offline, one-time)
  - CubeContact3D: dataclass holding per-timestep contact results (mirrors OBBContact)
  - cube_sphere_contact: runtime query (mirrors obb_contact_blend2)
  - contact_jacobians_3d: body twist -> contact velocity (mirrors contact_jacobians)

Coordinate conventions:
  - Object pose: q_pose = [px, py, pz, qw, qx, qy, qz]  (position + unit quaternion)
  - Body frame:  z-axis up, cube centered at origin
  - Normal:      points FROM cube surface TOWARD sphere center (outward from object)
  - Tangents:    {t1, t2} form right-handed frame with normal
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional


# ──────────────────────────────────────────────
# Rotation utilities
# ──────────────────────────────────────────────

def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """
    Unit quaternion -> 3x3 rotation matrix.

    Args:
        q: (4,) tensor [qw, qx, qy, qz]

    Returns:
        R: (3, 3) rotation matrix (body -> world)
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    R = torch.stack([
        torch.stack([1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)]),
        torch.stack([    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)]),
        torch.stack([    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]),
    ])  # (3, 3)

    return R


def tangent_frame(n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build an orthonormal tangent frame {t1, t2} given unit normal n.

    Prefer a gravity-aligned tangent when the contact normal is not vertical.
    This keeps side contacts from flipping tangent direction due to tiny
    numerical changes in n_z near zero.

    Args:
        n: (3,) unit normal vector

    Returns:
        t1, t2: (3,) unit tangent vectors s.t. {n, t1, t2} right-handed
    """
    down = torch.tensor([0.0, 0.0, -1.0], dtype=n.dtype, device=n.device)
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device)
    ref = torch.where(torch.abs(n[2]) < 0.9, down, x_axis)

    t1 = ref - torch.dot(ref, n) * n
    t1 = t1 / (torch.linalg.norm(t1) + 1e-12)
    t2 = torch.linalg.cross(n, t1)
    t2 = t2 / (torch.linalg.norm(t2) + 1e-12)

    return t1, t2


# ──────────────────────────────────────────────
# SDF Grid (offline, precomputed in body frame)
# ──────────────────────────────────────────────

class SDFGrid:
    """
    Precomputed signed-distance field in cube body frame.

    The grid is fixed once built.  At runtime, the sphere center is
    transformed into body frame and queried via trilinear interpolation.
    This makes the approach object-agnostic: swap from_cube() for
    from_mesh() to support arbitrary shapes without changing downstream code.

    Conventions:
      - phi > 0: outside the object
      - phi < 0: inside (penetrating)
      - phi = 0: on the surface
      - gradient of phi points outward (= contact normal direction)
    """

    def __init__(
        self,
        phi_grid: torch.Tensor,
        bounds: float,
        resolution: int,
    ):
        """
        Args:
            phi_grid:   (N, N, N) SDF values in body frame
            bounds:     grid spans [-bounds, +bounds] in each axis
            resolution: N (grid is N^3)
        """
        # Store as float32 — grid_sample requires float32 on CUDA.
        # Graph stays in float32 throughout; callers cast outputs as needed.
        self.phi_grid   = phi_grid.float()  # (N, N, N), float32
        self.bounds     = float(bounds)
        self.resolution = resolution
        self.device     = phi_grid.device

    @staticmethod
    def _auto_resolution(workspace: float, r_sphere: float) -> int:
        """
        Compute the minimum grid resolution such that cell size < r_sphere.

        Condition:  2 * workspace / (N - 1) < r_sphere
        → N > 2 * workspace / r_sphere + 1
        Round up to next power of 2 for GPU efficiency.

        Args:
            workspace: half-extent of the grid in each axis [m]
            r_sphere:  sphere radius [m]  (precision target)

        Returns:
            N: grid resolution
        """
        n_min = int(2.0 * workspace / r_sphere) + 2
        # Next power of 2 >= n_min
        N = 1
        while N < n_min:
            N *= 2
        return N

    @staticmethod
    def from_cube(
        half: float,
        r_sphere: float,
        workspace: float,
        resolution: int = None,
        smooth: bool = False,
        sharpness: float = 50.0,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float64,
    ) -> 'SDFGrid':
        """
        Build SDF grid from analytical cube SDF in body frame.

        The grid covers [-workspace, +workspace]^3 in body frame.
        workspace should be the physical extent of the space the sphere
        can reach — the SDF does not need to know the trajectory, only
        the reachable space.

        Resolution is auto-computed so that cell size < r_sphere (the
        smallest feature that matters for contact detection).  Pass an
        explicit resolution to override.

        Args:
            half:       cube half-side length [m]
            r_sphere:   sphere radius [m]  (sets precision requirement)
            workspace:  half-extent of grid in each body-frame axis [m]
                        e.g. 1.0 means grid spans [-1, 1]^3
            resolution: grid resolution N (N^3 cells); auto if None
            smooth:     if True, apply analytic smoothing at edges/corners
                        (3D extension of 2D _sdf_box_smoothed approach)
            sharpness:  smoothing sharpness — currently unused in analytic
                        smooth (reserved for future face-blending normal)
            device:     torch device
            dtype:      torch dtype
        """
        if resolution is None:
            resolution = SDFGrid._auto_resolution(workspace, r_sphere)

        cell_size = 2.0 * workspace / (resolution - 1)
        print(f"[SDFGrid] workspace={workspace:.3f}m  r_sphere={r_sphere:.4f}m  "
              f"N={resolution}  cell_size={cell_size*1000:.2f}mm  "
              f"({cell_size/r_sphere*100:.1f}% of r_sphere)")

        bounds = workspace

        # Build grid on CPU — avoids OOM on CUDA for large grids
        cpu = torch.device('cpu')
        coords = torch.linspace(-bounds, bounds, resolution, dtype=dtype, device=cpu)

        gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing='ij')
        pts = torch.stack([gx, gy, gz], dim=-1)  # (N, N, N, 3)

        if smooth:
            phi = SDFGrid._smooth_cube_sdf(pts, half)
            print(f"[SDFGrid]   SDF mode: smooth(sharpness={sharpness})")
        else:
            d = pts.abs() - half
            outside = d.clamp(min=0.0)
            phi = outside.norm(dim=-1) + d.max(dim=-1).values.clamp(max=0.0)
            print(f"[SDFGrid]   SDF mode: sharp")

        phi = phi.to(device=device)
        return SDFGrid(phi_grid=phi, bounds=bounds, resolution=resolution)

    @staticmethod
    def _smooth_cube_sdf(pts: torch.Tensor, half: float) -> torch.Tensor:
        """
        Analytic smoothed cube SDF. Mirrors 2D _sdf_box_smoothed exactly:
          d_i = smooth_abs(p_i) - half
          phi = ||max(d, 0)|| + smooth_max(dx, dy, dz).clamp(max=0)

        Uses eps=1e-6 for smoothing (same as 2D).
        """
        eps = 1e-6

        def smooth_abs(x):
            return torch.sqrt(x * x + eps * eps)

        def smooth_relu(x):
            return 0.5 * (x + torch.sqrt(x * x + eps * eps))

        def smooth_max2(a, b):
            return 0.5 * (a + b + torch.sqrt((a - b) ** 2 + eps * eps))

        dx = smooth_abs(pts[..., 0]) - half
        dy = smooth_abs(pts[..., 1]) - half
        dz = smooth_abs(pts[..., 2]) - half

        ox = smooth_relu(dx)
        oy = smooth_relu(dy)
        oz = smooth_relu(dz)
        phi_outside = torch.sqrt(ox * ox + oy * oy + oz * oz + eps * eps)

        m = smooth_max2(smooth_max2(dx, dy), dz)
        phi_inside = -smooth_relu(-m)

        return phi_outside + phi_inside

    @staticmethod
    def from_mesh(
        mesh_path: str,
        r_sphere: float,
        workspace: float,
        resolution: int = None,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float64,
    ) -> 'SDFGrid':
        """
        Build SDF grid from mesh file (future extension for arbitrary objects).

        Requires: mesh_to_sdf or pysdf library.
        """
        raise NotImplementedError(
            "from_mesh() not yet implemented. "
            "Install mesh_to_sdf and implement mesh SDF sampling here."
        )

    def _interp(self, p_norm_f32: torch.Tensor) -> torch.Tensor:
        """
        Trilinear interpolation on the SDF grid.

        Entirely in float32 to satisfy grid_sample on CUDA.

        Args:
            p_norm_f32: (3,) normalized coords in [-1, 1], float32, graph-connected

        Returns:
            phi: scalar float32
        """
        phi_grid = self.phi_grid.unsqueeze(0).unsqueeze(0)  # (1,1,N,N,N) float32
        grid = p_norm_f32.reshape(1, 1, 1, 1, 3)            # (1,1,1,1,3) float32

        return F.grid_sample(
            phi_grid,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).squeeze()   # scalar float32

    def query(self, p_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query SDF value and gradient at a body-frame point (no autograd needed).

        Gradient via central differences — robust to CUDA float32 limitation.

        Args:
            p_local: (3,) point in cube body frame, any dtype

        Returns:
            phi:  scalar float64
            grad: (3,) gradient of phi in body frame, float64
        """
        p_f32 = p_local.detach().float()
        p_norm = p_f32 / self.bounds          # float32, [-1,1]

        phi = self._interp(p_norm).double()

        # Central difference in normalised coords; step = 2/(N-1)
        h = 2.0 / (self.resolution - 1)
        grad = torch.zeros(3, dtype=torch.float64, device=self.device)
        for i in range(3):
            pp = p_norm.clone(); pp[i] += h
            pm = p_norm.clone(); pm[i] -= h
            # chain rule: dp_local = dp_norm * bounds
            grad[i] = ((self._interp(pp) - self._interp(pm)) / (2.0 * h * self.bounds)).double()

        return phi.detach(), grad.detach()

    def query_differentiable(self, p_local: torch.Tensor) -> torch.Tensor:
        """
        Differentiable phi — autograd flows through p_local.

        Works for both float32 and float64 inputs.

        Strategy: scale p_local to [-1,1] while keeping the graph alive,
        then pass through grid_sample (float32).  The scale op is the only
        part that needs to be in the graph; grid_sample carries the rest.

        Args:
            p_local: (3,) in body frame, float32 or float64, requires_grad OK

        Returns:
            phi: scalar float32, differentiable w.r.t. p_local
        """
        # Normalise in whatever dtype p_local is — graph stays alive through
        # the division because bounds is a plain Python float, not a tensor.
        p_norm = p_local / self.bounds          # same dtype as p_local

        # grid_sample needs float32.  We call it with p_norm cast to float32
        # but keep p_norm in the graph by computing phi as a linear combination
        # of grid values weighted by the interpolation coefficients.
        # The trick: use p_norm (graph-connected) to compute the 8 corner
        # weights analytically, then dot with the 8 corner phi values.
        return self._trilinear_differentiable(p_norm)

    def _trilinear_differentiable(self, p_norm: torch.Tensor) -> torch.Tensor:
        """
        Trilinear interpolation with full autograd support for any dtype.

        Computes the 8-corner weighted sum explicitly, so the gradient
        w.r.t. p_norm (and hence p_local) is always non-zero and correct.

        Args:
            p_norm: (3,) in [-1, 1], any dtype, graph-connected

        Returns:
            phi: scalar, same dtype as p_norm
        """
        N   = self.resolution
        bnd = self.bounds

        # Convert normalised [-1,1] coords to grid indices [0, N-1]
        # idx = (p_norm + 1) / 2 * (N - 1)
        idx = (p_norm + 1.0) * 0.5 * (N - 1)   # (3,) differentiable

        # Floor indices (clamped to valid range)
        i0 = idx[0].detach().long().clamp(0, N - 2)
        j0 = idx[1].detach().long().clamp(0, N - 2)
        k0 = idx[2].detach().long().clamp(0, N - 2)
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1

        # Fractional offsets (differentiable)
        tx = idx[0] - i0.float().to(p_norm.dtype)
        ty = idx[1] - j0.float().to(p_norm.dtype)
        tz = idx[2] - k0.float().to(p_norm.dtype)

        # 8 corner values from phi_grid (float32, detached — only weights carry grad)
        g = self.phi_grid   # (N,N,N) float32
        c000 = g[k0, j0, i0].to(p_norm.dtype)
        c001 = g[k0, j0, i1].to(p_norm.dtype)
        c010 = g[k0, j1, i0].to(p_norm.dtype)
        c011 = g[k0, j1, i1].to(p_norm.dtype)
        c100 = g[k1, j0, i0].to(p_norm.dtype)
        c101 = g[k1, j0, i1].to(p_norm.dtype)
        c110 = g[k1, j1, i0].to(p_norm.dtype)
        c111 = g[k1, j1, i1].to(p_norm.dtype)

        # Trilinear interpolation (differentiable w.r.t. tx, ty, tz)
        c00 = c000 * (1 - tx) + c001 * tx
        c01 = c010 * (1 - tx) + c011 * tx
        c10 = c100 * (1 - tx) + c101 * tx
        c11 = c110 * (1 - tx) + c111 * tx

        c0  = c00  * (1 - ty) + c01  * ty
        c1  = c10  * (1 - ty) + c11  * ty

        phi = c0   * (1 - tz) + c1   * tz

        return phi


# ──────────────────────────────────────────────
# Contact result dataclass (mirrors OBBContact)
# ──────────────────────────────────────────────

@dataclass
class CubeContact3D:
    """
    Contact information between a cube and a sphere.

    Mirrors OBBContact from geometry.py for API consistency.
    """
    phi:      torch.Tensor   # signed distance (>0: no contact, =0: touching, <0: penetrating)
    cp_world: torch.Tensor   # (3,) closest point on cube surface in world frame
    normal:   torch.Tensor   # (3,) unit outward normal (cube -> sphere direction)
    tangent1: torch.Tensor   # (3,) unit tangent 1
    tangent2: torch.Tensor   # (3,) unit tangent 2
    r_cp:     torch.Tensor   # (3,) vector from cube center to contact point (world frame)


# ──────────────────────────────────────────────
# Runtime contact query
# ──────────────────────────────────────────────

def cube_sphere_contact(
    q_pose:   torch.Tensor,
    p_sphere: torch.Tensor,
    sdf_grid: SDFGrid,
    r_sphere: float,
) -> CubeContact3D:
    """
    Compute contact between a cube (with pose) and a sphere.

    This is the 3D analogue of obb_contact_blend2().

    Args:
        q_pose:   (7,) = [px, py, pz, qw, qx, qy, qz]  cube pose in world frame
        p_sphere: (3,) sphere center in world frame
        sdf_grid: precomputed SDFGrid (body frame)
        r_sphere: sphere radius

    Returns:
        CubeContact3D with phi, normal, tangents, contact point
    """
    p_cube = q_pose[:3]
    q_rot  = q_pose[3:]          # [qw, qx, qy, qz]

    R = quat_to_rot(q_rot)       # (3,3) body->world

    # Transform sphere center to cube body frame
    p_local = R.T @ (p_sphere - p_cube)   # (3,)

    # query_differentiable requires float32 input to keep autograd graph alive.
    # p_local may be float64 (dynamics uses double); cast here, not inside query.
    # query_differentiable accepts any dtype now — pass p_local directly
    phi_raw = sdf_grid.query_differentiable(p_local)

    # Signed distance accounting for sphere radius
    phi = phi_raw - r_sphere

    # Contact normal via central-diff (no autograd needed, more robust)
    _, grad_body = sdf_grid.query(p_local.detach())
    grad_norm = torch.linalg.norm(grad_body) + 1e-8
    n_body = grad_body / grad_norm   # (3,) float64, body frame

    # Transform to world frame — cast to match q_pose dtype
    dt = q_pose.dtype
    n_world = R.to(dt) @ n_body.to(dt)
    n_world = n_world / (torch.linalg.norm(n_world) + 1e-8)

    # Closest point on cube surface in world frame
    cp_world = p_sphere - phi_raw.detach().to(dt) * n_world
    r_cp = cp_world - p_cube

    # Flip normal to match 2D obb_contact_blend2 convention:
    # normal points sphere→cube (inward).
    # Without flip: n=[+1,0,0] for sphere at +x → force +x on cube (wrong)
    # With flip:    n=[-1,0,0] for sphere at +x → force -x on cube (correct)
    n_contact = -n_world
    t1_contact, t2_contact = tangent_frame(n_contact)

    return CubeContact3D(
        phi=phi,
        cp_world=cp_world,
        normal=n_contact,
        tangent1=t1_contact,
        tangent2=t2_contact,
        r_cp=r_cp,
    )


def cube_ground_contact(
    q_pose: torch.Tensor,
    half: float,
    ground_z: float = 0.0,
    sharpness: float = 80.0,
) -> CubeContact3D:
    """
    Analytic cube-vs-ground contact using a soft minimum over cube vertices.

    This avoids a second SDF.  The contact point is a differentiable weighted
    average of the lowest cube vertices, and the contact normal points upward
    so positive normal force pushes the cube away from the ground.
    """
    p_cube = q_pose[:3]
    q_rot = q_pose[3:]
    R = quat_to_rot(q_rot)
    h = q_pose.new_tensor(half)

    vals = [-1.0, 1.0]
    verts_local = torch.stack([
        torch.stack([h * q_pose.new_tensor(x),
                     h * q_pose.new_tensor(y),
                     h * q_pose.new_tensor(z)])
        for x in vals for y in vals for z in vals
    ])
    verts_world = p_cube + verts_local @ R.T
    z_vals = verts_world[:, 2]

    weights = torch.softmax(-q_pose.new_tensor(sharpness) * z_vals, dim=0)
    cp_world = torch.sum(weights[:, None] * verts_world, dim=0)
    phi = torch.dot(weights, z_vals) - q_pose.new_tensor(ground_z)

    normal = torch.tensor([0.0, 0.0, 1.0], dtype=q_pose.dtype, device=q_pose.device)
    tangent1 = torch.tensor([1.0, 0.0, 0.0], dtype=q_pose.dtype, device=q_pose.device)
    tangent2 = torch.tensor([0.0, 1.0, 0.0], dtype=q_pose.dtype, device=q_pose.device)
    r_cp = cp_world - p_cube

    return CubeContact3D(
        phi=phi,
        cp_world=cp_world,
        normal=normal,
        tangent1=tangent1,
        tangent2=tangent2,
        r_cp=r_cp,
    )


def sphere_ground_contact(
    p_sphere: torch.Tensor,
    r_sphere: float,
    ground_z: float = 0.0,
) -> CubeContact3D:
    """Analytic sphere-vs-ground contact with upward normal."""
    normal = torch.tensor([0.0, 0.0, 1.0], dtype=p_sphere.dtype, device=p_sphere.device)
    tangent1 = torch.tensor([1.0, 0.0, 0.0], dtype=p_sphere.dtype, device=p_sphere.device)
    tangent2 = torch.tensor([0.0, 1.0, 0.0], dtype=p_sphere.dtype, device=p_sphere.device)
    cp_world = torch.stack([
        p_sphere[0],
        p_sphere[1],
        p_sphere.new_tensor(ground_z),
    ])
    phi = p_sphere[2] - p_sphere.new_tensor(r_sphere + ground_z)
    r_cp = cp_world - p_sphere

    return CubeContact3D(
        phi=phi,
        cp_world=cp_world,
        normal=normal,
        tangent1=tangent1,
        tangent2=tangent2,
        r_cp=r_cp,
    )


# ──────────────────────────────────────────────
# Contact Jacobians (mirrors contact_jacobians)
# ──────────────────────────────────────────────

def contact_jacobians_3d(
    n:    torch.Tensor,
    t1:   torch.Tensor,
    t2:   torch.Tensor,
    r_cp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute contact Jacobians mapping body twist to contact velocities.

    Body twist: [vx, vy, vz, wx, wy, wz]  (linear + angular, world frame)
    Contact velocity components:
      v_n  = Jn @ twist   (normal direction)
      v_t1 = Jt1 @ twist  (tangent 1)
      v_t2 = Jt2 @ twist  (tangent 2)

    Derivation:
      velocity at contact point cp = v + omega x r_cp
      v_n = n^T (v + omega x r_cp)
          = n^T v + n^T (omega x r_cp)
          = n^T v + (r_cp x n)^T omega    [scalar triple product identity]

    Args:
        n:    (3,) unit normal
        t1:   (3,) unit tangent 1
        t2:   (3,) unit tangent 2
        r_cp: (3,) vector from body COM to contact point (world frame)

    Returns:
        Jn:  (6,) normal Jacobian
        Jt1: (6,) tangent-1 Jacobian
        Jt2: (6,) tangent-2 Jacobian
    """
    # r_cp x n, r_cp x t1, r_cp x t2
    rxn  = torch.linalg.cross(r_cp, n)
    rxt1 = torch.linalg.cross(r_cp, t1)
    rxt2 = torch.linalg.cross(r_cp, t2)

    Jn  = torch.cat([n,  rxn])   # (6,)
    Jt1 = torch.cat([t1, rxt1])  # (6,)
    Jt2 = torch.cat([t2, rxt2])  # (6,)

    return Jn, Jt1, Jt2
