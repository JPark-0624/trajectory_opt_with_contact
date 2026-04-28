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

    Uses Duff et al. (2017) branchless method for numerical stability.

    Args:
        n: (3,) unit normal vector

    Returns:
        t1, t2: (3,) unit tangent vectors s.t. {n, t1, t2} right-handed
    """
    # Choose reference axis least aligned with n to avoid degeneracy
    sign = torch.where(n[2] >= 0,
                       torch.ones(1, dtype=n.dtype, device=n.device),
                       -torch.ones(1, dtype=n.dtype, device=n.device)).squeeze()
    a = -1.0 / (sign + n[2])
    b = n[0] * n[1] * a

    t1 = torch.stack([
        1.0 + sign * n[0]**2 * a,
        sign * b,
        -sign * n[0],
    ])
    t2 = torch.stack([
        b,
        sign + n[1]**2 * a,
        -n[1],
    ])

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
    def from_cube(
        half: float,
        resolution: int = 64,
        padding: float = 0.05,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float64,
    ) -> 'SDFGrid':
        """
        Build SDF grid from analytical cube SDF in body frame.

        The grid covers [-bounds, +bounds]^3 where bounds = half + padding,
        so there is a margin outside the cube surface for smooth gradients.

        Args:
            half:       cube half-side length
            resolution: grid resolution N (N^3 cells)
            padding:    extra margin beyond cube surface
            device:     torch device
            dtype:      torch dtype
        """
        bounds = half + padding
        coords = torch.linspace(-bounds, bounds, resolution, dtype=dtype, device=device)

        # Meshgrid: (N, N, N, 3)
        gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing='ij')
        pts = torch.stack([gx, gy, gz], dim=-1)  # (N, N, N, 3)

        # Analytical cube SDF (exact, no smoothing needed for offline grid)
        # d_i = |p_i| - half  per axis
        d = pts.abs() - half                              # (N, N, N, 3)
        outside = d.clamp(min=0.0)
        phi = outside.norm(dim=-1) + d.max(dim=-1).values.clamp(max=0.0)
        # phi > 0 outside, phi < 0 inside, phi = 0 on surface

        return SDFGrid(phi_grid=phi, bounds=bounds, resolution=resolution)

    @staticmethod
    def from_mesh(
        mesh_path: str,
        resolution: int = 64,
        padding: float = 0.05,
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

    t1_world, t2_world = tangent_frame(n_world)

    # Closest point on cube surface in world frame
    cp_world = p_sphere - phi_raw.detach().to(dt) * n_world
    r_cp = cp_world - p_cube

    return CubeContact3D(
        phi=phi,
        cp_world=cp_world,
        normal=n_world,
        tangent1=t1_world,
        tangent2=t2_world,
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