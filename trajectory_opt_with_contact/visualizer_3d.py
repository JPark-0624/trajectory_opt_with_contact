"""
3D visualization utilities for contact-gradient sanity tests.

The first target is the one-step 3D contact tests in tests/3D:
single side contact, vertical impact, and symmetric vertical lift.
"""

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation


class ContactVisualizer3D:
    """Diagnostic visualizer for one-step cube + two-sphere contact cases."""

    def __init__(self, half_size=0.1, sphere_radius=0.02, dt=0.01):
        self.half = float(half_size)
        self.r_sphere = float(sphere_radius)
        self.dt = float(dt)

    @staticmethod
    def _np(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _rot_from_quat(q):
        q = np.asarray(q, dtype=float)
        q = q / (np.linalg.norm(q) + 1e-12)
        qw, qx, qy, qz = q
        return np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ])

    def _cube_vertices(self, p, q):
        R = self._rot_from_quat(q)
        h = self.half
        local = np.array([
            [-h, -h, -h], [ h, -h, -h], [ h,  h, -h], [-h,  h, -h],
            [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h],
        ])
        return p + local @ R.T

    def _draw_cube(self, ax, p, q, color="0.55", alpha=0.18):
        verts = self._cube_vertices(p, q)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            ax.plot(*zip(verts[i], verts[j]), color=color, linewidth=1.2, alpha=0.95)

        faces = [
            [verts[i] for i in [0, 1, 2, 3]],
            [verts[i] for i in [4, 5, 6, 7]],
            [verts[i] for i in [0, 1, 5, 4]],
            [verts[i] for i in [2, 3, 7, 6]],
            [verts[i] for i in [1, 2, 6, 5]],
            [verts[i] for i in [0, 3, 7, 4]],
        ]
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        poly = Poly3DCollection(faces, facecolors=color, edgecolors="none", alpha=alpha)
        ax.add_collection3d(poly)

    def _draw_cube_xz(self, ax, p, q, color="0.55", linewidth=1.5, label=None):
        verts = self._cube_vertices(p, q)
        xz = verts[:, [0, 2]]
        center = xz.mean(axis=0)
        angles = np.arctan2(xz[:, 1] - center[1], xz[:, 0] - center[0])
        order = np.argsort(angles)

        hull = []
        for idx in order:
            point = xz[idx]
            if not any(np.linalg.norm(point - prev) < 1e-10 for prev in hull):
                hull.append(point)
        hull = np.asarray(hull)
        if len(hull) == 0:
            return
        hull = np.vstack([hull, hull[0]])
        ax.plot(hull[:, 0], hull[:, 1], color=color, linewidth=linewidth, label=label)

    def _draw_sphere(self, ax, center, color):
        ax.scatter(center[0], center[1], center[2], s=28, color="k", depthshade=True)
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 8)
        x = center[0] + self.r_sphere * np.outer(np.cos(u), np.sin(v))
        y = center[1] + self.r_sphere * np.outer(np.sin(u), np.sin(v))
        z = center[2] + self.r_sphere * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=0.55, linewidth=0, shade=True)

    def _draw_arrow(self, ax, start, vec, color, label=None, scale=1.0, linewidth=2.0):
        vec = np.asarray(vec, dtype=float) * scale
        if np.linalg.norm(vec) < 1e-12:
            return
        ax.quiver(
            start[0], start[1], start[2],
            vec[0], vec[1], vec[2],
            color=color,
            linewidth=linewidth,
            arrow_length_ratio=0.22,
            normalize=False,
            label=label,
        )

    def _set_equal_axes(self, ax, points, margin=0.08):
        pts = np.asarray(points, dtype=float)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = 0.5 * np.max(maxs - mins) + margin
        radius = max(radius, 0.18)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))

    def _contact_impulses(self, contact):
        n = self._np(contact["normal"])
        t1 = self._np(contact["tangent1"])
        t2 = self._np(contact["tangent2"])
        lam = float(self._np(contact["lamN"]))
        beta = self._np(contact["beta"])

        normal = self.dt * lam * n
        tangent = self.dt * ((beta[1] - beta[0]) * t1 + (beta[3] - beta[2]) * t2)
        total = normal + tangent
        return normal, tangent, total

    def plot_contact_case(self, case, save_path):
        """Save a one-step diagnostic figure for a contact-gradient case."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        pk = self._np(case["pk"])
        qk = self._np(case["qk"])
        vk = self._np(case["vk"])
        p1 = self._np(case["p1"])
        p2 = self._np(case["p2"])
        u1 = self._np(case["u1"])
        u2 = self._np(case["u2"])
        p_next = self._np(case["p_next"])
        q_next = self._np(case["q_next"])
        v_next = self._np(case["v_next"])
        goal_p = self._np(case.get("goal_p"))
        goal_q = self._np(case.get("goal_q"))

        contacts = case.get("contacts", [])
        gradients = case.get("gradients", {})

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3)
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_imp = fig.add_subplot(gs[0, 2])
        ax_force = fig.add_subplot(gs[1, 1])
        ax_grad = fig.add_subplot(gs[1, 2])

        # 3D scene
        self._draw_cube(ax3d, pk, qk, color="0.65", alpha=0.10)
        self._draw_cube(ax3d, p_next, q_next, color="tab:red", alpha=0.16)
        if goal_p is not None:
            if goal_q is None:
                goal_q = np.array([1.0, 0.0, 0.0, 0.0])
            self._draw_cube(ax3d, goal_p, goal_q, color="tab:green", alpha=0.08)
        self._draw_sphere(ax3d, p1, "tab:blue")
        self._draw_sphere(ax3d, p2, "tab:orange")
        self._draw_arrow(ax3d, pk, vk[:3], "black", "v initial", scale=0.06, linewidth=2.2)
        self._draw_arrow(ax3d, p_next, v_next[:3], "tab:red", "v next", scale=0.06, linewidth=2.2)
        self._draw_arrow(ax3d, p1, self.dt * u1, "tab:green", "u1 dt", scale=1.0, linewidth=1.7)
        self._draw_arrow(ax3d, p2, self.dt * u2, "tab:olive", "u2 dt", scale=1.0, linewidth=1.7)

        impulse_scale = case.get("impulse_scale", 8.0)
        scene_points = [pk, p_next, p1, p2]
        if goal_p is not None:
            scene_points.append(goal_p)
        impulse_rows = []
        for idx, contact in enumerate(contacts, start=1):
            cp = self._np(contact["cp_world"])
            normal_imp, tangent_imp, total_imp = self._contact_impulses(contact)
            scene_points.append(cp)
            impulse_rows.append((idx, cp, normal_imp, tangent_imp, total_imp))

        if impulse_scale is None:
            max_total = max([np.linalg.norm(row[4]) for row in impulse_rows] + [1e-12])
            impulse_scale = 0.10 / max_total

        for idx, cp, normal_imp, tangent_imp, total_imp in impulse_rows:
            self._draw_arrow(ax3d, cp, normal_imp, "crimson", f"N impulse c{idx}", scale=impulse_scale, linewidth=2.4)
            self._draw_arrow(ax3d, cp, tangent_imp, "purple", f"T impulse c{idx}", scale=impulse_scale, linewidth=1.7)
            self._draw_arrow(ax3d, cp, total_imp, "black", f"total impulse c{idx}", scale=impulse_scale, linewidth=3.0)

        self._set_equal_axes(ax3d, scene_points)
        ax3d.set_title(f"{case['name']} - 3D contact impulse")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.view_init(elev=22, azim=-55)

        # XZ projection
        self._draw_cube_xz(ax_xz, pk, qk, color="0.55", linewidth=1.4, label="cube k")
        self._draw_cube_xz(ax_xz, p_next, q_next, color="tab:red", linewidth=1.8, label="cube k+1")
        if goal_p is not None:
            self._draw_cube_xz(ax_xz, goal_p, goal_q, color="tab:green",
                               linewidth=1.8, label="goal")
        ax_xz.scatter([p1[0], p2[0]], [p1[2], p2[2]], c=["tab:blue", "tab:orange"], s=80)
        ax_xz.add_patch(plt.Circle((p1[0], p1[2]), self.r_sphere, fill=False,
                                   edgecolor="tab:blue", linewidth=1.8))
        ax_xz.add_patch(plt.Circle((p2[0], p2[2]), self.r_sphere, fill=False,
                                   edgecolor="tab:orange", linewidth=1.8))
        ax_xz.arrow(pk[0], pk[2], 0.06 * vk[0], 0.06 * vk[2], color="black", width=0.0015)
        ax_xz.arrow(p_next[0], p_next[2], 0.06 * v_next[0], 0.06 * v_next[2], color="tab:red", width=0.0015)
        for _, cp, _, _, total_imp in impulse_rows:
            ax_xz.scatter([cp[0]], [cp[2]], c="black", s=18, zorder=4)
            ax_xz.arrow(
                cp[0], cp[2],
                impulse_scale * total_imp[0], impulse_scale * total_imp[2],
                color="black",
                width=0.0012,
                length_includes_head=True,
                alpha=0.85,
            )
        ax_xz.set_title("XZ projection")
        ax_xz.set_xlabel("x")
        ax_xz.set_ylabel("z")
        ax_xz.axis("equal")
        ax_xz.grid(True, alpha=0.3)
        ax_xz.legend(fontsize=8)

        # Impulse components
        labels = []
        normal_vals = []
        tangent_vals = []
        total_vals = []
        for idx, _, normal_imp, tangent_imp, total_imp in impulse_rows:
            labels.append(f"c{idx}")
            normal_vals.append(np.linalg.norm(normal_imp))
            tangent_vals.append(np.linalg.norm(tangent_imp))
            total_vals.append(np.linalg.norm(total_imp))
        x = np.arange(len(labels))
        width = 0.25
        if labels:
            ax_imp.bar(x - width, normal_vals, width, label="normal")
            ax_imp.bar(x, tangent_vals, width, label="tangent")
            ax_imp.bar(x + width, total_vals, width, label="total")
            ax_imp.set_xticks(x)
            ax_imp.set_xticklabels(labels)
        ax_imp.set_title("Linear impulse magnitudes")
        ax_imp.set_ylabel("N*s")
        ax_imp.grid(True, axis="y", alpha=0.3)
        ax_imp.legend(fontsize=8)

        # Contact force and signed distance
        force_labels = []
        lam_vals = []
        phi_vals = []
        for idx, contact in enumerate(contacts, start=1):
            force_labels.append(f"c{idx}")
            lam_vals.append(float(self._np(contact["lamN"])))
            phi_vals.append(float(self._np(contact["phi"])))
        if force_labels:
            xx = np.arange(len(force_labels))
            ax_force.bar(xx - 0.18, lam_vals, 0.36, label="lamN")
            ax_force_t = ax_force.twinx()
            ax_force_t.bar(xx + 0.18, phi_vals, 0.36, color="tab:green", alpha=0.7, label="phi")
            ax_force.set_xticks(xx)
            ax_force.set_xticklabels(force_labels)
            ax_force_t.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            ax_force.set_ylabel("normal force")
            ax_force_t.set_ylabel("signed distance")
        ax_force.set_title("Contact force / gap")
        ax_force.grid(True, axis="y", alpha=0.3)

        # Gradients and state delta
        grad_names = list(gradients.keys())
        grad_vals = [float(gradients[k]) for k in grad_names]
        if grad_names:
            colors = ["tab:blue" if v >= 0 else "tab:red" for v in grad_vals]
            ax_grad.bar(np.arange(len(grad_names)), grad_vals, color=colors, alpha=0.8)
            ax_grad.axhline(0.0, color="k", linewidth=0.8)
            ax_grad.set_xticks(np.arange(len(grad_names)))
            ax_grad.set_xticklabels(grad_names, rotation=25, ha="right", fontsize=8)
        ax_grad.set_title("Gradient checks")
        ax_grad.grid(True, axis="y", alpha=0.3)

        fig.suptitle(
            f"{case['name']} | "
            f"p: {np.array2string(pk, precision=3)} -> {np.array2string(p_next, precision=3)}",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved 3D contact diagnostic: {save_path}")


def visualize_contact_case_3d(case, save_path, half_size=0.1, sphere_radius=0.02, dt=0.01):
    viz = ContactVisualizer3D(half_size=half_size, sphere_radius=sphere_radius, dt=dt)
    viz.plot_contact_case(case, save_path)


def animate_contact_rollout_3d(frames, save_path, half_size=0.1, sphere_radius=0.02,
                               dt=0.01, fps=15, impulse_scale=None,
                               ground_impulse_scale=None,
                               title="3D contact rollout",
                               goal_p=None, goal_q=None):
    """
    Save a multi-step cube + two-sphere contact rollout animation.

    Each frame is a dict with:
      p, q, p1, p2, v, contacts
    where contacts follow the same schema as plot_contact_case().
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    viz = ContactVisualizer3D(half_size=half_size, sphere_radius=sphere_radius, dt=dt)
    frames_np = []
    goal_p_np = viz._np(goal_p)
    goal_q_np = viz._np(goal_q)
    if goal_p_np is not None and goal_q_np is None:
        goal_q_np = np.array([1.0, 0.0, 0.0, 0.0])

    for frame in frames:
        contacts = frame.get("contacts", [])
        ground_contacts = frame.get("ground_contacts", [])
        impulse_rows = []
        for contact in contacts:
            cp = viz._np(contact["cp_world"])
            normal_imp, tangent_imp, total_imp = viz._contact_impulses(contact)
            impulse_rows.append((cp, normal_imp, tangent_imp, total_imp))
        ground_impulse_rows = []
        for contact in ground_contacts:
            cp = viz._np(contact["cp_world"])
            normal = viz._np(contact["normal"])
            lam = float(viz._np(contact["lamN"]))
            normal_imp = dt * lam * normal
            ground_impulse_rows.append((cp, normal_imp))

        frames_np.append({
            "p": viz._np(frame["p"]),
            "q": viz._np(frame["q"]),
            "p1": viz._np(frame["p1"]),
            "p2": viz._np(frame["p2"]),
            "v": viz._np(frame["v"]),
            "goal_p": viz._np(frame.get("goal_p")) if frame.get("goal_p") is not None else goal_p_np,
            "goal_q": viz._np(frame.get("goal_q")) if frame.get("goal_q") is not None else goal_q_np,
            "contacts": contacts,
            "ground_contacts": ground_contacts,
            "impulses": impulse_rows,
            "ground_impulses": ground_impulse_rows,
            "lam_sum": sum(float(viz._np(c["lamN"])) for c in contacts),
            "ground_lam_sum": sum(float(viz._np(c["lamN"])) for c in ground_contacts),
        })

    all_points = []
    all_impulses = []
    all_ground_impulses = []
    for frame in frames_np:
        all_points.extend([frame["p"], frame["p1"], frame["p2"]])
        if frame["goal_p"] is not None:
            all_points.append(frame["goal_p"])
        for cp, _, _, total_imp in frame["impulses"]:
            all_points.append(cp)
            all_impulses.append(np.linalg.norm(total_imp))
        for cp, normal_imp in frame["ground_impulses"]:
            all_points.append(cp)
            all_ground_impulses.append(np.linalg.norm(normal_imp))

    if impulse_scale is None:
        max_imp = max(all_impulses + [1e-12])
        impulse_scale = 0.14 / max_imp
    if ground_impulse_scale is None:
        max_ground_imp = max(all_ground_impulses + [1e-12])
        ground_impulse_scale = 0.10 / max_ground_imp

    pz_hist = np.array([frame["p"][2] for frame in frames_np])
    lam_hist = np.array([frame["lam_sum"] for frame in frames_np])
    ground_lam_hist = np.array([frame["ground_lam_sum"] for frame in frames_np])
    t_hist = np.arange(len(frames_np)) * dt

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 1])

    def draw_ground(ax):
        pts = np.asarray(all_points)
        xmin, xmax = pts[:, 0].min() - 0.12, pts[:, 0].max() + 0.12
        ymin, ymax = pts[:, 1].min() - 0.12, pts[:, 1].max() + 0.12
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(ymin, ymax, 2))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, color="0.82", alpha=0.25, linewidth=0)

    def draw_frame(i):
        frame = frames_np[i]
        ax3d.clear()
        ax_xz.clear()
        ax_hist.clear()

        p = frame["p"]
        q = frame["q"]
        p1 = frame["p1"]
        p2 = frame["p2"]
        v = frame["v"]
        goal_p_frame = frame["goal_p"]
        goal_q_frame = frame["goal_q"]

        if frame["ground_impulses"]:
            draw_ground(ax3d)
        viz._draw_cube(ax3d, p, q, color="tab:red", alpha=0.18)
        if goal_p_frame is not None:
            viz._draw_cube(ax3d, goal_p_frame, goal_q_frame,
                           color="tab:green", alpha=0.08)
        viz._draw_sphere(ax3d, p1, "tab:blue")
        viz._draw_sphere(ax3d, p2, "tab:orange")
        viz._draw_arrow(ax3d, p, v[:3], "tab:red", "cube velocity", scale=0.035, linewidth=2.0)

        for idx, (cp, normal_imp, tangent_imp, total_imp) in enumerate(frame["impulses"], start=1):
            viz._draw_arrow(ax3d, cp, normal_imp, "crimson", f"N c{idx}", scale=impulse_scale, linewidth=2.0)
            viz._draw_arrow(ax3d, cp, tangent_imp, "purple", f"T c{idx}", scale=impulse_scale, linewidth=1.6)
            viz._draw_arrow(ax3d, cp, total_imp, "black", f"total c{idx}", scale=impulse_scale, linewidth=2.6)
        for idx, (cp, normal_imp) in enumerate(frame["ground_impulses"], start=1):
            viz._draw_arrow(ax3d, cp, normal_imp, "forestgreen", f"ground c{idx}",
                            scale=ground_impulse_scale, linewidth=2.8)

        viz._set_equal_axes(ax3d, all_points, margin=0.10)
        ax3d.set_title(f"{title} | step {i:02d}/{len(frames_np)-1}")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.view_init(elev=20, azim=-55)

        viz._draw_cube_xz(ax_xz, p, q, color="tab:red", linewidth=1.8)
        if frame["ground_impulses"]:
            ax_xz.axhline(0.0, color="0.35", linewidth=1.2)
        if goal_p_frame is not None:
            viz._draw_cube_xz(ax_xz, goal_p_frame, goal_q_frame,
                              color="tab:green", linewidth=1.8, label="goal")
        ax_xz.scatter([p1[0], p2[0]], [p1[2], p2[2]], c=["tab:blue", "tab:orange"], s=80)
        ax_xz.add_patch(plt.Circle((p1[0], p1[2]), sphere_radius, fill=False,
                                   edgecolor="tab:blue", linewidth=1.8))
        ax_xz.add_patch(plt.Circle((p2[0], p2[2]), sphere_radius, fill=False,
                                   edgecolor="tab:orange", linewidth=1.8))
        ax_xz.arrow(p[0], p[2], 0.035 * v[0], 0.035 * v[2],
                    color="tab:red", width=0.0015, length_includes_head=True)
        for cp, _, _, total_imp in frame["impulses"]:
            ax_xz.scatter([cp[0]], [cp[2]], c="black", s=18, zorder=4)
            ax_xz.arrow(cp[0], cp[2],
                        impulse_scale * total_imp[0], impulse_scale * total_imp[2],
                        color="black", width=0.0012, length_includes_head=True, alpha=0.85)
        for cp, normal_imp in frame["ground_impulses"]:
            ax_xz.scatter([cp[0]], [cp[2]], c="forestgreen", s=22, zorder=5)
            ax_xz.arrow(cp[0], cp[2],
                        ground_impulse_scale * normal_imp[0],
                        ground_impulse_scale * normal_imp[2],
                        color="forestgreen", width=0.0014,
                        length_includes_head=True, alpha=0.9)
        ax_xz.set_title("XZ projection")
        ax_xz.set_xlabel("x")
        ax_xz.set_ylabel("z")
        ax_xz.axis("equal")
        ax_xz.grid(True, alpha=0.3)

        ax_hist.plot(t_hist, pz_hist, color="tab:red", linewidth=2.0, label="cube z")
        ax_hist.scatter([t_hist[i]], [pz_hist[i]], color="tab:red", s=35)
        ax_force = ax_hist.twinx()
        ax_force.plot(t_hist, lam_hist, color="tab:blue", linewidth=1.6, alpha=0.75, label="sphere lamN")
        if np.max(np.abs(ground_lam_hist)) > 0.0:
            ax_force.plot(t_hist, ground_lam_hist, color="forestgreen", linewidth=1.6,
                          alpha=0.75, label="ground lamN")
        ax_force.scatter([t_hist[i]], [lam_hist[i]], color="tab:blue", s=25)
        if np.max(np.abs(ground_lam_hist)) > 0.0:
            ax_force.scatter([t_hist[i]], [ground_lam_hist[i]], color="forestgreen", s=25)
        ax_hist.set_title("Lift height and contact force")
        ax_hist.set_xlabel("time [s]")
        ax_hist.set_ylabel("cube z [m]")
        ax_force.set_ylabel("sum normal force")
        ax_hist.grid(True, alpha=0.3)
        ax_force.legend(loc="upper right", fontsize=8)

        fig.tight_layout()

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(frames_np), interval=1000 / fps)

    try:
        if save_path.suffix.lower() == ".mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        else:
            writer = animation.PillowWriter(fps=fps)
        ani.save(save_path, writer=writer)
        saved_path = save_path
    except Exception as exc:
        fallback = save_path.with_suffix(".gif")
        print(f"Could not save {save_path} ({exc}); falling back to {fallback}")
        ani.save(fallback, writer=animation.PillowWriter(fps=fps))
        saved_path = fallback
    finally:
        plt.close(fig)

    print(f"✓ Saved 3D contact rollout animation: {saved_path}")
    return saved_path


def animate_ground_rollout_3d(frames, save_path, half_size=0.1, sphere_radius=0.02,
                              dt=0.01, fps=15, impulse_scale=None,
                              title="3D ground rollout"):
    """
    Save an animation for analytic ground-contact tests.

    Each frame is a dict with:
      kind: "cube" or "sphere"
      p, v, lamN, phi
      q for cube frames
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    viz = ContactVisualizer3D(half_size=half_size, sphere_radius=sphere_radius, dt=dt)
    frames_np = []
    for frame in frames:
        item = {
            "kind": frame["kind"],
            "p": viz._np(frame["p"]),
            "v": viz._np(frame["v"]),
            "lamN": float(viz._np(frame["lamN"])),
            "phi": float(viz._np(frame["phi"])),
        }
        if frame["kind"] == "cube":
            item["q"] = viz._np(frame["q"])
        frames_np.append(item)

    all_points = []
    all_lam = []
    for frame in frames_np:
        all_points.append(frame["p"])
        all_lam.append(abs(frame["lamN"]))
        if frame["kind"] == "cube":
            all_points.extend(viz._cube_vertices(frame["p"], frame["q"]))
        else:
            all_points.append(frame["p"] + np.array([0.0, 0.0, -sphere_radius]))

    if impulse_scale is None:
        impulse_scale = 0.08 / max(all_lam + [1e-12])

    z_hist = np.array([frame["p"][2] for frame in frames_np])
    vx_hist = np.array([frame["v"][0] for frame in frames_np])
    phi_hist = np.array([frame["phi"] for frame in frames_np])
    lam_hist = np.array([frame["lamN"] for frame in frames_np])
    t_hist = np.arange(len(frames_np)) * dt

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 1])

    def draw_ground(ax):
        pts = np.asarray(all_points)
        xmin, xmax = pts[:, 0].min() - 0.12, pts[:, 0].max() + 0.12
        ymin, ymax = pts[:, 1].min() - 0.12, pts[:, 1].max() + 0.12
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(ymin, ymax, 2))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, color="0.85", alpha=0.35, linewidth=0)

    def draw_frame(i):
        frame = frames_np[i]
        ax3d.clear()
        ax_xz.clear()
        ax_hist.clear()

        p = frame["p"]
        v = frame["v"]
        lamN = frame["lamN"]
        phi = frame["phi"]

        draw_ground(ax3d)
        if frame["kind"] == "cube":
            q = frame["q"]
            viz._draw_cube(ax3d, p, q, color="tab:red", alpha=0.18)
            viz._draw_cube_xz(ax_xz, p, q, color="tab:red", linewidth=1.8)
            contact_xz = p[0], p[2] - half_size
        else:
            viz._draw_sphere(ax3d, p, "tab:blue")
            ax_xz.scatter([p[0]], [p[2]], c=["tab:blue"], s=80)
            ax_xz.add_patch(plt.Circle((p[0], p[2]), sphere_radius, fill=False,
                                       edgecolor="tab:blue", linewidth=1.8))
            contact_xz = p[0], p[2] - sphere_radius

        contact_point = np.array([contact_xz[0], 0.0, 0.0])
        normal_imp = np.array([0.0, 0.0, dt * lamN])
        viz._draw_arrow(ax3d, contact_point, normal_imp, "black",
                        "ground impulse", scale=impulse_scale, linewidth=2.4)
        viz._draw_arrow(ax3d, p, v[:3], "tab:green",
                        "velocity", scale=0.035, linewidth=2.0)

        viz._set_equal_axes(ax3d, all_points, margin=0.12)
        ax3d.set_title(f"{title} | step {i:02d}/{len(frames_np)-1}")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.view_init(elev=20, azim=-55)

        ax_xz.axhline(0.0, color="0.35", linewidth=1.2)
        ax_xz.arrow(p[0], p[2], 0.035 * v[0], 0.035 * v[2],
                    color="tab:green", width=0.0015, length_includes_head=True)
        ax_xz.arrow(contact_xz[0], 0.0, 0.0, impulse_scale * normal_imp[2],
                    color="black", width=0.0012, length_includes_head=True, alpha=0.85)
        ax_xz.set_title(f"XZ projection | phi={phi:.2e}, lam={lamN:.2e}")
        ax_xz.set_xlabel("x")
        ax_xz.set_ylabel("z")
        ax_xz.axis("equal")
        ax_xz.grid(True, alpha=0.3)

        ax_hist.plot(t_hist, z_hist, color="tab:red", linewidth=2.0, label="z")
        ax_hist.plot(t_hist, vx_hist, color="tab:green", linewidth=1.6, label="vx")
        ax_hist.scatter([t_hist[i]], [z_hist[i]], color="tab:red", s=30)
        ax_force = ax_hist.twinx()
        ax_force.plot(t_hist, lam_hist, color="tab:blue", linewidth=1.4, alpha=0.75, label="lamN")
        ax_force.scatter([t_hist[i]], [lam_hist[i]], color="tab:blue", s=22)
        ax_hist.set_title("Height / horizontal speed / normal force")
        ax_hist.set_xlabel("time [s]")
        ax_hist.set_ylabel("z, vx")
        ax_force.set_ylabel("normal force")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(loc="upper left", fontsize=8)

        fig.tight_layout()

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(frames_np), interval=1000 / fps)

    try:
        if save_path.suffix.lower() == ".mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        else:
            writer = animation.PillowWriter(fps=fps)
        ani.save(save_path, writer=writer)
        saved_path = save_path
    except Exception as exc:
        fallback = save_path.with_suffix(".gif")
        print(f"Could not save {save_path} ({exc}); falling back to {fallback}")
        ani.save(fallback, writer=animation.PillowWriter(fps=fps))
        saved_path = fallback
    finally:
        plt.close(fig)

    print(f"✓ Saved 3D ground rollout animation: {saved_path}")
    return saved_path
