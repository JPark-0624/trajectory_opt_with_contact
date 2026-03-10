"""
Enhanced Visualization utilities for trajectory optimization results.

New features:
- 2x3 grid for trajectory plot with gradient and component-wise loss
- Side-by-side animation (initial vs optimal)
- Improved analysis plots with position/orientation error split
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.transforms as mtransforms


class TrajectoryVisualizer:
    """
    Enhanced visualizer for trajectory optimization results.
    """
    
    def __init__(self, half_size=0.1):
        """
        Initialize visualizer.
        
        Args:
            half_size: Half side length of the square object
        """
        self.half = half_size
        self.side = 2 * half_size
    
    def plot_trajectory(self, result, goal, xlim, ylim, save_path='trajectory.png'):
        """
        Plot the optimized trajectory with comprehensive analysis (2x3 grid).
        
        Args:
            result: Optimization result dictionary
            goal: Goal configuration [x, y, theta]
            xlim, ylim: Axis limits
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        traj = result['trajectory']
        pusher_traj = result['pusher_trajectory']
        forces = result['contact_forces']
        u_seq = result['u_seq']
        
        dt = 0.05  # Assume 0.05s timestep
        time = np.arange(len(forces)) * dt
        
        # ============================================================
        # Plot 1: XY Trajectory (with final pose)
        # ============================================================
        ax = axes[0, 0]
        ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1.5, label='Object', marker='o', markersize=3, alpha=0.8)
        ax.plot(pusher_traj[:, 0], pusher_traj[:, 1], 'b--', linewidth=1.5, label='Pusher', marker='s', markersize=3, alpha=0.6)
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=150, marker='o', label='Start', zorder=5)
        ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', label='Goal', zorder=5)
        
        # Draw start square
        start_rect = Rectangle((traj[0, 0]-self.half, traj[0, 1]-self.half), 
                               self.side, self.side, fc='green', alpha=0.3, ec='green', lw=1)
        ax.add_patch(start_rect)
        
        # Draw goal square
        goal_rect = Rectangle((goal[0]-self.half, goal[1]-self.half), 
                              self.side, self.side, fc='gold', alpha=0.3, ec='gold', lw=2, ls='--')
        goal_transform = mtransforms.Affine2D().rotate_around(goal[0], goal[1], goal[2])
        goal_rect.set_transform(goal_transform + ax.transData)
        ax.add_patch(goal_rect)
        
        # **NEW: Draw final pose square (semi-transparent, overlapping is OK)**
        final_x, final_y, final_theta = traj[-1]
        final_rect = Rectangle((final_x-self.half, final_y-self.half), 
                              self.side, self.side, fc='red', alpha=0.25, ec='red', lw=1.5, ls='-')
        final_transform = mtransforms.Affine2D().rotate_around(final_x, final_y, final_theta)
        final_rect.set_transform(final_transform + ax.transData)
        ax.add_patch(final_rect)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Trajectory in XY Plane', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.axis('equal')
        
        # ============================================================
        # Plot 2: Contact Forces
        # ============================================================
        ax = axes[0, 1]
        ax.plot(time, forces[:, 0], 'b-', linewidth=2, label='Normal force (λ_n)')
        ax.plot(time, forces[:, 1], 'r-', linewidth=2, label='Tangential force (λ_t)')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Force (N)', fontsize=12)
        ax.set_title('Contact Forces', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # ============================================================
        # Plot 3: **NEW** Control Gradients
        # ============================================================
        ax = axes[0, 2]
        if 'control_gradients' in result and result['control_gradients'] is not None:
            grads = result['control_gradients']
            ax.plot(time, grads[:, 0], 'b-', linewidth=2, label='∂L/∂u_x', alpha=0.7)
            ax.plot(time, grads[:, 1], 'r-', linewidth=2, label='∂L/∂u_y', alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Gradient', fontsize=12)
            ax.set_title('Loss Gradient w.r.t. Control', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('symlog', linthresh=1e-3)
        else:
            ax.text(0.5, 0.5, 'No gradient data\navailable', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Loss Gradient w.r.t. Control', fontsize=14, fontweight='bold')
        
        # ============================================================
        # Plot 4: Configuration vs Time
        # ============================================================
        ax = axes[1, 0]
        ax.plot(time, traj[1:, 0], 'b-', linewidth=2, label='x')
        ax.plot(time, traj[1:, 1], 'orange', linewidth=2, label='y')
        ax.plot(time, traj[1:, 2], 'g-', linewidth=2, label='θ')
        ax.axhline(y=goal[0], color='b', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=goal[1], color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=goal[2], color='g', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Configuration', fontsize=12)
        ax.set_title('Configuration vs Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # ============================================================
        # Plot 5: Control Inputs
        # ============================================================
        ax = axes[1, 1]
        ax.plot(time, u_seq[:, 0], 'b-', linewidth=2, label='u_x (pusher vel)')
        ax.plot(time, u_seq[:, 1], 'r-', linewidth=2, label='u_y (pusher vel)')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Control (m/s)', fontsize=12)
        ax.set_title('Control Inputs', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # ============================================================
        # Plot 6: **NEW** Component-wise Loss (Bar chart)
        # ============================================================
        ax = axes[1, 2]
        if 'loss_components' in result and result['loss_components'] is not None:
            comp = result['loss_components']
            labels = []
            values = []
            colors = []
            
            # Total (always first)
            labels.append('Total')
            values.append(comp.get('total', 0))
            colors.append('darkblue')
            
            # Soft cost components (WEIGHTED!)
            if 'control_energy' in comp and comp.get('w_control', 0) > 0:
                labels.append('Control')
                values.append(comp['control_energy'] * comp.get('w_control', 1.0))
                colors.append('steelblue')
            
            if 'control_smooth' in comp and comp.get('w_smooth', 0) > 0:
                labels.append('Smooth')
                values.append(comp['control_smooth'] * comp.get('w_smooth', 1.0))
                colors.append('skyblue')
            
            if 'obj_vel' in comp and comp.get('w_objvel', 0) > 0:
                labels.append('ObjVel')
                values.append(comp['obj_vel'] * comp.get('w_objvel', 1.0))
                colors.append('lightcoral')
            
            if 'target_xy' in comp and comp.get('w_targetxy', 0) > 0:
                labels.append('TargetXY')
                values.append(comp['target_xy'] * comp.get('w_targetxy', 1.0))
                colors.append('orange')
            
            if 'target_orient' in comp and comp.get('w_orient', 0) > 0:
                labels.append('Orient')
                values.append(comp['target_orient'] * comp.get('w_orient', 1.0))
                colors.append('gold')
            
            # ALM defect
            if 'alm_defect' in comp and comp.get('alm_defect', 0) > 0:
                labels.append('Defect')
                values.append(comp['alm_defect'])
                colors.append('darkred')
            
            bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.7, edgecolor='k')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Weighted Loss', fontsize=12)
            ax.set_title('Component-wise Loss Breakdown', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars (smaller font)
            for bar, val in zip(bars, values):
                if val > 0:  # Only show non-zero
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1e}', ha='center', va='bottom', fontsize=7, rotation=0)
        else:
            ax.text(0.5, 0.5, 'No loss component\ndata available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Component-wise Loss Breakdown', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Trajectory plot saved to '{save_path}'")
        plt.show()
    
    def animate_trajectory(self, result, goal, save_path='animation.mp4', fps=60, 
                          obstacle_pos=None, xlim=(-1, 1), ylim=(-1, 1)):
        """
        Create side-by-side animation: initial trajectory (left) vs optimal trajectory (right).
        
        Args:
            result: Optimization result dictionary
            goal: Goal configuration [x, y, theta]
            save_path: Path to save the animation
            fps: Frames per second
            obstacle_pos: Optional obstacle position [x, y]
            xlim, ylim: Axis limits
        """
        traj_opt = result['trajectory']
        pusher_opt = result['pusher_trajectory']
        
        # Get initial trajectory if available
        if 'initial_trajectory' in result and result['initial_trajectory'] is not None:
            traj_init = result['initial_trajectory']
            pusher_init = result.get('initial_pusher_trajectory', pusher_opt)  # fallback
        else:
            # If no initial data, just show optimal on both sides
            print("⚠ No initial trajectory data; showing optimal on both sides")
            traj_init = traj_opt
            pusher_init = pusher_opt
        
        # Create figure with 1 row, 2 columns
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
        
        for ax, title in zip([ax_left, ax_right], ['Initial Trajectory', 'Optimal Trajectory']):
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Goal visualization
            goal_patch = Rectangle((goal[0]-self.half, goal[1]-self.half), 
                                  self.side, self.side, 
                                  fc='none', ec='g', lw=2.0, ls='--', zorder=1)
            rotpatch = mtransforms.Affine2D().rotate_around(goal[0], goal[1], goal[2])
            goal_patch.set_transform(rotpatch + ax.transData)
            ax.add_patch(goal_patch)
            
            # Obstacle if provided
            if obstacle_pos is not None:
                obs = Circle(obstacle_pos, 0.025, fc='gray', ec='k', lw=1.2, zorder=1)
                ax.add_patch(obs)
        
        # Create dynamic patches for LEFT (initial)
        box_init = Rectangle((-self.half, -self.half), self.side, self.side, 
                            fc='lightblue', ec='k', lw=1.2, zorder=2, alpha=0.8)
        robot_init = Circle((pusher_init[0, 0], pusher_init[0, 1]), 0.008, 
                           fc='blue', ec='k', lw=1.2, zorder=3, alpha=0.8)
        ax_left.add_patch(box_init)
        ax_left.add_patch(robot_init)
        
        traj_line_init = Line2D([], [], linestyle='-', linewidth=2, alpha=0.5, 
                               color='lightblue', label='Object path')
        pusher_line_init = Line2D([], [], linestyle='--', linewidth=2, alpha=0.5, 
                                 color='blue', label='Pusher path')
        ax_left.add_line(traj_line_init)
        ax_left.add_line(pusher_line_init)
        
        # Create dynamic patches for RIGHT (optimal)
        box_opt = Rectangle((-self.half, -self.half), self.side, self.side, 
                           fc='C1', ec='k', lw=1.2, zorder=2)
        robot_opt = Circle((pusher_opt[0, 0], pusher_opt[0, 1]), 0.008, 
                          fc='C0', ec='k', lw=1.2, zorder=3)
        ax_right.add_patch(box_opt)
        ax_right.add_patch(robot_opt)
        
        traj_line_opt = Line2D([], [], linestyle='-', linewidth=2, alpha=0.5, 
                              color='red', label='Object path')
        pusher_line_opt = Line2D([], [], linestyle='--', linewidth=2, alpha=0.5, 
                                color='blue', label='Pusher path')
        ax_right.add_line(traj_line_opt)
        ax_right.add_line(pusher_line_opt)
        
        def animate(i):
            # Update LEFT (initial)
            x_init, y_init, th_init = traj_init[i]
            t_init = mtransforms.Affine2D().rotate(th_init).translate(x_init, y_init)
            box_init.set_transform(t_init + ax_left.transData)
            robot_init.center = (pusher_init[i, 0], pusher_init[i, 1])
            traj_line_init.set_data(traj_init[:i+1, 0], traj_init[:i+1, 1])
            pusher_line_init.set_data(pusher_init[:i+1, 0], pusher_init[:i+1, 1])
            
            # Update RIGHT (optimal)
            x_opt, y_opt, th_opt = traj_opt[i]
            t_opt = mtransforms.Affine2D().rotate(th_opt).translate(x_opt, y_opt)
            box_opt.set_transform(t_opt + ax_right.transData)
            robot_opt.center = (pusher_opt[i, 0], pusher_opt[i, 1])
            traj_line_opt.set_data(traj_opt[:i+1, 0], traj_opt[:i+1, 1])
            pusher_line_opt.set_data(pusher_opt[:i+1, 0], pusher_opt[:i+1, 1])
            
            return box_init, robot_init, traj_line_init, pusher_line_init, \
                   box_opt, robot_opt, traj_line_opt, pusher_line_opt
        
        print(f"Creating side-by-side animation with {len(traj_opt)} frames at {fps} fps...")
        ani = FuncAnimation(fig, animate, frames=len(traj_opt), 
                          interval=1000/fps, blit=False, repeat=True)
        
        # Save animation
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2000)
            ani.save(save_path, writer=writer, dpi=150)
            print(f"✓ Animation saved to '{save_path}'")
        except Exception as e:
            print(f"✗ Failed to save animation: {e}")
            print("  Try: conda install ffmpeg")
        
        plt.close(fig)
        return ani
    
    def plot_analysis(self, result, goal, save_path='analysis.png'):
        """
        Detailed analysis plots with improved error metrics.
        
        Args:
            result: Optimization result dictionary
            goal: Goal configuration [x, y, theta]
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        dt = 0.05
        
        traj = result['trajectory']
        forces = result['contact_forces']
        phis = result['signed_distances']
        u_seq = result['u_seq']
        
        # Prefer state velocity if provided
        if 'velocity_trajectory' in result and result['velocity_trajectory'] is not None:
            vel = result['velocity_trajectory']
            vel = np.asarray(vel)
            if vel.ndim == 2 and vel.shape[1] >= 2:
                vel_xy = vel[:, :2]
            else:
                raise ValueError(f"velocity_trajectory has unexpected shape: {vel.shape}")
            vel_mag = np.linalg.norm(vel_xy, axis=1)
            tVel = np.arange(len(vel_mag)) * dt
        else:
            vel = np.diff(traj[:, :2], axis=0) / dt
            vel_mag = np.linalg.norm(vel, axis=1)
            tVel = np.arange(len(vel_mag)) * dt

        time = np.arange(len(forces)) * dt
        
        # ============================================================
        # Plot 1: **IMPROVED** Position & Orientation Error to GOAL
        # ============================================================
        ax = axes[0, 0]
        pos_error = np.linalg.norm(traj[1:, :2] - goal[:2], axis=1)
        orient_error = np.abs(traj[1:, 2] - goal[2])
        
        ax.plot(time, pos_error, 'b-', linewidth=2, label='Position error (m)')
        ax_twin = ax.twinx()
        ax_twin.plot(time, orient_error, 'r-', linewidth=2, label='Orientation error (rad)')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Position Error (m)', fontsize=12, color='b')
        ax_twin.set_ylabel('Orientation Error (rad)', fontsize=12, color='r')
        ax.set_title('Tracking Error to Goal', fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='b')
        ax_twin.tick_params(axis='y', labelcolor='r')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # ============================================================
        # Plot 2: Force magnitude
        # ============================================================
        ax = axes[0, 1]
        force_mag = np.linalg.norm(forces, axis=1)
        ax.plot(time, force_mag, 'r-', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Force Magnitude (N)', fontsize=12)
        ax.set_title('Contact Force Magnitude', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # ============================================================
        # Plot 3: Signed distance
        # ============================================================
        ax = axes[0, 2]
        ax.plot(time, phis, 'g-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Signed Distance (m)', fontsize=12)
        ax.set_title('Signed Distance (Contact Detection)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # ============================================================
        # Plot 4: Control magnitude
        # ============================================================
        ax = axes[1, 0]
        control_mag = np.linalg.norm(u_seq, axis=1)
        ax.plot(time, control_mag, 'm-', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Control Magnitude (m/s)', fontsize=12)
        ax.set_title('Control Input Magnitude', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # ============================================================
        # Plot 5: Friction cone check
        # ============================================================
        ax = axes[1, 1]
        mu = 0.5
        friction_ratio = np.abs(forces[:, 1]) / (forces[:, 0] + 1e-6)
        ax.plot(time, friction_ratio, 'c-', linewidth=2, label='|λ_t| / λ_n')
        ax.axhline(y=mu, color='r', linestyle='--', linewidth=2, label=f'μ = {mu}')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Friction Ratio', fontsize=12)
        ax.set_title('Friction Cone Constraint', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ============================================================
        # Plot 6: Object velocity
        # ============================================================
        ax = axes[1, 2]
        ax.plot(tVel, vel_mag, 'orange', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Velocity Magnitude (m/s)', fontsize=12)
        ax.set_title('Object Velocity', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Analysis plot saved to '{save_path}'")
        plt.show()


def visualize_result(result, goal, half_size=0.1, 
                    save_trajectory='trajectory.png',
                    save_animation='animation.mp4',
                    save_analysis='analysis.png',
                    obstacle_pos=None,
                    xlim=(-1, 1),
                    ylim=(-1, 1)):
    """
    Convenience function to visualize optimization results.
    
    Args:
        result: Optimization result dictionary
        goal: Goal configuration [x, y, theta]
        half_size: Half side length of square
        save_trajectory: Path for trajectory plot
        save_animation: Path for animation
        save_analysis: Path for analysis plot
        obstacle_pos: Optional obstacle position [x, y]
        xlim: X-axis limits for animation
        ylim: Y-axis limits for animation
    """
    viz = TrajectoryVisualizer(half_size=half_size)
    
    print("\n" + "="*60)
    print("Generating Enhanced Visualizations")
    print("="*60)
    
    viz.plot_trajectory(result, goal, xlim=xlim, ylim=ylim, save_path=save_trajectory)
    viz.plot_analysis(result, goal, save_path=save_analysis)
    viz.animate_trajectory(result, goal, save_path=save_animation, 
                          fps=60, obstacle_pos=obstacle_pos,
                          xlim=xlim, ylim=ylim)
    
    print("="*60)
    print("✓ All enhanced visualizations complete!")
    print("="*60)