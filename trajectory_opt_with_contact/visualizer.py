"""
Visualization utilities for trajectory optimization results.
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
    Visualizer for trajectory optimization results.
    
    Provides methods to:
    - Plot trajectories
    - Visualize contact forces
    - Create animations
    - Generate analysis plots
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
        Plot the optimized trajectory with analysis.
        
        Args:
            result: Optimization result dictionary
            goal: Goal configuration [x, y, theta]
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        traj = result['trajectory']
        pusher_traj = result['pusher_trajectory']
        forces = result['contact_forces']
        u_seq = result['u_seq']
        
        dt = 0.05  # Assume 0.05s timestep
        time = np.arange(len(forces)) * dt
        
        # Plot 1: XY Trajectory
        ax = axes[0, 0]
        ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1, label='Object', marker='o', markersize=3)
        ax.plot(pusher_traj[:, 0], pusher_traj[:, 1], 'b--', linewidth=1, label='Pusher', marker='s', markersize=3)
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=150, marker='o', label='Start', zorder=5)
        ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', label='Goal', zorder=5)
        
        
        # Draw start and goal squares
        start_rect = Rectangle((traj[0, 0]-self.half, traj[0, 1]-self.half), 
                               self.side, self.side, fc='green', alpha=0.3, ec='green', lw=1)
        ax.add_patch(start_rect)
        
        goal_rect = Rectangle((goal[0]-self.half, goal[1]-self.half), 
                              self.side, self.side, fc='gold', alpha=0.3, ec='gold', lw=2, ls='--')
        goal_transform = mtransforms.Affine2D().rotate_around(goal[0], goal[1], goal[2])
        goal_rect.set_transform(goal_transform + ax.transData)
        ax.add_patch(goal_rect)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Trajectory in XY Plane', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set limits AFTER adding all elements
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal', adjustable='box')
            print("✓ Axis limits set.")
        else:
            ax.axis('equal')
            print("✓ Axis set to equal aspect ratio.")
        
        
        # Plot 2: Contact Forces
        ax = axes[0, 1]
        ax.plot(time, forces[:, 0], 'b-', linewidth=2, label='Normal force (λ_n)')
        ax.plot(time, forces[:, 1], 'r-', linewidth=2, label='Tangential force (λ_t)')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Force (N)', fontsize=12)
        ax.set_title('Contact Forces', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Configuration vs Time
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
        
        # Plot 4: Control Inputs
        ax = axes[1, 1]
        ax.plot(time, u_seq[:, 0], 'b-', linewidth=2, label='u_x (pusher vel)')
        ax.plot(time, u_seq[:, 1], 'r-', linewidth=2, label='u_y (pusher vel)')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Control (m/s)', fontsize=12)
        ax.set_title('Control Inputs', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Trajectory plot saved to '{save_path}'")
        plt.show()
    
    def animate_trajectory(self, result, goal, save_path='animation.mp4', fps=60, 
                          obstacle_pos=None, xlim=(-1, 1), ylim=(-1, 1)):
        """
        Create an animation of the trajectory.
        
        Args:
            result: Optimization result dictionary
            goal: Goal configuration [x, y, theta]
            save_path: Path to save the animation
            fps: Frames per second
            obstacle_pos: Optional obstacle position [x, y]
            xlim: X-axis limits (default: (-1, 1))
            ylim: Y-axis limits (default: (-1, 1))
        """
        traj = result['trajectory']
        pusher_traj = result['pusher_trajectory']
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.set_title("Trajectory optimization of pushing a square", fontsize=14)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Create patches
        box = Rectangle((-self.half, -self.half), self.side, self.side, 
                       fc='C1', ec='k', lw=1.2, zorder=2)
        robot = Circle((pusher_traj[0, 0], pusher_traj[0, 1]), 0.008, 
                      fc='C0', ec='k', lw=1.2, zorder=3)
        
        # Add obstacle if provided
        if obstacle_pos is not None:
            obs = Circle(obstacle_pos, 0.025, fc='gray', ec='k', lw=1.2, zorder=1)
            ax.add_patch(obs)
        
        # Goal visualization
        goal_patch = Rectangle((goal[0]-self.half, goal[1]-self.half), 
                              self.side, self.side, 
                              fc='none', ec='g', lw=2.0, ls='--', zorder=1)
        rotpatch = mtransforms.Affine2D().rotate_around(goal[0], goal[1], goal[2])
        goal_patch.set_transform(rotpatch + ax.transData)
        
        ax.add_patch(box)
        ax.add_patch(robot)
        ax.add_patch(goal_patch)
        
        # Trajectory lines (will be updated during animation)
        traj_line = Line2D([], [], linestyle='-', linewidth=2, alpha=0.5, 
                          color='red', label='Object path')
        pusher_line = Line2D([], [], linestyle='--', linewidth=2, alpha=0.5, 
                            color='blue', label='Pusher path')
        ax.add_line(traj_line)
        ax.add_line(pusher_line)
        
        def animate(i):
            # Update object
            x, y, th = traj[i]
            t = mtransforms.Affine2D().rotate(th).translate(x, y)
            box.set_transform(t + ax.transData)
            
            # Update pusher
            robot.center = (pusher_traj[i, 0], pusher_traj[i, 1])
            
            # Update trajectory lines
            traj_line.set_data(traj[:i+1, 0], traj[:i+1, 1])
            pusher_line.set_data(pusher_traj[:i+1, 0], pusher_traj[:i+1, 1])
            
            return box, robot, traj_line, pusher_line
        
        print(f"Creating animation with {len(traj)} frames at {fps} fps...")
        ani = FuncAnimation(fig, animate, frames=len(traj), 
                          interval=1000/fps, blit=False, repeat=True)
        
        # Save animation
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2000)
            ani.save(save_path, writer=writer, dpi=200)
            print(f"✓ Animation saved to '{save_path}'")
        except Exception as e:
            print(f"✗ Failed to save animation: {e}")
            print("  Try: conda install ffmpeg")
        
        plt.close(fig)
        return ani
    
    def plot_analysis(self, result, save_path='analysis.png'):
        """
        Detailed analysis plots.
        
        Args:
            result: Optimization result dictionary
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        

        dt = 0.05
        

        traj = result['trajectory']
        forces = result['contact_forces']
        phis = result['signed_distances']
        u_seq = result['u_seq']
        
        # Prefer state velocity if provided; otherwise finite-diff
        if 'velocity_trajectory' in result and result['velocity_trajectory'] is not None:
            vel = result['velocity_trajectory']
            vel = np.asarray(vel)

            # Accept [T+1, 3] or [T+1, 2] etc. Use xy components if available.
            if vel.ndim == 2 and vel.shape[1] >= 2:
                vel_xy = vel[:, :2]
            else:
                raise ValueError(f"velocity_trajectory has unexpected shape: {vel.shape}")

            vel_mag = np.linalg.norm(vel_xy, axis=1)
            tVel = np.arange(len(vel_mag)) * dt
        else:
            vel = np.diff(traj[:, :2], axis=0) / dt  # length T
            vel_mag = np.linalg.norm(vel, axis=1)
            tVel = np.arange(len(vel_mag)) * dt


        time = np.arange(len(forces)) * dt
        
        # 1. Position error magnitude
        ax = axes[0, 0]
        pos_error = np.linalg.norm(traj[1:, :2] - traj[-1:, :2], axis=1)
        ax.plot(time, pos_error, 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Position Error to Final State')
        ax.grid(True, alpha=0.3)
        
        # 2. Force magnitude
        ax = axes[0, 1]
        force_mag = np.linalg.norm(forces, axis=1)
        ax.plot(time, force_mag, 'r-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force Magnitude (N)')
        ax.set_title('Contact Force Magnitude')
        ax.grid(True, alpha=0.3)
        
        # 3. Signed distance
        ax = axes[0, 2]
        ax.plot(time, phis, 'g-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signed Distance (m)')
        ax.set_title('Signed Distance (Contact Detection)')
        ax.grid(True, alpha=0.3)
        
        # 4. Control magnitude
        ax = axes[1, 0]
        control_mag = np.linalg.norm(u_seq, axis=1)
        ax.plot(time, control_mag, 'm-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Control Magnitude (m/s)')
        ax.set_title('Control Input Magnitude')
        ax.grid(True, alpha=0.3)
        
        # 5. Friction cone check
        ax = axes[1, 1]
        mu = 0.5  # Friction coefficient
        friction_ratio = np.abs(forces[:, 1]) / (forces[:, 0] + 1e-6)
        ax.plot(time, friction_ratio, 'c-', linewidth=2, label='|λ_t| / λ_n')
        ax.axhline(y=mu, color='r', linestyle='--', linewidth=2, label=f'μ = {mu}')
        ax.set_xlabel('Time (s)')# Set limits AFTER adding all elements
        
        ax.set_ylabel('Friction Ratio')
        ax.set_title('Friction Cone Constraint')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Energy-like quantity
        ax = axes[1, 2]
        # Kinetic energy proxy (velocity magnitude over time)
        # Compute velocity from position differences
        vel_mag = np.linalg.norm(vel, axis=1)
        ax.plot(tVel, vel_mag, 'orange', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Magnitude (m/s)')
        ax.set_title('Object Velocity')
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
    print("Generating Visualizations")
    print("="*60)
    
    viz.plot_trajectory(result, goal, xlim=xlim, ylim=ylim, save_path=save_trajectory)
    viz.plot_analysis(result, save_path=save_analysis)
    viz.animate_trajectory(result, goal, save_path=save_animation, 
                          fps=60, obstacle_pos=obstacle_pos,
                          xlim=xlim, ylim=ylim)
    
    print("="*60)
    print("✓ All visualizations complete!")
    print("="*60)