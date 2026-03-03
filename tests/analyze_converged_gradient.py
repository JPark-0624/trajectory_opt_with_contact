"""
Converged Gradient Analysis

목적: Optimized solution의 gradient pattern 분석
     - Loss component별 기여도
     - Timestep별 중요도
     - target_mu별 비교
"""

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from trajectory_opt_with_contact.dynamics import rollout, IPMOptions
from trajectory_opt_with_contact import TrajectoryOptimizer

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME = time.time()


# Setup
Q0 = [0.0, 0.0, 0.0]
V0 = [0.0, 0.0, 0.0]
PUSHER0 = [-0.1, -0.3]
GOAL = [0.2, 0.5, -0.3]

W_TARGET = 20.0
W_ORIENT = 0.5
W_V = 1.0
W_CTRL = 0.01


def optimize_to_convergence(optimizer, target_mu, max_iters=300, tol=1e-4):
    """
    Optimize until convergence.
    
    Returns:
        u_star: Optimized control
        loss_history: Loss over iterations
        converged: Whether converged
    """
    print(f"\n{'─'*80}")
    print(f"Optimizing with target_mu = {target_mu:.0e}")
    print(f"{'─'*80}")
    
    # Initial guess
    u_init_np = optimizer.compute_geometric_initial_trajectory(
        robot_pos=PUSHER0, box_pos=Q0, goal_pos=GOAL,
    )
    u = torch.tensor(u_init_np, dtype=DTYPE, device=DEVICE, requires_grad=True)
    
    # Optimizer
    opt = torch.optim.Adam([u], lr=0.01)
    
    # Scheduler: Reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 
        mode='min',           # Minimize loss
        factor=0.5,           # Reduce LR by half
        patience=10,          # Wait 10 iterations
        threshold=1e-4,       # Significant improvement threshold
        min_lr=1e-5,          # Don't go below this
    )
    
    loss_history = []
    prev_loss = float('inf')
    
    for iter in range(max_iters):
        opt.zero_grad()
        
        # Rollout
        q0 = torch.tensor(Q0, dtype=DTYPE, device=DEVICE)
        v0 = torch.tensor(V0, dtype=DTYPE, device=DEVICE)
        pusher0 = torch.tensor(PUSHER0, dtype=DTYPE, device=DEVICE)
        goal = torch.tensor(GOAL, dtype=DTYPE, device=DEVICE)
        
        loss, *_ = rollout(
            u_seq=u, q0=q0, v0=v0, pr0=pusher0,
            horizon=optimizer.horizon, h=optimizer.dt, m=optimizer.m, Izz=optimizer.Izz,
            half=optimizer.half, mu=optimizer.mu, goal_xy=goal,
            w_target=W_TARGET, w_orient=W_ORIENT, w_v=W_V, w_ctrl=W_CTRL, w_obs=0.0,
            qp_solver=None, dynamics_solver='IP',
            obstacle_pos=None, device=DEVICE,
            ipm_opts=IPMOptions(
                    target_mu=target_mu,
                    max_newton=20,
                    tol=1e-6,
                    smooth_sdf=50.0,
                    enable_viscous_ground_friction=True,
                    c_lin=1.0,
                    c_ang=0.00667,
                ),
        )
        
        loss.backward()
        opt.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # Update learning rate based on loss
        scheduler.step(loss_val)
        grad_norm = u.grad.norm().item()

        # Check convergence
        if iter % 10 == 0:
            current_lr = opt.param_groups[0]['lr']
            print(f"  Iter {iter:3d}: Loss = {loss_val:.6f}, ||∇|| = {grad_norm:.3e}, LR = {current_lr:.2e}")
        
        # More tolerant convergence check
        if abs(prev_loss - loss_val) < tol or grad_norm < 1e-3:
            print(f"\n  ✓ Converged at iteration {iter} (loss={loss_val:.6f}), grad)_norm={grad_norm:.3e}")
            converged = True
            break
        
        prev_loss = loss_val
    else:
        print(f"\n  ⚠ Max iterations reached")
        converged = False
    
    return u.detach().clone(), loss_history, converged


def compute_fd_gradient(u_opt, target_mu, eps=1e-2):
    """Compute FD gradient at converged solution."""
    q0 = torch.tensor(Q0, dtype=DTYPE, device=DEVICE)
    v0 = torch.tensor(V0, dtype=DTYPE, device=DEVICE)
    pusher0 = torch.tensor(PUSHER0, dtype=DTYPE, device=DEVICE)
    goal = torch.tensor(GOAL, dtype=DTYPE, device=DEVICE)
    
    horizon, dim = u_opt.shape
    grad_fd = torch.zeros_like(u_opt)
    
    print(f"    FD: ", end='', flush=True)
    for t in range(horizon):
        if t % 5 == 0:
            print(f"{t}...", end='', flush=True)
        for d in range(dim):
            u_plus = u_opt.clone()
            u_plus[t, d] += eps
            loss_plus, *_ = rollout(
                u_seq=u_plus, q0=q0, v0=v0, pr0=pusher0,
                horizon=horizon, h=0.05, m=1.0, Izz=0.00667,
                half=0.1, mu=0.5, goal_xy=goal,
                w_target=W_TARGET, w_orient=W_ORIENT, w_v=W_V, w_ctrl=W_CTRL, w_obs=0.0,
                qp_solver=None, dynamics_solver='IP',
                obstacle_pos=None, device=DEVICE,
                ipm_opts=IPMOptions(
                    target_mu=target_mu, max_newton=20, tol=1e-6,
                    smooth_sdf=50.0, enable_viscous_ground_friction=True,
                    c_lin=1.0, c_ang=0.00667,
                ),
            )
            
            u_minus = u_opt.clone()
            u_minus[t, d] -= eps
            loss_minus, *_ = rollout(
                u_seq=u_minus, q0=q0, v0=v0, pr0=pusher0,
                horizon=horizon, h=0.05, m=1.0, Izz=0.00667,
                half=0.1, mu=0.5, goal_xy=goal,
                w_target=W_TARGET, w_orient=W_ORIENT, w_v=W_V, w_ctrl=W_CTRL, w_obs=0.0,
                qp_solver=None, dynamics_solver='IP',
                obstacle_pos=None, device=DEVICE,
                ipm_opts=IPMOptions(
                    target_mu=target_mu, max_newton=20, tol=1e-6,
                    smooth_sdf=50.0, enable_viscous_ground_friction=True,
                    c_lin=1.0, c_ang=0.00667,
                ),
            )
            
            grad_fd[t, d] = (loss_plus.item() - loss_minus.item()) / (2 * eps)
    
    print("Done")
    return grad_fd


def compute_component_gradients(u_opt, target_mu, tol=1e-4):
    """
    Compute gradient contribution from each loss component.
    Also compute FD gradient and extract trajectory data.
    """
    q0 = torch.tensor(Q0, dtype=DTYPE, device=DEVICE)
    v0 = torch.tensor(V0, dtype=DTYPE, device=DEVICE)
    pusher0 = torch.tensor(PUSHER0, dtype=DTYPE, device=DEVICE)
    goal = torch.tensor(GOAL, dtype=DTYPE, device=DEVICE)
    
    horizon = u_opt.shape[0]
    
    # Helper to compute gradient for one component
    def grad_component(w_target, w_orient, w_v, w_ctrl, return_result=False):
        u_var = u_opt.clone().requires_grad_(True)
        
        result = rollout(
            u_seq=u_var, q0=q0, v0=v0, pr0=pusher0,
            horizon=optimizer.horizon, h=optimizer.dt, m=optimizer.m, Izz=optimizer.Izz,
            half=optimizer.half, mu=optimizer.mu, goal_xy=goal,
            w_target=w_target, w_orient=w_orient, w_v=w_v, w_ctrl=w_ctrl, w_obs=0.0,
            qp_solver=None, dynamics_solver='IP',
            obstacle_pos=None, device=DEVICE,
            ipm_opts=IPMOptions(
                    target_mu=target_mu,
                    max_newton=20,
                    tol=1e-6,
                    smooth_sdf=50.0,
                    enable_viscous_ground_friction=True,
                    c_lin=1.0,
                    c_ang=0.00667,
                ),
        )
        
        loss = result[0]
        loss.backward()
        
        if return_result:
            return u_var.grad.clone(), result
        return u_var.grad.clone()
    
    # Compute each component
    print("  Computing component gradients...")
    grad_goal = grad_component(W_TARGET, 0, 0, 0)
    grad_orient = grad_component(0, W_ORIENT, 0, 0)
    grad_v = grad_component(0, 0, W_V, 0)
    grad_ctrl = grad_component(0, 0, 0, W_CTRL)
    grad_total, result = grad_component(W_TARGET, W_ORIENT, W_V, W_CTRL, return_result=True)
    
    # Compute FD gradient
    print("  Computing FD gradient...")
    grad_fd = compute_fd_gradient(u_opt, target_mu, eps=1e-4)
    
    # FD vs IFT metrics
    diff = (grad_total - grad_fd).abs()
    mean_err = diff.mean().item()
    max_err = diff.max().item()
    corr = torch.corrcoef(torch.stack([
        grad_total.flatten(), grad_fd.flatten()
    ]))[0, 1].item()
    
    print(f"  FD-IFT: err={mean_err:.3e}, corr={corr:.3f}")
    
    # Extract trajectory
    qs = result[4]  # qs
    pusher_traj = result[5]  # pusher_traj
    
    return {
        'goal': grad_goal,
        'orient': grad_orient,
        'v': grad_v,
        'ctrl': grad_ctrl,
        'total': grad_total,
        'fd': grad_fd,
        'fd_metrics': {'mean_err': mean_err, 'max_err': max_err, 'correlation': corr},
        'qs': qs,
        'pusher_traj': pusher_traj,
    }


def visualize_converged_gradients(results, save_path="./result/dynamics/converged_gradient_analysis.png"):
    """
    Visualize converged gradient patterns with FD comparison.
    
    Layout:
      Row 1: Loss convergence curves
      Row 2: Component gradient magnitudes (stacked bar)
      Row 3: IFT vs FD comparison (scatter plot)
    """

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    n_mu = len(results)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, n_mu, hspace=0.4, wspace=0.3)
    
    for col, result in enumerate(results):
        target_mu = result['target_mu']
        loss_history = result['loss_history']
        grads = result['component_gradients']
        fd_metrics = grads['fd_metrics']
        
        # Row 1: Loss convergence
        ax = fig.add_subplot(gs[0, col])
        ax.plot(loss_history, linewidth=2, color='steelblue')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'Convergence (μ={target_mu:.0e})\nFinal={loss_history[-1]:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Row 2: Component magnitudes (per timestep)
        ax = fig.add_subplot(gs[1, col])
        
        horizon = grads['goal'].shape[0]
        timesteps = np.arange(horizon)
        
        # Compute norms per timestep
        norm_goal = grads['goal'].norm(dim=1).cpu().numpy()
        norm_orient = grads['orient'].norm(dim=1).cpu().numpy()
        norm_v = grads['v'].norm(dim=1).cpu().numpy()
        norm_ctrl = grads['ctrl'].norm(dim=1).cpu().numpy()
        norm_total = grads['total'].norm(dim=1).cpu().numpy()  # Actual total gradient norm
        
        # Stacked bar chart
        ax.bar(timesteps, norm_goal, label='Goal', alpha=0.8, color='red')
        ax.bar(timesteps, norm_orient, bottom=norm_goal, label='Orient', alpha=0.8, color='orange')
        ax.bar(timesteps, norm_v, bottom=norm_goal+norm_orient, label='Velocity', alpha=0.8, color='green')
        ax.bar(timesteps, norm_ctrl, bottom=norm_goal+norm_orient+norm_v, label='Control', alpha=0.8, color='blue')
        
        # Add total gradient norm as a line
        ax.plot(timesteps, norm_total, 'k-', linewidth=2.5, alpha=0.7, label='||∇L|| (actual)', marker='o', markersize=4)
        
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('||∂L/∂u||', fontsize=10)
        ax.set_title(f'Component Contributions', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Row 3: IFT vs FD scatter
        ax = fig.add_subplot(gs[2, col])
        
        grad_ift = grads['total'].detach().cpu().numpy()
        grad_fd = grads['fd'].detach().cpu().numpy()
        
        ax.scatter(grad_ift.flatten(), grad_fd.flatten(), alpha=0.5, s=20, color='steelblue')
        lim = max(abs(grad_ift).max(), abs(grad_fd).max())
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('IFT Gradient', fontsize=10)
        ax.set_ylabel('FD Gradient', fontsize=10)
        ax.set_title(f'IFT vs FD\nCorr={fd_metrics["correlation"]:.3f}, Err={fd_metrics["mean_err"]:.2e}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    save_path_time = save_path.replace(".png", f"_{int(TIME)}.png")
    plt.savefig(save_path_time, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to {save_path_time}")


def analyze_component_importance(grads):
    """
    Analyze which component dominates.
    """
    # Total magnitude per component
    mag_goal = grads['goal'].norm().item()
    mag_orient = grads['orient'].norm().item()
    mag_v = grads['v'].norm().item()
    mag_ctrl = grads['ctrl'].norm().item()
    mag_total = grads['total'].norm().item()
    
    print(f"\n  Component magnitudes:")
    print(f"    Goal:     {mag_goal:.3e} ({100*mag_goal/mag_total:.1f}%)")
    print(f"    Orient:   {mag_orient:.3e} ({100*mag_orient/mag_total:.1f}%)")
    print(f"    Velocity: {mag_v:.3e} ({100*mag_v/mag_total:.1f}%)")
    print(f"    Control:  {mag_ctrl:.3e} ({100*mag_ctrl/mag_total:.1f}%)")
    print(f"    Total:    {mag_total:.3e}")
    
    # Dominant component
    components = {'Goal': mag_goal, 'Orient': mag_orient, 'Velocity': mag_v, 'Control': mag_ctrl}
    dominant = max(components, key=components.get)
    print(f"\n  → Dominant: {dominant}")
    
    return components


def animate_optimal_trajectory(results, save_dir="./result/dynamics", fps=20):
    """Create animation for each optimized trajectory."""
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.lines import Line2D
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Creating optimal trajectory animations")
    print("="*80)
    
    half = 0.1
    side = 0.2
    
    for result in results:
        target_mu = result['target_mu']
        grads = result['component_gradients']
        
        # Get trajectory
        qs = grads['qs']
        pusher_traj = grads['pusher_traj']
        
        # Handle both list and tensor
        if isinstance(qs, list):
            traj = torch.stack(qs).detach().cpu().numpy()
        else:
            traj = qs.detach().cpu().numpy()
            
        if isinstance(pusher_traj, list):
            pusher = torch.stack(pusher_traj).detach().cpu().numpy()
        else:
            pusher = pusher_traj.detach().cpu().numpy()
        
        print(f"\n  Animation: target_mu={target_mu:.0e}, frames={len(traj)}")
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"Optimal Trajectory (μ={target_mu:.0e})", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Patches
        box = Rectangle((-half, -half), side, side,
                       fc='orange', ec='black', lw=2, zorder=2, alpha=0.8)
        robot = Circle((pusher[0, 0], pusher[0, 1]), 0.02,
                      fc='blue', ec='black', lw=2, zorder=3)
        
        # Goal
        goal_patch = Rectangle((GOAL[0]-half, GOAL[1]-half), side, side,
                              fc='none', ec='green', lw=2.5, ls='--', zorder=1)
        goal_transform = mtransforms.Affine2D().rotate_around(GOAL[0], GOAL[1], GOAL[2])
        goal_patch.set_transform(goal_transform + ax.transData)
        
        ax.add_patch(box)
        ax.add_patch(robot)
        ax.add_patch(goal_patch)
        
        # Trajectory lines
        traj_line = Line2D([], [], linestyle='-', linewidth=2.5, alpha=0.6,
                          color='red', label='Box path')
        pusher_line = Line2D([], [], linestyle='--', linewidth=2.5, alpha=0.6,
                            color='blue', label='Pusher path')
        ax.add_line(traj_line)
        ax.add_line(pusher_line)
        ax.legend(loc='upper right', fontsize=11)
        
        # Info text
        info_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                           fontsize=10, verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(i):
            x, y, theta = traj[i]
            
            # FIXED: rotate then translate
            transform = mtransforms.Affine2D().rotate(theta).translate(x, y)
            box.set_transform(transform + ax.transData)
            
            robot.center = (pusher[i, 0], pusher[i, 1])
            
            traj_line.set_data(traj[:i+1, 0], traj[:i+1, 1])
            pusher_line.set_data(pusher[:i+1, 0], pusher[:i+1, 1])
            
            time_val = i * 0.05
            info_text.set_text(f'Time: {time_val:.2f}s\nStep: {i}/{len(traj)-1}')
            
            return box, robot, traj_line, pusher_line, info_text
        
        ani = FuncAnimation(fig, animate, frames=len(traj),
                          interval=1000/fps, blit=False, repeat=True)
        
        save_path = f"{save_dir}/optimal_trajectory_mu{target_mu:.0e}_{int(TIME)}.mp4"
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2000)
            ani.save(save_path, writer=writer, dpi=150)
            print(f"    ✓ Saved: {save_path}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        
        plt.close(fig)
    
    print("\n✓ All animations created!")


def visualize_gradient_structure_detail(results, save_path="./result/dynamics/converged_gradient_structure_detailed.png"):
    """
    Detailed gradient structure visualization (like test_realistic_box_push).
    
    Shows IFT vs FD gradients component-wise for each target_mu.
    Uses dual y-axis to handle different scales.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    n_mu = len(results)
    fig, axes = plt.subplots(n_mu, 1, figsize=(14, 3.5 * n_mu))
    
    if n_mu == 1:
        axes = [axes]
    
    # Color scheme
    color_ift = 'steelblue'
    color_fd = 'orangered'
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        ax2 = ax.twinx()  # Create secondary y-axis for FD
        
        target_mu = result['target_mu']
        grads = result['component_gradients']
        
        grad_ift = grads['total'].detach().cpu().numpy()  # (horizon, 2)
        grad_fd = grads['fd'].detach().cpu().numpy()
        
        horizon = grad_ift.shape[0]
        timesteps = np.arange(horizon)
        
        # IFT gradients on left y-axis
        l1 = ax.plot(timesteps, grad_ift[:, 0], 
                marker='o', linestyle='-', linewidth=2, markersize=6,
                color=color_ift, alpha=0.8, label='IFT ∂L/∂u_x')
        l2 = ax.plot(timesteps, grad_ift[:, 1], 
                marker='^', linestyle='-', linewidth=2, markersize=6,
                color=color_ift, alpha=0.6, label='IFT ∂L/∂u_y')
        
        # FD gradients on right y-axis
        l3 = ax2.plot(timesteps, grad_fd[:, 0], 
                marker='s', linestyle='--', linewidth=1.5, markersize=5,
                color=color_fd, alpha=0.8, label='FD ∂L/∂u_x')
        l4 = ax2.plot(timesteps, grad_fd[:, 1], 
                marker='v', linestyle='--', linewidth=1.5, markersize=5,
                color=color_fd, alpha=0.6, label='FD ∂L/∂u_y')
        
        # Styling for left y-axis (IFT)
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('IFT Gradient', fontsize=11, color=color_ift)
        ax.tick_params(axis='y', labelcolor=color_ift)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ift_max = max(abs(grad_ift).min(), abs(grad_ift).max())
        ax.set_ylim(-1.2*ift_max, 1.2*ift_max)
        
        # Styling for right y-axis (FD)
        ax2.set_ylabel('FD Gradient', fontsize=11, color=color_fd)
        ax2.tick_params(axis='y', labelcolor=color_fd)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        fd_max = max(abs(grad_fd.min()), abs(grad_fd.max()))
        if fd_max > 0:  # Avoid division by zero if FD not computed
            ax2.set_ylim(-fd_max, fd_max)

        # Title
        ax.set_title(f'Gradient Structure: target_mu = {target_mu:.0e}', 
                     fontsize=12, fontweight='bold')
        
        # Combined legend
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right', fontsize=9, ncol=2)
        
        # Metrics
        fd_metrics = grads['fd_metrics']
        mean_err = fd_metrics['mean_err']
        max_err = fd_metrics['max_err']
        corr = fd_metrics['correlation']
        
        # Add metrics box
        ax.text(0.02, 0.98, 
                f'Mean err: {mean_err:.3e}\nMax err: {max_err:.3e}\nCorr: {corr:.3f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path_time = save_path.replace(".png", f"_{int(TIME)}.png")
    plt.savefig(save_path_time, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved gradient structure detail to {save_path_time}")


# Main
if __name__ == "__main__":
    print("="*80)
    print("Converged Gradient Analysis")
    print("="*80)
    
    optimizer = TrajectoryOptimizer(
        mass=1.0, side_length=0.2, mu=0.5,
        horizon=60, dt=0.05, device=DEVICE,
        TO_solver='shooting', dynamics_solver='IP',
    )
    
    # target_mu_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    target_mu_list = [1e-3, 1e-6]
    
    results = []
    
    for target_mu in target_mu_list:
        # Optimize to convergence
        u_opt, loss_history, converged = optimize_to_convergence(
            optimizer, target_mu, max_iters=300, tol=1e-6
        )
        
        if not converged:
            print(f"  ⚠ Max iterations reached (keeping result anyway)")
            # Continue with this result anyway for analysis
        
        # Compute component gradients + FD
        component_grads = compute_component_gradients(u_opt, target_mu, tol=1e-4)
        
        # Analyze
        importance = analyze_component_importance(component_grads)
        
        results.append({
            'target_mu': target_mu,
            'u_opt': u_opt,
            'loss_history': loss_history,
            'component_gradients': component_grads,
            'importance': importance,
        })
    
    # Visualize
    if len(results) == 0:
        print("\n" + "="*80)
        print("✗ NO RESULTS TO VISUALIZE")
        print("="*80)
        print("All optimizations failed to converge. Try:")
        print("  - Increase max_iters (현재: 300)")
        print("  - Increase tol (현재: 1e-4)")
        print("  - Check initial guess quality")
        sys.exit(1)
    
    visualize_converged_gradients(results)
    
    # Visualize detailed gradient structure (like test_realistic_box_push)
    visualize_gradient_structure_detail(results)
    
    # Animate
    animate_optimal_trajectory(results)
    
    print("\n" + "="*80)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\n결과:")
    print("  1. converged_gradient_analysis_{timestamp}.png")
    print("     - Row 1: Loss convergence")
    print("     - Row 2: Component contributions (stacked bar)")
    print("     - Row 3: IFT vs FD comparison (scatter)")
    print("\n  2. converged_gradient_structure_detailed_{timestamp}.png")
    print("     - Timestep-by-timestep IFT vs FD")
    print("     - Component-wise (u_x, u_y)")
    print("     - Same format as test_realistic_box_push.py")
    print("\n  3. optimal_trajectory_mu{value}_{timestamp}.mp4")
    print("     - Converged trajectory animation")
    print("\n인사이트:")
    print("  ✓ Converged state에서 어느 component가 dominant?")
    print("  ✓ Goal vs Control trade-off")
    print("  ✓ Timestep별 중요도 (어느 phase가 critical?)")
    print("  ✓ FD vs IFT at optimum (수렴 후 gradient 정확도)")
    print("  ✓ Optimal trajectory visualization")