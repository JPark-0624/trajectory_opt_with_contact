"""
DDP Optimizer for Box Pusher System

Example usage of DDP with contact-implicit dynamics (IPM solver)

Author: Juneil Park
Date: 2026-03-12
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from trajectory_opt_with_contact.ddp_refactored import DDPOptimizer, CholeskyQPSolver
from trajectory_opt_with_contact.dynamics import step_square_pos_ip, IPMOptions


# ==============================================================================
# Problem Setup
# ==============================================================================

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.double

# Time parameters
dt = 0.05
horizon = 100
total_time = horizon * dt

# State/Control dimensions
n_x = 8  # [q(3), v(3), pusher_pos(2)]
n_u = 2  # [pusher_vel(2)]

# Initial state
q_init = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)  # Box at origin
v_init = torch.zeros(3, device=device, dtype=dtype)  # At rest
pr_init = torch.tensor([-0.15, 0.0], device=device, dtype=dtype)  # Pusher behind box

x0 = torch.cat([q_init, v_init, pr_init])

# Goal state
q_goal = torch.tensor([0.3, 0.0, 0.0], device=device, dtype=dtype)  # Move forward 30cm
v_goal = torch.zeros(3, device=device, dtype=dtype)  # At rest
pr_goal = torch.tensor([0.15, 0.0], device=device, dtype=dtype)  # Pusher at final position

x_goal = torch.cat([q_goal, v_goal, pr_goal])

print("="*80)
print("DDP Optimizer for Box Pusher")
print("="*80)
print(f"\nProblem setup:")
print(f"  Horizon: {horizon} steps ({total_time:.2f}s)")
print(f"  dt: {dt}s")
print(f"  State dim: {n_x}")
print(f"  Control dim: {n_u}")
print(f"\nPhysics parameters (defined in dynamics_fn):")
print(f"  Box mass: 0.5 kg")
print(f"  Box inertia: 0.01 kg·m²")
print(f"  Box half-width: 0.05 m")
print(f"  Friction coefficient: 0.3")
print(f"  IPM target_mu: 1e-6")
print(f"\nInitial state:")
print(f"  Box position: {q_init.cpu().numpy()}")
print(f"  Box velocity: {v_init.cpu().numpy()}")
print(f"  Pusher position: {pr_init.cpu().numpy()}")
print(f"\nGoal state:")
print(f"  Box position: {q_goal.cpu().numpy()}")
print(f"  Box velocity: {v_goal.cpu().numpy()}")
print(f"  Pusher position: {pr_goal.cpu().numpy()}")


# ==============================================================================
# Dynamics Wrapper
# ==============================================================================

# Physics parameters (defined once for dynamics wrapper)
# IMPORTANT: These MUST be defined in the same scope as dynamics_fn
# so that autograd can trace through them during differentiation!
m = 1        # Box mass (kg)
Izz = (1.0/6.0) * 1.0 * (0.2**2 + 0.2**2)     # Box moment of inertia (kg·m²)
half = 0.1    # Box half-width (m)
mu = 0.5        # Friction coefficient

# IPM solver options

ipm_opts = IPMOptions(
        target_mu=1e-4,           # Tight complementarity
        max_newton=20,
        tol=1e-6,
        enable_viscous_ground_friction=True,
        smooth_sdf=50.0,
        c_lin=1.0,
        c_ang=0.00667
    )

skip_solving_threshold = 100.0  # Skip IPM if forces too small


def dynamics_fn(state, control):
    """
    Wrapper for step_square_pos_ip dynamics
    
    CRITICAL: All parameters needed by step_square_pos_ip are defined 
    in the outer scope (above) to ensure they're accessible during 
    autograd differentiation. Don't move them outside this module!
    
    Args:
        state: [8] = [q(3), v(3), pusher_pos(2)]
        control: [2] = [pusher_vel(2)]
    
    Returns:
        next_state: [8]
    """
    # Unpack state
    qk = state[:3]    # [q_x, q_y, θ]
    vk = state[3:6]   # [vx, vy, ω]
    pusher_pos = state[6:8]  # [pr_x, pr_y]
    
    # Control
    u_push = control  # [u_x, u_y]
    
    # Call IPM contact solver with all parameters from outer scope
    # q_next, v_next, pusher_pos_next, lam_vec, phi, z
    q_next, v_next, pusher_pos_next, _, _ , _= step_square_pos_ip(
        qk=qk,
        vk=vk,
        pusher_pos=pusher_pos,
        u_push=u_push,
        h=dt,                # Time step (from global scope)
        m=m,                 # Mass (from outer scope)
        Izz=Izz,            # Inertia (from outer scope)
        half=half,          # Half-width (from outer scope)
        mu=mu,               # Friction (from outer scope)
        ipm_opts=ipm_opts,  # IPM options (from outer scope)
        skip_solving_threshold=skip_solving_threshold,
        z_prev=None,        # No warm start
        jacobian_type="autograd",  # Use autograd for differentiation
        debugOut=None,      # No debug output
        modeAConfig=None,   # No mode A config
    )
    
    # Pack next state
    next_state = torch.cat([q_next, v_next, pusher_pos_next])
    
    return next_state


# ==============================================================================
# Cost Functions
# ==============================================================================

# Cost weights
w_pos = 100.0       # Position error weight
w_orient = 10.0     # Orientation error weight
w_vel = 1.0         # Velocity penalty weight
w_control = 0.01    # Control effort weight
w_terminal = 1000.0 # Terminal cost weight

def stage_cost_fn(state, control):
    """
    Running cost for each timestep
    
    Args:
        state: [8] = [q(3), v(3), pusher_pos(2)]
        control: [2] = [pusher_vel(2)]
    
    Returns:
        cost: scalar
    """
    # Unpack state
    q = state[:3]
    v = state[3:6]
    
    # Position error (x, y only - not orientation yet)
    pos_err = q[:2] - q_goal[:2]
    
    # Orientation error
    theta_err = q[2] - q_goal[2]
    
    # Velocity penalty (should be small)
    vel_penalty = torch.sum(v**2)
    
    # Control effort
    control_penalty = torch.sum(control**2)
    
    # Total cost
    cost = (w_pos * torch.sum(pos_err**2) + 
            w_orient * theta_err**2 +
            w_vel * vel_penalty +
            w_control * control_penalty)
    
    return cost


def terminal_cost_fn(state, goal_state):
    """
    Terminal cost at final timestep
    
    Args:
        state: [8] = [q(3), v(3), pusher_pos(2)]
        goal_state: [8] = target state
    
    Returns:
        cost: scalar
    """
    # Unpack
    q = state[:3]
    v = state[3:6]
    
    q_g = goal_state[:3]
    v_g = goal_state[3:6]
    
    # Position error
    pos_err = q[:2] - q_g[:2]
    theta_err = q[2] - q_g[2]
    
    # Velocity error
    vel_err = v - v_g
    
    # Terminal cost
    cost = w_terminal * (torch.sum(pos_err**2) + 
                         theta_err**2 + 
                         0.1 * torch.sum(vel_err**2))
    
    return cost


# ==============================================================================
# DDP Optimizer Setup
# ==============================================================================

print("\n" + "="*80)
print("Creating DDP Optimizer")
print("="*80)

optimizer = DDPOptimizer(
    dynamics_fn=dynamics_fn,
    stage_cost_fn=stage_cost_fn,
    terminal_cost_fn=terminal_cost_fn,
    horizon=horizon,
    dt=dt,
    n_x=n_x,
    n_u=n_u,
    qp_solver=CholeskyQPSolver(reg=1e-6),
    u_min=-0.5,  # Max pusher velocity: 0.5 m/s
    u_max=0.5,
    max_iters=50,
    reg_init=1e-6,
    reg_scale=10.0,
    device=device,
    verbose=True,
)

print("\nOptimizer created successfully!")
print(f"  QP Solver: Cholesky (sequential)")
print(f"  Control limits: [{optimizer.u_min}, {optimizer.u_max}]")
print(f"  Regularization: {optimizer.reg:.2e}")
print(f"  Max iterations: {optimizer.max_iters}")


# ==============================================================================
# Run Optimization
# ==============================================================================

print("\n" + "="*80)
print("Running DDP Optimization")
print("="*80)

result = optimizer.optimize(x0, x_goal)

# Extract results
X_opt = result['trajectory']  # [T+1, n_x]
U_opt = result['controls']    # [T, n_u]
cost_final = result['cost']
K_gains = result['K_gains']   # Feedback gains

print("\n" + "="*80)
print("Optimization Complete!")
print("="*80)
print(f"\nFinal cost: {cost_final:.6f}")
print(f"\nFinal state:")
print(f"  Box position: {X_opt[-1, :3]}")
print(f"  Box velocity: {X_opt[-1, 3:6]}")
print(f"  Pusher position: {X_opt[-1, 6:8]}")
print(f"\nGoal state:")
print(f"  Box position: {x_goal[:3].cpu().numpy()}")
print(f"  Box velocity: {x_goal[3:6].cpu().numpy()}")
print(f"  Pusher position: {x_goal[6:8].cpu().numpy()}")

# Error analysis
q_final = X_opt[-1, :3]
q_err = np.linalg.norm(q_final - x_goal[:3].cpu().numpy())
print(f"\nPosition error: {q_err:.6f} m")

# Control statistics
U_mean = np.mean(np.abs(U_opt), axis=0)
U_max = np.max(np.abs(U_opt), axis=0)
print(f"\nControl statistics:")
print(f"  Mean |u|: {U_mean}")
print(f"  Max |u|: {U_max}")


# ==============================================================================
# Save Results
# ==============================================================================

output_file = Path(__file__).parent.parent / "results" / "ddp_pusher_results.npz"
output_file.parent.mkdir(exist_ok=True)

np.savez(
    output_file,
    trajectory=X_opt,
    controls=U_opt,
    cost=cost_final,
    K_gains=K_gains if K_gains else None,
    horizon=horizon,
    dt=dt,
    q_goal=q_goal.cpu().numpy(),
)

print(f"\n✅ Results saved to: {output_file}")


# ==============================================================================
# Visualization (Optional)
# ==============================================================================

try:
    from trajectory_opt_with_contact.visualizer import visualize_trajectory
    
    print("\n" + "="*80)
    print("Visualizing Results")
    print("="*80)
    
    # Prepare data for visualizer
    # Note: visualizer expects specific format from multiple shooting
    # We'll create a compatible structure
    
    viz_data = {
        'trajectory': X_opt,
        'controls': U_opt,
        'cost': cost_final,
        'horizon': horizon,
        'dt': dt,
    }
    
    # Create simple plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Box position
    axes[0, 0].plot(X_opt[:, 0], X_opt[:, 1], 'b-', linewidth=2, label='Trajectory')
    axes[0, 0].plot(q_init[0].cpu(), q_init[1].cpu(), 'go', markersize=10, label='Start')
    axes[0, 0].plot(q_goal[0].cpu(), q_goal[1].cpu(), 'r*', markersize=15, label='Goal')
    axes[0, 0].set_xlabel('x (m)')
    axes[0, 0].set_ylabel('y (m)')
    axes[0, 0].set_title('Box Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')
    
    # Box orientation
    t = np.arange(horizon + 1) * dt
    axes[0, 1].plot(t, X_opt[:, 2], 'b-', linewidth=2)
    axes[0, 1].axhline(q_goal[2].cpu(), color='r', linestyle='--', label='Goal')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Orientation (rad)')
    axes[0, 1].set_title('Box Orientation')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Controls
    t_u = np.arange(horizon) * dt
    axes[1, 0].plot(t_u, U_opt[:, 0], 'b-', linewidth=2, label='u_x')
    axes[1, 0].plot(t_u, U_opt[:, 1], 'r-', linewidth=2, label='u_y')
    axes[1, 0].axhline(optimizer.u_min, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axhline(optimizer.u_max, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Control (m/s)')
    axes[1, 0].set_title('Pusher Velocity')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Velocities
    axes[1, 1].plot(t, X_opt[:, 3], 'b-', linewidth=2, label='v_x')
    axes[1, 1].plot(t, X_opt[:, 4], 'r-', linewidth=2, label='v_y')
    axes[1, 1].plot(t, X_opt[:, 5], 'g-', linewidth=2, label='ω')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Velocity')
    axes[1, 1].set_title('Box Velocity')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    plot_file = output_file.parent / "ddp_pusher_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_file}")
    
    plt.show()
    
except ImportError as e:
    print(f"\n⚠️  Visualization skipped: {e}")
    print("   Install matplotlib to enable plotting")


print("\n" + "="*80)
print("DDP Optimization Complete!")
print("="*80)