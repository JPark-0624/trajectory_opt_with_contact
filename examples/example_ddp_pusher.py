"""
DDP Optimizer for Box Pusher System

Example usage of DDP with contact-implicit dynamics (IPM solver)
Compatible with TrajectoryVisualizer (same as multiple_shooting example)

Author: Juneil Park
Date: 2026-03-16
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
dt = 0.05  # Same as multiple_shooting
horizon = 60
total_time = horizon * dt

# State/Control dimensions
n_x = 8  # [q(3), v(3), pusher_pos(2)]
n_u = 2  # [pusher_vel(2)]

# Physics parameters (box)
m = 1.0
side_length = 0.2
half = side_length / 2.0
Izz = (1.0 / 6.0) * m * (side_length ** 2 + side_length ** 2)

# Initial state
q_init = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)  # Box at origin
v_init = torch.zeros(3, device=device, dtype=dtype)  # At rest
pr_init = torch.tensor([-0.15, 0.0], device=device, dtype=dtype)  # Pusher behind box

x0 = torch.cat([q_init, v_init, pr_init])

# Goal state
q_goal = torch.tensor([0.5, 0.0, 0.0], device=device, dtype=dtype)  # Move forward 50cm
v_goal = torch.zeros(3, device=device, dtype=dtype)  # At rest
pr_goal = torch.tensor([0.35, 0.0], device=device, dtype=dtype)  # Pusher at final position

x_goal = torch.cat([q_goal, v_goal, pr_goal])

print("="*80)
print("DDP Optimizer for Box Pusher")
print("="*80)
print(f"\nProblem setup:")
print(f"  Horizon: {horizon} steps ({total_time:.2f}s)")
print(f"  dt: {dt}s")
print(f"  State dim: {n_x}")
print(f"  Control dim: {n_u}")
print(f"\nPhysics parameters:")
print(f"  Box mass: {m} kg")
print(f"  Box side length: {side_length} m")
print(f"  Box inertia: {Izz:.6f} kg·m²")
print(f"  Friction coefficient: 0.5")
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
mu = 0.5        # Friction coefficient

# IPM solver options
ipm_opts = IPMOptions(
        target_mu=1e-6,           # Tight complementarity
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
    
    All parameters needed by step_square_pos_ip are defined 
    in the outer scope to ensure autograd can trace them.
    
    Args:
        state: [8] = [q(3), v(3), pusher_pos(2)]
        control: [2] = [pusher_vel(2)]
    
    Returns:
        next_state: [8]
    """
    # Unpack state
    qk = state[:3]
    vk = state[3:6]
    pusher_pos = state[6:8]
    
    # Control
    u_push = control
    
    # Call IPM contact solver
    q_next, v_next, pusher_pos_next, _, _, _ = step_square_pos_ip(
        qk=qk,
        vk=vk,
        pusher_pos=pusher_pos,
        u_push=u_push,
        h=dt,
        m=m,
        Izz=Izz,
        half=half,
        mu=mu,
        ipm_opts=ipm_opts,
        skip_solving_threshold=skip_solving_threshold,
        z_prev=None,
        jacobian_type="autograd",
        debugOut=None,
        modeAConfig=None,
    )
    
    # Pack next state
    next_state = torch.cat([q_next, v_next, pusher_pos_next])
    
    return next_state


# ==============================================================================
# Cost Functions
# ==============================================================================

# Cost weights (similar to multiple_shooting)
w_control = 1.0
w_target_xy = 100.0
w_target_orient = 10.0
w_vel = 1.0
w_terminal = 1000.0

def stage_cost_fn(state, control):
    """
    Running cost for each timestep
    
    Args:
        state: [8]
        control: [2]
    
    Returns:
        cost: scalar
    """
    q = state[:3]
    v = state[3:6]
    
    # Position error (x, y only)
    pos_err = q[:2] - q_goal[:2]
    
    # Orientation error
    theta_err = q[2] - q_goal[2]
    
    # Velocity penalty
    vel_penalty = torch.sum(v**2)
    
    # Control effort
    control_penalty = torch.sum(control**2)
    
    # Total cost
    cost = (w_target_xy * torch.sum(pos_err**2) + 
            w_target_orient * theta_err**2 +
            w_vel * vel_penalty +
            w_control * control_penalty)
    
    return cost


def terminal_cost_fn(state, goal_state):
    """
    Terminal cost at final timestep
    
    Args:
        state: [8]
        goal_state: [8]
    
    Returns:
        cost: scalar
    """
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
    u_min=-0.5,  # Max pusher velocity
    u_max=0.5,
    max_iters=10,
    reg_init=1e-4,
    reg_scale=10.0,
    device=device,
    verbose=True,
)

print("\nOptimizer created successfully!")
print(f"  Method: DDP with geometric initialization")
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

# Optimize with automatic geometric initialization
result = optimizer.optimize(
    x0, 
    x_goal,
    use_geometric_init=True,
)

# Extract results
cost_final = result['loss']
q_final = result['q_final']

print("\n" + "="*80)
print("Optimization Complete!")
print("="*80)
print(f"\nFinal cost: {cost_final:.6f}")
print(f"Method: {result['method']}")
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"\nFinal state:")
print(f"  Box position: {q_final}")
print(f"\nGoal state:")
print(f"  Box position: {q_goal.cpu().numpy()}")

# Error analysis
q_err = np.linalg.norm(q_final - q_goal.cpu().numpy())
print(f"\nPosition error: {q_err:.6f} m")

# Control statistics
U_opt = result['u_seq']
U_mean = np.mean(np.abs(U_opt), axis=0)
U_max = np.max(np.abs(U_opt), axis=0)
print(f"\nControl statistics:")
print(f"  Mean |u|: {U_mean}")
print(f"  Max |u|: {U_max}")


# ==============================================================================
# Save Results
# ==============================================================================

output_dir = Path(__file__).parent.parent / "results"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "ddp_pusher_results.npz"

# Save in format compatible with visualizer
np.savez(
    output_file,
    **result  # Unpack all result fields
)

print(f"\n✅ Results saved to: {output_file}")


# ==============================================================================
# Visualization with TrajectoryVisualizer
# ==============================================================================

print("\n" + "="*80)
print("Visualizing Results")
print("="*80)

try:
    from trajectory_opt_with_contact.visualizer import visualize_result
    import time
    
    # Create visualizer (same half_size as physics)
    save_time = time.time()
    visualize_result(
        result,
        goal=q_goal.detach().cpu().numpy(),
        save_trajectory="trajectory_ddp_" + str(save_time) + ".png",
        save_analysis="analysis_ddp_" + str(save_time) + ".png",
        save_animation="animation_ddp_" + str(save_time) + ".mp4",
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
    )
    
    print("\n" + "="*80)
    print("DDP Optimization and Visualization Complete!")
    print("="*80)
    
except ImportError as e:
    print(f"⚠️  TrajectoryVisualizer not available: {e}")
    print("   Place visualizer.py in the parent directory")
    print("   Results still saved to:", output_file)