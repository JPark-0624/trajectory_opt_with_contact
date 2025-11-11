"""
Example: Trajectory optimization with full visualization
Matches the original main.py visualization style
"""

import torch
from trajectory_opt_with_contact import TrajectoryOptimizer, visualize_result

# Set random seed
torch.manual_seed(0)

print("="*60)
print("Trajectory Optimization with Visualization")
print("="*60)

# Create optimizer (matching original parameters)

TO_solver = 'iLQR' #'shooting'
dynamics_solver = 'IP' #'LCP'
obs = '' #'obs' #

print("\nCreating optimizer...")
optimizer = TrajectoryOptimizer(
    mass=1.0, side_length=0.2, mu=0.6,
    horizon=30, dt=0.05,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    TO_solver=TO_solver,
    dynamics_solver=dynamics_solver, #'LCP',
    # ALM params
    use_second_order=True,         # LBFGS solver for Augmented Lagrangian
    alm_enabled=True,
    alm_rho_init=1e2,
    alm_target_tol=1e-6,
    alm_outer_iters=8,
    lbfgs_inner_steps=10
)

print(f"Using device: {optimizer.device}")

# Define problem (matching original main.py)
q0 = [0.0, 0.0, 0.0]
v0 = [0.0, 0.0, 0.0]
pusher0 = [0.3, -0.3]
goal = [-0.2, 0.5, -0.3]
obstacle = None # [0.2, -0.2] # # Obstacle position from original code
u_init = [[-0.2, 0.2]] * optimizer.horizon  # Initial guess from original code


print(f"\nInitial pose: {q0}")
print(f"Goal pose: {goal}")
print(f"Obstacle at: {obstacle}")

# Optimize
print("\nOptimizing trajectory...")
result = optimizer.optimize(
    q0=q0,
    v0=v0,
    pusher0=pusher0,
    goal=goal,
    u_init=u_init,
    max_iters= 10, #iLQR doesn't need any many iterations
    lr=0.05,
    lr_decay_step=10,
    lr_decay_gamma=0.8,
    obstacle_pos=obstacle,
    verbose=True
)

print(f"\n{'='*60}")
print("Optimization Complete!")
print(f"{'='*60}")
print(f"Final loss: {result['loss']:.4f}")
print(f"Final pose [x,y,theta]: {result['q_final']}")
print(f"Position error: {result['q_final'] - goal}")
print(f"{'='*60}")

# Visualize (with original style parameters)
visualize_result(
    result, 
    goal, 
    half_size=optimizer.half,
    save_trajectory='trajectory' + TO_solver + dynamics_solver+ obs+'.png',
    save_animation='animation'+ TO_solver + dynamics_solver+obs+  '.mp4',
    save_analysis='analysis' + TO_solver + dynamics_solver+obs+ '.png',
    obstacle_pos=obstacle,
    xlim=(-1.0, 1.0),
    ylim=(-1.0, 1.0)
)

print("\n✓ Done! Check the generated files:")
print("  - trajectory.png")
print("  - analysis.png")
print("  - animation.mp4")
print("\nContact forces summary:")
print(f"  Mean normal impulse: {result['contact_forces'][:, 0].mean():.4f}")
print(f"  Mean tangential impulse: {result['contact_forces'][:, 1].mean():.4f}")