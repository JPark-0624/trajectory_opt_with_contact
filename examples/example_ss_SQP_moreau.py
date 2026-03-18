"""
Example: Single Shooting SQP with Moreau solver

Matches visualize_example.py API for easy comparison.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trajectory_opt_with_contact.single_shooting_SQP_moreau import SingleShootingSQP, SQPConfig, CostWeights
from trajectory_opt_with_contact import visualize_result

# Import dynamics (matching existing code)
from trajectory_opt_with_contact.dynamics import step_square, step_square_pos_ip, IPMOptions


print("="*60)
print("Single Shooting SQP with Moreau")
print("="*60)

# Choose dynamics solver
dynamics_solver = 'IP'  # 'IP' or 'LCP'

if dynamics_solver == 'IP':
    dynamics_fn = step_square_pos_ip
    print("Using Interior Point dynamics")
else:
    dynamics_fn = step_square
    print("Using LCP dynamics")

# Create optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

ipmOpts = IPMOptions(
        target_mu=1e-4,           # Tight complementarity
        max_newton=20,
        tol=1e-6,
        enable_viscous_ground_friction=True,
        smooth_sdf=50.0,
        c_lin=1.0,
        c_ang=0.00667
    )


optimizer = SingleShootingSQP(
    mass=1.0,
    side_length=0.2,
    mu=0.5,
    horizon=100,
    dt=0.05,
    device=device,
    dynamics_module=dynamics_fn,
    ipmOpts = ipmOpts
)

# Problem setup (matching visualize_example.py)
# Diagonal push
q0 = [0.0, 0.0, 0.0]
v0 = [0.0, 0.0, 0.0]
pusher0 = [-0.1, -0.3]
goal = [0.2, 0.5, -0.3]

# Or X-axis push:
# q0 = [0.0, 0.0, 0.0]
# v0 = [0.0, 0.0, 0.0]
# pusher0 = [-0.3, 0.0]
# goal = [0.5, 0.0, 0.0]

print(f"\nInitial pose: {q0}")
print(f"Goal pose: {goal}")

# SQP configuration
cfg = SQPConfig(
    maxIters=5,
    tol=1e-4,
    use_line_search=True,
    line_search_max_iters=10,
    line_search_beta=0.5,
    u_min=-0.3,
    u_max=0.3,
    use_trust_region=True,
    trust_region_iters=3,
    trust_region_size=0.5,
)

# Cost weights (matching visualize_example: w_target=20, w_orient=0.5, w_v=1.0, w_ctrl=0.01)
w = CostWeights(
    wControl=0.01,         # Matches w_ctrl
    wControlSmooth=0.0,
    wObjVel=1.0,          # Matches w_v (but now terminal velocity!)
    wTargetXY=20.0,       # Matches w_target
    wTargetOrient=1.0,    # Matches w_orient
)

# Initial guess (optional - will use simple init if None)
u_init = None

print("\nOptimizing trajectory...")
result = optimizer.optimize(
    q0=q0,
    v0=v0,
    pusher0=pusher0,
    goal=goal,
    u_init=u_init,
    cfg=cfg,
    w=w,
    verbose=True,
)

print(f"\n{'='*60}")
print("Optimization Complete!")
print(f"{'='*60}")
print(f"Final loss: {result['loss']:.4f}")
print(f"Final pose [x,y,theta]: {result['q_final']}")
print(f"Goal: {goal}")
print(f"Position error: {result['q_final'] - goal}")
print(f"Solve time: {result['solve_time']:.2f}s")
print(f"{'='*60}")

# Visualize (matching visualize_example.py)
print("\nGenerating visualizations...")

# Note: visualize_result expects specific keys
# We may need to add dummy values for compatibility
if 'contact_forces' not in result:
    import numpy as np
    result['contact_forces'] = np.zeros((optimizer.horizon, 2))

try:
    visualize_result(
        result,
        goal,
        half_size=optimizer.half,
        save_trajectory='trajectory_SQP_moreau.png',
        save_animation='animation_SQP_moreau.mp4',
        save_analysis='analysis_SQP_moreau.png',
        obstacle_pos=None,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0)
    )
    
    print("\n✓ Done! Check the generated files:")
    print("  - trajectory_SQP_moreau.png")
    print("  - analysis_SQP_moreau.png")
    print("  - animation_SQP_moreau.mp4")
except Exception as e:
    print(f"\nNote: Visualization failed (may need adjustments): {e}")
    print("But optimization completed successfully!")

print("\nSummary:")
print(f"  Final position error: {result['q_final'][:2] - goal[:2]}")
print(f"  Final orientation error: {result['q_final'][2] - goal[2]:.4f} rad")
print(f"  Total solve time: {result['solve_time']:.2f}s")
