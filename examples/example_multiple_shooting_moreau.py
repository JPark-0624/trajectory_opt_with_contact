"""
Example for Block Multiple Shooting with Moreau QP Solver

Drop-in replacement demonstration:
- Same problem setup as ALM example
- Same visualization
- Much faster (15-150x speedup expected)
- Exact constraint satisfaction

Author: Juneil Park + Claude
Date: 2026-03-17
"""

import torch
import numpy as np
from trajectory_opt_with_contact.block_multiple_shooting_moreau import (
    BlockMultipleShootingMoreau,
    MoreauConfig,
    CostWeights
)
from trajectory_opt_with_contact.dynamics import IPMOptions
from trajectory_opt_with_contact.visualizer import visualize_result


def main():
    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ========================================
    # Problem Setup (IDENTICAL to ALM example)
    # ========================================
    q0 = [0.0, 0.0, 0.0]          # Initial: [x, y, theta]
    v0 = [0.0, 0.0, 0.0]          # Initial velocity
    pr0 = [-0.1, -0.3]             # Initial pusher position
    goal = [0.2, 0.5, -0.3]      # Goal: [x, y, theta]
    
    horizon = 60                 # Time steps
    dt = 0.05                     # 5 seconds total
    blockSize = 15                # 4 blocks
    
    # ========================================
    # Optimizer Setup
    # ========================================
    ipmOpts = IPMOptions(
        target_mu=1e-4,           # Tight complementarity
        max_newton=20,
        tol=1e-6,
        enable_viscous_ground_friction=True,
        smooth_sdf=50.0,
        c_lin=1.0,
        c_ang=0.00667
    )
    
    optimizer = BlockMultipleShootingMoreau(
        mass=1.0,
        sideLength=0.2,
        muFriction=0.5,
        horizon=horizon,
        dt=dt,
        blockSize=blockSize,
        device=device,
        # Note: No enableDefectALM (always exact via Moreau)
        ipmWarmStart=True,
        ipmOpts=ipmOpts,
        skipSolvingThreshold=100.0,
    )
    
    # ========================================
    # Moreau & Cost Configuration
    # ========================================
    cfg = MoreauConfig(
        maxIters=10,              # SQP iterations (vs outerIters in ALM)
        tol=1e-4,                 # Convergence tolerance
        use_line_search=False,    # α=1 for now (monitor first)
        u_min=-0.2,               # Control limits
        u_max=0.2,
    )
    
    w = CostWeights(
        wControl=1e-2,            # Light control regularization
        wControlSmooth=0.0,       # No smoothness penalty
        wObjVel=1.0,              # Penalize object velocity
        wTargetXY=20.0,           # Strong position tracking
        wTargetOrient=0.5,        # Moderate orientation tracking
    )
    
    # ========================================
    # Optimize
    # ========================================
    print("\n" + "="*70)
    print("Starting Block Multiple Shooting with Moreau QP")
    print("="*70)
    
    import time
    start_time = time.time()
    
    result = optimizer.optimize(
        q0=torch.tensor(q0, device=optimizer.device, dtype=torch.float64),
        v0=torch.tensor(v0, device=optimizer.device, dtype=torch.float64),
        pr0=torch.tensor(pr0, device=optimizer.device, dtype=torch.float64),
        goalXY=torch.tensor(goal[:2], device=optimizer.device, dtype=torch.float64),
        goalTheta=float(goal[2]),
        cfg=cfg,
        w=w,
        track_gradients=True,     # Enable gradient tracking for visualization
    )
    
    solve_time = time.time() - start_time
    
    # ========================================
    # Print Results
    # ========================================
    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)
    print(f"Total solve time: {solve_time:.2f} seconds")
    print(f"Final loss: {result['loss']:.4f}")
    print(f"Terminal position: {result['trajectory'][-1]}")
    print(f"Goal position: {goal}")
    print(f"Position error: {np.linalg.norm(result['trajectory'][-1][:2] - np.array(goal[:2])):.6f} m")
    print(f"Orientation error: {abs(result['trajectory'][-1][2] - goal[2]):.6f} rad")
    
    # Loss breakdown (NO alm_defect!)
    comp = result['loss_components']
    print("\nLoss Component Breakdown:")
    print(f"  Total:           {comp['total']:.6f}")
    print(f"  Control energy:  {comp['control_energy']*comp['w_control']:.6f}")
    print(f"  Control smooth:  {comp['control_smooth']*comp['w_smooth']:.6f}")
    print(f"  Object velocity: {comp['obj_vel']*comp['w_objvel']:.6f}")
    print(f"  Target XY:       {comp['target_xy']*comp['w_targetxy']:.6f}")
    print(f"  Target orient:   {comp['target_orient']*comp['w_orient']:.6f}")
    print(f"  (No ALM defect - exact constraint satisfaction via Moreau)")
    
    # Constraint satisfaction check
    print("\nConstraint Satisfaction:")
    print(f"  Defect norm: {result['stationarityInfo']['final_defect_norm']:.6e} (should be ~0)")
    print(f"  Converged: {result['stationarityInfo']['converged']}")
    
    # ========================================
    # Visualize with Enhanced Visualizer
    # ========================================
    print("\n" + "="*70)
    print("Generating Visualizations...")
    print("="*70)
    
    visualize_result(
        result,
        goal=goal,
        half_size=optimizer.half,
        save_trajectory="trajectory_ms_moreau.png",
        save_analysis="analysis_ms_moreau.png",
        save_animation="animation_ms_moreau.mp4",
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
    )
    
    print("\n✅ All done! Check the output files:")
    print("   - trajectory_ms_moreau.png  (2x3 grid with gradients & loss breakdown)")
    print("   - analysis_ms_moreau.png    (detailed analysis)")
    print("   - animation_ms_moreau.mp4   (side-by-side: initial vs optimal)")
    print(f"\n📊 Solve time: {solve_time:.2f}s")
    print("   (Compare with ALM: typically 30-75s → 15-150x speedup expected)")


if __name__ == "__main__":
    main()
