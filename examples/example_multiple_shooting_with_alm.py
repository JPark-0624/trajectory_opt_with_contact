"""
Clean example for Block Multiple Shooting with ALM
- Automatic initialization (no manual uInit/prKnotInit setup)
- Enhanced visualization with initial vs optimal comparison
"""

import torch
import numpy as np
from trajectory_opt_with_contact.multiple_shooting_with_alm import (
    BlockMultipleShootingWithALM,
    ALMConfig,
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
    # Problem Setup
    # ========================================
    q0 = [0.0, 0.0, 0.0]          # Initial: [x, y, theta]
    v0 = [0.0, 0.0, 0.0]          # Initial velocity
    pr0 = [-0.1, -0.3]             # Initial pusher position
    goal = [0.2, 0.5, -0.3]      # Goal: [x, y, theta]
    
    horizon = 60                 # Time steps
    dt = 0.05                     # 5 seconds total
    blockSize = 60                # 5 blocks
    
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
    
    optimizer = BlockMultipleShootingWithALM(
        mass=1.0,
        sideLength=0.2,
        muFriction=0.5,
        horizon=horizon,
        dt=dt,
        blockSize=blockSize,
        device=device,
        enableDefectALM=True,
        ipmWarmStart=True,
        ipmOpts=ipmOpts,
        skipSolvingThreshold=100.0,
    )
    
    # ========================================
    # ALM & Cost Configuration
    # ========================================
    cfg = ALMConfig(
        outerIters=1,
        innerIters=2,
        useLbfgs=False,           # Use Adam
        lr=0.01,
        rhoDefectInit=2.0,
        rhoDefectEta=2.0,
        rhoDefectMax=1e4,
        tolDefect=1e-3,
    )
    
    w = CostWeights(
        wControl=1e-2,            # Light control regularization
        wControlSmooth=0.0,       # No smoothness penalty
        wObjVel=1.0,              # Penalize object velocity
        wTargetXY=20.0,           # Strong position tracking
        wTargetOrient=0.5,        # Moderate orientation tracking
    )
    
    # ========================================
    # Optimize (CLEAN! No manual init needed)
    # ========================================
    print("\n" + "="*70)
    print("Starting Block Multiple Shooting Optimization")
    print("="*70)
    
    result = optimizer.optimize(
        q0=torch.tensor(q0, device=optimizer.device),
        v0=torch.tensor(v0, device=optimizer.device),
        pr0=torch.tensor(pr0, device=optimizer.device),
        goalXY=torch.tensor(goal[:2], device=optimizer.device),
        goalTheta=float(goal[2]),
        cfg=cfg,
        w=w,
        track_gradients=True,     # Enable gradient tracking for visualization
        # uInit and prKnotInit are auto-generated!
    )
    
    # ========================================
    # Print Results
    # ========================================
    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)
    print(f"Final loss: {result['loss']:.4f}")
    print(f"Terminal position: {result['trajectory'][-1]}")
    print(f"Goal position: {goal}")
    print(f"Position error: {np.linalg.norm(result['trajectory'][-1][:2] - goal[:2]):.6f} m")
    print(f"Orientation error: {abs(result['trajectory'][-1][2] - goal[2]):.6f} rad")
    
    # Loss breakdown
    comp = result['loss_components']
    print("\nLoss Component Breakdown:")
    print(f"  Total:           {comp['total']:.6f}")
    print(f"  Control energy:  {comp['control_energy']*comp['w_control']:.6f}")
    print(f"  Control smooth:  {comp['control_smooth']*comp['w_smooth']:.6f}")
    print(f"  Object velocity: {comp['obj_vel']*comp['w_objvel']:.6f}")
    print(f"  Target XY:       {comp['target_xy']*comp['w_targetxy']:.6f}")
    print(f"  Target orient:   {comp['target_orient']*comp['w_orient']:.6f}")
    print(f"  ALM defect:      {comp['alm_defect']:.6f}")
    
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
        save_trajectory="trajectory_ms_alm.png",
        save_analysis="analysis_ms_alm.png",
        save_animation="animation_ms_alm.mp4",
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
    )
    
    print("\n✅ All done! Check the output files:")
    print("   - trajectory_ms_alm.png  (2x3 grid with gradients & loss breakdown)")
    print("   - analysis_ms_alm.png    (detailed analysis)")
    print("   - animation_ms_alm.mp4   (side-by-side: initial vs optimal)")


if __name__ == "__main__":
    main()