"""
Example usage for BlockMultipleShootingWithALM.

This script is designed to "match" the interface of:
  - multiple_shooting_with_alm.py (optimizer)
  - visualizer.py (TrajectoryVisualizer / visualize_result)

It demonstrates your incremental plan:

Step 1 (default in this example):
  - Single-shooting-equivalent by using blockSize=horizon (1 block).
  - Terminal (x,y) enforced via ALM (hard-ish).
  - Orientation NOT enforced via ALM yet.
  - Defect ALM enabled (so the terminal knot cannot "cheat" without dynamics).

How to switch to true block multi-shooting:
  - Set blockSize=5 (or 10)
  - Keep enableDefectALM=True
  - Optionally enable terminal theta ALM.

Outputs:
  - trajectory_ms_alm.png
  - analysis_ms_alm.png
  - animation_ms_alm.mp4
"""

import os
import sys
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Robust imports (works whether you run from repo root or as an installed module)
# -----------------------------------------------------------------------------
from trajectory_opt_with_contact import multiple_shooting_with_alm
from trajectory_opt_with_contact.dynamics import IPMOptions
from trajectory_opt_with_contact.visualizer import visualize_result


# -----------------------------------------------------------------------------
# Simple geometric initializer (pusher position path -> velocity inputs)
# -----------------------------------------------------------------------------
def computeGeometricVelocityInit(pusher0, box0, goal, horizon, dt, contactOffset=1e-4):
    """
    Create a piecewise-linear pusher POSITION path: approach then push,
    then convert it to VELOCITY inputs u[t] via finite difference.

    Returns:
      uInit: (T,2) velocities
      pInit: (T+1,2) positions (for debugging)
    """
    p0 = np.array(pusher0[:2], dtype=float)
    b0 = np.array(box0[:2], dtype=float)
    g0 = np.array(goal[:2], dtype=float)

    distToBox = np.linalg.norm(b0 - p0)
    distBoxToGoal = np.linalg.norm(g0 - b0)
    total = distToBox + distBoxToGoal + 1e-9

    steps1 = max(5, min(horizon - 5, int(horizon * (distToBox / total))))
    steps2 = horizon - steps1

    # approach target: just shy of contact
    dirToBox = (b0 - p0) / (distToBox + 1e-9)
    pContact = b0 - dirToBox * contactOffset

    # push target: just shy of goal alignment
    dirToGoal = (g0 - b0) / (distBoxToGoal + 1e-9)
    pFinal = g0 - dirToGoal * contactOffset

    pList = [p0.copy()]
    # phase 1
    for i in range(steps1):
        a = (i + 1) / steps1
        p = (1 - a) * p0 + a * pContact
        pList.append(p)
    # phase 2
    for i in range(steps2):
        a = (i + 1) / steps2
        p = (1 - a) * pContact + a * pFinal
        pList.append(p)

    pArr = np.stack(pList, axis=0)  # (T+1,2)
    uArr = (pArr[1:] - pArr[:-1]) / dt  # (T,2)
    return uArr.astype(np.float32), pArr.astype(np.float32)



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Problem definition
    # -------------------------
    # q0 = [0.0, 0.0, 0.0]    # object pose
    # v0 = [0.0, 0.0, 0.0]     # object vel (unused by QS much)
    # pr0 = [-0.3, 0.0]        # pusher start
    # goal = [0.5, 0.0, 0.0]   # desired object pose

    # Task 2: Diagonal push (uncomment to try)
    q0 = [0.0, 0.0, 0.0]
    v0 = [0.0, 0.0, 0.0]
    pr0 = [0.3, -0.3]
    goal = [-0.2, 0.5, -0.3]


    horizon = 100
    dt = 0.05

    # -------------------------
    # Step 1: single-shooting-equivalent
    #   Use ONE block, but keep defect ALM ON so knots cannot cheat.
    # -------------------------
    blockSize = 20  # 1 block => equivalent to single shooting

    ipmOpts = IPMOptions(target_mu=1e-6, max_newton=20, tol=1e-3, smooth_sdf=50.0, #smooth_sdf is unused
                    enable_viscous_ground_friction=True,
                    c_lin=1.0,
                    c_ang=0.00667 ### c_ang = c_lin * (Izz/m)
                    )  # use defaults from your dynamics
    
    optimizer = multiple_shooting_with_alm.BlockMultipleShootingWithALM(
        mass=1.0,
        sideLength=0.2,
        muFriction=0.5,
        horizon=horizon,
        dt=dt,
        blockSize=blockSize,
        device=device,
        enableDefectALM=True,          # IMPORTANT for single-shooting-equivalent correctness
        ipmWarmStart=True,
        ipmOpts=ipmOpts,
        skipSolvingThreshold=0.003,
    )

    # ALM configuration
    cfg = multiple_shooting_with_alm.ALMConfig(
        outerIters=1,
        innerIters=200,
        useLbfgs=False,
        lr=0.01,
        # Defects (1 block => only one defect)
        rhoDefectInit=2.0,
        rhoDefectEta=2.0,
        rhoDefectMax=1e4,
        tolDefect=1e-3,
    )

    # Soft costs (tune as needed)
    w = multiple_shooting_with_alm.CostWeights(
        wControl=1e-2,
        wControlSmooth=0.0,
        wObjVel=1.0,
        wTargetXY=20.0,
        wTargetOrient=0.5,
    )

    # Initial guess (velocity)
    uInitNp, pInitNp = computeGeometricVelocityInit(
        pusher0=pr0,
        box0=q0,
        goal=goal,
        horizon=horizon,
        dt=dt,
        contactOffset=1e-4,
    )
    uInit = torch.tensor(uInitNp, device=optimizer.device)

    # Optional: initialize knots using the geometric position path
    # For single block, we only need knot 0 and knot 1. We'll initialize knot 1
    # by "pretending" object reaches goal (helps ALM start closer); defect ALM will correct it.
    qKnotInit = torch.zeros(optimizer.numKnots, 3, device=optimizer.device)
    prKnotInit = torch.zeros(optimizer.numKnots, 2, device=optimizer.device)
    qKnotInit[0] = torch.tensor(q0, device=optimizer.device)
    prKnotInit[0] = torch.tensor(pr0, device=optimizer.device)
    qKnotInit[-1] = torch.tensor(goal, device=optimizer.device)
    prKnotInit[-1] = torch.tensor(pInitNp[-1], device=optimizer.device)


    # Run optimization
    out = optimizer.optimize(
        q0=torch.tensor(q0, device=optimizer.device),
        v0=torch.tensor(v0, device=optimizer.device),
        pr0=torch.tensor(pr0, device=optimizer.device),
        goalXY=torch.tensor(goal[:2], device=optimizer.device),
        goalTheta=float(goal[2]),
        cfg=cfg,
        w=w,
        uInit=uInit,
        qKnotInit=qKnotInit,
        prKnotInit=prKnotInit,
    )

    uOpt = out["u_seq"]
    qKnots = out["qKnots"]
    prKnots = out["prKnots"]
    qsOpt = out["trajectory"]

    print(f"uOpt: {uOpt}")
    print(f"qKnots: {qKnots}")
    print(f"prKnots: {prKnots}")
    print(f"qsOpt: {qsOpt}")

    print("=" * 70)
    print("Multiple-shooting-with-ALM (single-shooting-equivalent) complete")
    print("=" * 70)
    print(f"Device: {optimizer.device}")
    print(f"Terminal XY residual norm (last outer): {out['history']['terminalXYNorm'][-1].item():.6e}")
    print(f"Terminal knot (object): {qKnots[-1]}")
    print(f"Goal: {goal}")


    # Visualize
    visualize_result(
        out,
        goal=goal,
        half_size=optimizer.half,
        save_trajectory="trajectory_ms_alm.png",
        save_analysis="analysis_ms_alm.png",
        save_animation="animation_ms_alm.mp4",
        obstacle_pos=None,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
    )

    print("\n✓ Done! Generated:")
    print("  - trajectory_ms_alm.png")
    print("  - analysis_ms_alm.png")
    print("  - animation_ms_alm.mp4")


if __name__ == "__main__":
    main()
