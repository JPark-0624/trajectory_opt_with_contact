"""
Block Multiple-Shooting Trajectory Optimization with Augmented Lagrangian (ALM)
REFACTORED VERSION - with automatic initialization and enhanced result tracking

Key improvements:
  - Automatic geometric initialization (no need for manual uInit/prKnotInit in example)
  - Initial trajectory tracking for visualization comparison
  - Loss component breakdown for analysis
  - Control gradient history (optional)

Decision variables:
  - uSeq [T, 2]: pusher velocity controls
  - prKnots [M+1, 2]: pusher position knots
  
Object state (q, v) propagates via physics, NOT decision variables.

Author: Juneil Park
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import numpy as np

# Local project imports
from .dynamics import step_square_pos_ip, IPMOptions


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to (-pi, pi]. Works elementwise."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class ALMConfig:
    # Outer iterations
    outerIters: int = 10
    # Inner iterations per outer iter
    innerIters: int = 200
    # Optimizer choice
    useLbfgs: bool = False
    lbfgsMaxIter: int = 20
    lbfgsHistorySize: int = 20

    # Defect ALM
    rhoDefectInit: float = 1.0
    rhoDefectMax: float = 1e6
    rhoDefectEta: float = 10.0
    tolDefect: float = 1e-4

    # Learning rate for Adam (if not LBFGS)
    lr: float = 5e-2
    lrDecayStep: int = 20
    lrDecayGamma: float = 0.5


@dataclass
class CostWeights:
    # Soft costs
    wControl: float = 1.0
    wControlSmooth: float = 1.0
    wObjVel: float = 0.0
    wTargetXY: float = 1.0  
    wTargetOrient: float = 1.0


class BlockMultipleShootingWithALM:
    def __init__(
        self,
        mass: float = 1.0,
        sideLength: float = 0.2,
        muFriction: float = 0.5,
        horizon: int = 100,
        dt: float = 0.05,
        blockSize: int = 5,
        device: str = "cuda",
        enableDefectALM: bool = True,
        # Physics / solver tuning
        ipmWarmStart: bool = True,
        ipmOpts: Optional[IPMOptions] = None,
        skipSolvingThreshold: float = 100.0,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.m = float(mass)
        self.side = float(sideLength)
        self.half = float(sideLength) / 2.0
        self.mu = float(muFriction)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.Izz = (1.0 / 6.0) * self.m * (self.side ** 2 + self.side ** 2)

        self.blockSize = int(blockSize)
        if self.blockSize <= 0:
            raise ValueError("blockSize must be positive")

        # knot count M such that last knot covers T
        self.numBlocks = math.ceil(self.horizon / self.blockSize)
        self.numKnots = self.numBlocks + 1  # includes knot 0

        self.enableDefectALM = bool(enableDefectALM)
        self.ipmWarmStart = bool(ipmWarmStart)

        self.ipmOpts = ipmOpts if ipmOpts is not None else IPMOptions()
        self.skipSolvingThreshold = float(skipSolvingThreshold)

        self.rhoDefect: float = 0.0

    def resetAlmState(self, cfg: ALMConfig) -> None:
        self.rhoDefect = cfg.rhoDefectInit
        # Defect: [dpx, dpy] = 2 components (pusher position only)
        self.lamDefect = torch.zeros(self.numBlocks, 2, device=self.device)

    # -----------------------------------------------------------------------------
    # Simple geometric initializer (pusher position path -> velocity inputs)
    # -----------------------------------------------------------------------------
    def computeGeometricVelocityInit(self, pusher0, box0, goal, contactOffset=1e-4):
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

        steps1 = max(5, min(self.horizon - 5, int(self.horizon * (distToBox / total))))
        steps2 = self.horizon - steps1

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
        uArr = (pArr[1:] - pArr[:-1]) / self.dt  # (T,2)
        return uArr.astype(np.float32), pArr.astype(np.float32)
    
    def initializeKnotsFromForwardSim(self, q0, v0, pr0, uInit):
        """Forward simulation으로 physics-consistent knot 초기화"""
        print(f"Running forward simulation for initial trajectory")
        K = self.numKnots
        B = self.blockSize
        
        prKnotInit = torch.zeros(K, 2, device=self.device)
        
        prKnotInit[0] = pr0

        qsInit = []
        vsInit = []
        prsInit = []
        
        # Forward simulate
        q, v, pr = q0.clone(), v0.clone(), pr0.clone()
        z_prev = None
        
        for j in range(self.numBlocks):
            t0 = j * B
            t1 = min((j + 1) * B, self.horizon)
            
            with torch.no_grad():
                for t in range(t0, t1):
                    u = uInit[t]
                    q, v, pr, _, _, z_prev = step_square_pos_ip(
                        q, v, pr, u, 
                        h=self.dt, m=self.m, Izz=self.Izz,
                        half=self.half, mu=self.mu,
                        ipm_opts=self.ipmOpts,
                        skip_solving_threshold=self.skipSolvingThreshold,
                        z_prev=z_prev if self.ipmWarmStart else None
                    )
                    if self.ipmWarmStart and z_prev is not None:
                        z_prev = z_prev.detach()
                    qsInit.append(q)
                    vsInit.append(v)
                    prsInit.append(pr)
            
            prKnotInit[j + 1] = pr

        return prKnotInit, torch.stack(qsInit), torch.stack(vsInit), torch.stack(prsInit)



    def _simulateBlock(
        self,
        qStart: torch.Tensor,     # [3]
        vStart: torch.Tensor,     # [3]
        prStart: torch.Tensor,    # [2]
        uBlock: torch.Tensor,     # [B,2] velocities
        steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate one block of length 'steps' using IPM step.
        Returns (qEnd, vEnd, prEnd, objVelEnergySum, lamdas, phis, qs, vs, qrobot_hist).
        """
        q = qStart
        v = vStart
        pr = prStart
        lamdas = []
        phis = []
        qs = [qStart]
        vs = [vStart]
        qrobot_hist = [prStart]
        objVelEnergy = torch.zeros((), device=self.device)

        # Warm-start variable for IPM step (z_prev)
        zPrev = None

        for k in range(steps):
            uVel = uBlock[k]
            q, v, pr, lam, phi, zPrev = step_square_pos_ip(
                    q, v, pr, uVel, h=self.dt, m=self.m, Izz=self.Izz,
                    half=self.half, mu=self.mu,
                    ipm_opts=self.ipmOpts, skip_solving_threshold=self.skipSolvingThreshold,
                    z_prev=zPrev if (self.ipmWarmStart and zPrev is not None) else None,
                )
            if self.ipmWarmStart and zPrev is not None:
                zPrev = zPrev.detach()
            lamdas.append(lam)
            phis.append(phi)
            qs.append(q)
            vs.append(v)
            qrobot_hist.append(pr)
            objVelEnergy = objVelEnergy + (v[:2] ** 2).sum()

        return q, v, pr, objVelEnergy, lamdas, phis, qs, vs, qrobot_hist

    def check_stationarity(self, outer, uSeq, prKnots, 
                        defectNorm, terminalNorm, cfg):
        """
        Check KKT-like stationarity conditions for ALM.
        """
        with torch.no_grad():
            # 1. Gradient norms
            grad_u = uSeq.grad.norm().item() if uSeq.grad is not None else 0.0  
            grad_pr = prKnots.grad[1:].norm().item() if prKnots.grad is not None else 0.0

            total_grad = grad_u + grad_pr

            # 2. Constraint violations
            defect_viol = defectNorm.item()
            total_viol = defect_viol
            
            # 3. Stationarity criteria
            grad_threshold = 1e-2
            feas_threshold = cfg.tolDefect
            
            grad_ok = total_grad < grad_threshold
            feas_ok = total_viol < feas_threshold
            
            print(f"\n[Stationarity Check - Outer {outer+1}]")
            print(f"  Gradients: ||∇u||={grad_u:.3e}, ||∇pr||={grad_pr:.3e}")
            print(f"    Total: {total_grad:.3e} {'✅' if grad_ok else '❌'} (threshold: {grad_threshold:.3e})")
            print(f"  Constraints: Defect={defect_viol:.3e}")
            print(f"    Total: {total_viol:.3e} {'✅' if feas_ok else '❌'} (threshold: {feas_threshold:.3e})")

            is_stationary = grad_ok and feas_ok
            print(f"  Stationarity: {'✅ CONVERGED' if is_stationary else '❌ Not converged'}")
            
            return is_stationary, {
                "grad_u": grad_u,
                "grad_pr": grad_pr,
                "total_grad": total_grad,
                "defect_viol": defect_viol,
                "is_stationary": is_stationary,
            }

    def optimize(
        self,
        q0: torch.Tensor,
        v0: torch.Tensor,
        pr0: torch.Tensor,
        goalXY: torch.Tensor,
        goalTheta: Optional[float] = None,
        cfg: Optional[ALMConfig] = None,
        w: Optional[CostWeights] = None,
        # **NEW: Optional overrides for initialization**
        uInit: Optional[torch.Tensor] = None,
        prKnotInit: Optional[torch.Tensor] = None,
        # **NEW: Track control gradients**
        track_gradients: bool = False,
    ) -> Dict:
        """
        Optimize trajectory with automatic initialization.
        
        Args:
            q0, v0, pr0: Initial state
            goalXY: Goal position [2]
            goalTheta: Goal orientation (optional)
            cfg: ALM configuration
            w: Cost weights
            uInit: Override control initialization [T, 2] (optional)
            prKnotInit: Override pusher knot initialization [M+1, 2] (optional)
            track_gradients: If True, store control gradient history
        
        Returns:
            Dictionary with:
                - Optimized trajectory and controls
                - Initial trajectory for comparison
                - Loss component breakdown
                - (Optional) Control gradient history
        """
        if cfg is None:
            cfg = ALMConfig()
        if w is None:
            w = CostWeights()
        
        
        if uInit is None:
            print("🔧 Computing geometric initialization...")
            uInitNp, pInitNp = self.computeGeometricVelocityInit(
                pusher0=pr0.detach().cpu().numpy(),
                box0=q0.detach().cpu().numpy(),
                goal=goalXY.detach().cpu().numpy(),
                contactOffset=1e-2,
            )
            uInit = torch.tensor(uInitNp, device = self.device)


        if prKnotInit is None:
            prKnotInit, qs_init, vs_init, prs_init  = self.initializeKnotsFromForwardSim(
            q0, v0, pr0,
            uInit
            )

        
        # Initialize decision variables
        uSeq = torch.nn.Parameter(uInit.clone())
        prKnots = torch.nn.Parameter(prKnotInit.clone())
        prKnots.requires_grad = True
        
        # ALM state
        self.resetAlmState(cfg)
        
        # Optimizer
        if cfg.useLbfgs:
            optimizer = torch.optim.LBFGS(
                [uSeq, prKnots],
                lr=1.0,
                max_iter=cfg.lbfgsMaxIter,
                history_size=cfg.lbfgsHistorySize,
                line_search_fn="strong_wolfe"
            )
            sched = None
        else:
            optimizer = torch.optim.Adam([uSeq, prKnots], lr=cfg.lr)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',factor=0.5,patience=25,min_lr=1e-6
            )
        
        # History tracking
        history = {
            "outerLoss": [],
            "terminalXYNorm": [],
            "terminalThetaAbs": [],
            "defectNorm": [],
            "rhoDefect": [],
        }
        
        # **NEW: Final gradient storage**
        final_grad_u = None
        
        # Constants
        B = self.blockSize
        T = self.horizon
        
        def _enforceInitialKnot():
            """Enforce prKnots[0] = pr0"""
            with torch.no_grad():
                prKnots[0] = pr0
        
        # ========================================================================
        # OUTER ALM LOOP
        # ========================================================================
        for outer in range(cfg.outerIters):
            print(f"\n{'='*70}")
            print(f"OUTER ITERATION {outer+1}/{cfg.outerIters}")
            print(f"{'='*70}")
            
            _enforceInitialKnot()
            
            # State dict to share between closure and outer loop
            state = {}
            
            def closure():
                optimizer.zero_grad(set_to_none=True)
                
                # Control energy
                controlEnergyCost = w.wControl * (uSeq ** 2).sum()
                controlSmoothCost = w.wControlSmooth * ((uSeq[1:] - uSeq[:-1]) ** 2).sum()
                
                # Simulate all blocks
                qs = []
                vs = []
                qrobot_hist = []
                lamdas = []
                phis = []
                objVelEnergy = torch.zeros((), device=self.device)
                
                almDefect = torch.zeros((), device=self.device)
                defectNormSq = torch.zeros((), device=self.device)
                
                qCurr, vCurr = q0, v0
                
                for j in range(self.numBlocks):
                    t0 = j * B
                    t1 = min(t0 + B, T)
                    steps = t1 - t0
                    
                    qStart = qCurr
                    vStart = vCurr
                    prStart = prKnots[j]
                    
                    qEndPred, vEndPred, prEndPred, objVelEnergyBlock, lamdaBlock, phisBlock, qsBlock, vsBlock, qrobot_histBlock = self._simulateBlock(
                        qStart, vStart, prStart, uSeq[t0:t1], steps
                    )
                    objVelEnergy = objVelEnergy + objVelEnergyBlock
                    
                    if j == 0:
                        qs.extend(qsBlock)
                        vs.extend(vsBlock)
                        qrobot_hist.extend(qrobot_histBlock)
                    else:
                        qs.extend(qsBlock[1:])
                        vs.extend(vsBlock[1:])
                        qrobot_hist.extend(qrobot_histBlock[1:])
                    
                    lamdas.extend(lamdaBlock)
                    phis.extend(phisBlock)
                    
                    if self.enableDefectALM:
                        dpr = prKnots[j + 1] - prEndPred
                        defectNormSq = defectNormSq + (dpr ** 2).sum()
                        
                        lam = self.lamDefect[j]
                        almDefect = almDefect + (lam * dpr).sum() + 0.5 * self.rhoDefect * (dpr ** 2).sum()
                    
                    qCurr, vCurr = qEndPred, vEndPred
                
                # Terminal cost (soft)
                objVelEnergyCost = w.wObjVel * objVelEnergy
                rXY = qCurr[:2] - goalXY
                terminalXYNormSq = (rXY ** 2).sum()
                targetCost = w.wTargetXY * terminalXYNormSq
                
                targetOrientCost = torch.zeros((), device=self.device)
                if goalTheta is not None:
                    rTh = wrap_to_pi(qCurr[2] - torch.as_tensor(goalTheta, device=self.device))
                    terminalOrientNormSq = (rTh ** 2)
                    targetOrientCost = w.wTargetOrient * terminalOrientNormSq
                
                # Total objective
                soft = (
                    controlEnergyCost
                    + controlSmoothCost
                    + objVelEnergyCost
                    + targetCost
                    + targetOrientCost
                )
                
                total = soft + almDefect
                
                total.backward()
                
                # Store state for outer loop
                state["total"] = total
                state["soft"] = soft
                state["almDefect"] = almDefect
                state["controlEnergy"] = controlEnergyCost
                state["controlSmooth"] = controlSmoothCost
                state["objVelEnergy"] = objVelEnergyCost
                state["targetDiffNorm"] = terminalXYNormSq
                state["targetOrientDiffNorm"] = terminalOrientNormSq if goalTheta is not None else torch.zeros(())
                state["lamdas"] = torch.stack(lamdas)
                state["phis"] = torch.stack(phis)
                state["qs"] = torch.stack(qs)
                state["vs"] = torch.stack(vs)
                state["qrobot_hist"] = torch.stack(qrobot_hist)
                
                return total
            
            # Inner optimization
            lossVal = None
            if cfg.useLbfgs:
                for _ in range(cfg.innerIters):
                    lossVal = optimizer.step(closure)
            else:
                for _ in range(cfg.innerIters):
                    lossVal = closure().item()
                    optimizer.step()
                    sched.step(lossVal)
                    
            # Extract final results
            lossVal = state["total"]
            qs = state["qs"]
            vs = state["vs"]
            pusherTraj = state["qrobot_hist"]
            contactImpulses = state["lamdas"]
            signedDistances = state["phis"]
            controlEnergy = state["controlEnergy"]
            controlSmooth = state["controlSmooth"]
            objVelEnergy = state["objVelEnergy"]
            targetDiffNorm = state["targetDiffNorm"]
            targetOrientDiffNorm = state["targetOrientDiffNorm"]
            
            # ========================================================================
            # OUTER LOOP: Update ALM multipliers
            # ========================================================================
            with torch.no_grad():
                _enforceInitialKnot()
                
                # Terminal residual
                rXY = (qs[-1][:2] - goalXY)
                terminalXYNorm = rXY.norm()
                dpr_debug = []
                
                # Defect residual
                defectNorm = torch.zeros((), device=self.device)
                if self.enableDefectALM:
                    for j in range(self.numBlocks):
                        t_knot = (j + 1) * B
                        t_knot = min(t_knot, len(qs) - 1)
                        
                        pr_sim = pusherTraj[t_knot]
                        dpr = prKnots[j + 1] - pr_sim
                        dpr_debug.append(dpr.norm())
                        defectNorm = defectNorm + dpr.norm()
                        
                        # Update Lagrange multiplier
                        self.lamDefect[j] = self.lamDefect[j] + self.rhoDefect * dpr
                    
                    # Rho schedule
                    if defectNorm.item() > cfg.tolDefect:
                        self.rhoDefect = min(self.rhoDefect * cfg.rhoDefectEta, cfg.rhoDefectMax)
                
                # Terminal theta residual
                terminalThetaAbs = torch.zeros((), device=self.device)
                if goalTheta is not None:
                    rTh = wrap_to_pi(qs[-1][2] - torch.as_tensor(goalTheta, device=self.device))
                    terminalThetaAbs = rTh.abs()
                
                # Check stationarity
                is_stationary, stat_info = self.check_stationarity(
                    outer, uSeq, prKnots, 
                    defectNorm, terminalXYNorm, cfg
                )
                print(f"Pusher position defect norms at knots: {torch.stack(dpr_debug).cpu().numpy()}")
                
                print(
                    f"Loss={lossVal.item():.4f}, "
                    f"Final obj pose: {qs[-1].detach().cpu().numpy()}, "
                    f"Goal pose: {[*goalXY.detach().cpu().numpy(), goalTheta]}"
                )
                
                # Log
                history["outerLoss"].append(lossVal.detach().cpu())
                history["terminalXYNorm"].append(terminalXYNorm.detach().cpu())
                history["terminalThetaAbs"].append(terminalThetaAbs.detach().cpu())
                history["defectNorm"].append(defectNorm.detach().cpu())
                history["rhoDefect"].append(torch.tensor(self.rhoDefect))
                
                if is_stationary and outer >= 2:
                    print("Early stopping: stationarity achieved!")
                    break
                
                print(
                    f"Outer iter {outer+1}/{cfg.outerIters}: "
                    f"Loss={lossVal.item():.4f}, "
                    f"||rXY||={terminalXYNorm.item():.6f}, "
                    f"|rTh|={terminalThetaAbs.item():.6f}, "
                    f"DefectNorm={defectNorm.item():.6f}, "
                    f"rhoDefect={self.rhoDefect:.1f}"
                )
                print(
                    f"Cost details: total = {lossVal.item():.4f}, " 
                    f"wControlEnergy={w.wControl*controlEnergy.item():.4f}, "
                    f"wControlSmooth={w.wControlSmooth*controlSmooth.item():.4f}, "
                    f"wObjVelEnergy={w.wObjVel*objVelEnergy.item():.4f}, "
                    f"wTargetXY={w.wTargetXY*targetDiffNorm.item():.4f}, "
                    f"wTargetOrient={w.wTargetOrient*targetOrientDiffNorm.item():.4f}"
                )
        
        # **NEW: Capture final gradient after inner loop**
        if track_gradients and uSeq.grad is not None:
            final_grad_u = uSeq.grad.detach().clone().cpu().numpy()

        print("ALM optimization completed.")
        
        return {
            # Optimized trajectory
            "loss": lossVal.detach().cpu(),
            "q_final": qs[-1].detach().cpu(),
            "u_seq": uSeq.detach().cpu().numpy(),
            "trajectory": qs.detach().cpu().numpy(),
            "velocity_trajectory": vs.detach().cpu().numpy(),
            "pusher_trajectory": pusherTraj.detach().cpu().numpy(),
            "prKnots": prKnots.detach().cpu().numpy(),
            "contact_forces": contactImpulses.detach().cpu().numpy(),
            "signed_distances": signedDistances.detach().cpu().numpy(),
            
            # **NEW: Initial trajectory for visualization**
            "initial_trajectory": qs_init.cpu().numpy(),
            "initial_velocity_trajectory": vs_init.cpu().numpy(),
            "initial_pusher_trajectory": prs_init.cpu().numpy(),
            
            # **NEW: Loss component breakdown**
            "loss_components": {
                "total": lossVal.item(),
                "control_energy": controlEnergy.item(),
                "control_smooth": controlSmooth.item(),
                "obj_vel": objVelEnergy.item(),
                "target_xy": targetDiffNorm.item(),
                "target_orient": targetOrientDiffNorm.item(),
                "alm_defect": state["almDefect"].item(),
                # Weights for legend
                "w_control": w.wControl,
                "w_smooth": w.wControlSmooth,
                "w_objvel": w.wObjVel,
                "w_targetxy": w.wTargetXY,
                "w_orient": w.wTargetOrient,
                "rho_defect": self.rhoDefect,
            },
            
            # **NEW: Final control gradient (last outer iter, last inner iter)**
            "control_gradients": final_grad_u,
            
            # History and diagnostics
            "history": {k: torch.stack(v).detach().cpu().numpy() for k, v in history.items()},
            "stationarityInfo": {k: v for k, v in stat_info.items()},
        }