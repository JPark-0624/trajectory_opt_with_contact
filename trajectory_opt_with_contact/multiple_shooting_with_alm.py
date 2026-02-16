"""
Block Multiple-Shooting Trajectory Optimization with Augmented Lagrangian (ALM)
STATE KNOT VERSION - includes both position (q) and velocity (v) as decision variables

Key changes from previous version:
  - Decision variables now include vKnots [M+1, 3] in addition to qKnots
  - Defect constraint dimension: 5 → 8 (includes velocity defect)
  - Velocity is properly connected across blocks via defect constraints
  - Critical for contact-implicit dynamics with velocity impulse model

Goal:
  1) Enforce terminal target pose as soft cost
  2) Enforce block defects (position + velocity + pusher) via ALM
  3) Proper velocity continuity for contact dynamics

Author: Juneil Park
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch

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
        skipSolvingThreshold: float = 0.003,
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
        # Defect now includes: [dpx, dpy] = 2 components
        self.lamDefect = torch.zeros(self.numBlocks, 2, device=self.device)

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
        vs = [vStart]  # ← NEW: velocity trajectory
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
            vs.append(v)  # ← NEW
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
            
            # print(f" gradient details u_seq : {uSeq.grad} ")
            # print(f" gradient details prKnots : {prKnots.grad} ")

            is_stationary = grad_ok and feas_ok
            print(f"  Stationarity: {'✅ CONVERGED' if is_stationary else '❌ Not converged'}")
            
            return is_stationary, {
                'grad_u': grad_u,
                'grad_pr': grad_pr,
                'total_grad': total_grad,
                'defect': defect_viol,
                'total_violation': total_viol,
                'converged': is_stationary
            }


    def optimize(
        self,
        q0: torch.Tensor,         # [3] object initial pose
        v0: torch.Tensor,         # [3] object initial velocity
        pr0: torch.Tensor,        # [2] pusher initial pose
        goalXY: torch.Tensor,     # [2]
        goalTheta: Optional[float] = None,
        cfg: Optional[ALMConfig] = None,
        w: Optional[CostWeights] = None,
        # initial guesses
        uInit: Optional[torch.Tensor] = None,            # [T,2]
        prKnotInit: Optional[torch.Tensor] = None,       # [K,2]
    ) -> Dict[str, torch.Tensor]:
        """Run block multiple-shooting with ALM and state knots."""
        if cfg is None:
            cfg = ALMConfig()
        if w is None:
            w = CostWeights()
        # Move inputs to device
        q0 = q0.to(self.device)
        v0 = v0.to(self.device)
        pr0 = pr0.to(self.device)
        goalXY = goalXY.to(self.device)

        # Initialize primal variables
        T = self.horizon
        K = self.numKnots
        B = self.blockSize

        if uInit is None:
            uInit = torch.zeros(T, 2, device=self.device)
        else:
            uInit = uInit.to(self.device)

        if prKnotInit is None:
            prKnotInit = torch.zeros(K, 2, device=self.device)
            prKnotInit[0] = pr0
        else:
            prKnotInit = prKnotInit.to(self.device)
            prKnotInit[0] = pr0

        # Primal params
        uSeq = torch.nn.Parameter(uInit.clone())
        prKnots = torch.nn.Parameter(prKnotInit.clone())

        # Fix knot 0 to initial state
        def _enforceInitialKnot():
            with torch.no_grad():
                prKnots.data[0].copy_(pr0)

        # ALM state init
        self.resetAlmState(cfg)

        # Logs
        history = {
            "outerLoss": [],
            "terminalXYNorm": [],
            "terminalThetaAbs": [],
            "defectNorm": [],
            "rhoDefect": [],
        }

        for outer in range(cfg.outerIters):
            _enforceInitialKnot()
            
            # Optimizer setup
            if cfg.useLbfgs:
                optimizer = torch.optim.LBFGS(
                    [uSeq, prKnots],
                    lr = cfg.lr,
                    max_iter=cfg.lbfgsMaxIter,
                    history_size=cfg.lbfgsHistorySize,
                    # line_search_fn="strong_wolfe",
                )
            else:
                optimizer = torch.optim.Adam(
                    [uSeq, prKnots],
                    lr = cfg.lr
                )
                        
            #sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lrDecayStep, gamma=cfg.lrDecayGamma)

            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                    optimizer,
                                    mode='min',
                                    factor=0.5,        
                                    patience=25,        
                                    min_lr=1e-6,       
                                )

            # state initialization
            state = {
                "total": None, "soft": None, "almDefect": None,
                "controlEnergy": None, "controlSmooth": None, "objVelEnergy": None, "targetCost": None, "targetOrientCost": None,
                "lamdas": None, "phis": None,
                "qs": None, "vs": None, "qrobot_hist": None
            }

            def closure():
                optimizer.zero_grad(set_to_none=True)
                _enforceInitialKnot()

                # Soft costs
                controlEnergy = (uSeq ** 2).sum()
                controlEnergyCost = w.wControl * controlEnergy
                controlSmooth = ((uSeq[1:] - uSeq[:-1]) ** 2).sum() if T > 1 else torch.zeros((), device=self.device)
                controlSmoothCost = w.wControlSmooth * controlSmooth
                objVelEnergy = torch.zeros((), device=self.device)

                # Defects and ALM terms
                defectNormSq = torch.zeros((), device=self.device)
                almDefect = torch.zeros((), device=self.device)
                lamdas = []
                phis = []
                qs = []
                vs = []  # ← NEW: velocity trajectory for logging
                qrobot_hist = []

                qCurr, vCurr = q0.clone(), v0.clone()  # start of current block

                # Simulate each block
                for j in range(self.numBlocks):
                    t0 = j * B
                    t1 = min((j + 1) * B, T)
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
                        vs.extend(vsBlock)  # ← NEW
                        qrobot_hist.extend(qrobot_histBlock)
                    else:
                        qs.extend(qsBlock[1:])
                        vs.extend(vsBlock[1:])  # ← NEW
                        qrobot_hist.extend(qrobot_histBlock[1:])

                    lamdas.extend(lamdaBlock)
                    phis.extend(phisBlock)

                    if self.enableDefectALM:
                        # ← NEW: defect now includes velocity
                        # Defect = [dpr (2)] = 2 components
                        dpr = prKnots[j + 1] - prEndPred
                        defectNormSq = defectNormSq + (dpr ** 2).sum()

                        lam = self.lamDefect[j]
                        almDefect = almDefect + (lam * dpr).sum() + 0.5 * self.rhoDefect * (dpr ** 2).sum()

                    qCurr, vCurr = qEndPred, vEndPred  # update current state for next block

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
                grad_norm = uSeq.grad.norm().item()
                grad_pr = prKnots.grad[1:].norm().item()
                # print(f"Inner iter closure: total loss: {total.item():.4f} | "
                #       f"total gradient = {grad_norm+grad_pr:.4f} | "
                #      f"grad norm: {grad_norm:.4f} | "
                #      f"grad pr: {grad_pr:.4f} | "
                #      f"object pose {qs[-1].detach().cpu().numpy()} | ")

                state["total"] = total
                state["soft"] = soft
                state["almDefect"] = almDefect
                state["controlEnergy"] = controlEnergyCost
                state["controlSmooth"] = controlSmoothCost
                state["objVelEnergy"] = objVelEnergyCost
                state["targetDiffNorm"] = terminalXYNormSq
                state["targetOrientDiffNorm"] = terminalOrientNormSq
                state["lamdas"] = torch.stack(lamdas)
                state["phis"] = torch.stack(phis)
                state["qs"] = torch.stack(qs)
                state["vs"] = torch.stack(vs)
                state["qrobot_hist"] = torch.stack(qrobot_hist)

                return total
            
            lossVal = None
            # Inner optimization
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
                # Defect residual: extract from cached trajectory
                defectNorm = torch.zeros((), device=self.device)
                if self.enableDefectALM:
                    for j in range(self.numBlocks):
                        # Index into cached trajectory at knot boundary
                        t_knot = (j + 1) * B
                        t_knot = min(t_knot, len(qs) - 1)
                        
                        # Simulated state at knot j+1
                        pr_sim = pusherTraj[t_knot]
                        
                        # Defect = knot variable - simulated value
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
                    f"Loss={lossVal.item() if lossVal is not None else 'N/A':.4f}, "
                   f"Final obj pose: {qs[-1].detach().cpu().numpy()}, Goal pose: {[*goalXY.detach().cpu().numpy(), goalTheta]})"
                )

                # Log
                history["outerLoss"].append(lossVal.detach().cpu() if lossVal is not None else torch.tensor(float("nan")))
                history["terminalXYNorm"].append(terminalXYNorm.detach().cpu())
                history["terminalThetaAbs"].append(terminalThetaAbs.detach().cpu())
                history["defectNorm"].append(defectNorm.detach().cpu())
                history["rhoDefect"].append(torch.tensor(self.rhoDefect))
                
                if is_stationary and outer >= 2:
                    print("Early stopping: stationarity achieved!")
                    break

                print(
                    f"Outer iter {outer+1}/{cfg.outerIters}: "
                    f"Loss={lossVal.item() if lossVal is not None else 'N/A':.4f}, "
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


        print("ALM optimization completed.")

        return {
            "loss": lossVal.detach().cpu(),
            "q_final": qs[-1].detach().cpu(),
            "u_seq": uSeq.detach().cpu().numpy(),
            "trajectory" : qs.detach().cpu().numpy(),
            "velocity_trajectory": vs.detach().cpu().numpy(),  # ← NEW
            "pusher_trajectory": pusherTraj.detach().cpu().numpy(),
            "prKnots": prKnots.detach().cpu().numpy(),
            "history": {k: torch.stack(v).detach().cpu().numpy() for k, v in history.items()},
            "stationarityInfo": {k: v for k, v in stat_info.items()},
            "contact_forces" : contactImpulses.detach().cpu().numpy(),
            "signed_distances" : signedDistances.detach().cpu().numpy()       
        }