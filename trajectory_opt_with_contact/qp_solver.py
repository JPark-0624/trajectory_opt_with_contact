"""
QP Solver for Contact Forces

Implements a differentiable quadratic programming layer for computing
contact forces with Coulomb friction constraints.
"""

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

class ContactQPSolver:
    """
    Differentiable QP solver for contact forces with friction cone constraints.
    
    Solves: min 0.5 * λ^T G λ + b^T λ
            s.t. λ_n >= 0
                 |λ_t| <= μ * λ_n  (friction cone)
    
    Args:
        mu: Coefficient of friction
        n_contacts: Number of contact points (default: 1)
    """
    
    def __init__(self, mu=0.6, n_contacts=1,
                backend: str = "cvxpy",          # "cvxpy" or "qpth"
                ipm_eps: float = 1e-4,           # target duality gap / accuracy for qpth
                ipm_max_iter: int = 50):
        self.mu = mu
        self.n_contacts = n_contacts
        self.dim = 2 * n_contacts  # [λ_n, λ_t] per contact
        self.backend = backend
        self.ipm_eps = ipm_eps
        self.ipm_max_iter = ipm_max_iter

        # Build the QP problem
        self._build_qp()
    
    def _build_qp(self):
        if self.backend == "cvxpy":
            """Construct the CVXPY problem and create CvxpyLayer."""
            # Decision variable: contact forces [λ_n1, λ_t1, λ_n2, λ_t2, ...]
            lam = cp.Variable(self.dim)
            
            # Parameters (passed from PyTorch)
            Q_param = cp.Parameter((self.dim, self.dim), PSD=True)  # Cholesky factor
            b_param = cp.Parameter(self.dim)                         # Linear term
            
            # Objective: 0.5 * ||Q @ λ||^2 + b^T λ
            # (reformulated to avoid quad_form for better numerical stability)
            objective = cp.Minimize(0.5 * cp.sum_squares(Q_param @ lam) + b_param @ lam)
            
            # Constraints
            constraints = []
            
            # Non-penetration: λ_n >= 0 for all contacts
            for i in range(self.n_contacts):
                idx_n = 2 * i
                constraints.append(lam[idx_n] >= 0)
            
            # Friction cone: |λ_t| <= μ * λ_n for all contacts
            for i in range(self.n_contacts):
                idx_n = 2 * i
                idx_t = 2 * i + 1
                constraints.append(lam[idx_t] <= self.mu * lam[idx_n])
                constraints.append(lam[idx_t] >= -self.mu * lam[idx_n])
            
            # Create the problem and layer
            problem = cp.Problem(objective, constraints)
            self.qp_layer = CvxpyLayer(problem, parameters=[Q_param, b_param], variables=[lam])
        elif self.backend == "qpth":
            # qpth is a differentiable primal-dual interior-point method
            from qpth.qp import QPFunction
            self.QPFunction = QPFunction
        else:
            raise ValueError("backend must be 'cvxpy' or 'qpth'")

    def _build_ineq_mats(self, device, dtype):
        # G λ ≤ h  for λ_n ≥ 0 and |λ_t| ≤ μ λ_n
        # For each contact i: rows:
        # [-1, 0]·[λ_n,λ_t] ≤ 0           (λ_n ≥ 0)
        # [-μ, 1]·[λ_n,λ_t] ≤ 0           ( λ_t - μ λ_n ≤ 0 )
        # [-μ,-1]·[λ_n,λ_t] ≤ 0           ( -λ_t - μ λ_n ≤ 0 )
        rows = []
        for i in range(self.n_contacts):
            r = torch.zeros(3, self.dim, device=device, dtype=dtype)
            # λ_n index
            in_n = 2 * i
            in_t = in_n + 1
            # -λ_n ≤ 0
            r[0, in_n] = -1.0
            # -μ λ_n + 1*λ_t ≤ 0
            r[1, in_n] = -self.mu
            r[1, in_t] = 1.0
            # -μ λ_n - 1*λ_t ≤ 0
            r[2, in_n] = -self.mu
            r[2, in_t] = -1.0
            rows.append(r)
        G = torch.vstack(rows)                                # (3*n, 2*n)
        h = torch.zeros(G.size(0), device=device, dtype=dtype) # zeros
        return G, h

    def solve(self, Q_chol, b):
        """
        Solve the contact QP.
        
        Args:
            Q: Cholesky factor of the Delassus matrix (dim x dim)
            b: Linear term (dim,)
        
        Returns:
            lam_star: Optimal contact forces (dim,)
        """
        if self.backend == 'cvxpy':
            lam_star, = self.qp_layer(Q_chol, b)
            return lam_star
        elif self.backend == 'qpth':
             # qpth path (interior-point)
            device, dtype = Q_chol.device, Q_chol.dtype
            dim = self.dim
            # H = Q^T Q (make PSD -> add tiny reg to ensure PD)
            H = Q_chol.transpose(0, 1) @ Q_chol + 1e-9 * torch.eye(dim, device=device, dtype=dtype)
            f = b  # linear term
            G, h = self._build_ineq_mats(device, dtype)
            # No equalities (A,b) -> use empty tensors with proper shape
            A = torch.empty(0, dim, device=device, dtype=dtype)
            a = torch.empty(0, device=device, dtype=dtype)

            # qpth expects batch; unsqueeze and squeeze
            Q = H.unsqueeze(0)
            p = f.unsqueeze(0)
            G_ = G.unsqueeze(0)
            h_ = h.unsqueeze(0)
            A_ = A.unsqueeze(0)
            a_ = a.unsqueeze(0)

            lam = self.QPFunction(
                eps=self.ipm_eps,          # accuracy / (duality gap target-ish)
                verbose=False,
                maxIter=self.ipm_max_iter,
                notImprovedLim=10,
                #check_Q_spd=False
            )(Q, p, G_, h_, A_, a_).squeeze(0)
            return lam