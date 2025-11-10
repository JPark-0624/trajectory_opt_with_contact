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
                backend: str = "cvxpy"):
        self.mu = mu
        self.n_contacts = n_contacts
        self.dim = 2 * n_contacts  # [λ_n, λ_t] per contact
        self.backend = backend

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
        else:
            raise ValueError("backend must be 'cvxpy' or 'qpth'")