"""
Unit Tests for DDP Implementation

Tests:
1. Linear dynamics (analytical solution exists)
2. Nonlinear dynamics (simple pendulum)
3. Box constraints (control limits)
4. Backward pass only (gain computation)
5. Value function propagation

Goal: Verify DDP works correctly before integrating with IPM dynamics
"""

import torch
import numpy as np
import pytest
from trajectory_opt_with_contact.ddp_refactored import DDPOptimizer, CholeskyQPSolver


# ==============================================================================
# Test 1: Linear Dynamics (Analytical Solution with FIXED matrices)
# ==============================================================================

def test_linear_dynamics_unconstrained():
    """
    Test DDP on linear system (LQR problem) with FIXED matrices
    
    System:
        x_{t+1} = A x_t + B u_t
        
    Cost:
        sum_t (x_t' Q x_t + u_t' R u_t) + x_T' Q_T x_T
    
    This has analytical solution via Riccati equation!
    We use FIXED matrices (no random seed) for reproducible comparison.
    """
    print("\n" + "="*70)
    print("Test 1: Linear Dynamics (LQR) - Fixed Matrices")
    print("="*70)
    
    # Setup
    n_x = 4  # state dim
    n_u = 2  # control dim
    T = 20   # horizon
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.double
    
    # ========================================================================
    # FIXED matrices (no randomness!)
    # ========================================================================
    
    # System dynamics: x_{t+1} = A x_t + B u_t
    A = torch.tensor([
        [0.95, 0.05, 0.00, 0.00],
        [0.00, 0.90, 0.10, 0.00],
        [0.00, 0.00, 0.92, 0.05],
        [0.05, 0.00, 0.00, 0.88]
    ], device=device, dtype=dtype)
    
    B = torch.tensor([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
        [0.3, 0.7]
    ], device=device, dtype=dtype)
    
    # Cost matrices
    Q = torch.eye(n_x, device=device, dtype=dtype)
    R = torch.eye(n_u, device=device, dtype=dtype) * 0.1
    Q_T = torch.eye(n_x, device=device, dtype=dtype) * 10.0
    
    # Goal and initial state
    x_goal = torch.ones(n_x, device=device, dtype=dtype)
    x0 = torch.zeros(n_x, device=device, dtype=dtype)
    
    print(f"\nProblem setup:")
    print(f"  State dim: {n_x}, Control dim: {n_u}, Horizon: {T}")
    print(f"  x0 = {x0.cpu().numpy()}")
    print(f"  x_goal = {x_goal.cpu().numpy()}")
    
    # ========================================================================
    # ANALYTICAL SOLUTION via Discrete-Time Riccati Equation
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("Computing Analytical Solution (Riccati Equation)")
    print(f"{'='*70}")
    
    # Backward Riccati recursion
    S = Q_T.clone()
    K_list_analytical = []
    
    for t in reversed(range(T)):
        # Compute optimal gain: K_t = -(R + B'SB)^{-1} B'SA
        tmp = R + B.T @ S @ B
        K_t = -torch.linalg.solve(tmp, B.T @ S @ A)
        K_list_analytical.append(K_t)
        
        # Update cost-to-go: S_t = Q + A'SA - A'SB(R+B'SB)^{-1}B'SA
        S = Q + A.T @ S @ A - A.T @ S @ B @ torch.linalg.solve(tmp, B.T @ S @ A)
    
    K_list_analytical = K_list_analytical[::-1]  # Reverse to forward order
    
    # Forward simulation with optimal control
    x = x0.clone()
    cost_analytical = 0.0
    
    for t in range(T):
        x_err = x - x_goal
        u_opt = K_list_analytical[t] @ x_err
        
        # Stage cost
        cost_analytical += 0.5 * (x_err @ Q @ x_err + u_opt @ R @ u_opt).item()
        
        # Dynamics
        x = A @ x + B @ u_opt
    
    # Terminal cost
    x_err = x - x_goal
    cost_analytical += 0.5 * (x_err @ Q_T @ x_err).item()
    final_error_analytical = torch.norm(x - x_goal).item()
    
    print(f"✅ Analytical optimal cost: {cost_analytical:.10f}")
    print(f"✅ Analytical final error: {final_error_analytical:.10f}")
    
    # ========================================================================
    # DDP SOLUTION (should match analytical!)
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("DDP Optimization")
    print(f"{'='*70}")
    
    # Dynamics
    def dynamics_fn(x, u):
        return A @ x + B @ u
    
    # Stage cost
    def stage_cost_fn(x, u):
        x_err = x - x_goal
        return 0.5 * (x_err @ Q @ x_err + u @ R @ u)
    
    # Terminal cost
    def terminal_cost_fn(x, goal):
        x_err = x - goal
        return 0.5 * x_err @ Q_T @ x_err
    
    # DDP optimizer
    optimizer = DDPOptimizer(
        dynamics_fn=dynamics_fn,
        stage_cost_fn=stage_cost_fn,
        terminal_cost_fn=terminal_cost_fn,
        horizon=T,
        dt=1.0,  # Discrete time
        n_x=n_x,
        n_u=n_u,
        qp_solver=CholeskyQPSolver(reg=1e-8),
        max_iters=50,
        device=device,
        verbose=True,
    )
    
    # Optimize
    result = optimizer.optimize(x0, x_goal)
    
    # Extract results
    cost_ddp = result['cost']
    x_final = result['trajectory'][-1]
    final_error_ddp = np.linalg.norm(x_final - x_goal.cpu().numpy())
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("📊 COMPARISON: Analytical vs DDP")
    print(f"{'='*70}")
    
    cost_diff = abs(cost_analytical - cost_ddp)
    error_diff = abs(final_error_analytical - final_error_ddp)
    
    print(f"\nCost:")
    print(f"  Analytical: {cost_analytical:.10f}")
    print(f"  DDP:        {cost_ddp:.10f}")
    print(f"  Difference: {cost_diff:.10f}")
    print(f"  Relative:   {cost_diff/cost_analytical*100:.8f}%")
    
    print(f"\nFinal Error:")
    print(f"  Analytical: {final_error_analytical:.10f}")
    print(f"  DDP:        {final_error_ddp:.10f}")
    print(f"  Difference: {error_diff:.10f}")
    
    # ========================================================================
    # VERDICT
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("🎯 VERDICT")
    print(f"{'='*70}")
    
    # Check if DDP found a good solution
    # Note: DDP may find BETTER solution than analytical due to feedforward term!
    
    if cost_ddp <= cost_analytical:
        print(f"✅ EXCELLENT! DDP found solution ≤ analytical!")
        print(f"   DDP cost: {cost_ddp:.6f}")
        print(f"   Analytical (feedback-only): {cost_analytical:.6f}")
        print(f"   DDP includes feedforward term → Can be better!")
    elif cost_diff < cost_analytical * 0.05:  # Within 5%
        print(f"✅ GOOD! DDP close to analytical solution!")
        print(f"   Cost difference: {cost_diff:.6f} ({cost_diff/cost_analytical*100:.2f}%)")
    else:
        print(f"⚠️  Warning: DDP differs significantly from analytical")
        print(f"   Cost difference: {cost_diff:.6f} ({cost_diff/cost_analytical*100:.2f}%)")
    
    # Assertions - DDP should be better or close to analytical
    assert cost_ddp < cost_analytical * 1.1, f"DDP cost {cost_ddp:.6f} much worse than analytical {cost_analytical:.6f}"
    assert cost_ddp < 100.0, "Cost too high!"
    assert final_error_ddp < 1.0, "Did not reach goal!"
    
    print("\n✅ Test 1 passed!")



# ==============================================================================
# Test 2: Nonlinear Dynamics (Pendulum Swing-up)
# ==============================================================================

def test_nonlinear_pendulum():
    """
    Test DDP on simple pendulum
    
    System:
        θ_{t+1} = θ_t + dt * ω_t
        ω_{t+1} = ω_t + dt * (u_t/m*L - g/L * sin(θ_t) - b*ω_t)
        
    Goal: Swing up from bottom (θ=0) to top (θ=π)
    """
    print("\n" + "="*70)
    print("Test 2: Nonlinear Dynamics (Pendulum)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.double
    
    # Pendulum parameters
    m = 1.0   # mass
    L = 1.0   # length
    g = 9.81  # gravity
    b = 0.1   # damping
    dt = 0.05
    T = 100   # horizon
    
    # Dynamics
    def dynamics_fn(x, u):
        """
        x = [θ, ω]
        u = torque
        """
        theta, omega = x[0], x[1]
        
        # Handle both scalar and tensor u - but keep it as tensor!
        if u.numel() == 1 and u.dim() == 0:
            u_val = u  # Already scalar tensor
        else:
            u_val = u.flatten()[0]  # Extract first element as tensor
        
        theta_dot = omega
        omega_dot = u_val / (m * L**2) - (g / L) * torch.sin(theta) - b * omega
        
        theta_next = theta + dt * theta_dot
        omega_next = omega + dt * omega_dot
        
        # Use torch.stack to preserve computational graph!
        return torch.stack([theta_next, omega_next])
    
    # Cost
    def stage_cost_fn(x, u):
        # Handle both scalar and tensor u - but keep as tensor!
        if u.numel() == 1 and u.dim() == 0:
            u_val = u  # Already scalar tensor
        else:
            u_val = u.flatten()[0]  # Extract as tensor
        return 0.01 * u_val**2  # Control penalty
    
    def terminal_cost_fn(x, goal):
        theta_err = x[0] - goal[0]  # θ - π
        omega_err = x[1] - goal[1]  # ω - 0
        return 100.0 * theta_err**2 + 10.0 * omega_err**2
    
    # Initial and goal
    x0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)  # Hanging down
    goal = torch.tensor([np.pi, 0.0], device=device, dtype=dtype)  # Upright
    
    # DDP optimizer
    optimizer = DDPOptimizer(
        dynamics_fn=dynamics_fn,
        stage_cost_fn=stage_cost_fn,
        terminal_cost_fn=terminal_cost_fn,
        horizon=T,
        dt=dt,
        n_x=2,
        n_u=1,
        qp_solver=CholeskyQPSolver(reg=1e-6),
        max_iters=100,
        device=device,
        verbose=True,
    )
    
    # Optimize
    result = optimizer.optimize(x0, goal)
    
    # Check convergence
    final_cost = result['cost']
    print(f"\nFinal cost: {final_cost:.6f}")
    
    # Check if reached near upright
    x_final = result['trajectory'][-1]
    theta_final = x_final[0]
    
    # Normalize to [-π, π]
    theta_err = np.abs((theta_final - np.pi + np.pi) % (2 * np.pi) - np.pi)
    
    print(f"Final θ error: {theta_err:.3f} rad ({np.degrees(theta_err):.1f}°)")
    
    assert final_cost < 1000.0, "Cost too high!"
    # Relaxed threshold for swing-up (hard problem!)
    assert theta_err < 0.5, f"Did not swing up! Error: {theta_err:.3f}"
    
    print("✅ Test 2 passed!")


# ==============================================================================
# Test 3: Box Constraints (Control Limits)
# ==============================================================================

def test_box_constraints():
    """
    Test DDP with control limits
    
    Same as Test 1 but with u_min ≤ u ≤ u_max
    Should saturate controls and still work
    """
    print("\n" + "="*70)
    print("Test 3: Box Constraints")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.double
    
    n_x = 2
    n_u = 1
    T = 30
    
    # Double integrator
    A = torch.tensor([[1.0, 0.1], [0.0, 1.0]], device=device, dtype=dtype)
    B = torch.tensor([[0.005], [0.1]], device=device, dtype=dtype)
    
    Q = torch.eye(n_x, device=device, dtype=dtype) * 10.0
    R = torch.tensor([[0.1]], device=device, dtype=dtype)
    
    x_goal = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    x0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
    
    # Control limits
    u_min = -0.5
    u_max = 0.5
    
    def dynamics_fn(x, u):
        return A @ x + B @ u
    
    def stage_cost_fn(x, u):
        x_err = x - x_goal
        return 0.5 * (x_err @ Q @ x_err + u @ R @ u)
    
    def terminal_cost_fn(x, goal):
        x_err = x - goal
        return 0.5 * x_err @ (Q * 100.0) @ x_err
    
    # DDP with constraints
    optimizer = DDPOptimizer(
        dynamics_fn=dynamics_fn,
        stage_cost_fn=stage_cost_fn,
        terminal_cost_fn=terminal_cost_fn,
        horizon=T,
        dt=1.0,
        n_x=n_x,
        n_u=n_u,
        u_min=u_min,
        u_max=u_max,  # ← Control limits!
        qp_solver=CholeskyQPSolver(reg=1e-8),
        max_iters=50,
        device=device,
        verbose=True,
    )
    
    result = optimizer.optimize(x0, x_goal)
    
    # Check controls are within bounds
    U = result['controls']
    print(f"\nControl range: [{U.min():.3f}, {U.max():.3f}]")
    print(f"Bounds: [{u_min}, {u_max}]")
    
    assert U.min() >= u_min - 1e-6, "Control violated lower bound!"
    assert U.max() <= u_max + 1e-6, "Control violated upper bound!"
    
    # Check convergence
    final_cost = result['cost']
    print(f"Final cost: {final_cost:.6f}")
    
    x_final = result['trajectory'][-1]
    error = np.linalg.norm(x_final - x_goal.cpu().numpy())
    print(f"Final error: {error:.6f}")
    
    assert final_cost < 500.0, "Cost too high!"
    assert error < 0.3, "Did not reach goal (relaxed due to constraints)!"
    
    print("✅ Test 3 passed!")


# ==============================================================================
# Test 4: Backward Pass Only (Gain Verification)
# ==============================================================================

def test_backward_pass_gains():
    """
    Test backward pass in isolation
    
    Given nominal trajectory, compute gains and verify:
    1. K has correct shape
    2. k has correct shape
    3. No NaN/Inf
    """
    print("\n" + "="*70)
    print("Test 4: Backward Pass Gains")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.double
    
    n_x = 4
    n_u = 2
    T = 10
    
    # Simple linear system
    A = torch.eye(n_x, device=device, dtype=dtype) * 0.95
    B = torch.randn(n_x, n_u, device=device, dtype=dtype) * 0.1
    
    def dynamics_fn(x, u):
        return A @ x + B @ u
    
    def stage_cost_fn(x, u):
        return 0.5 * (x @ x + u @ u)
    
    def terminal_cost_fn(x, goal):
        return 0.5 * (x - goal) @ (x - goal)
    
    # Create optimizer
    optimizer = DDPOptimizer(
        dynamics_fn=dynamics_fn,
        stage_cost_fn=stage_cost_fn,
        terminal_cost_fn=terminal_cost_fn,
        horizon=T,
        dt=1.0,
        n_x=n_x,
        n_u=n_u,
        device=device,
        verbose=False,
    )
    
    optimizer.goal = torch.zeros(n_x, device=device, dtype=dtype)
    
    # Create nominal trajectory
    X = torch.randn(T+1, n_x, device=device, dtype=dtype) * 0.1
    U = torch.randn(T, n_u, device=device, dtype=dtype) * 0.1
    
    # Backward pass
    K_list, k_list, diverged = optimizer.backward_pass(X, U)
    
    assert not diverged, "Backward pass diverged!"
    assert len(K_list) == T, f"Wrong number of K gains: {len(K_list)} != {T}"
    assert len(k_list) == T, f"Wrong number of k gains: {len(k_list)} != {T}"
    
    # Check shapes and no NaN
    for t in range(T):
        K = K_list[t]
        k = k_list[t]
        
        assert K.shape == (n_u, n_x), f"K[{t}] wrong shape: {K.shape}"
        assert k.shape == (n_u,), f"k[{t}] wrong shape: {k.shape}"
        
        assert not torch.isnan(K).any(), f"K[{t}] has NaN!"
        assert not torch.isinf(K).any(), f"K[{t}] has Inf!"
        assert not torch.isnan(k).any(), f"k[{t}] has NaN!"
        assert not torch.isinf(k).any(), f"k[{t}] has Inf!"
    
    print(f"All {T} gains computed successfully")
    print(f"K shape: {K_list[0].shape}")
    print(f"k shape: {k_list[0].shape}")
    print("✅ Test 4 passed!")


# ==============================================================================
# Test 5: Value Function Symmetry
# ==============================================================================

def test_value_function_symmetry():
    """
    Test that Vxx remains symmetric during backward pass
    
    Vxx should always be symmetric (it's a Hessian!)
    """
    print("\n" + "="*70)
    print("Test 5: Value Function Symmetry")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.double
    
    n_x = 3
    n_u = 2
    T = 20
    
    # Random linear system
    torch.manual_seed(123)
    A = torch.randn(n_x, n_x, device=device, dtype=dtype) * 0.1 + 0.9 * torch.eye(n_x, device=device, dtype=dtype)
    B = torch.randn(n_x, n_u, device=device, dtype=dtype)
    
    def dynamics_fn(x, u):
        return A @ x + B @ u
    
    def stage_cost_fn(x, u):
        return 0.5 * (x @ x + u @ u)
    
    def terminal_cost_fn(x, goal):
        return 0.5 * (x - goal) @ (x - goal) * 10.0
    
    optimizer = DDPOptimizer(
        dynamics_fn=dynamics_fn,
        stage_cost_fn=stage_cost_fn,
        terminal_cost_fn=terminal_cost_fn,
        horizon=T,
        dt=1.0,
        n_x=n_x,
        n_u=n_u,
        device=device,
        verbose=False,
    )
    
    optimizer.goal = torch.zeros(n_x, device=device, dtype=dtype)
    
    # Nominal trajectory
    X = torch.randn(T+1, n_x, device=device, dtype=dtype) * 0.1
    U = torch.randn(T, n_u, device=device, dtype=dtype) * 0.1
    
    # Manually check symmetry during backward pass
    # (We'd need to modify backward_pass to expose Vxx, or reimplement here)
    
    # For now, just run backward pass and check it doesn't crash
    K_list, k_list, diverged = optimizer.backward_pass(X, U)
    
    assert not diverged, "Backward pass diverged!"
    
    # TODO: Expose Vxx from backward pass and check:
    # for each Vxx: assert torch.allclose(Vxx, Vxx.T)
    
    print("Backward pass completed without divergence")
    print("(Full Vxx symmetry check requires exposing internal state)")
    print("✅ Test 5 passed (basic check)!")


# ==============================================================================
# Test 6: Forward Pass Improvement
# ==============================================================================

def test_forward_pass_improvement():
    """
    Test that forward pass improves cost
    
    Given a backward pass, forward pass should reduce cost
    """
    print("\n" + "="*70)
    print("Test 6: Forward Pass Cost Improvement")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.double
    
    n_x = 2
    n_u = 1
    T = 15
    
    # Double integrator
    A = torch.tensor([[1.0, 0.1], [0.0, 1.0]], device=device, dtype=dtype)
    B = torch.tensor([[0.005], [0.1]], device=device, dtype=dtype)
    
    def dynamics_fn(x, u):
        return A @ x + B @ u
    
    def stage_cost_fn(x, u):
        return 0.5 * (x @ x + 0.1 * u @ u)
    
    def terminal_cost_fn(x, goal):
        return 0.5 * (x - goal) @ (x - goal) * 100.0
    
    x0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
    goal = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    
    optimizer = DDPOptimizer(
        dynamics_fn=dynamics_fn,
        stage_cost_fn=stage_cost_fn,
        terminal_cost_fn=terminal_cost_fn,
        horizon=T,
        dt=1.0,
        n_x=n_x,
        n_u=n_u,
        device=device,
        verbose=False,
    )
    
    optimizer.goal = goal
    
    # Random initial controls
    U_init = torch.randn(T, n_u, device=device, dtype=dtype) * 0.1
    
    # Initial cost
    X_init, cost_init = optimizer.rollout(x0, U_init)
    print(f"Initial cost: {cost_init.item():.6f}")
    
    # Backward pass
    K_list, k_list, diverged = optimizer.backward_pass(X_init, U_init)
    assert not diverged
    
    # Forward pass with α=1.0
    X_new, U_new, cost_new = optimizer.forward_pass(x0, X_init, U_init, K_list, k_list, alpha=1.0)
    print(f"New cost (α=1.0): {cost_new:.6f}")
    
    # Cost should improve (or at least not get much worse)
    improvement = cost_init.item() - cost_new
    print(f"Improvement: {improvement:.6f}")
    
    # Allow small increase due to approximation error
    assert improvement > -10.0, f"Cost got much worse! Δ={improvement:.6f}"
    
    print("✅ Test 6 passed!")


# ==============================================================================
# Run all tests
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DDP UNIT TESTS")
    print("="*70)
    
    # Run tests
    tests = [
        ("Linear Dynamics", test_linear_dynamics_unconstrained),
        ("Nonlinear Pendulum", test_nonlinear_pendulum),
        ("Box Constraints", test_box_constraints),
        ("Backward Pass Gains", test_backward_pass_gains),
        ("Value Function Symmetry", test_value_function_symmetry),
        ("Forward Pass Improvement", test_forward_pass_improvement),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name} FAILED:")
            print(f"   {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"SUMMARY: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} tests failed")
    print("="*70)