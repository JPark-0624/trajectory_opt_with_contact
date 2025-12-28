"""
Component-level profiling to identify exact bottlenecks.

This script isolates and times each major component:
1. Pure geometry computation (obb_contact_blend2)
2. Residual evaluation  
3. Jacobian computation (autograd)
4. Newton iteration overhead
5. Full dynamics step

Usage:
    python test_profile_components.py
"""

import torch
import time
import numpy as np
from datetime import datetime
import sys

from trajectory_opt_with_contact import (
    step_square_pos_ip, IPMOptions
)
from trajectory_opt_with_contact.geometry import obb_contact_blend2, contact_jacobians, perp


class TeeWriter:
    """Writer that outputs to both console and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def setup_test_case(device='cuda', dtype=torch.float32):
    """Setup a typical pusher-slider test case."""
    # Physical parameters
    m = 1.0
    Izz = 1.0 / 6.0
    half = 0.1
    mu = 0.6
    h = 0.02
    
    # Initial state - pusher in contact with slider
    q0 = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    v0 = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    pr0 = torch.tensor([-0.3, -0.05], dtype=dtype, device=device)
    
    # Control input
    u = torch.tensor([0.3, 0.0], dtype=dtype, device=device)
    
    # IPM options
    ipm_opts = IPMOptions(
        target_mu=1e-4,
        max_newton=20,
        tol=1e-3,
        smooth_sdf=50.0,
        enable_viscous_ground_friction=True,
        c_lin=8.0,
        c_ang=8.0 * half,
    )
    
    return q0, v0, pr0, u, h, m, Izz, half, mu, ipm_opts


def time_function(func, *args, n_iters=100, warmup=10, name="Function"):
    """Time a function with warmup."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Actual timing
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # Convert to ms
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n{name}:")
    print(f"  Mean: {mean_time:.3f} ms")
    print(f"  Std:  {std_time:.3f} ms")
    print(f"  Min:  {min_time:.3f} ms")
    print(f"  Max:  {max_time:.3f} ms")
    
    return mean_time, result


def test_1_pure_geometry():
    """Test pure geometry computation."""
    print("\n" + "="*70)
    print("TEST 1: Pure Geometry Computation (obb_contact_blend2)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # Test case
    q = torch.tensor([0.05, 0.03, 0.1], dtype=dtype, device=device)
    pusher_pos = torch.tensor([0.15, 0.05], dtype=dtype, device=device)
    half = torch.tensor(0.1, dtype=dtype, device=device)
    
    mean_time, result = time_function(
        obb_contact_blend2, q, pusher_pos, half,
        n_iters=1000,
        warmup=100,
        name="obb_contact_blend2 (single call)"
    )
    
    print(f"\nResult: phi={result.phi.item():.6f}")
    return mean_time


def create_residual_function(qk, vk, pusher_pos_next, h, m, Izz, half, mu, 
                              target_mu, enable_viscous_ground_friction, c_lin, c_ang):
    """
    Create a residual function that matches the one used inside step_square_pos_ip.
    This is extracted from StepSquarePosIPFn.forward.
    """
    device = qk.device
    dtype = qk.dtype
    
    # Convert to tensors
    hT = torch.tensor(h, dtype=dtype, device=device)
    muT = torch.tensor(mu, dtype=dtype, device=device)
    mT = torch.tensor(m, dtype=dtype, device=device)
    IzzT = torch.tensor(Izz, dtype=dtype, device=device)
    halfT = torch.tensor(half, dtype=dtype, device=device)
    c_linT = torch.tensor(c_lin, dtype=dtype, device=device)
    c_angT = torch.tensor(c_ang, dtype=dtype, device=device)
    
    # Mass inverse
    M_inv = torch.diag(torch.stack([1.0/mT, 1.0/mT, 1.0/IzzT]))
    
    def unpack(z):
        _q = z[0:3]
        _v = z[3:6]
        _lamN = z[6]
        _beta = z[7:9]
        _r = z[9]
        _y = z[10]
        _s = z[11]
        _w = z[12:14]
        return _q, _v, _lamN, _beta, _r, _y, _s, _w
    
    def residual(z):
        _q, _v, _lamN, _beta, _r, _y, _s, _w = unpack(z)
        
        cnt = obb_contact_blend2(_q, pusher_pos_next, halfT)
        n, t, r_cp = cnt.normal, cnt.tangent, cnt.r_cp
        Jn, Jt = contact_jacobians(n, t, r_cp)
        
        # Tangential rel velocity at cp
        v_cp = _v[0:2] + _v[2] * perp(r_cp)
        v_rel_t = torch.dot(t, v_cp - torch.zeros_like(v_cp))  # u_push is zero for this test
        
        # Two facets +/-
        v_facets = torch.stack([v_rel_t, -v_rel_t])
        
        lamT_total = _beta[0] - _beta[1]
        impulse = Jn * _lamN + Jt * lamT_total
        
        # Option A: viscous ground friction
        Fg = torch.zeros(3, dtype=dtype, device=device)
        if enable_viscous_ground_friction and (c_lin > 0.0 or c_ang > 0.0):
            Fg = torch.stack([
                -c_linT * _v[0],
                -c_linT * _v[1],
                -c_angT * _v[2],
            ])
        
        dv = M_inv @ (impulse + hT * Fg)
        
        r_dyn = _v - vk - dv
        r_kin = _q - qk - hT * _v
        r_gap = _y - cnt.phi
        r_cone = _s - (muT * _lamN - torch.sum(_beta))
        r_slip = _w - (v_facets + _r)
        
        mu_star = torch.tensor(target_mu, dtype=dtype, device=device)
        r_c1 = _y * _lamN - mu_star
        r_c2 = _r * _s - mu_star
        r_c3 = _beta * _w - mu_star
        
        return torch.cat([r_dyn, r_kin, r_gap.view(1), r_cone.view(1), r_slip,
                          r_c1.view(1), r_c2.view(1), r_c3])
    
    return residual


def test_2_residual_evaluation():
    """Test residual evaluation (includes geometry)."""
    print("\n" + "="*70)
    print("TEST 2: Residual Evaluation")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # Setup
    q, v, pr, u, h, m, Izz, half, mu, ipm_opts = setup_test_case(device, dtype)
    
    # Create a test z vector (typical values from Newton iteration)
    z = torch.tensor([
        0.01, 0.01, 0.05,  # q
        0.1, 0.1, 0.2,      # v
        5.0,                # lambda_N
        2.0, 2.0,           # beta
        0.5,                # r
        0.01,               # y
        0.5,                # s
        1.0, 1.0            # w
    ], dtype=dtype, device=device)
    
    pusher_pos_next = pr + u * h
    
    # Create residual function
    residual_fn = create_residual_function(
        q, v, pusher_pos_next, h, m, Izz, half, mu,
        ipm_opts.target_mu, ipm_opts.enable_viscous_ground_friction,
        ipm_opts.c_lin, ipm_opts.c_ang
    )
    
    mean_time, R = time_function(
        residual_fn, z,
        n_iters=500,
        warmup=50,
        name="Residual evaluation (1 geometry call)"
    )
    
    print(f"\nResidual norm: {torch.linalg.norm(R).item():.6f}")
    return mean_time


def test_3_jacobian_autograd():
    """Test Jacobian computation with autograd."""
    print("\n" + "="*70)
    print("TEST 3: Jacobian Computation (Autograd)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # Setup
    q, v, pr, u, h, m, Izz, half, mu, ipm_opts = setup_test_case(device, dtype)
    
    z = torch.tensor([
        0.01, 0.01, 0.05,
        0.1, 0.1, 0.2,
        5.0,
        2.0, 2.0,
        0.5,
        0.01,
        0.5,
        1.0, 1.0
    ], dtype=dtype, device=device)
    
    pusher_pos_next = pr + u * h
    
    # Create residual function
    residual_fn = create_residual_function(
        q, v, pusher_pos_next, h, m, Izz, half, mu,
        ipm_opts.target_mu, ipm_opts.enable_viscous_ground_friction,
        ipm_opts.c_lin, ipm_opts.c_ang
    )
    
    # Define Jacobian function
    def jacobian_fn(z_in):
        z_req = z_in.detach().requires_grad_(True)
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(
                residual_fn,
                z_req,
                strict=False,
                create_graph=False,
                vectorize=True,
            )
        R = residual_fn(z_req)
        return J.reshape(R.numel(), z_req.numel())
    
    mean_time, J = time_function(
        jacobian_fn, z,
        n_iters=100,
        warmup=10,
        name="Jacobian computation (autograd)"
    )
    
    print(f"\nJacobian shape: {J.shape}")
    print(f"Jacobian norm: {torch.linalg.norm(J).item():.6f}")
    return mean_time


def test_4_jacobian_analytical():
    """Test Jacobian computation with analytical method."""
    print("\n" + "="*70)
    print("TEST 4: Jacobian Computation (Analytical)")
    print("="*70)
    
    try:
        from trajectory_opt_with_contact.analytical_jacobian import AnalyticalJacobian
    except ImportError:
        print("WARNING: Analytical Jacobian not available, skipping...")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    # Setup
    q, v, pr, u, h, m, Izz, half, mu, ipm_opts = setup_test_case(device, dtype)
    
    z = torch.tensor([
        0.01, 0.01, 0.05,
        0.1, 0.1, 0.2,
        5.0,
        2.0, 2.0,
        0.5,
        0.01,
        0.5,
        1.0, 1.0
    ], dtype=dtype, device=device)
    
    pusher_pos_next = pr + u * h
    
    # Create analytical Jacobian computer
    analytical_jac = AnalyticalJacobian(
        mass=m, Izz=Izz, half_length=half, mu=mu, dt=h, device=device, dtype=dtype
    )
    
    # Create residual function
    residual_fn = create_residual_function(
        q, v, pusher_pos_next, h, m, Izz, half, mu,
        ipm_opts.target_mu, ipm_opts.enable_viscous_ground_friction,
        ipm_opts.c_lin, ipm_opts.c_ang
    )
    
    # Define Jacobian function
    def jacobian_fn(z_in):
        R = residual_fn(z_in)
        J = analytical_jac.compute_jacobian(z_in, R, pusher_pos=pusher_pos_next)
        return J
    
    mean_time, J = time_function(
        jacobian_fn, z,
        n_iters=100,
        warmup=10,
        name="Jacobian computation (analytical)"
    )
    
    print(f"\nJacobian shape: {J.shape}")
    print(f"Jacobian norm: {torch.linalg.norm(J).item():.6f}")
    return mean_time


def test_5_full_step():
    """Test full dynamics step."""
    print("\n" + "="*70)
    print("TEST 5: Full Dynamics Step (Autograd)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    q, v, pr, u, h, m, Izz, half, mu, ipm_opts = setup_test_case(device, dtype)
    
    def step_fn():
        return step_square_pos_ip(
            q, v, pr, u, h=h, m=m, Izz=Izz, half=half, mu=mu,
            ipm_opts=ipm_opts, skip_solving_threshold=0.3,
            use_analytical_jacobian=False
        )
    
    mean_time, result = time_function(
        step_fn,
        n_iters=100,
        warmup=10,
        name="Full step (autograd)"
    )
    
    q_next, v_next, pr_next, lam, phi = result
    print(f"\nFinal q: [{q_next[0].item():.3f}, {q_next[1].item():.3f}, {q_next[2].item():.3f}]")
    return mean_time


def test_6_full_step_analytical():
    """Test full dynamics step with analytical Jacobian."""
    print("\n" + "="*70)
    print("TEST 6: Full Dynamics Step (Analytical)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    q, v, pr, u, h, m, Izz, half, mu, ipm_opts = setup_test_case(device, dtype)
    
    def step_fn():
        return step_square_pos_ip(
            q, v, pr, u, h=h, m=m, Izz=Izz, half=half, mu=mu,
            ipm_opts=ipm_opts, skip_solving_threshold=0.3,
            use_analytical_jacobian=True
        )
    
    try:
        mean_time, result = time_function(
            step_fn,
            n_iters=100,
            warmup=10,
            name="Full step (analytical)"
        )
        
        q_next, v_next, pr_next, lam, phi = result
        print(f"\nFinal q: [{q_next[0].item():.3f}, {q_next[1].item():.3f}, {q_next[2].item():.3f}]")
        return mean_time
    except Exception as e:
        print(f"WARNING: Analytical Jacobian failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_analysis(results):
    """Print detailed analysis of results."""
    # Summary
    print("\n" + "="*70)
    print(" " * 25 + "SUMMARY")
    print("="*70)
    
    print("\nComponent Timings:")
    print(f"  1. Pure geometry (1 call):        {results['geometry']:.3f} ms")
    print(f"  2. Residual (1 geometry):          {results['residual']:.3f} ms")
    print(f"  3. Jacobian (autograd):            {results['jacobian_autograd']:.3f} ms")
    if results['jacobian_analytical']:
        print(f"  4. Jacobian (analytical):          {results['jacobian_analytical']:.3f} ms")
    print(f"  5. Full step (autograd):           {results['full_step_autograd']:.3f} ms")
    if results['full_step_analytical']:
        print(f"  6. Full step (analytical):         {results['full_step_analytical']:.3f} ms")
    
    # Analysis
    print("\n" + "="*70)
    print(" " * 25 + "ANALYSIS")
    print("="*70)
    
    # Estimate geometry calls in full step
    geom_time = results['geometry']
    full_time = results['full_step_autograd']
    
    estimated_geom_calls = full_time / geom_time
    print(f"\nEstimated geometry calls per step: {estimated_geom_calls:.1f}")
    print(f"Geometry time in full step: {estimated_geom_calls * geom_time:.1f} ms ({estimated_geom_calls * geom_time / full_time * 100:.1f}%)")
    
    # Jacobian comparison
    jac_auto = results['jacobian_autograd']
    print(f"\nJacobian (autograd) breakdown:")
    print(f"  Total time: {jac_auto:.3f} ms")
    print(f"  Geometry component: ~{geom_time:.3f} ms (1 forward pass)")
    print(f"  AD overhead: ~{jac_auto - geom_time:.3f} ms")
    
    if results['jacobian_analytical']:
        jac_ana = results['jacobian_analytical']
        print(f"\nJacobian (analytical) breakdown:")
        print(f"  Total time: {jac_ana:.3f} ms")
        print(f"  Estimated geometry calls: {jac_ana / geom_time:.1f}x")
        print(f"  Speedup vs autograd: {jac_auto / jac_ana:.2f}x")
    
    # Full step comparison
    if results['full_step_analytical']:
        speedup = results['full_step_autograd'] / results['full_step_analytical']
        print(f"\nFull step speedup (analytical): {speedup:.2f}x")
        if speedup < 1.0:
            print(f"  WARNING: Analytical is {1/speedup:.2f}x SLOWER!")
        else:
            print(f"  SUCCESS: Analytical is {speedup:.2f}x faster!")
    
    # Bottleneck identification
    print("\n" + "="*70)
    print(" " * 22 + "BOTTLENECK ANALYSIS")
    print("="*70)
    
    print(f"\nIn a typical Newton iteration (~10 iterations):")
    print(f"  Residual: 10 x {results['residual']:.3f} ms = {10 * results['residual']:.1f} ms")
    print(f"  Jacobian: 10 x {jac_auto:.3f} ms = {10 * jac_auto:.1f} ms")
    print(f"  Total: {10 * (results['residual'] + jac_auto):.1f} ms")
    print(f"\nActual full step time: {full_time:.1f} ms")
    print(f"Unaccounted time (linear solve, line search, etc): {full_time - 10 * (results['residual'] + jac_auto):.1f} ms")
    
    jacobian_percentage = (10 * jac_auto) / full_time * 100
    print(f"\nJacobian is ~{jacobian_percentage:.1f}% of total time")
    
    if jacobian_percentage > 60:
        print("SUCCESS: Jacobian is the PRIMARY bottleneck!")
    elif jacobian_percentage > 40:
        print("WARNING: Jacobian is A MAJOR bottleneck")
    else:
        print("WARNING: Jacobian is NOT the main bottleneck")
    
    # Recommendations
    print("\n" + "="*70)
    print(" " * 23 + "RECOMMENDATIONS")
    print("="*70)
    
    if results['jacobian_analytical'] and results['jacobian_analytical'] > jac_auto:
        print("\nWARNING: Current analytical Jacobian is slower than autograd!")
        print("Reasons:")
        print(f"  - Geometry calls: {results['jacobian_analytical'] / geom_time:.1f}x vs 1x (autograd)")
        print(f"  - Overhead: Finite differences for geometry gradients")
        print("\nSolutions:")
        print("  1. Implement geometry caching (reuse baseline)")
        print("  2. Implement analytical geometry gradients (eliminate FD)")
    
    geom_percentage = (estimated_geom_calls * geom_time) / full_time * 100
    if geom_percentage > 50:
        print(f"\nWARNING: Geometry computation is {geom_percentage:.1f}% of total time!")
        print("Solutions:")
        print("  1. Optimize obb_contact_blend2 implementation")
        print("  2. Cache geometry calls within Newton iteration")
        print("  3. Use simpler geometry (e.g., sphere approximation)")


def main():
    """Run all component tests."""
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"profile_components_{timestamp}.txt"
    
    # Redirect stdout to both console and file
    tee = TeeWriter(output_file)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("\n" + "="*70)
        print(" " * 20 + "COMPONENT-LEVEL PROFILING")
        print("="*70)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output file: {output_file}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"\nSUCCESS: Device: {torch.cuda.get_device_name(0)}")
        else:
            print("\nWARNING: Device: CPU (will be slower)")
        
        results = {}
        
        # Run tests
        results['geometry'] = test_1_pure_geometry()
        results['residual'] = test_2_residual_evaluation()
        results['jacobian_autograd'] = test_3_jacobian_autograd()
        results['jacobian_analytical'] = test_4_jacobian_analytical()
        results['full_step_autograd'] = test_5_full_step()
        results['full_step_analytical'] = test_6_full_step_analytical()
        
        # Print analysis
        print_analysis(results)
        
        print("\n" + "="*70)
        print("PROFILING COMPLETE")
        print("="*70)
        print(f"\nSUCCESS: Results saved to: {output_file}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        # Restore original stdout and close file
        sys.stdout = original_stdout
        tee.close()
        print(f"\nSUCCESS: Results saved to: {output_file}")


if __name__ == "__main__":
    main()