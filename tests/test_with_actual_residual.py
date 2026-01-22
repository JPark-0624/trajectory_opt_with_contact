"""
Test analytical Jacobian with actual residual function.

Simple, clean version - directly uses AnalyticalJacobian from analytical_jacobian.py.
"""

import torch
import sys
sys.path.insert(0, '/tmp')

from trajectory_opt_with_contact.geometry import obb_contact_blend2, contact_jacobians, perp
from analytical_jacobian import AnalyticalJacobian, pack_z


# ============================================================
# Residual Wrapper (from your dynamics.py)
# ============================================================

class ContactImplicitResidualWrapper:
    """
    Wrapper for the residual function from dynamics.py.
    
    Matches the signature: residual(z, qk, vk, pusher_pos_next, u_push)
    """
    
    def __init__(self, m, Izz, half, mu, h, target_mu=1e-4, enable_viscous=False, device='cpu'):
        self.m = m
        self.Izz = Izz
        self.half = half
        self.mu = mu
        self.h = h
        self.target_mu = target_mu
        self.enable_viscous = enable_viscous
        self.device = device
        
        # Mass matrix
        self.M = torch.diag(torch.tensor([m, m, Izz], dtype=torch.float64, device=device))
        self.M_inv = torch.diag(torch.tensor([1/m, 1/m, 1/Izz], dtype=torch.float64, device=device))
    
    def __call__(self, z, qk, vk, pusher_pos_next, u_push):
        """
        Compute residual r(z).
        
        This matches the residual function in dynamics.py (lines 205-245).
        """
        # Unpack z: [q, v, λN, β, r, y, s, w]
        _q = z[0:3]
        _v = z[3:6]
        _lamN = z[6]
        _beta = z[7:9]
        _r = z[9]
        _y = z[10]
        _s = z[11]
        _w = z[12:14]
        
        # Contact geometry
        cnt = obb_contact_blend2(_q, pusher_pos_next, self.half)
        
        # Contact Jacobians
        Jn, Jt = contact_jacobians(cnt.normal, cnt.tangent, cnt.r_cp)
        
        # Dynamics residual: r_dyn = v - vk - h * M_inv @ impulse
        lamT_total = _beta[0] - _beta[1]
        impulse = Jn * _lamN + Jt * lamT_total
        r_dyn = _v - vk - self.h * (self.M_inv @ impulse)
        
        # Kinematics residual: r_kin = q - qk - h * v
        r_kin = _q - qk - self.h * _v
        
        # Gap residual: r_gap = y - phi(q)
        r_gap = _y - cnt.phi
        
        # Friction cone residual: r_cone = -mu*λN + β[0] + β[1] + s
        r_cone = -self.mu * _lamN + _beta[0] + _beta[1] + _s
        
        # Slip residual: r_slip = w - v_facets - r*t
        # v_facets = tangential velocities at contact point
        v_cp = _v[:2] + _v[2] * perp(cnt.r_cp)
        v_t = torch.dot(cnt.tangent, v_cp)
        v_facets = torch.stack([v_t, -v_t])
        r_slip = _w - v_facets - _r * cnt.tangent
        
        # Complementarity constraints
        comp1 = _y * _lamN
        comp2 = _r * _s
        comp3 = _beta * _w
        
        # Pack residual
        r = torch.cat([
            r_dyn,      # (3,)
            r_kin,      # (3,)
            r_gap.unsqueeze(0),  # (1,)
            r_cone.unsqueeze(0), # (1,)
            r_slip,     # (2,)
            comp1.unsqueeze(0),  # (1,)
            comp2.unsqueeze(0),  # (1,)
            comp3,      # (2,)
        ])
        
        return r


# ============================================================
# Test Helper Functions
# ============================================================

def create_test_case(device='cpu'):
    """Create a single realistic test case."""
    
    # Configuration
    q = torch.tensor([
        torch.randn(1).item() * 0.1,
        torch.randn(1).item() * 0.1,
        torch.randn(1).item() * 0.3,
    ], dtype=torch.float64, device=device)
    
    # Velocity
    v = torch.tensor([
        torch.randn(1).item() * 0.2,
        torch.randn(1).item() * 0.2,
        torch.randn(1).item() * 0.5,
    ], dtype=torch.float64, device=device)
    
    # Contact forces (positive)
    λN = torch.abs(torch.randn(1, dtype=torch.float64, device=device) * 5.0 + 1.0)
    β = torch.abs(torch.randn(2, dtype=torch.float64, device=device) * 2.0 + 0.5)
    
    # Slack variables (positive)
    r = torch.abs(torch.randn(1, dtype=torch.float64, device=device) * 0.5 + 0.1)
    y = torch.abs(torch.randn(1, dtype=torch.float64, device=device) * 0.01 + 1e-4)
    s = torch.abs(torch.randn(1, dtype=torch.float64, device=device) * 0.5 + 0.1)
    w = torch.abs(torch.randn(2, dtype=torch.float64, device=device) * 2.0 + 0.5)
    
    # Pack z
    z = pack_z(q, v, λN[0], β, r[0], y[0], s[0], w)
    
    # Previous state
    qk = q - torch.randn(3, dtype=torch.float64, device=device) * 0.01
    vk = v - torch.randn(3, dtype=torch.float64, device=device) * 0.05
    pusher_pos = torch.tensor([0.15, 0.0], dtype=torch.float64, device=device)
    u_push = torch.randn(2, dtype=torch.float64, device=device) * 0.1
    
    return {
        'z': z,
        'qk': qk,
        'vk': vk,
        'pusher_pos_next': pusher_pos,
        'u_push': u_push
    }


def test_single_point(device='cpu', verbose=True):
    """Test analytical Jacobian at a single point."""
    
    # Setup
    params = {
        'mass': 1.0,
        'Izz': 0.02,
        'half_length': 0.1,
        'mu': 0.6,
        'dt': 0.05,
        'device': device
    }
    
    # Create residual and Jacobian
    residual_wrapper = ContactImplicitResidualWrapper(
        m=params['mass'],
        Izz=params['Izz'],
        half=params['half_length'],
        mu=params['mu'],
        h=params['dt'],
        device=device
    )
    
    analytical_jac = AnalyticalJacobian(**params)
    
    # Create test case
    test_data = create_test_case(device)
    z = test_data['z']
    
    if verbose:
        print("="*80)
        print("SINGLE POINT JACOBIAN TEST")
        print("="*80)
        print(f"\nTest configuration:")
        print(f"  Device: {device}")
        print(f"  z shape: {z.shape}")
    
    # Compute autograd Jacobian
    if verbose:
        print(f"\nComputing autograd Jacobian...")
    
    import time
    t0 = time.time()
    z_grad = z.clone().detach().requires_grad_(True)
    
    def residual_fn(z_):
        return residual_wrapper(
            z_,
            test_data['qk'],
            test_data['vk'],
            test_data['pusher_pos_next'],
            test_data['u_push']
        )
    
    J_autograd = torch.autograd.functional.jacobian(residual_fn, z_grad)
    t_autograd = time.time() - t0
    
    if verbose:
        print(f"  Time: {t_autograd*1000:.3f} ms")
    
    # Compute analytical Jacobian
    if verbose:
        print(f"\nComputing analytical Jacobian...")
    
    t0 = time.time()
    J_analytical = analytical_jac.compute_jacobian(
        z,
        q_k=test_data['qk'],
        v_k=test_data['vk'],
        pusher_pos=test_data['pusher_pos_next']
    )
    t_analytical = time.time() - t0
    
    if verbose:
        print(f"  Time: {t_analytical*1000:.3f} ms")
        print(f"\nSpeedup: {t_autograd/t_analytical:.2f}x")
    
    # Compare
    diff = torch.abs(J_analytical - J_autograd)
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    
    if verbose:
        print(f"\nError analysis:")
        print(f"  Max absolute error:  {max_error:.6e}")
        print(f"  Mean absolute error: {mean_error:.6e}")
    
    # Test with relaxed tolerance
    rtol = 1e-3
    atol = 1e-5
    passed = torch.allclose(J_analytical, J_autograd, rtol=rtol, atol=atol)
    
    if verbose:
        print(f"\nTest result (rtol={rtol:.0e}, atol={atol:.0e}):")
        if passed:
            print(f"  ✓ PASSED")
        else:
            print(f"  ✗ FAILED")
    
    return {
        'passed': passed,
        'max_error': max_error,
        'mean_error': mean_error,
        'speedup': t_autograd / t_analytical,
        'time_autograd': t_autograd,
        'time_analytical': t_analytical
    }


# ============================================================
# Main Test
# ============================================================

def main():
    """Run simple test."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    result = test_single_point(device=device, verbose=True)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Max error: {result['max_error']:.6e}")
    print(f"Speedup:   {result['speedup']:.2f}x")
    print(f"Status:    {'PASSED ✓' if result['passed'] else 'FAILED ✗'}")
    print("="*80)


if __name__ == "__main__":
    main()