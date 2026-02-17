"""
Simple test for IRL weight recovery.

Quick sanity check to verify the IRL implementation works correctly.
"""

import torch
import numpy as np

from trajectory_opt_with_contact.optimizer import TrajectoryOptimizer
from trajectory_opt_with_contact.irl_weight_recovery import IRLWeightRecovery, IRLOptions, pack_demo


def test_feature_matching():
    """
    Test IRL with feature matching method.
    
    This test:
    1. Creates a simple scenario
    2. Generates demo with known weights
    3. Recovers weights using feature matching
    4. Verifies recovery is close to ground truth
    """
    print("\n" + "="*80)
    print("TEST: IRL Weight Recovery (Feature Matching)")
    print("="*80)
    
    # Create optimizer with short horizon for speed
    optimizer = TrajectoryOptimizer(
        mass=1.0,
        side_length=0.2,
        mu=0.6,
        horizon=30,  # Short horizon for fast testing
        dt=0.05,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        TO_solver='shooting',
        dynamics_solver='IP'
    )
    
    # Simple scenario
    q0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.0])
    pusher0 = np.array([0.2, -0.2])
    goal = np.array([0.2, 0.0, 0.0])
    
    # True weights (unnormalized)
    w_true_dict = {
        'w_target': 20,
        'w_orient': 0.5,
        'w_v': 1.0,
        'w_ctrl': 0.01,
        'w_obs': 0.0
    }
    
    scale = 0.0
    for k, v in w_true_dict.items():
        scale += v

    # Scale up for actual optimization
    w_true_scaled = {k: v / scale for k, v in w_true_dict.items()}
    
    print("\n1. Generating expert demonstration...")
    print(f"   True weights (scaled): {w_true_scaled}")
    
    # Generate demonstration
    result = optimizer.optimize(
        q0=q0, v0=v0, pusher0=pusher0, goal=goal,
        w_target=w_true_dict['w_target'],
        w_orient=w_true_dict['w_orient'],
        w_v=w_true_dict['w_v'],
        w_ctrl=w_true_dict['w_ctrl'],
        w_obs=w_true_dict['w_obs'],
        max_iters=100,
        lr=0.01,
        verbose=False
    )
    
    u_demo = result['u_seq']
    print(f"   Demo generated: loss = {result['loss']:.6f}")
    
    # Pack demo
    demo = pack_demo(q0, v0, pusher0, goal, u_demo)
    
    print("\n2. Recovering weights using IRL...")
    
    # Create IRL solver
    irl = IRLWeightRecovery(optimizer)
    
    # Set options for fast test
    opts = IRLOptions(
        max_outer_iters=30,
        max_inner_iters=50,
        lr_weights=0.02,
        lr_trajectory=0.01,
        #method="feature_matching",
        method="control_matching",  # Optional: test control matching instead
        weight_tol=1e-3,
        control_tol=1e-2,
        verbose=False
    )
    
    # Recover
    irl_result = irl.recover_weights(demo, opts=opts)
    
    w_recovered = irl_result['w_recovered']
    
    print(f"   Converged: {irl_result['converged']}")
    print(f"   Iterations: {irl_result['num_iters']}")
    print(f"   Final control error: {irl_result['control_error_history'][-1]:.6f}")
    
    print("\n3. Comparing weights...")
    
    # Normalize true weights for comparison
    w_true_array = np.array([
        w_true_dict['w_target'],
        w_true_dict['w_orient'],
        w_true_dict['w_v'],
        w_true_dict['w_ctrl'],
        w_true_dict['w_obs']
    ])
    w_true_norm = w_true_array / w_true_array.sum()
    
    w_rec_np = w_recovered.cpu().numpy()
    
    feature_names = ['w_target', 'w_orient', 'w_v', 'w_ctrl', 'w_obs']
    
    print(f"\n   {'Feature':12s} {'True':>10s} {'Recovered':>10s} {'Error':>10s}")
    print("   " + "-"*50)
    
    total_error = 0.0
    for i, name in enumerate(feature_names):
        true_val = w_true_norm[i]
        rec_val = w_rec_np[i]
        error = abs(rec_val - true_val)
        total_error += error
        
        print(f"   {name:12s} {true_val:10.4f} {rec_val:10.4f} {error:10.4f}")
    
    print("   " + "-"*50)
    print(f"   Total L1 error: {total_error:.6f}")
    
    # Determine pass/fail
    tolerance = 0.15  # Allow 15% total error
    passed = total_error < tolerance
    
    print(f"\n4. Test Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    if not passed:
        print(f"   Error {total_error:.4f} exceeds tolerance {tolerance:.4f}")
    
    return passed


def test_control_matching():
    """
    Test IRL with direct control matching.
    
    Note: This is more computationally expensive as it backprops through
    the entire inner optimization loop.
    """
    print("\n" + "="*80)
    print("TEST: IRL Weight Recovery (Control Matching)")
    print("="*80)
    print("Note: This test is more expensive due to bi-level optimization")
    
    # Create optimizer with very short horizon
    optimizer = TrajectoryOptimizer(
        mass=1.0,
        side_length=0.2,
        mu=0.6,
        horizon=20,  # Very short for speed
        dt=0.05,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        TO_solver='shooting',
        dynamics_solver='IP'
    )
    
    # Simple scenario
    q0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.0])
    pusher0 = np.array([0.15, -0.15])
    goal = np.array([0.15, 0.0, 0.0])
    
    # True weights
    w_true = {
        'w_target': 20.0,
        'w_orient': 1.0,
        'w_v': 0.1,
        'w_ctrl': 1e-3,
        'w_obs': 0.5
    }
    
    print("\n1. Generating demonstration...")
    result = optimizer.optimize(
        q0=q0, v0=v0, pusher0=pusher0, goal=goal,
        **w_true,
        max_iters=30,
        verbose=False
    )
    
    demo = pack_demo(q0, v0, pusher0, goal, result['u_seq'])
    print(f"   Demo loss: {result['loss']:.6f}")
    
    print("\n2. Recovering weights (control matching)...")
    
    irl = IRLWeightRecovery(optimizer)
    opts = IRLOptions(
        max_outer_iters=20,
        max_inner_iters=30,
        lr_weights=0.01,
        lr_trajectory=0.01,
        method="control_matching",
        verbose=False
    )
    
    irl_result = irl.recover_weights(demo, opts=opts)
    
    print(f"   Converged: {irl_result['converged']}")
    print(f"   Final control error: {irl_result['control_error_history'][-1]:.6f}")
    
    # Basic check: control error should be small
    passed = irl_result['control_error_history'][-1] < 1.0
    
    print(f"\n3. Test Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


if __name__ == "__main__":
    print("\n" + "="*80)
    print("IRL WEIGHT RECOVERY TESTS")
    print("="*80)
    
    try:
        # Test 1: Feature Matching
        test1_passed = test_feature_matching()
        
        # Test 2: Control Matching (optional, more expensive)
        # Uncomment to run:
        # test2_passed = test_control_matching()
        # 
        # all_passed = test1_passed and test2_passed
        
        all_passed = test1_passed
        
        print("\n" + "="*80)
        print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
