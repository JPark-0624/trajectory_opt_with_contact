"""
Comprehensive Jacobian test with detailed logging.

Clean version - directly uses AnalyticalJacobian.
"""

import torch
import sys
import datetime
import numpy as np
sys.path.insert(0, '/tmp')

from trajectory_opt_with_contact.geometry import obb_contact_blend2, contact_jacobians, perp
from analytical_jacobian import AnalyticalJacobian, pack_z
import time


# Import test utilities from test_with_actual_residual
from test_with_actual_residual import ContactImplicitResidualWrapper, create_test_case


class TestLogger:
    """Logger that writes to both console and file."""
    
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
    
    def log(self, message):
        """Write to both console and file."""
        print(message)
        self.file.write(message + '\n')
        self.file.flush()
    
    def close(self):
        self.file.close()


def analyze_jacobian_blocks(J_analytical, J_autograd, logger):
    """Detailed block-by-block analysis with value comparison."""
    
    blocks = {
        'A': (slice(0,3), slice(0,3), 'Medium', 'r_dyn/q'),
        'B': (slice(0,3), slice(3,6), 'Easy', 'r_dyn/v'),
        'C': (slice(0,3), 6, 'Medium', 'r_dyn/λN'),
        'X': (slice(0,3), slice(7,9), 'Medium', 'r_dyn/β'),  # ADDED: Missing block!
        'D': (slice(3,6), slice(0,3), 'Easy', 'r_kin/q'),
        'E': (slice(3,6), slice(3,6), 'Easy', 'r_kin/v'),
        'G': (6, slice(0,3), 'Medium', 'r_gap/q'),
        'H': (6, 10, 'Easy', 'r_gap/y'),
        'I': (7, 6, 'Easy', 'r_cone/λN'),
        'J': (7, slice(7,9), 'Easy', 'r_cone/β'),
        'K': (7, 11, 'Easy', 'r_cone/s'),
        'F': (slice(8,10), slice(0,3), 'Medium', 'r_slip/q'),  # ADDED: Missing block!
        'L': (slice(8,10), slice(3,6), 'Medium', 'r_slip/v'),
        'M': (slice(8,10), 9, 'Easy', 'r_slip/r'),
        'N': (slice(8,10), slice(12,14), 'Easy', 'r_slip/w'),
        'O': (10, 6, 'Easy', '(y*λN)/λN'),
        'P': (10, 10, 'Easy', '(y*λN)/y'),
        'Q': (11, 9, 'Easy', '(r*s)/r'),
        'R': (11, 11, 'Easy', '(r*s)/s'),
        'S': (slice(12,14), slice(7,9), 'Easy', '(β∘w)/β'),
        'T': (slice(12,14), slice(12,14), 'Easy', '(β∘w)/w'),
    }
    
    logger.log("\n" + "="*80)
    logger.log("BLOCK-BY-BLOCK ERROR ANALYSIS")
    logger.log("="*80)
    logger.log(f"\n{'Block':<6} {'Type':<8} {'Shape':<10} {'Max Error':<12} {'Mean Error':<12} {'Status'}")
    logger.log("-"*80)
    
    results = {}
    non_perfect_blocks = []
    
    for block_name, (row, col, difficulty, description) in blocks.items():
        block_ana = J_analytical[row, col]
        block_auto = J_autograd[row, col]
        
        diff = block_ana - block_auto
        max_error = torch.abs(diff).max().item()
        mean_error = torch.abs(diff).mean().item()
        
        shape_str = str(block_ana.shape if hasattr(block_ana, 'shape') else tuple())
        
        results[block_name] = {
            'max_error': max_error,
            'mean_error': mean_error,
            'difficulty': difficulty,
            'analytical': block_ana,
            'autograd': block_auto,
        }
        
        # Status
        if max_error < 1e-10:
            status = "✓✓✓ Perfect"
        elif max_error < 1e-6:
            status = "✓✓ Good"
        elif max_error < 1e-4:
            status = "✓ OK"
        elif max_error < 1e-2:
            status = "⚠ Warning"
        else:
            status = "✗ Bad"
        
        # Track non-perfect blocks
        if max_error >= 1e-10:
            non_perfect_blocks.append((block_name, max_error, block_ana, block_auto, description))
        
        logger.log(f"{block_name:<6} {difficulty:<8} {shape_str:<10} "
                  f"{max_error:<12.2e} {mean_error:<12.2e} {status}")
    
    # Show detailed comparison for non-perfect blocks
    if non_perfect_blocks:
        logger.log("\n" + "="*80)
        logger.log("DETAILED COMPARISON FOR NON-PERFECT BLOCKS")
        logger.log("="*80)
        
        for block_name, max_err, ana, auto, desc in non_perfect_blocks:
            logger.log(f"\n{'-'*80}")
            logger.log(f"Block {block_name}: {desc}")
            logger.log(f"Max error: {max_err:.6e}")
            logger.log(f"{'-'*80}")
            
            # Convert to numpy for better printing
            if hasattr(ana, 'shape'):
                if len(ana.shape) == 0:  # scalar
                    logger.log(f"  Analytical: {ana.item():.10f}")
                    logger.log(f"  Autograd:   {auto.item():.10f}")
                    logger.log(f"  Difference: {(ana - auto).item():.10e}")
                elif len(ana.shape) == 1:  # 1D
                    np.set_printoptions(precision=10, suppress=True)
                    logger.log(f"  Analytical: {ana.cpu().numpy()}")
                    logger.log(f"  Autograd:   {auto.cpu().numpy()}")
                    logger.log(f"  Difference: {(ana - auto).cpu().numpy()}")
                elif len(ana.shape) == 2:  # 2D
                    np.set_printoptions(precision=10, suppress=True, linewidth=120)
                    logger.log(f"  Analytical:")
                    for i, row in enumerate(ana.cpu().numpy()):
                        logger.log(f"    Row {i}: {row}")
                    logger.log(f"  Autograd:")
                    for i, row in enumerate(auto.cpu().numpy()):
                        logger.log(f"    Row {i}: {row}")
                    logger.log(f"  Difference:")
                    diff_arr = (ana - auto).cpu().numpy()
                    for i, row in enumerate(diff_arr):
                        logger.log(f"    Row {i}: {row}")
            else:  # scalar
                logger.log(f"  Analytical: {float(ana):.10f}")
                logger.log(f"  Autograd:   {float(auto):.10f}")
                logger.log(f"  Difference: {float(ana - auto):.10e}")
    
    # Summary by difficulty
    logger.log("\n" + "="*80)
    logger.log("SUMMARY BY DIFFICULTY")
    logger.log("="*80)
    
    for diff_level in ['Easy', 'Medium']:
        blocks_in_level = {k: v for k, v in results.items() if v['difficulty'] == diff_level}
        
        if blocks_in_level:
            max_errors = [v['max_error'] for v in blocks_in_level.values()]
            mean_errors = [v['mean_error'] for v in blocks_in_level.values()]
            
            logger.log(f"\n{diff_level} blocks ({len(blocks_in_level)} total):")
            logger.log(f"  Max error range:  [{min(max_errors):.2e}, {max(max_errors):.2e}]")
            logger.log(f"  Mean error range: [{min(mean_errors):.2e}, {max(mean_errors):.2e}]")
            
            # Count by status
            perfect = sum(1 for e in max_errors if e < 1e-10)
            good = sum(1 for e in max_errors if 1e-10 <= e < 1e-6)
            ok = sum(1 for e in max_errors if 1e-6 <= e < 1e-4)
            warning = sum(1 for e in max_errors if 1e-4 <= e < 1e-2)
            bad = sum(1 for e in max_errors if e >= 1e-2)
            
            logger.log(f"  Status count: {perfect} perfect, {good} good, {ok} ok, {warning} warning, {bad} bad")
    
    return results


def run_comprehensive_test(logger):
    """Run full test suite with detailed logging."""
    
    logger.log("="*80)
    logger.log("COMPREHENSIVE JACOBIAN TEST WITH DETAILED LOGGING")
    logger.log("="*80)
    logger.log(f"\nTest time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.log(f"Device: {device}")
    
    # Setup
    params = {
        'mass': 1.0,
        'Izz': 0.02,
        'half_length': 0.1,
        'mu': 0.6,
        'dt': 0.05,
        'device': device
    }
    
    logger.log(f"\nPhysics parameters:")
    for k, v in params.items():
        logger.log(f"  {k}: {v}")
    
    # Create objects
    logger.log(f"\nCreating residual wrapper and analytical Jacobian...")
    
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
    logger.log(f"\nGenerating test case...")
    test_data = create_test_case(device=device)
    
    # Log test data
    logger.log(f"\nTest data:")
    z = test_data['z']
    logger.log(f"  z shape: {z.shape}")
    logger.log(f"  z = [q(3), v(3), λN(1), β(2), r(1), y(1), s(1), w(2)]")
    logger.log(f"  q:    [{z[0]:.4f}, {z[1]:.4f}, {z[2]:.4f}]")
    logger.log(f"  v:    [{z[3]:.4f}, {z[4]:.4f}, {z[5]:.4f}]")
    logger.log(f"  λN:   {z[6]:.4f}")
    logger.log(f"  β:    [{z[7]:.4f}, {z[8]:.4f}]")
    logger.log(f"  r:    {z[9]:.4f}")
    logger.log(f"  y:    {z[10]:.4f}")
    logger.log(f"  s:    {z[11]:.4f}")
    logger.log(f"  w:    [{z[12]:.4f}, {z[13]:.4f}]")
    
    # Compute Jacobians
    logger.log(f"\n" + "="*80)
    logger.log("COMPUTING JACOBIANS")
    logger.log("="*80)
    
    logger.log(f"\n1. Computing autograd Jacobian...")
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
    logger.log(f"   Time: {t_autograd*1000:.3f} ms")
    
    logger.log(f"\n2. Computing analytical Jacobian...")
    t0 = time.time()
    J_analytical = analytical_jac.compute_jacobian(
        z,
        q_k=test_data['qk'],
        v_k=test_data['vk'],
        pusher_pos=test_data['pusher_pos_next']
    )
    t_analytical = time.time() - t0
    logger.log(f"   Time: {t_analytical*1000:.3f} ms")
    
    logger.log(f"\n3. Performance:")
    logger.log(f"   Speedup: {t_autograd/t_analytical:.2f}x")
    logger.log(f"   Autograd:   {t_autograd*1000:.3f} ms")
    logger.log(f"   Analytical: {t_analytical*1000:.3f} ms")
    
    logger.log(f"\nJacobian shape: {J_analytical.shape}")
    
    # Show full Jacobian matrices with block labels
    logger.log("\n" + "="*80)
    logger.log("FULL JACOBIAN MATRIX (14x14) - ANALYTICAL")
    logger.log("="*80)
    
    # Column headers with variable names
    logger.log("\n           q_x       q_y       q_θ       v_x       v_y       v_ω       λN        β+        β-        r         y         s         w+        w-")
    logger.log("         [  0  ][  1  ][  2  ][  3  ][  4  ][  5  ][  6  ][  7  ][  8  ][  9  ][ 10  ][ 11  ][ 12  ][ 13  ]")
    logger.log("         " + "-"*145)
    
    row_labels = [
        "r_dyn_x  [0]", "r_dyn_y  [1]", "r_dyn_ω  [2]",
        "r_kin_x  [3]", "r_kin_y  [4]", "r_kin_ω  [5]",
        "r_gap    [6]",
        "r_cone   [7]",
        "r_slip+  [8]", "r_slip-  [9]",
        "y*λN    [10]",
        "r*s     [11]",
        "β+*w+   [12]", "β-*w-   [13]"
    ]
    
    for i in range(14):
        row_str = f"{row_labels[i]}: "
        for j in range(14):
            val = J_analytical[i, j].item()
            if abs(val) < 1e-10:
                row_str += "     .    "
            else:
                row_str += f"{val:10.5f}"
        logger.log(row_str)
    
    logger.log("\n" + "="*80)
    logger.log("FULL JACOBIAN MATRIX (14x14) - AUTOGRAD")
    logger.log("="*80)
    logger.log("\n           q_x       q_y       q_θ       v_x       v_y       v_ω       λN        β+        β-        r         y         s         w+        w-")
    logger.log("         [  0  ][  1  ][  2  ][  3  ][  4  ][  5  ][  6  ][  7  ][  8  ][  9  ][ 10  ][ 11  ][ 12  ][ 13  ]")
    logger.log("         " + "-"*145)
    
    for i in range(14):
        row_str = f"{row_labels[i]}: "
        for j in range(14):
            val = J_autograd[i, j].item()
            if abs(val) < 1e-10:
                row_str += "     .    "
            else:
                row_str += f"{val:10.5f}"
        logger.log(row_str)
    
    logger.log("\n" + "="*80)
    logger.log("DIFFERENCE MATRIX (Analytical - Autograd)")
    logger.log("="*80)
    logger.log("\n           q_x       q_y       q_θ       v_x       v_y       v_ω       λN        β+        β-        r         y         s         w+        w-")
    logger.log("         [  0  ][  1  ][  2  ][  3  ][  4  ][  5  ][  6  ][  7  ][  8  ][  9  ][ 10  ][ 11  ][ 12  ][ 13  ]")
    logger.log("         " + "-"*145)
    
    diff_matrix = J_analytical - J_autograd
    max_diff = 0.0
    max_loc = (-1, -1)
    
    for i in range(14):
        row_str = f"{row_labels[i]}: "
        for j in range(14):
            val = diff_matrix[i, j].item()
            abs_val = abs(val)
            
            if abs_val > max_diff:
                max_diff = abs_val
                max_loc = (i, j)
            
            if abs_val < 1e-10:
                row_str += "     .    "
            elif abs_val > 1e-3:
                row_str += f"**{val:8.5f}**"  # Highlight large errors
            elif abs_val > 1e-6:
                row_str += f" {val:9.5f} "
            else:
                row_str += f"{val:10.5f}"
        logger.log(row_str)
    
    logger.log("\n" + "="*80)
    logger.log("BLOCK STRUCTURE REFERENCE")
    logger.log("="*80)
    logger.log("""
         q(3)  v(3)  λN(1) β(2)  r(1)  y(1)  s(1)  w(2)
         0:3   3:6   6     7:9   9     10    11    12:14
r_dyn    A     B     C     X     -     -     -     -      (0:3)
r_kin    D     E     -     -     -     -     -     -      (3:6)
r_gap    G     -     -     -     -     H     -     -      (6)
r_cone   -     -     I     J     -     -     K     -      (7)
r_slip   F     L     -     -     M     -     -     N      (8:10)
y*λN     O     -     -     -     -     P     -     -      (10)
r*s      -     -     -     -     Q     -     R     -      (11)
β∘w      -     -     -     S     -     -     -     T      (12:14)

Total: 22 blocks
- Easy: 14 blocks (B, D, E, H, I, J, K, M, N, O, P, Q, R, S, T)
- Medium: 8 blocks (A, C, X, F, G, L)
    """)
    
    # Detailed analysis
    results = analyze_jacobian_blocks(J_analytical, J_autograd, logger)
    
    # Overall error
    diff = torch.abs(J_analytical - J_autograd)
    max_error = diff.max().item()
    max_loc = torch.where(diff == diff.max())
    max_row, max_col = max_loc[0][0].item(), max_loc[1][0].item()
    
    # Test tolerance
    logger.log("\n" + "="*80)
    logger.log("TOLERANCE TEST")
    logger.log("="*80)
    
    rtol, atol = 1e-3, 1e-5
    passed = torch.allclose(J_analytical, J_autograd, rtol=rtol, atol=atol)
    
    logger.log(f"\nTolerance: rtol={rtol:.0e}, atol={atol:.0e}")
    logger.log(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    # Overall error location
    logger.log("\n" + "="*80)
    logger.log("OVERALL ERROR ANALYSIS")
    logger.log("="*80)
    logger.log(f"\nMax absolute error: {max_error:.6e}")
    logger.log(f"Location: J[{max_row}, {max_col}]")
    logger.log(f"  Analytical value: {J_analytical[max_row, max_col].item():.10f}")
    logger.log(f"  Autograd value:   {J_autograd[max_row, max_col].item():.10f}")
    logger.log(f"  Difference:       {(J_analytical[max_row, max_col] - J_autograd[max_row, max_col]).item():.10e}")
    
    # Identify which block this belongs to
    blocks_map = {
        'A': (slice(0,3), slice(0,3)),
        'B': (slice(0,3), slice(3,6)),
        'C': (slice(0,3), 6),
        'X': (slice(0,3), slice(7,9)),  # ADDED
        'D': (slice(3,6), slice(0,3)),
        'E': (slice(3,6), slice(3,6)),
        'G': (6, slice(0,3)),
        'H': (6, 10),
        'I': (7, 6),
        'J': (7, slice(7,9)),
        'K': (7, 11),
        'F': (slice(8,10), slice(0,3)),  # ADDED: r_slip/q
        'L': (slice(8,10), slice(3,6)),
        'M': (slice(8,10), 9),
        'N': (slice(8,10), slice(12,14)),
        'O': (10, 6),
        'P': (10, 10),
        'Q': (11, 9),
        'R': (11, 11),
        'S': (slice(12,14), slice(7,9)),
        'T': (slice(12,14), slice(12,14)),
    }
    
    found_block = "Unknown"
    for block_name, (row_slice, col_slice) in blocks_map.items():
        # Convert slice to range
        if isinstance(row_slice, slice):
            row_range = range(row_slice.start or 0, row_slice.stop or 14)
        else:
            row_range = [row_slice]
        
        if isinstance(col_slice, slice):
            col_range = range(col_slice.start or 0, col_slice.stop or 14)
        else:
            col_range = [col_slice]
        
        if max_row in row_range and max_col in col_range:
            found_block = block_name
            break
    
    logger.log(f"  Belongs to Block: {found_block}")
    
    # Show nearby elements for context
    logger.log(f"\nContext (3x3 region around error):")
    r_start = max(0, max_row - 1)
    r_end = min(14, max_row + 2)
    c_start = max(0, max_col - 1)
    c_end = min(14, max_col + 2)
    
    logger.log(f"  Analytical:")
    for i in range(r_start, r_end):
        row_str = "    "
        for j in range(c_start, c_end):
            marker = ">" if (i == max_row and j == max_col) else " "
            row_str += f"{marker}{J_analytical[i, j].item():10.6f} "
        logger.log(row_str)
    
    logger.log(f"  Autograd:")
    for i in range(r_start, r_end):
        row_str = "    "
        for j in range(c_start, c_end):
            marker = ">" if (i == max_row and j == max_col) else " "
            row_str += f"{marker}{J_autograd[i, j].item():10.6f} "
        logger.log(row_str)
    
    # Final summary
    logger.log("\n" + "="*80)
    logger.log("FINAL SUMMARY")
    logger.log("="*80)
    
    logger.log(f"\nMax absolute error: {max_error:.6e}")
    logger.log(f"Speedup: {t_autograd/t_analytical:.2f}x")
    
    # Overall assessment
    if max_error < 1e-4:
        logger.log(f"\n✓ ASSESSMENT: Excellent for Phase 1 (finite differences)")
    elif max_error < 1e-2:
        logger.log(f"\n⚠ ASSESSMENT: Acceptable but could be improved")
    else:
        logger.log(f"\n✗ ASSESSMENT: Errors too large - check implementation")
    
    logger.log("\n" + "="*80)
    logger.log("TEST COMPLETED")
    logger.log("="*80)


def main():
    # Create logger with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'jacobian_test_results_{timestamp}.txt'
    
    logger = TestLogger(log_filename)
    
    try:
        logger.log("Starting comprehensive test...\n")
        run_comprehensive_test(logger)
        logger.log(f"\n\n✓ Results saved to: {log_filename}")
        
    except Exception as e:
        logger.log(f"\n\n✗ ERROR: {str(e)}")
        import traceback
        logger.log("\nTraceback:")
        logger.log(traceback.format_exc())
    
    finally:
        logger.close()
    
    print(f"\n{'='*80}")
    print(f"Log file saved: {log_filename}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()