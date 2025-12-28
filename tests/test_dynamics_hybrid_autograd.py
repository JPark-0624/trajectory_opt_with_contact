"""
Test Hybrid Jacobian Accuracy During Full Trajectory

This script:
1. Runs the same trajectory TWICE (autograd vs hybrid)
2. Compares results at each step
3. Logs Jacobian method performance
4. Visualizes where hybrid fails

Based on your test_dynamics.py
"""

import torch
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from trajectory_opt_with_contact import (
    step_square_pos_ip, IPMOptions
)

# ===========================================================
# Setup
# ===========================================================
device = torch.device("cuda")
dtype = torch.float64  # FIXED: Use float64 for trajectory optimization!
                       # float32 causes exponential error accumulation
                       # in contact dynamics over 150 steps

# Physical parameters
m = 1.0
Izz = 1.0 / 6.0
half = 0.1
mu = 0.9
h = 0.05
horizon = 150  # Shorter for testing

# Initial state
q0 = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
v0 = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
pr0 = torch.tensor([-3.0, 0.0], dtype=dtype, device=device)

# Control sequence
u_seq = torch.tensor([[2.0, 0.0]] * horizon, dtype=dtype, device=device)

# IPM options
ipm_opts = IPMOptions(
    target_mu=1e-4, 
    max_newton=20, 
    tol=1e-6,  # FIXED: Tightened from 1e-3 for better convergence
    smooth_sdf=50.0,
    enable_viscous_ground_friction=True,
    c_lin=8.0,          
    c_ang=8.0 * half     
)

os.makedirs("results", exist_ok=True)

# ===========================================================
# Run trajectory with AUTOGRAD
# ===========================================================
print("\n" + "="*70)
print("RUNNING TRAJECTORY WITH AUTOGRAD")
print("="*70)

qs_auto = [q0]
vs_auto = [v0]
prs_auto = [pr0]
lams_auto = []
phis_auto = []

start_time = time.time()

q, v, pr = q0.clone(), v0.clone(), pr0.clone()
for k in range(horizon):
    q_next, v_next, pr_next, lam, phi = step_square_pos_ip(
        q, v, pr, u_seq[k], 
        h=h, m=m, Izz=Izz, half=half, mu=mu,
        skip_solving_threshold=0.3,
        ipm_opts=ipm_opts,
        jacobian_type="autograd"  # AUTOGRAD
    )
    
    qs_auto.append(q_next)
    vs_auto.append(v_next)
    prs_auto.append(pr_next)
    lams_auto.append(lam)
    phis_auto.append(phi)
    q, v, pr = q_next, v_next, pr_next
    
    if (k+1) % 10 == 0:
        print(f"  Step {k+1}/{horizon} complete")

time_auto = time.time() - start_time
print(f"\nAutograd total time: {time_auto:.3f} seconds")
print(f"Average per step: {time_auto/horizon*1000:.1f} ms")

# ===========================================================
# Run trajectory with HYBRID
# ===========================================================
print("\n" + "="*70)
print("RUNNING TRAJECTORY WITH HYBRID")
print("="*70)

qs_hybrid = [q0]
vs_hybrid = [v0]
prs_hybrid = [pr0]
lams_hybrid = []
phis_hybrid = []

start_time = time.time()

q, v, pr = q0.clone(), v0.clone(), pr0.clone()
converged_steps = 0
max_iter_steps = 0

for k in range(horizon):
    q_next, v_next, pr_next, lam, phi = step_square_pos_ip(
        q, v, pr, u_seq[k], 
        h=h, m=m, Izz=Izz, half=half, mu=mu,
        skip_solving_threshold=0.3,
        ipm_opts=ipm_opts,
        jacobian_type="hybrid"  # HYBRID
    )
    
    qs_hybrid.append(q_next)
    vs_hybrid.append(v_next)
    prs_hybrid.append(pr_next)
    lams_hybrid.append(lam)
    phis_hybrid.append(phi)
    q, v, pr = q_next, v_next, pr_next
    
    if (k+1) % 10 == 0:
        print(f"  Step {k+1}/{horizon} complete")

time_hybrid = time.time() - start_time
print(f"\nHybrid total time: {time_hybrid:.3f} seconds")
print(f"Average per step: {time_hybrid/horizon*1000:.1f} ms")

# ===========================================================
# Compare results
# ===========================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

# Convert to arrays
qs_auto_arr = torch.stack(qs_auto).cpu().numpy()
qs_hybrid_arr = torch.stack(qs_hybrid).cpu().numpy()
lams_auto_arr = torch.stack(lams_auto).cpu().numpy()
lams_hybrid_arr = torch.stack(lams_hybrid).cpu().numpy()
phis_auto_arr = torch.tensor(phis_auto).cpu().numpy()
phis_hybrid_arr = torch.tensor(phis_hybrid).cpu().numpy()

# Compute differences
q_diff = np.linalg.norm(qs_auto_arr - qs_hybrid_arr, axis=1)
lam_diff = np.linalg.norm(lams_auto_arr - lams_hybrid_arr, axis=1)
phi_diff = np.abs(phis_auto_arr - phis_hybrid_arr)

print(f"\nPosition difference (L2 norm):")
print(f"  Mean: {q_diff.mean():.6e}")
print(f"  Max:  {q_diff.max():.6e}")
print(f"  Steps with diff > 1e-3: {np.sum(q_diff > 1e-3)}/{horizon}")

print(f"\nContact force difference:")
print(f"  Mean: {lam_diff.mean():.6e}")
print(f"  Max:  {lam_diff.max():.6e}")

print(f"\nGap difference:")
print(f"  Mean: {phi_diff.mean():.6e}")
print(f"  Max:  {phi_diff.max():.6e}")

print(f"\nPerformance:")
print(f"  Autograd: {time_auto/horizon*1000:.1f} ms/step")
print(f"  Hybrid:   {time_hybrid/horizon*1000:.1f} ms/step")
print(f"  Ratio:    {time_hybrid/time_auto:.2f}x")

# ===========================================================
# Analyze contact regions
# ===========================================================
print("\n" + "="*70)
print("CONTACT ANALYSIS")
print("="*70)

# Find contact steps (phi < 0.01)
contact_mask = phis_auto_arr < 0.01
contact_steps = np.where(contact_mask)[0]
no_contact_steps = np.where(~contact_mask)[0]

print(f"\nSteps in contact (φ < 0.01): {len(contact_steps)}/{horizon}")
if len(contact_steps) > 0:
    print(f"  Position diff (mean): {q_diff[contact_steps].mean():.6e}")
    print(f"  Position diff (max):  {q_diff[contact_steps].max():.6e}")
    print(f"  Force diff (mean):    {lam_diff[contact_steps].mean():.6e}")
    print(f"  Force diff (max):     {lam_diff[contact_steps].max():.6e}")

print(f"\nSteps NOT in contact (φ ≥ 0.01): {len(no_contact_steps)}/{horizon}")
if len(no_contact_steps) > 0:
    print(f"  Position diff (mean): {q_diff[no_contact_steps].mean():.6e}")
    print(f"  Position diff (max):  {q_diff[no_contact_steps].max():.6e}")
    print(f"  Force diff (mean):    {lam_diff[no_contact_steps].mean():.6e}")
    print(f"  Force diff (max):     {lam_diff[no_contact_steps].max():.6e}")

if len(contact_steps) > 0 and len(no_contact_steps) > 0:
    contact_error = q_diff[contact_steps].mean()
    no_contact_error = q_diff[no_contact_steps].mean()
    if no_contact_error > 0:
        ratio = contact_error / no_contact_error
        print(f"\n⚠️  Error ratio (contact / no-contact): {ratio:.1f}x")
        if ratio > 10:
            print("    → Hybrid SIGNIFICANTLY WORSE in contact!")
        elif ratio > 2:
            print("    → Hybrid moderately worse in contact")

# ===========================================================
# Visualization
# ===========================================================
print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Autograd vs Hybrid Jacobian Comparison', fontsize=16)

# Steps for plotting states (includes initial state, so horizon+1 points)
steps = np.arange(horizon+1)

# Row 1: Position difference
ax = axes[0, 0]
ax.semilogy(steps, q_diff, 'r-', linewidth=2, label='Position diff')
ax.axhline(1e-6, color='g', linestyle='--', alpha=0.5, label='Good (1e-6)')
ax.axhline(1e-3, color='orange', linestyle='--', alpha=0.5, label='Warning (1e-3)')
ax.set_xlabel('Step')
ax.set_ylabel('||q_auto - q_hybrid||')
ax.set_title('Position Difference')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
# lam_diff has horizon elements (no initial value), so use steps[1:]
ax.semilogy(steps[1:], lam_diff, 'b-', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('||λ_auto - λ_hybrid||')
ax.set_title('Contact Force Difference')
ax.grid(True, alpha=0.3)

# Row 2: Contact state
ax = axes[1, 0]
# phi arrays have horizon elements (no initial value), so use steps[1:]
ax.plot(steps[1:], phis_auto_arr, 'b-', linewidth=2, label='Autograd', alpha=0.7)
ax.plot(steps[1:], phis_hybrid_arr, 'r--', linewidth=2, label='Hybrid', alpha=0.7)
ax.axhline(0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel('Step')
ax.set_ylabel('φ (gap)')
ax.set_title('Contact Gap')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
# lam arrays have horizon elements (no initial value), so use steps[1:]
ax.plot(steps[1:], lams_auto_arr[:, 0], 'b-', linewidth=2, label='Autograd', alpha=0.7)
ax.plot(steps[1:], lams_hybrid_arr[:, 0], 'r--', linewidth=2, label='Hybrid', alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('λN (normal force)')
ax.set_title('Contact Force')
ax.legend()
ax.grid(True, alpha=0.3)

# Row 3: Trajectory
ax = axes[2, 0]
ax.plot(qs_auto_arr[:, 0], qs_auto_arr[:, 1], 'b-', linewidth=2, 
        label='Autograd', alpha=0.7)
ax.plot(qs_hybrid_arr[:, 0], qs_hybrid_arr[:, 1], 'r--', linewidth=2, 
        label='Hybrid', alpha=0.7)
ax.plot(q0[0].cpu(), q0[1].cpu(), 'go', markersize=10, label='Start')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Trajectory (x-y)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Correlation plot
ax = axes[2, 1]
# phi_diff has horizon elements, use steps[1:] for color mapping
scatter = ax.scatter(phis_auto_arr, q_diff[1:], c=steps[1:], cmap='viridis', s=50, alpha=0.6)
ax.set_xlabel('φ (gap)')
ax.set_ylabel('Position difference')
ax.set_title('Error vs Contact Gap')
ax.set_yscale('log')
ax.axvline(0.01, color='r', linestyle='--', alpha=0.5, label='Contact threshold')
plt.colorbar(scatter, ax=ax, label='Step')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/hybrid_vs_autograd_comparison.png', dpi=150)
print("  Saved: results/hybrid_vs_autograd_comparison.png")

# ===========================================================
# Final verdict
# ===========================================================
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

if q_diff.max() > 1e-2:
    print("\n❌ HYBRID JACOBIAN FAILED")
    print("   Large trajectory divergence detected!")
    print(f"   Max position error: {q_diff.max():.6e}")
elif q_diff.max() > 1e-4:
    print("\n⚠️  HYBRID JACOBIAN QUESTIONABLE")
    print("   Moderate trajectory divergence")
    print(f"   Max position error: {q_diff.max():.6e}")
else:
    print("\n✓ Hybrid Jacobian produced similar results")
    print(f"  Max position error: {q_diff.max():.6e}")

if time_hybrid > time_auto:
    print(f"\n⏱️  Performance: Hybrid is {time_hybrid/time_auto:.2f}x SLOWER")
    print("   → No benefit from hybrid approach!")
else:
    print(f"\n⏱️  Performance: Hybrid is {time_auto/time_hybrid:.2f}x faster")

# Recommendation
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if q_diff.max() > 1e-3 or time_hybrid > time_auto:
    print("\n🚫 DO NOT USE HYBRID JACOBIAN")
    print("   Reasons:")
    if q_diff.max() > 1e-3:
        print("   - Significant accuracy loss")
    if time_hybrid > time_auto:
        print("   - No performance benefit")
    print("\n   → Stick with AUTOGRAD")
else:
    print("\n✓ Hybrid Jacobian is viable")
    print("  (But verify on your specific use cases)")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)