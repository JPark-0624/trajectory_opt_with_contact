"""
Example: IRL Weight Recovery for Contact-Implicit Trajectory Optimization

This example demonstrates:
1. Generate expert demonstration with known weights
2. Recover weights from demonstration using IRL
3. Compare recovered weights with ground truth
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Assuming these are in your project structure
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from trajectory_opt_with_contact.optimizer import TrajectoryOptimizer
from trajectory_opt_with_contact.irl_weight_recovery import IRLWeightRecovery, IRLOptions, pack_demo
from trajectory_opt_with_contact.utils import save_demo_npz, load_demo_npz


def generate_expert_demonstration(
    optimizer: TrajectoryOptimizer,
    q0, v0, pusher0, goal,
    w_true: dict,
    obstacle_pos=None,
    max_iters=100
):
    """
    Generate expert demonstration using known weights.
    
    Args:
        optimizer: TrajectoryOptimizer instance
        q0, v0, pusher0, goal: Initial state and goal
        w_true: True weights used by expert
        obstacle_pos: Optional obstacle position
        max_iters: Max optimization iterations
    
    Returns:
        demo: Dictionary containing demonstration data
        result: Full optimization result from expert
    """
    print("\n" + "="*80)
    print("GENERATING EXPERT DEMONSTRATION")
    print("="*80)
    print(f"True weights:")
    for name, val in w_true.items():
        print(f"  {name:12s}: {val:.6f}")
    
    u_init = optimizer.compute_geometric_initial_trajectory(
        robot_pos=pusher0,      # [x, y] - initial robot position
        box_pos=q0,        # [x, y] - initial box position
        goal_pos=goal,       # [x, y] - goal position (only x, y used)
        )


    # Run trajectory optimization with true weights
    result = optimizer.optimize(
        q0=q0, v0=v0, pusher0=pusher0, goal=goal,
        w_target=w_true['w_target'],
        w_orient=w_true['w_orient'],
        w_v=w_true['w_v'],
        w_ctrl=w_true['w_ctrl'],
        w_obs=w_true['w_obs'],
        u_init=u_init,
        obstacle_pos=obstacle_pos,
        max_iters=max_iters,
        lr=0.01,
        lr_decay_gamma=0.5,
        lr_decay_step=20,
        verbose=True
    )
    
    # Extract demonstration control sequence
    u_demo = result['u_seq']
    
    # Pack into demo dict
    demo = pack_demo(q0, v0, pusher0, goal, u_demo, obstacle_pos)
    
    print(f"\nDemonstration generated:")
    print(f"  Final loss: {result['loss']:.6f}")
    print(f"  Control sequence shape: {u_demo.shape}")
    print(f"  Final position: {result['q_final']}")
    
    return demo, result


def recover_weights_from_demo(
    optimizer: TrajectoryOptimizer,
    demo: dict,
    method: str = "feature_matching"
):
    """
    Recover weights from demonstration using IRL.
    
    Args:
        optimizer: TrajectoryOptimizer instance
        demo: Demonstration dictionary
        method: "feature_matching" or "control_matching"
    
    Returns:
        irl_result: Dictionary containing recovered weights and diagnostics
    """
    print("\n" + "="*80)
    print(f"RECOVERING WEIGHTS ({method.upper()})")
    print("="*80)
    
    # Create IRL solver
    irl = IRLWeightRecovery(optimizer)
    
    # Set options
    opts = IRLOptions(
        max_outer_iters=50,
        max_inner_iters=100,
        lr_weights=0.01,
        lr_trajectory=0.01,
        method=method,
        weight_tol=1e-4,
        control_tol=1e-3,
        warm_start_inner=True,
        verbose=True
    )
    
    # Recover weights
    irl_result = irl.recover_weights(demo, opts=opts)
    
    return irl_result


def compare_weights(w_true: dict, w_recovered: torch.Tensor, feature_names: list):
    """
    Compare true and recovered weights.
    
    Args:
        w_true: Dictionary of true weights
        w_recovered: Recovered weight tensor
        feature_names: List of feature names
    """
    print("\n" + "="*80)
    print("WEIGHT COMPARISON")
    print("="*80)
    print(f"{'Feature':12s} {'True':>10s} {'Recovered':>10s} {'Error':>10s} {'Rel Error':>10s}")
    print("-"*60)
    
    w_true_tensor = torch.tensor([
        w_true['w_target'], w_true['w_orient'], w_true['w_v'], 
        w_true['w_ctrl'], w_true['w_obs']
    ])
    
    # Normalize true weights for fair comparison
    w_true_norm = w_true_tensor / w_true_tensor.sum()
    
    total_error = 0.0
    for i, name in enumerate(feature_names):
        true_val = w_true_norm[i].item()
        rec_val = w_recovered[i].item()
        error = abs(rec_val - true_val)
        rel_error = error / (true_val + 1e-10) * 100
        total_error += error
        
        print(f"{name:12s} {true_val:10.6f} {rec_val:10.6f} {error:10.6f} {rel_error:9.2f}%")
    
    print("-"*60)
    print(f"Total L1 error: {total_error:.6f}")
    print(f"L2 error: {torch.norm(w_recovered - w_true_norm).item():.6f}")


def plot_irl_convergence(irl_result: dict, save_path: str = None):
    """
    Plot IRL convergence diagnostics.
    
    Args:
        irl_result: Result dictionary from IRL
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss history
    axes[0].plot(irl_result['loss_history'], 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('IRL Loss Convergence')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Control error history
    axes[1].plot(irl_result['control_error_history'], 'r-', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Control Error (L2 norm)')
    axes[1].set_title('Control Matching Error')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nConvergence plot saved to: {save_path}")
    
    plt.show()


def plot_feature_comparison(irl_result: dict, feature_names: list, save_path: str = None):
    """
    Plot feature values: demo vs recovered.
    
    Args:
        irl_result: Result dictionary from IRL
        feature_names: List of feature names
        save_path: Optional path to save figure
    """
    if 'features_demo' not in irl_result:
        print("Feature comparison only available for feature_matching method")
        return
    
    features_demo = irl_result['features_demo'].cpu().numpy()
    features_final = irl_result['features_final'].cpu().numpy()
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, features_demo, width, label='Demo', alpha=0.8)
    bars2 = ax.bar(x + width/2, features_final, width, label='Recovered', alpha=0.8)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Feature Value')
    ax.set_title('Feature Comparison: Demo vs Recovered')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature comparison plot saved to: {save_path}")
    
    plt.show()


def main():
    """
    Main example workflow:
    1. Create optimizer
    2. Generate expert demonstration
    3. Recover weights using IRL
    4. Compare and visualize results
    """
    
    # ========== Setup ==========
    
    # Create optimizer (same settings for demo generation and recovery)
    optimizer = TrajectoryOptimizer(
        mass=1.0,
        side_length=0.2,
        mu=0.5,
        horizon=100,
        dt=0.05,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        TO_solver='shooting',
        dynamics_solver='IP'
    )
    
    q0 = [0.0, 0.0, 0.0]
    v0 = [0.0, 0.0, 0.0]
    pusher0 = [0.3, -0.3]
    goal = [-0.2, 0.5, -0.3]
    obstacle_pos = None  # [0.2, -0.2]  # Optional obstacle
    
    # True weights used by expert
    w_true = {
        'w_target': 20.0,
        'w_orient': 0.5,
        'w_v': 1.0,
        'w_ctrl': 1e-3,
        'w_obs': 0.0
    }
    
    # ========== Generate Demonstration ==========
    
    demo, expert_result = generate_expert_demonstration(
        optimizer, q0, v0, pusher0, goal, w_true,
        obstacle_pos=obstacle_pos,
        max_iters=100
    )
    
    # Optionally save demonstration

    #save_demo_npz(demo, demo_path)
    #print(f"\nDemonstration saved to: {demo_path}")
    
    # ========== Recover Weights ==========
    
    # Method 1: Feature Matching (faster, simpler)
    print("\n\n")
    irl_result_fm = recover_weights_from_demo(
        optimizer, demo, method="control_matching"
    )
    
    # Method 2: Control Matching (more accurate, slower)
    # Note: This requires backprop through entire optimization!
    # print("\n\n")
    # irl_result_cm = recover_weights_from_demo(
    #     optimizer, demo, method="control_matching"
    # )
    
    # ========== Analysis ==========
    
    feature_names = ['w_target', 'w_orient', 'w_v', 'w_ctrl', 'w_obs']
    
    # Compare weights
    print("\n\nFEATURE MATCHING RESULTS:")
    compare_weights(w_true, irl_result_fm['w_recovered'], feature_names)
    
    # Plot convergence
    plot_irl_convergence(irl_result_fm, save_path="/home/claude/irl_convergence.png")
    
    # Plot feature comparison
    plot_feature_comparison(irl_result_fm, feature_names, 
                           save_path="/home/claude/feature_comparison.png")
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
