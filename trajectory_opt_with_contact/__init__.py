"""
Contact-Implicit Trajectory Optimization Package

A differentiable contact physics simulator with trajectory optimization
for planar manipulation tasks.
"""

from .qp_solver import ContactQPSolver
from .geometry import rot, closest_point_on_square_and_normal, contact_frame_and_J
from .dynamics import step_square, rollout, step_square_pos_ip, IPMOptions, step_square_pos_ip_lin
from .optimizer import TrajectoryOptimizer
from .transcription import DirectTranscriptionOptimizer
from .visualizer import TrajectoryVisualizer, visualize_result

__version__ = "0.2.0"
__all__ = [
    "ContactQPSolver",
    "rot",
    "closest_point_on_square_and_normal",
    "contact_frame_and_J",
    "step_square",
    "step_square_ip"
    "rollout",
    "TrajectoryOptimizer",
    "DirectTranscriptionOptimizer", 
    "TrajectoryVisualizer",
    "visualize_result",
]