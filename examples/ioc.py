from trajectory_opt_with_contact import TrajectoryOptimizer, IOCFitter, rollout
import torch
import numpy as np
from icecream import ic


# 1) Generate a demonstration with your current optimizer (any weights)
optimizer = TrajectoryOptimizer(
    mass=1.0, side_length=0.2, mu=0.6,
    horizon=30, dt=0.05,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    TO_solver='shooting',
    dynamics_solver='IP',           # or 'LCP'
    use_second_order=True,
    alm_enabled=True, alm_rho_init=1e2, alm_target_tol=1e-6,
    alm_outer_iters=8, lbfgs_inner_steps=10
)

q0 = [0.0, 0.0, 0.0]
v0 = [0.0, 0.0, 0.0]
pusher0 = [0.3, -0.3]
goal = [-0.2, 0.5, -0.3]
obstacle = None
u_init = [[-0.2, 0.2]] * optimizer.horizon

demo_res = optimizer.optimize(
    q0=q0, v0=v0, pusher0=pusher0, goal=goal,
    w_target = 20.0, w_v = 0.1, w_ctrl = 10, w_obs = 1.0,
    u_init=u_init, max_iters=30, lr=0.05,
    lr_decay_step=10,
    lr_decay_gamma=0.8,
    obstacle_pos=obstacle, verbose=True
)

u_demo = demo_res["u_seq"]
np.savez_compressed('demo.npz', u_demo=np.asarray(u_demo))

u_demo = np.load('demo.npz', allow_pickle=True)['u_demo']

ic(u_demo)
# # Compute per-term ∂f/∂u at the demo controls
# u_demo_var = torch.as_tensor(u_demo, dtype=torch.double, device=optimizer.device).detach().requires_grad_(True)

# def control_grad_of(term_name):
#     # reuse your rollout, but zero out other weights to isolate a single term
#     w = dict(w_target=0.0, w_v=0.0, w_ctrl=0.0, w_obs=0.0)
#     w[term_name] = 1.0  # isolate this term's contribution
#     total, qf, lamb, phi, qs, pr, goal_term, ctrl_term, v_term, obs_term, pen_term = rollout(
#         u_demo_var, optimizer._to_tensor(q0,requires_grad=True), optimizer._to_tensor(v0,requires_grad=True),
#         optimizer._to_tensor(pusher0,requires_grad=True), optimizer.horizon, optimizer.dt,
#         optimizer.m, optimizer.Izz, optimizer.half, optimizer.mu,
#         optimizer._to_tensor(goal,requires_grad=True),
#         w_target=w["w_target"], w_v=w["w_v"], w_ctrl=w["w_ctrl"], w_obs=w["w_obs"],
#         qp_solver=optimizer.qp_solver, dynamics_solver=optimizer.dynamics_solver,
#         obstacle_pos=None, device=optimizer.device
#     )
#     term = {"w_target": goal_term, "w_v": v_term, "w_ctrl": ctrl_term, "w_obs": obs_term}[term_name]
#     (g,) = torch.autograd.grad(term, u_demo_var, create_graph=False, retain_graph=True)
#     return g.detach()

# g_target = control_grad_of("w_target")
# g_v      = control_grad_of("w_v")

# # cosine similarity ~ 1.0 => colinear
# cos = torch.nn.functional.cosine_similarity(g_target.view(-1), g_v.view(-1), dim=0)
# print("cosine(g_target, g_v) =", float(cos))
# exit()
# torch.autograd.set_detect_anomaly(True)
# 2) Fit weights so that u_demo satisfies stationarity for J_w
demo = dict(q0=q0, v0=v0, pusher0=pusher0, goal=goal, u_demo=u_demo, obstacle_pos=obstacle)
ioc = IOCFitter(optimizer, demo, learnable_terms=(), normalize_weights=True) #"w_target","w_v","w_ctrl"

learned_weights = ioc.fit_weights_kkt(max_outer_iters=50, lr=1, verbose=True)
# learned_weights = ioc.fit_weights_unrolled(max_outer_iters=50)
print("Learned weights (KKT):", learned_weights)
