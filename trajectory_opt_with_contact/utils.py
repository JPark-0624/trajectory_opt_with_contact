# ioc_io.py
import os
import numpy as np
import torch
from typing import Dict, Union, Optional

ArrayLike = Union[np.ndarray, torch.Tensor, list, tuple, float]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def pack_demo(q0, v0, pusher0, goal, u_demo, obstacle_pos=None) -> Dict:
    """
    Create a canonical demo dict compatible with IOCFitter.
    Shapes:
      q0, v0, goal: (3,), pusher0: (2,), u_demo: (T,2)
      obstacle_pos: (2,) or None
    """
    return dict(
        q0=_to_numpy(q0),
        v0=_to_numpy(v0),
        pusher0=_to_numpy(pusher0),
        goal=_to_numpy(goal),
        u_demo=_to_numpy(u_demo),
        obstacle_pos=None if obstacle_pos is None else _to_numpy(obstacle_pos),
    )

# ---------- NPZ (portable, language-agnostic) ----------
def save_demo_npz(demo: Dict, path: str):
    ensure_dir(path)
    save_dict = {}
    for k, v in demo.items():
        if v is None:
            continue   # skip None fields (e.g. obstacle_pos)
        save_dict[k] = _to_numpy(v)
    np.savez_compressed(path, **save_dict)

def load_demo_npz(path: str) -> Dict:
    data = np.load(path, allow_pickle=True)
    demo = {k: data[k] for k in data.files}
    # Normalize None for obstacle_pos if absent
    if "obstacle_pos" not in demo:
        demo["obstacle_pos"] = None
    return demo

# ---------- PT (Torch-native; preserves dtypes exactly) ----------
def save_demo_pt(demo: Dict, path: str):
    ensure_dir(path)
    # Convert numpy to tensors for torch.save (still portable within PyTorch)
    t_demo = {}
    for k, v in demo.items():
        if v is None:
            t_demo[k] = None
        else:
            t_demo[k] = torch.as_tensor(v, dtype=torch.double)
    torch.save(t_demo, path)

def load_demo_pt(path: str, device: Optional[torch.device] = None) -> Dict:
    d = torch.load(path, map_location="cpu")
    # Return numpy for consistency with pack_demo
    demo = {}
    for k, v in d.items():
        if v is None:
            demo[k] = None
        else:
            demo[k] = v.detach().cpu().numpy()
    return demo