from __future__ import annotations

from typing import Optional

import math
import torch

from core.q_funcs import q_arr_dist


def ensure_signal_shape(x_arr: Optional[torch.Tensor], ref_z: torch.Tensor) -> Optional[torch.Tensor]:
    if x_arr is None:
        return None
    x = x_arr.to(device=ref_z.device, dtype=ref_z.dtype)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.shape != ref_z.shape:
        raise ValueError(f"x_arr shape mismatch: expected {tuple(ref_z.shape)}, got {tuple(x.shape)}")
    return x


def print_iter_log(
    alg_name: str,
    iter_idx: int,
    total_iters: int,
    A_mat: torch.Tensor,
    psi: torch.Tensor,
    z: torch.Tensor,
    x_true: Optional[torch.Tensor],
) -> None:
    _ = A_mat, psi
    msg = f"[{alg_name}] epoch {iter_idx}/{total_iters}"
    if x_true is not None:
        dist = float(q_arr_dist(z, x_true).item())
        msg += f" dist_T={dist:.6e} ln_dist_T={math.log(max(dist, 1e-300)):.6f}"
    print(msg)
