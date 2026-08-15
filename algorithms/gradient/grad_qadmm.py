"""Quaternion ADMM (QADMM) for intensity phase retrieval — single-trial core loop."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from algorithms.gradient.grad_qrkm import quat_total_loss
from core.q_funcs import phase_factor_estimate
from core.quat_linalg_minimal import quat_least_squares_conj_left, quat_matvec_conj_left
from core.quat_real_embed import quat_matrix_conj_left_real_block
from core.quat_metrics import quat_dist_right_phase, quat_raw_distance
from core.quat_ops import quat_abs
from utils.common import assert_float64, to_float64


def qadmm_safe_phase(w: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Unit phase aligned with w where |w| >= eps; else identity quaternion (1,0,0,0).
    """
    w = to_float64(w, name="w")
    assert_float64(w, name="w")
    mag = quat_abs(w)
    safe = mag >= float(eps)
    id_q = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=w.dtype, device=w.device)
    return torch.where(
        safe.unsqueeze(-1),
        w / mag.clamp_min(float(eps)).unsqueeze(-1),
        id_q.expand_as(w),
    )


def qadmm_z_update_from_w(
    w_z: torch.Tensor,
    b: torch.Tensor,
    rho: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Closed-form z-step: min_z 1/2 |||z|-b||^2 + (rho/2)||z - w_z||^2 per row.
    """
    w_z = to_float64(w_z, name="w_z")
    b = to_float64(b, name="b")
    assert_float64(w_z, name="w_z")
    assert_float64(b, name="b")
    mag_w = quat_abs(w_z)
    num_fallback = int((mag_w < float(eps)).sum().item())
    phi_w = qadmm_safe_phase(w_z, eps)
    amp_new = (b + float(rho) * mag_w) / (1.0 + float(rho))
    z_new = amp_new.unsqueeze(-1) * phi_w
    aux = {
        "num_zero_phase_fallback": num_fallback,
        "mean_mag_w": float(torch.mean(mag_w).item()),
        "mean_amp_new": float(torch.mean(amp_new).item()),
    }
    return z_new, aux


def grad_qadmm(
    A_mat: torch.Tensor,
    y_k: torch.Tensor,
    q0: torch.Tensor,
    T: int,
    *,
    rho: float = 0.6,
    eps: float = 1e-12,
    return_history: bool = True,
    verbose: bool = True,
    progress_every: Optional[int] = None,
    x_true: Optional[torch.Tensor] = None,
    use_pfe: bool = False,
) -> tuple[list[torch.Tensor], dict[str, list[Any]]]:
    """
    QADMM iterations for min_q sum (|Aq|_l - b_l)^2 with b = sqrt(y_k), split z = Aq.

    history['loss'] matches QRKM: sum_l (|Aq|_l - b_l)^2 (no 1/2 factor).

    history['dual_residual'] (simplified, fixed in v1): rho * ||z^{k+1} - z^k||_F
    """
    A_mat = to_float64(A_mat, name="A_mat")
    y_k = to_float64(y_k, name="y_k")
    q0 = to_float64(q0, name="q0")
    assert_float64(A_mat, name="A_mat")
    assert_float64(y_k, name="y_k")
    assert_float64(q0, name="q0")
    if y_k.dim() == 2 and y_k.shape[1] == 1:
        y_k = y_k.squeeze(1)
    if A_mat.ndim != 3 or A_mat.shape[-1] != 4:
        raise ValueError(f"Expected A_mat (n,d,4), got {tuple(A_mat.shape)}")
    n, d, _ = A_mat.shape
    if y_k.ndim != 1 or y_k.shape[0] != n:
        raise ValueError(f"Expected y_k (n,), n={n}, got {tuple(y_k.shape)}")
    if q0.shape != (d, 4):
        raise ValueError(f"Expected q0 (d,4) with d={d}, got {tuple(q0.shape)}")
    if x_true is not None:
        x_true = to_float64(x_true, name="x_true")
        assert_float64(x_true, name="x_true")
        if x_true.shape != (d, 4):
            raise ValueError(f"Expected x_true (d,4), got {tuple(x_true.shape)}")
    if progress_every is None:
        progress_every = max(1, int(T) // 10)
    else:
        progress_every = max(1, int(progress_every))

    b = torch.sqrt(torch.clamp(y_k, min=0.0))
    z = quat_matvec_conj_left(A_mat, q0)
    lam = torch.zeros_like(z)
    q = q0.clone()
    # A is fixed over ADMM; building Ar once avoids O(n d) Python nested loops per q-step.
    Ar = quat_matrix_conj_left_real_block(A_mat)

    q_path: list[torch.Tensor] = [q.clone()]
    history: dict[str, list[Any]] = {
        "loss": [],
        "dist_T": [],
        "dist_raw": [],
        "primal_residual": [],
        "dual_residual": [],
        "q_norm": [],
        "z_norm": [],
        "lambda_norm": [],
        "ls_residual_norm": [],
        "ls_solution_norm": [],
        "zero_phase_fallbacks": [],
    }

    for k in range(int(T)):
        z_old = z
        w_q = z + lam / float(rho)
        q_new, ls_info = quat_least_squares_conj_left(A_mat, w_q, Ar=Ar)
        if use_pfe:
            q_new = phase_factor_estimate(q_new)
        u_new = quat_matvec_conj_left(A_mat, q_new)
        w_z = u_new - lam / float(rho)
        z_new, z_aux = qadmm_z_update_from_w(w_z, b, rho, eps=float(eps))
        lam_new = lam + float(rho) * (z_new - u_new)

        diff_zu = z_new - u_new
        primal = float(torch.sqrt(torch.sum(diff_zu * diff_zu)).item())
        diff_z = z_new - z_old
        dual = float(rho) * float(torch.sqrt(torch.sum(diff_z * diff_z)).item())

        loss_val = float(quat_total_loss(A_mat, q_new, b).item())
        q_norm = float(torch.sqrt(torch.sum(q_new * q_new)).item())
        z_norm = float(torch.sqrt(torch.sum(z_new * z_new)).item())
        lam_norm = float(torch.sqrt(torch.sum(lam_new * lam_new)).item())
        ls_rn = ls_info["residual_norm"]
        ls_res = float(ls_rn.item()) if isinstance(ls_rn, torch.Tensor) else float(ls_rn)
        ls_sn = ls_info["solution_norm"]
        ls_sol = float(ls_sn.item()) if isinstance(ls_sn, torch.Tensor) else float(ls_sn)

        q = q_new
        z = z_new
        lam = lam_new
        q_path.append(q.clone())

        if return_history:
            history["loss"].append(loss_val)
            history["primal_residual"].append(primal)
            history["dual_residual"].append(dual)
            history["q_norm"].append(q_norm)
            history["z_norm"].append(z_norm)
            history["lambda_norm"].append(lam_norm)
            history["ls_residual_norm"].append(ls_res)
            history["ls_solution_norm"].append(ls_sol)
            history["zero_phase_fallbacks"].append(int(z_aux["num_zero_phase_fallback"]))
            if x_true is not None:
                history["dist_T"].append(float(quat_dist_right_phase(q, x_true).item()))
                history["dist_raw"].append(float(quat_raw_distance(q, x_true).item()))

        if verbose and (((k + 1) % progress_every == 0) or (k + 1 == int(T))):
            msg = f"[alg_qadmm] epoch {k + 1}/{int(T)}"
            if x_true is not None and return_history and history["dist_T"]:
                d_t = float(history["dist_T"][-1])
                ln_d_t = math.log(max(d_t, 1e-300))
                msg += f" dist_T={d_t:.6e} ln_dist_T={ln_d_t:.6f}"
            print(msg)

    if not return_history:
        history = {key: [] for key in history}
    else:
        assert len(history["loss"]) == int(T)

    assert len(q_path) == int(T) + 1

    return q_path, history
