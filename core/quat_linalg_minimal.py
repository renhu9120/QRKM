"""Minimal quaternion linear algebra for QADMM via real embedding."""

from __future__ import annotations

import torch

from core.quat_ops import quat_mv_conj_left
from core.quat_real_embed import (
    quat_matrix_conj_left_real_block,
    quat_vec_from_real,
    quat_vec_to_real,
)
from utils.common import assert_float64, to_float64


def quat_matvec_conj_left(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Semantic alias of quat_mv_conj_left for QADMM readability.

    This wrapper keeps exactly the same conjugated-left map semantics.
    """
    return quat_mv_conj_left(A, x)


def quat_matH_vec_conj_left(A: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Real-adjoint action g = A^* w under current conjugated-left embedding.

    This implementation is defined by Ar^T wr in real embedding.
    """
    A = to_float64(A, name="A")
    w = to_float64(w, name="w")
    assert_float64(A, name="A")
    assert_float64(w, name="w")
    if A.ndim != 3 or A.shape[-1] != 4:
        raise ValueError(f"Expected A shape (n,d,4), got {A.shape}")
    if w.ndim != 2 or w.shape[-1] != 4 or w.shape[0] != A.shape[0]:
        raise ValueError(f"Expected w shape ({A.shape[0]},4), got {w.shape}")
    d = A.shape[1]
    Ar = quat_matrix_conj_left_real_block(A)
    wr = quat_vec_to_real(w)
    gr = Ar.transpose(0, 1) @ wr
    return quat_vec_from_real(gr, d=d)


def quat_least_squares_conj_left(
    A: torch.Tensor,
    w: torch.Tensor,
    rcond: float | None = None,
    *,
    Ar: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Solve min_q ||A q - w||_2^2 under current conjugated-left model.

    For n < d, this returns the minimum-norm least-squares solution from
    torch.linalg.lstsq on the real embedding.

    Parameters
    ----------
    Ar
        If provided, must be ``quat_matrix_conj_left_real_block(A)`` for the same
        ``A``. Reusing ``Ar`` avoids rebuilding the real block matrix on every call
        (large win inside ADMM loops where ``A`` is fixed).
    """
    A = to_float64(A, name="A")
    w = to_float64(w, name="w")
    assert_float64(A, name="A")
    assert_float64(w, name="w")
    if A.ndim != 3 or A.shape[-1] != 4:
        raise ValueError(f"Expected A shape (n,d,4), got {A.shape}")
    if w.ndim != 2 or w.shape[-1] != 4 or w.shape[0] != A.shape[0]:
        raise ValueError(f"Expected w shape ({A.shape[0]},4), got {w.shape}")
    n, d, _ = A.shape
    ar_reused = Ar is not None
    if ar_reused:
        Ar = to_float64(Ar, name="Ar")
        assert_float64(Ar, name="Ar")
        exp_rows, exp_cols = 4 * n, 4 * d
        if Ar.shape != (exp_rows, exp_cols):
            raise ValueError(f"Ar must have shape ({exp_rows}, {exp_cols}), got {tuple(Ar.shape)}")
    else:
        Ar = quat_matrix_conj_left_real_block(A)
    wr = quat_vec_to_real(w)
    sol = torch.linalg.lstsq(Ar, wr.unsqueeze(-1), rcond=rcond)
    xr = sol.solution.squeeze(-1)
    q = quat_vec_from_real(xr, d=d)
    # Real residual matches quaternion residual in this embedding.
    residual_norm_real = torch.sqrt(torch.sum((Ar @ xr - wr) * (Ar @ xr - wr)))
    if ar_reused:
        residual_norm = residual_norm_real
    else:
        fit = quat_matvec_conj_left(A, q)
        residual_norm = torch.sqrt(torch.sum((fit - w) * (fit - w)))
    solution_norm = torch.sqrt(torch.sum(xr * xr))

    rank_val = None
    rank_raw = None
    if hasattr(sol, "rank"):
        rank_t = sol.rank
        rank_raw = rank_t
        if isinstance(rank_t, torch.Tensor):
            if rank_t.numel() > 0:
                rank_val = int(rank_t.item())
        elif rank_t is not None:
            rank_val = int(rank_t)

    singular_values = None
    if hasattr(sol, "singular_values") and sol.singular_values is not None:
        singular_values = sol.singular_values

    num_singular_values = None
    min_singular_value = None
    max_singular_value = None
    condition_est = None
    if singular_values is not None and singular_values.numel() > 0:
        num_singular_values = int(singular_values.numel())
        min_singular_value = torch.min(singular_values)
        max_singular_value = torch.max(singular_values)
        if float(min_singular_value.item()) > 0.0:
            condition_est = max_singular_value / min_singular_value

    info = {
        "residual_norm": residual_norm,
        "residual_norm_real": residual_norm_real,
        "solution_norm": solution_norm,
        "rank": rank_val,
        "rank_raw": rank_raw,
        "singular_values": singular_values,
        "num_singular_values": num_singular_values,
        "min_singular_value": min_singular_value,
        "max_singular_value": max_singular_value,
        "condition_est": condition_est,
        "solve_method": "real_lstsq",
        "shape_A": (n, d, 4),
        "shape_Ar": tuple(Ar.shape),
        "shape_wr": tuple(wr.shape),
        "rcond": rcond,
    }
    return q, info


def quat_normal_eq_conj_left(A: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (G, rhs) with G = Ar^T Ar and rhs = Ar^T wr for debugging."""
    A = to_float64(A, name="A")
    w = to_float64(w, name="w")
    assert_float64(A, name="A")
    assert_float64(w, name="w")
    if A.ndim != 3 or A.shape[-1] != 4:
        raise ValueError(f"Expected A shape (n,d,4), got {A.shape}")
    if w.ndim != 2 or w.shape[-1] != 4 or w.shape[0] != A.shape[0]:
        raise ValueError(f"Expected w shape ({A.shape[0]},4), got {w.shape}")
    Ar = quat_matrix_conj_left_real_block(A)
    wr = quat_vec_to_real(w)
    G = Ar.transpose(0, 1) @ Ar
    rhs = Ar.transpose(0, 1) @ wr
    return G, rhs
