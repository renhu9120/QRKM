"""Right-phase distance and alignment metrics."""

from __future__ import annotations

import torch

from core.quat_ops import quat_abs, quat_conj, quat_mul
from utils.common import assert_float64, to_float64


def quat_right_phase_align(x_hat: torch.Tensor, x_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x_hat = to_float64(x_hat, name="x_hat")
    x_true = to_float64(x_true, name="x_true")
    assert_float64(x_hat, name="x_hat")
    assert_float64(x_true, name="x_true")
    if x_hat.shape != x_true.shape or x_hat.ndim != 2 or x_hat.shape[-1] != 4:
        raise ValueError(f"Expected (d,4) matching shapes; got {x_hat.shape}, {x_true.shape}")
    conj_true = quat_conj(x_true)
    prod = quat_mul(conj_true, x_hat)
    s = torch.sum(prod, dim=0)
    mag = quat_abs(s.unsqueeze(0)).squeeze(0)
    if float(mag.item()) < float(eps):
        return torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float64, device=s.device)
    # Strict normalization in the nonzero regime: avoid artificial 1e-12 floor.
    return s / mag


def quat_dist_right_phase(x_hat: torch.Tensor, x_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q_opt = quat_right_phase_align(x_hat, x_true, eps=float(eps))
    aligned = quat_mul(x_true, q_opt.unsqueeze(0))
    diff = x_hat - aligned
    return torch.sqrt(torch.sum(diff * diff))


def quat_raw_distance(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    x_hat = to_float64(x_hat, name="x_hat")
    x_true = to_float64(x_true, name="x_true")
    assert_float64(x_hat, name="x_hat")
    assert_float64(x_true, name="x_true")
    diff = x_hat - x_true
    return torch.sqrt(torch.sum(diff * diff))


def success_under_right_phase(x_hat: torch.Tensor, x_true: torch.Tensor, tol: float = 1e-5) -> bool:
    d = quat_dist_right_phase(x_hat, x_true)
    return bool(float(d.item()) <= float(tol))
