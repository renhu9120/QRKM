"""Verify PR loss identities (selected-row, gradient-step, energy)."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from core.quat_ops import quat_abs
from algorithms.gradient.grad_qrkm import (
    quat_full_gradient,
    quat_row_response,
    quat_selected_row_gradient,
    quat_total_loss,
    row_energy_beta,
    single_row_u_and_T,
)


def _assert_close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-10, rtol: float = 1e-10) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"Mismatch max={float(torch.max(torch.abs(a-b))):.6e}")


def _assert_close_scalar(a: float, b: float, atol: float = 1e-10, rtol: float = 1e-10) -> None:
    denom = max(abs(b), 1.0)
    if abs(a - b) > atol + rtol * denom:
        raise AssertionError(f"Scalar mismatch |a-b|={abs(a-b):.6e}")


def _sample_nonsingular_x(A: torch.Tensor, b: torch.Tensor, *, min_mag: float = 1e-3, max_trials: int = 200) -> torch.Tensor:
    n, d, _ = A.shape
    g = torch.Generator(device=A.device)
    for t in range(max_trials):
        g.manual_seed(t)
        x = torch.randn(d, 4, dtype=torch.float64, device=A.device, generator=g)
        u = quat_row_response(A, x)
        mags = quat_abs(u)
        beta = row_energy_beta(A)
        if bool(torch.all(mags >= min_mag)) and bool(torch.all(beta >= 1e-10)):
            return x
    raise RuntimeError("Failed to sample nonsingular x.")


def verify_selected_row_and_gradient_step() -> None:
    dev = torch.device("cpu")
    g = torch.Generator(device=dev)
    g.manual_seed(11)
    n, d = 12, 9
    A = torch.randn(n, d, 4, dtype=torch.float64, device=dev, generator=g) * 0.3
    x = _sample_nonsingular_x(A, torch.ones(n, dtype=torch.float64, device=dev))
    b = torch.rand(n, dtype=torch.float64, device=dev, generator=g) * 0.6 + 0.5
    for l in range(n):
        a_l = A[l]
        beta_l = float(torch.sum(a_l * a_l).item())
        b_l = float(b[l].item())
        u_l, Tlx, u_sharp, did_skip = single_row_u_and_T(a_l, x, b_l, beta_l, eps=1e-12, eps_beta=1e-12)
        if did_skip:
            continue
        u_after = quat_row_response(a_l.unsqueeze(0), Tlx).squeeze(0)
        _assert_close(u_after, u_sharp, atol=1e-10, rtol=1e-10)
        grad_l, _aux = quat_selected_row_gradient(a_l, x, b_l, eps=1e-12)
        T_grad = x - (1.0 / (2.0 * beta_l)) * grad_l
        _assert_close(Tlx, T_grad, atol=1e-10, rtol=1e-10)
        amp = float(quat_abs(u_after.unsqueeze(0)).item())
        _assert_close_scalar(amp, b_l, atol=1e-10, rtol=1e-10)


def verify_energy_identity() -> None:
    dev = torch.device("cpu")
    g = torch.Generator(device=dev)
    g.manual_seed(21)
    n, d = 14, 10
    A = torch.randn(n, d, 4, dtype=torch.float64, device=dev, generator=g) * 0.25
    x = _sample_nonsingular_x(A, torch.ones(n, dtype=torch.float64, device=dev), min_mag=1e-2)
    b = torch.rand(n, dtype=torch.float64, device=dev, generator=g) * 0.7 + 0.4
    beta = row_energy_beta(A)
    F = quat_total_loss(A, x, b)
    acc = torch.zeros((), dtype=torch.float64, device=dev)
    for l in range(n):
        a_l = A[l]
        grad_l, _aux = quat_selected_row_gradient(a_l, x, float(b[l].item()), eps=1e-12)
        gn = torch.sum(grad_l * grad_l)
        acc = acc + (0.25 / float(beta[l].item())) * gn
    _assert_close(acc.unsqueeze(0), F.unsqueeze(0), atol=1e-10, rtol=1e-10)


def verify_full_gradient_sum() -> None:
    dev = torch.device("cpu")
    g = torch.Generator(device=dev)
    g.manual_seed(31)
    n, d = 11, 8
    A = torch.randn(n, d, 4, dtype=torch.float64, device=dev, generator=g) * 0.2
    x = _sample_nonsingular_x(A, torch.ones(n, dtype=torch.float64, device=dev))
    b = torch.rand(n, dtype=torch.float64, device=dev, generator=g) * 0.5 + 0.5
    g_sum = torch.zeros_like(x)
    for l in range(n):
        grad_l, _ = quat_selected_row_gradient(A[l], x, float(b[l].item()), eps=1e-12)
        g_sum = g_sum + grad_l
    g_full = quat_full_gradient(A, x, b, eps=1e-12)
    _assert_close(g_sum, g_full, atol=1e-10, rtol=1e-10)


def run_all() -> None:
    verify_selected_row_and_gradient_step()
    verify_energy_identity()
    verify_full_gradient_sum()
    print("[verify_quat_pr_loss] all checks passed.")


if __name__ == "__main__":
    run_all()
