"""Verify quaternion core operations."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from core.quat_ops import (
    quat_abs,
    quat_abs_sq,
    quat_conj,
    quat_inner_conj_left,
    quat_mul,
)


def _assert_close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-12, rtol: float = 1e-12) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"Mismatch max={float(torch.max(torch.abs(a-b))):.6e}")


def verify_conj_involution() -> None:
    q = torch.randn(5, 4, dtype=torch.float64)
    _assert_close(quat_conj(quat_conj(q)), q)


def verify_mul_norm() -> None:
    g = torch.Generator()
    g.manual_seed(0)
    p = torch.randn(8, 4, dtype=torch.float64, generator=g)
    q = torch.randn(8, 4, dtype=torch.float64, generator=g)
    pq = quat_mul(p, q)
    lhs = torch.sqrt(quat_abs_sq(pq))
    rhs = torch.sqrt(quat_abs_sq(p)) * torch.sqrt(quat_abs_sq(q))
    _assert_close(lhs, rhs, atol=1e-10, rtol=1e-10)


def verify_conj_square() -> None:
    g = torch.Generator()
    g.manual_seed(1)
    q = torch.randn(6, 4, dtype=torch.float64, generator=g)
    prod = quat_mul(q, quat_conj(q))
    mag2 = quat_abs_sq(q).unsqueeze(-1).expand_as(q)
    expected = torch.zeros_like(q)
    expected[..., 0] = mag2[..., 0]
    _assert_close(prod, expected, atol=1e-12, rtol=1e-12)


def verify_inner_conj_left_manual() -> None:
    g = torch.Generator()
    g.manual_seed(2)
    d = 7
    a = torch.randn(3, d, 4, dtype=torch.float64, generator=g)
    x = torch.randn(3, d, 4, dtype=torch.float64, generator=g)
    out = quat_inner_conj_left(a, x)
    manual = torch.zeros(3, 4, dtype=torch.float64)
    for bi in range(3):
        acc = torch.zeros(4, dtype=torch.float64)
        for j in range(d):
            acc = acc + quat_mul(quat_conj(a[bi, j]), x[bi, j])
        manual[bi] = acc
    _assert_close(out, manual, atol=1e-12, rtol=1e-12)


def run_all() -> None:
    verify_conj_involution()
    verify_mul_norm()
    verify_conj_square()
    verify_inner_conj_left_manual()
    print("[verify_quat_ops] all checks passed.")


if __name__ == "__main__":
    run_all()
