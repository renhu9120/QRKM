"""Quaternion algebra and conjugated-left linear maps (explicit, float64)."""

from __future__ import annotations

import torch

from utils.common import assert_float64, to_float64


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    q = to_float64(q, name="q")
    assert_float64(q, name="q")
    out = q.clone()
    out[..., 1:4] *= -1.0
    return out


def quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = to_float64(p, name="p")
    q = to_float64(q, name="q")
    assert_float64(p, name="p")
    assert_float64(q, name="q")
    a, b, c, d = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    e, f, g, h = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    o0 = a * e - b * f - c * g - d * h
    o1 = a * f + b * e + c * h - d * g
    o2 = a * g - b * h + c * e + d * f
    o3 = a * h + b * g - c * f + d * e
    return torch.stack((o0, o1, o2, o3), dim=-1)


def quat_abs_sq(q: torch.Tensor) -> torch.Tensor:
    q = to_float64(q, name="q")
    assert_float64(q, name="q")
    return torch.sum(q * q, dim=-1)


def quat_abs(q: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(quat_abs_sq(q))


def quat_inv(q: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    q = to_float64(q, name="q")
    assert_float64(q, name="q")
    denom = quat_abs_sq(q) + float(eps)
    return quat_conj(q) / denom.unsqueeze(-1)


def quat_normalize_safe(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = to_float64(q, name="q")
    assert_float64(q, name="q")
    n = quat_abs(q).unsqueeze(-1) + float(eps)
    return q / n


def quat_normalize_exact(q: torch.Tensor) -> torch.Tensor:
    q = to_float64(q, name="q")
    assert_float64(q, name="q")
    n = quat_abs(q).unsqueeze(-1)
    return q / n


def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Backward-compatible alias: keep historical safe behavior.
    return quat_normalize_safe(q, eps=eps)


def quat_inner_conj_left(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    a = to_float64(a, name="a")
    x = to_float64(x, name="x")
    assert_float64(a, name="a")
    assert_float64(x, name="x")
    if a.shape[-1] != 4 or x.shape[-1] != 4:
        raise ValueError("Last dimension must be 4 for quaternion tensors.")
    a_b, x_b = torch.broadcast_tensors(a, x)
    conj_a = quat_conj(a_b)
    prod = quat_mul(conj_a, x_b)
    return torch.sum(prod, dim=-2)


def quat_mv_conj_left(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    A = to_float64(A, name="A")
    x = to_float64(x, name="x")
    assert_float64(A, name="A")
    assert_float64(x, name="x")
    if A.ndim != 3 or x.ndim != 2:
        raise ValueError(f"Expected A (n,d,4) and x (d,4); got {A.shape}, {x.shape}")
    n, d, four = A.shape
    if four != 4 or x.shape != (d, 4):
        raise ValueError(f"Shape mismatch: A {A.shape}, x {x.shape}")
    conj_A = quat_conj(A)
    x_row = x.unsqueeze(0)
    prod = quat_mul(conj_A, x_row)
    return torch.sum(prod, dim=1)


def quat_bmv_conj_left(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    A = to_float64(A, name="A")
    X = to_float64(X, name="X")
    assert_float64(A, name="A")
    assert_float64(X, name="X")
    if A.ndim != 3 or X.ndim != 3:
        raise ValueError(f"Expected A (n,d,4) and X (B,d,4); got {A.shape}, {X.shape}")
    n, d, four = A.shape
    B, d2, four2 = X.shape
    if four != 4 or four2 != 4 or d != d2:
        raise ValueError(f"Shape mismatch: A {A.shape}, X {X.shape}")
    conj_A = quat_conj(A).unsqueeze(0)
    X_exp = X.unsqueeze(1)
    prod = quat_mul(conj_A, X_exp)
    return torch.sum(prod, dim=2)
