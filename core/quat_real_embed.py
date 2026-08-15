"""Real embedding utilities for conjugated-left quaternion linear maps."""

from __future__ import annotations

import torch

from core.quat_ops import quat_conj
from utils.common import assert_float64, to_float64


def quat_left_mul_matrix(q: torch.Tensor) -> torch.Tensor:
    """Return real 4x4 matrix L(q) such that vec(q*x) = L(q) @ vec(x)."""
    q = to_float64(q, name="q")
    assert_float64(q, name="q")
    if q.ndim != 1 or q.shape[0] != 4:
        raise ValueError(f"Expected q shape (4,), got {q.shape}")
    a, b, c, d = q[0], q[1], q[2], q[3]
    return torch.stack(
        (
            torch.stack((a, -b, -c, -d)),
            torch.stack((b, a, -d, c)),
            torch.stack((c, d, a, -b)),
            torch.stack((d, -c, b, a)),
        ),
        dim=0,
    )


def quat_conj_left_mul_matrix(q: torch.Tensor) -> torch.Tensor:
    """Return L(conj(q)) for current conjugated-left forward model."""
    q = to_float64(q, name="q")
    assert_float64(q, name="q")
    if q.ndim != 1 or q.shape[0] != 4:
        raise ValueError(f"Expected q shape (4,), got {q.shape}")
    return quat_left_mul_matrix(quat_conj(q))


def quat_vec_to_real(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten quaternion vector to real vector.

    This real embedding matches the current conjugated-left forward model.
    """
    x = to_float64(x, name="x")
    assert_float64(x, name="x")
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError(f"Expected x shape (d,4), got {x.shape}")
    return x.contiguous().reshape(-1)


def quat_vec_from_real(xr: torch.Tensor, d: int) -> torch.Tensor:
    """
    Recover quaternion vector from real embedding.

    This real embedding matches the current conjugated-left forward model.
    """
    xr = to_float64(xr, name="xr")
    assert_float64(xr, name="xr")
    if xr.ndim != 1:
        raise ValueError(f"Expected xr shape (4d,), got {xr.shape}")
    if int(d) <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if xr.shape[0] != 4 * int(d):
        raise ValueError(f"Expected xr length {4*int(d)}, got {xr.shape[0]}")
    return xr.reshape(int(d), 4)


def quat_matrix_conj_left_real_block(A: torch.Tensor) -> torch.Tensor:
    """
    Build real block matrix Ar for quaternion map x -> quat_mv_conj_left(A, x).

    This real embedding matches the current conjugated-left forward model.
    """
    A = to_float64(A, name="A")
    assert_float64(A, name="A")
    if A.ndim != 3 or A.shape[-1] != 4:
        raise ValueError(f"Expected A shape (n,d,4), got {A.shape}")
    n, d, _ = A.shape
    Ar = torch.zeros(4 * n, 4 * d, dtype=torch.float64, device=A.device)
    for row in range(n):
        for col in range(d):
            block = quat_conj_left_mul_matrix(A[row, col])
            Ar[4 * row : 4 * (row + 1), 4 * col : 4 * (col + 1)] = block
    return Ar
