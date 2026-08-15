"""Synthetic quaternion Gaussian data (Chen–Ng style: Var per component = 1/4)."""

from __future__ import annotations

import torch

from core.quat_ops import quat_abs_sq, quat_mv_conj_left
from utils.common import assert_float64, pick_device, to_float64


def sample_quaternion_gaussian(
    shape: tuple[int, ...],
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Proper quaternion Gaussian entries.

    ``generator`` keeps the data stream independent of the global RNG, so that a
    solver consuming random numbers cannot shift the measurement matrices seen by
    a later batch (required for paired comparisons).
    """
    if dtype != torch.float64:
        raise TypeError("sample_quaternion_gaussian requires torch.float64")
    dev = pick_device(device)
    out = torch.randn(*shape, 4, device=dev, dtype=torch.float64, generator=generator) * 0.5
    return out


def make_quaternion_gaussian_matrix(
    n: int,
    d: int,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if dtype != torch.float64:
        raise TypeError("make_quaternion_gaussian_matrix requires torch.float64")
    return sample_quaternion_gaussian((n, d), device=device, dtype=dtype)


def make_random_quaternion_signal(
    d: int,
    normalize: bool = True,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if dtype != torch.float64:
        raise TypeError("make_random_quaternion_signal requires torch.float64")
    x = sample_quaternion_gaussian((d,), device=device, dtype=dtype)
    if normalize:
        nrm = torch.sqrt(torch.sum(x * x)) + 1e-12
        x = x / nrm
    return x


def make_noiseless_measurements(A: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    A = to_float64(A, name="A")
    x_true = to_float64(x_true, name="x_true")
    assert_float64(A, name="A")
    assert_float64(x_true, name="x_true")
    u = quat_mv_conj_left(A, x_true)
    return quat_abs_sq(u)
