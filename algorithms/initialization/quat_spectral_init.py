"""Quaternion spectral initialization for QRKM."""

from __future__ import annotations

import math

import torch

from core.quat_ops import quat_conj, quat_mul, quat_mv_conj_left
from algorithms.gradient.grad_qrkm import row_energy_beta
from utils.common import assert_float64, pick_device, to_float64


def quat_spectral_init_shared_batch(
    A: torch.Tensor,
    y: torch.Tensor,
    *,
    power_iters: int = 50,
    eps: float = 1e-12,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Intensity-based quaternion spectral initializer, Eq. (2), over a stack of instances.

    ``A`` is ``(B, n, d, 4)`` and ``y`` is ``(B, n)`` with ``y_k = |a_k^* x|^2``;
    the returned estimate is ``(B, d, 4)``.  The instance axis ``B`` is never
    reduced, so instance ``i`` of the output depends only on instance ``i`` of the
    input.  This is the single point from which *every* compared method is started
    (Sec. IV of the letter): it is computed once per instance and passed to each
    solver via ``z0=``, so the comparison isolates the iterative update.

    ``generator`` seeds the random unit vector that starts the power iteration.
    Keeping it on a stream of its own is what makes the initialization independent
    of anything a solver later draws.
    """
    if A.ndim != 4 or A.shape[-1] != 4:
        raise ValueError(f"A must be (B,n,d,4), got {tuple(A.shape)}")
    if y.ndim != 2 or y.shape[:2] != A.shape[:2]:
        raise ValueError(f"y must be (B,n)={tuple(A.shape[:2])}, got {tuple(y.shape)}")
    if A.dtype != torch.float64 or y.dtype != torch.float64:
        raise TypeError("quat_spectral_init_shared_batch requires float64 tensors")

    B, n, d, _ = A.shape
    beta = torch.sum(A * A, dim=(2, 3))
    x = torch.randn((B, d, 4), dtype=torch.float64, device=A.device, generator=generator)
    x = x / (torch.sqrt(torch.sum(x * x, dim=(1, 2), keepdim=True)) + float(eps))
    for _ in range(int(power_iters)):
        u = torch.sum(quat_mul(quat_conj(A), x.unsqueeze(1)), dim=2)
        w = (y / (beta + float(eps))).unsqueeze(-1) * u
        x_new = torch.sum(quat_mul(A, w.unsqueeze(2)), dim=1)
        x = x_new / (torch.sqrt(torch.sum(x_new * x_new, dim=(1, 2), keepdim=True)) + float(eps))
    gamma = torch.sqrt(torch.clamp(torch.mean(y, dim=1, keepdim=True), min=0.0)).unsqueeze(-1)
    return gamma * x


def quat_spectral_init_shared(
    A: torch.Tensor,
    y: torch.Tensor,
    *,
    power_iters: int = 50,
    eps: float = 1e-12,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Single-instance form of :func:`quat_spectral_init_shared_batch`.

    ``A`` is ``(n, d, 4)``, ``y`` is ``(n,)``, and the result is ``(d, 4)``.
    """
    return quat_spectral_init_shared_batch(
        A.unsqueeze(0), y.unsqueeze(0),
        power_iters=power_iters, eps=eps, generator=generator,
    )[0]


def quat_spectral_init(
    A: torch.Tensor,
    y: torch.Tensor,
    num_power_iters: int = 50,
    init_mode: str = "random",
    seed: int | None = None,
    eps: float = 1e-12,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> torch.Tensor:
    if dtype != torch.float64:
        raise TypeError("quat_spectral_init requires torch.float64")
    if init_mode != "random":
        raise NotImplementedError("Only init_mode='random' is supported in this phase.")
    dev = pick_device(device)
    A = to_float64(A, name="A").to(dev)
    y = to_float64(y, name="y").to(dev)
    assert_float64(A, name="A")
    assert_float64(y, name="y")
    if A.ndim != 3 or y.ndim != 1 or A.shape[0] != y.shape[0]:
        raise ValueError(f"Incompatible A {A.shape} and y {y.shape}")
    n, d, _ = A.shape
    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(int(seed))
    beta = row_energy_beta(A)
    x = torch.randn(d, 4, dtype=torch.float64, device=dev, generator=gen)
    nrm = torch.sqrt(torch.sum(x * x)) + float(eps)
    x = x / nrm
    for _ in range(int(num_power_iters)):
        u = quat_mv_conj_left(A, x)
        w = (y / (beta + float(eps))).unsqueeze(-1) * u
        prod = quat_mul(A, w.unsqueeze(1))
        x_new = torch.sum(prod, dim=0)
        nrm_new = torch.sqrt(torch.sum(x_new * x_new)) + float(eps)
        x = x_new / nrm_new
    gamma = math.sqrt(float(torch.mean(y).item()))
    x = float(gamma) * x
    if verbose:
        print(f"[quat_spectral_init] d={d}, n={n}, num_power_iters={num_power_iters}")
        print(
            f"[quat_spectral_init] beta: mean={float(torch.mean(beta)):.6e} "
            f"min={float(torch.min(beta)):.6e} max={float(torch.max(beta)):.6e}"
        )
        print(f"[quat_spectral_init] ||x0||={float(torch.sqrt(torch.sum(x*x))):.6e} gamma={gamma:.6e}")
    return x
