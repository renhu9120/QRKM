"""Verify real-embedding minimal linalg for conjugated-left model."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from core.quat_linalg_minimal import (
    quat_least_squares_conj_left,
    quat_matH_vec_conj_left,
    quat_matvec_conj_left,
)
from core.quat_ops import quat_mv_conj_left
from core.quat_real_embed import (
    quat_matrix_conj_left_real_block,
    quat_vec_from_real,
    quat_vec_to_real,
)
from core.quat_sampling import make_noiseless_measurements, make_quaternion_gaussian_matrix, make_random_quaternion_signal
from utils.common import pick_device


def _assert_close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-11, rtol: float = 1e-11) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        err = float(torch.max(torch.abs(a - b)).item())
        raise AssertionError(f"Mismatch max={err:.6e}, atol={atol:.1e}, rtol={rtol:.1e}")


def _real_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.dot(quat_vec_to_real(x), quat_vec_to_real(y))


def verify_forward_embedding_consistency(dev: torch.device) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(11)
    n, d = 9, 5
    A = torch.randn(n, d, 4, dtype=torch.float64, device=dev, generator=g)
    x = torch.randn(d, 4, dtype=torch.float64, device=dev, generator=g)
    u_q = quat_mv_conj_left(A, x)
    Ar = quat_matrix_conj_left_real_block(A)
    ur = Ar @ quat_vec_to_real(x)
    u_r = quat_vec_from_real(ur, d=n)
    _assert_close(u_q, u_r, atol=1e-12, rtol=1e-12)


def verify_adjoint_consistency(dev: torch.device) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(12)
    n, d = 8, 6
    A = torch.randn(n, d, 4, dtype=torch.float64, device=dev, generator=g)
    x = torch.randn(d, 4, dtype=torch.float64, device=dev, generator=g)
    w = torch.randn(n, 4, dtype=torch.float64, device=dev, generator=g)
    Ax = quat_matvec_conj_left(A, x)
    lhs = _real_dot(Ax, w)
    g_adj = quat_matH_vec_conj_left(A, w)
    rhs = _real_dot(x, g_adj)
    if abs(float((lhs - rhs).item())) > 1e-10:
        raise AssertionError(f"Adjoint mismatch |lhs-rhs|={abs(float((lhs-rhs).item())):.6e}")


def verify_least_squares_self_consistency(dev: torch.device) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(13)
    n, d = 20, 7
    A = torch.randn(n, d, 4, dtype=torch.float64, device=dev, generator=g)
    x_true = torch.randn(d, 4, dtype=torch.float64, device=dev, generator=g)
    w = quat_mv_conj_left(A, x_true)
    x_rec, info = quat_least_squares_conj_left(A, w, rcond=None)
    residual = torch.sqrt(torch.sum((quat_mv_conj_left(A, x_rec) - w) ** 2))
    dist_raw = torch.sqrt(torch.sum((x_rec - x_true) ** 2))
    if float(residual.item()) > 1e-10:
        raise AssertionError(f"Residual too large: {float(residual.item()):.6e}")
    if float(dist_raw.item()) > 1e-9:
        raise AssertionError(f"Recovery raw error too large: {float(dist_raw.item()):.6e}")
    if info["solve_method"] != "real_lstsq":
        raise AssertionError("Unexpected solve_method in info.")


def verify_qrkm_measurement_compatibility(dev: torch.device) -> None:
    d, n = 10, 50
    A = make_quaternion_gaussian_matrix(n, d, device=dev, dtype=torch.float64)
    x_true = make_random_quaternion_signal(d, normalize=True, device=dev, dtype=torch.float64)
    y = make_noiseless_measurements(A, x_true)
    b = torch.sqrt(torch.clamp(y, min=0.0))
    w = quat_mv_conj_left(A, x_true)
    if not torch.allclose(torch.sqrt(torch.sum(w * w, dim=-1)), b, atol=1e-11, rtol=1e-11):
        raise AssertionError("Amplitude mismatch with QRKM measurement semantics.")
    x_rec, _ = quat_least_squares_conj_left(A, w)
    residual = torch.sqrt(torch.sum((quat_mv_conj_left(A, x_rec) - w) ** 2))
    if float(residual.item()) > 1e-10:
        raise AssertionError(f"Compatibility LS residual too large: {float(residual.item()):.6e}")


def verify_ill_conditioned_behavior(dev: torch.device) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(14)
    cases = [
        ("underdetermined", 6, 12),
        ("square", 8, 8),
        ("overdetermined", 24, 9),
        ("near_rank_deficient", 12, 10),
    ]
    for name, n, d in cases:
        A = torch.randn(n, d, 4, dtype=torch.float64, device=dev, generator=g)
        if name == "near_rank_deficient":
            A[:, 1] = A[:, 0] + 1e-10 * torch.randn(n, 4, dtype=torch.float64, device=dev, generator=g)
        if n > 0:
            A[0] = A[0] * 1e-8
        x_true = torch.randn(d, 4, dtype=torch.float64, device=dev, generator=g)
        w = quat_mv_conj_left(A, x_true)
        x_rec, info = quat_least_squares_conj_left(A, w, rcond=None)
        fit = quat_mv_conj_left(A, x_rec)
        residual = torch.sqrt(torch.sum((fit - w) ** 2))
        if not torch.isfinite(residual):
            raise AssertionError(f"{name}: non-finite residual")
        rank = info["rank"]
        rank_msg = "None" if rank is None else str(rank)
        print(
            f"[verify_quat_real_embed][{name}] n={n} d={d} "
            f"residual={float(residual.item()):.6e} rank={rank_msg}"
        )


def verify_minimum_norm_behavior_nullspace(dev: torch.device) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(15)
    n, d = 6, 12
    A = torch.randn(n, d, 4, dtype=torch.float64, device=dev, generator=g)
    x_true = torch.randn(d, 4, dtype=torch.float64, device=dev, generator=g)
    w = quat_mv_conj_left(A, x_true)
    q_ref, _ = quat_least_squares_conj_left(A, w, rcond=None)

    Ar = quat_matrix_conj_left_real_block(A)
    _, _, vh = torch.linalg.svd(Ar, full_matrices=True)
    vr = vh[-1]
    vr = vr / (torch.sqrt(torch.sum(vr * vr)) + 1e-15)
    v = quat_vec_from_real(vr, d=d)

    null_residual = torch.sqrt(torch.sum((quat_mv_conj_left(A, v)) ** 2))
    if float(null_residual.item()) > 1e-8:
        raise AssertionError(f"Nullspace residual too large: {float(null_residual.item()):.6e}")

    residual_ref = torch.sqrt(torch.sum((quat_mv_conj_left(A, q_ref) - w) ** 2))
    norm_ref = torch.sqrt(torch.sum(q_ref * q_ref))
    alpha_list = [1e-3, 1e-2, 1e-1, 1.0]

    print(f"[verify_min_norm_nullspace] n={n} d={d}")
    print(f"[verify_min_norm_nullspace] null_residual={float(null_residual.item()):.6e}")
    for alpha in alpha_list:
        q_alt = q_ref + float(alpha) * v
        residual_alt = torch.sqrt(torch.sum((quat_mv_conj_left(A, q_alt) - w) ** 2))
        norm_alt = torch.sqrt(torch.sum(q_alt * q_alt))
        norm_gap = norm_alt - norm_ref
        if float(residual_alt.item()) > 1e-8:
            raise AssertionError(f"alpha={alpha}: residual_alt too large {float(residual_alt.item()):.6e}")
        if float(norm_gap.item()) < -1e-10:
            raise AssertionError(f"alpha={alpha}: norm_alt < norm_ref by {float(norm_gap.item()):.6e}")
        print(
            f"[verify_min_norm_nullspace] alpha={alpha:.1e} "
            f"residual_ref={float(residual_ref.item()):.6e} "
            f"residual_alt={float(residual_alt.item()):.6e} "
            f"norm_ref={float(norm_ref.item()):.6e} "
            f"norm_alt={float(norm_alt.item()):.6e} "
            f"norm_gap={float(norm_gap.item()):.6e}"
        )

    norm_true = torch.sqrt(torch.sum(x_true * x_true))
    dist_true_ref = torch.sqrt(torch.sum((x_true - q_ref) ** 2))
    residual_true = torch.sqrt(torch.sum((quat_mv_conj_left(A, x_true) - w) ** 2))
    print(
        f"[verify_min_norm_optional] norm_true={float(norm_true.item()):.6e} "
        f"norm_ref={float(norm_ref.item()):.6e} "
        f"dist_true_ref={float(dist_true_ref.item()):.6e} "
        f"residual_true={float(residual_true.item()):.6e} "
        f"residual_ref={float(residual_ref.item()):.6e}"
    )


def verify_least_squares_debug_info(dev: torch.device) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(16)
    cases = [
        ("underdetermined", 6, 12),
        ("square", 8, 8),
        ("overdetermined", 24, 9),
        ("near_rank_deficient", 12, 10),
    ]
    for name, n, d in cases:
        A = torch.randn(n, d, 4, dtype=torch.float64, device=dev, generator=g)
        if name == "near_rank_deficient":
            A[:, 1] = A[:, 0] + 1e-10 * torch.randn(n, 4, dtype=torch.float64, device=dev, generator=g)
        x_true = torch.randn(d, 4, dtype=torch.float64, device=dev, generator=g)
        w = quat_mv_conj_left(A, x_true)
        _, info = quat_least_squares_conj_left(A, w, rcond=None)
        print(f"[verify_ls_debug][{name}]")
        print(f"  shape_A={info['shape_A']} shape_Ar={info['shape_Ar']} shape_wr={info['shape_wr']}")
        print(f"  residual_norm={float(info['residual_norm'].item()):.6e}")
        print(f"  residual_norm_real={float(info['residual_norm_real'].item()):.6e}")
        print(f"  solution_norm={float(info['solution_norm'].item()):.6e}")
        print(f"  rank={info['rank']}")
        print(f"  rank_raw={info['rank_raw']}")
        print(f"  num_singular_values={info['num_singular_values']}")
        if info["min_singular_value"] is None:
            print("  min_singular_value=None")
        else:
            print(f"  min_singular_value={float(info['min_singular_value'].item()):.6e}")
        if info["max_singular_value"] is None:
            print("  max_singular_value=None")
        else:
            print(f"  max_singular_value={float(info['max_singular_value'].item()):.6e}")
        if info["condition_est"] is None:
            print("  condition_est=None")
        else:
            print(f"  condition_est={float(info['condition_est'].item()):.6e}")

        rr = float(info["residual_norm_real"].item())
        rq = float(info["residual_norm"].item())
        if not (abs(rr - rq) <= 1e-8 * (1.0 + abs(rq))):
            raise AssertionError(f"{name}: residual_norm and residual_norm_real mismatch")


def run_all() -> None:
    dev = pick_device("cuda" if torch.cuda.is_available() else "cpu")
    verify_forward_embedding_consistency(dev)
    verify_adjoint_consistency(dev)
    verify_least_squares_self_consistency(dev)
    verify_qrkm_measurement_compatibility(dev)
    verify_ill_conditioned_behavior(dev)
    verify_minimum_norm_behavior_nullspace(dev)
    verify_least_squares_debug_info(dev)
    print("[verify_quat_real_embed] all checks passed.")


if __name__ == "__main__":
    run_all()
