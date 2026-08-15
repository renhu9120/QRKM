"""Quaternion phase-retrieval: loss, gradients, and randomized Kaczmarz (QRKM) sweeps."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from core.quat_metrics import quat_dist_right_phase, quat_raw_distance
from core.quat_ops import quat_abs, quat_mul, quat_mv_conj_left
from core.q_funcs import phase_factor_estimate
from utils.common import assert_float64, to_float64


def quat_row_response(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return quat_mv_conj_left(A, x)


def quat_row_amplitude(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    u = quat_row_response(A, x)
    return quat_abs(u)


def quat_row_residual(
    A: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = to_float64(A, name="A")
    x = to_float64(x, name="x")
    b = to_float64(b, name="b")
    assert_float64(A, name="A")
    assert_float64(x, name="x")
    assert_float64(b, name="b")
    u = quat_row_response(A, x)
    mag = quat_abs(u)
    safe = (mag >= float(eps)).unsqueeze(-1)
    inv = torch.where(mag.unsqueeze(-1) > 0, 1.0 / mag.unsqueeze(-1), torch.zeros_like(mag.unsqueeze(-1)))
    phi = torch.where(safe, u * inv, torch.zeros_like(u))
    phi = torch.where(safe, phi, torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=u.dtype, device=u.device).expand_as(phi))
    r = torch.where(safe, b.unsqueeze(-1) * phi - u, torch.zeros_like(u))
    return u, phi, r


def quat_row_loss(A: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    amp = quat_row_amplitude(A, x)
    return (amp - b) ** 2


def quat_total_loss(A: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(quat_row_loss(A, x, b))


def quat_selected_row_gradient(
    a_l: torch.Tensor,
    x: torch.Tensor,
    b_l: float | torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    a_l = to_float64(a_l, name="a_l")
    x = to_float64(x, name="x")
    assert_float64(a_l, name="a_l")
    assert_float64(x, name="x")
    if isinstance(b_l, torch.Tensor):
        b_l_t = to_float64(b_l, name="b_l")
        assert_float64(b_l_t, name="b_l")
        b_val = float(b_l_t.item())
    else:
        b_val = float(b_l)
    u_l = quat_row_response(a_l.unsqueeze(0), x).squeeze(0)
    mag = float(quat_abs(u_l.unsqueeze(0)).item())
    beta_l = float(torch.sum(quat_abs_sq_rows(a_l)).item())
    f_l = (mag - b_val) ** 2
    if mag < float(eps):
        zero = torch.zeros_like(x)
        aux = {
            "u_l": u_l,
            "phi_l": torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float64, device=x.device),
            "r_l": torch.zeros(4, dtype=torch.float64, device=x.device),
            "beta_l": beta_l,
            "f_l": f_l,
            "skipped_gradient": True,
        }
        return zero, aux
    inv = 1.0 / mag
    phi_l = u_l * inv
    r_l = b_val * phi_l - u_l
    grad_l = -2.0 * quat_mul(a_l, r_l.unsqueeze(0))
    aux = {
        "u_l": u_l,
        "phi_l": phi_l,
        "r_l": r_l,
        "beta_l": beta_l,
        "f_l": f_l,
        "skipped_gradient": False,
    }
    return grad_l, aux


def quat_abs_sq_rows(a_l: torch.Tensor) -> torch.Tensor:
    return torch.sum(a_l * a_l, dim=-1)


def quat_full_gradient(A: torch.Tensor, x: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    A = to_float64(A, name="A")
    x = to_float64(x, name="x")
    b = to_float64(b, name="b")
    assert_float64(A, name="A")
    assert_float64(x, name="x")
    assert_float64(b, name="b")
    _, _, r = quat_row_residual(A, x, b, eps=float(eps))
    contrib = -2.0 * quat_mul(A, r.unsqueeze(1))
    return torch.sum(contrib, dim=0)


def single_row_u_and_T(
    a_l: torch.Tensor,
    x: torch.Tensor,
    b_l: float,
    beta_l: float,
    eps: float = 1e-12,
    eps_beta: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Return u_l(x), T_l(x), u_l^sharp(x), did_skip for diagnostics."""
    a_l = to_float64(a_l, name="a_l")
    x = to_float64(x, name="x")
    assert_float64(a_l, name="a_l")
    assert_float64(x, name="x")
    u_l = quat_row_response(a_l.unsqueeze(0), x).squeeze(0)
    mag = float(quat_abs(u_l.unsqueeze(0)).item())
    if float(beta_l) < float(eps_beta) or mag < float(eps):
        return u_l, x.clone(), torch.zeros_like(u_l), True
    inv = 1.0 / mag
    phi = u_l * inv
    u_sharp = float(b_l) * phi
    r_l = float(b_l) * phi - u_l
    step = (1.0 / float(beta_l)) * quat_mul(a_l, r_l.unsqueeze(0))
    T_x = x + step
    return u_l, T_x, u_sharp, False


def row_energy_beta(A: torch.Tensor) -> torch.Tensor:
    A = to_float64(A, name="A")
    assert_float64(A, name="A")
    return torch.sum(A * A, dim=(1, 2))


def grad_qrkm(
        A_mat: torch.Tensor,
        y_k: torch.Tensor,
        z0: torch.Tensor,
        T: int,
        *,
        eps: float = 1e-12,
        eps_beta: float = 1e-12,
        return_history: bool = True,
        verbose: bool = True,
        diagnostic_mode: bool = False,
        diag_tol: float = 1e-10,
        x_true: Optional[torch.Tensor] = None,
        use_pfe: bool = False,
) -> tuple[list[torch.Tensor], dict[str, list[Any]]]:
    """
    Randomized quaternion Kaczmarz sweeps (one full pass per outer iteration).

    Each outer step draws an ordered multiset of row indices (with replacement),
    applies the row-wise Kaczmarz update in that order, then optionally logs
    loss and diagnostics.

    Parameters
    ----------
    A_mat : torch.Tensor, shape (n, d, 4)
        Quaternion measurement rows.
    y_k : torch.Tensor, shape (n,)
        Intensity measurements y_k = |a_k^* x|^2 (same convention as the driver).
    z0 : torch.Tensor, shape (d, 4)
        Initial estimate (typically from spectral init).
    T : int
        Number of outer sweeps (epochs).
    x_true : torch.Tensor, optional
        Ground truth of shape (d, 4), used only for distance logging when provided.
    use_pfe : bool, default=False
        If True, apply phase factor estimate after each full randomized sweep.

    Returns
    -------
    z_path : list of torch.Tensor
        ``[z^{(0)}, z^{(1)}, ..., z^{(T)}]`` where ``z^{(0)} = z0`` and each next
        entry is the state after one randomized full sweep.
    history : dict
        Per-epoch scalars (``loss``, ``num_skips``, …); empty values if
        ``return_history`` is False.
    """
    A_mat = to_float64(A_mat, name="A_mat")
    y_k = to_float64(y_k, name="y_k")
    z0 = to_float64(z0, name="z0")
    assert_float64(A_mat, name="A_mat")
    assert_float64(y_k, name="y_k")
    assert_float64(z0, name="z0")
    if x_true is not None:
        x_true = to_float64(x_true, name="x_true")
        assert_float64(x_true, name="x_true")

    n, d, _ = A_mat.shape
    b = torch.sqrt(torch.clamp(y_k, min=0.0))
    beta = row_energy_beta(A_mat)
    p = beta / (torch.sum(beta) + float(eps))

    x = z0.clone()
    z_path: list[torch.Tensor] = [x.clone()]

    history: dict[str, list[Any]] = {
        "loss": [],
        "dist_T": [],
        "dist_raw": [],
        "num_skips": [],
        "mean_step_norm": [],
        "mean_row_mismatch": [],
    }

    for k in range(int(T)):
        z = x.clone()
        idx = torch.multinomial(p, num_samples=n, replacement=True)
        skip_count = 0
        step_norms: list[float] = []
        for t in range(n):
            l = int(idx[t].item())
            a_l = A_mat[l]
            beta_l = float(beta[l].item())
            b_l = float(b[l].item())
            u_l = quat_row_response(a_l.unsqueeze(0), z).squeeze(0)
            mag = float(quat_abs(u_l.unsqueeze(0)).item())
            if beta_l < float(eps_beta) or mag < float(eps):
                skip_count += 1
                continue
            inv = 1.0 / mag
            phi_l = u_l * inv
            r_l = b_l * phi_l - u_l
            step = (1.0 / beta_l) * quat_mul(a_l, r_l.unsqueeze(0))
            if diagnostic_mode:
                _u_before, Tlx, u_sharp, did_skip = single_row_u_and_T(
                    a_l, z, b_l, beta_l, eps=float(eps), eps_beta=float(eps_beta)
                )
                if did_skip:
                    raise RuntimeError("Diagnostic inconsistency: row should not be skipped.")
                grad_l, _aux = quat_selected_row_gradient(a_l, z, b_l, eps=float(eps))
                T_grad = z - (1.0 / (2.0 * beta_l)) * grad_l
                u_after = quat_row_response(a_l.unsqueeze(0), Tlx).squeeze(0)
                _check_close(u_after, u_sharp, float(diag_tol), "selected-row identity")
                _check_close(Tlx, T_grad, float(diag_tol), "gradient-step identity")
                amp_after = float(quat_abs(u_after.unsqueeze(0)).item())
                _check_close_scalar(amp_after, b_l, float(diag_tol), "amplitude enforcement")
            z = z + step
            step_norms.append(float(torch.sqrt(torch.sum(step * step)).item()))
        x = phase_factor_estimate(z) if use_pfe else z
        z_path.append(x.clone())

        loss_val = float(quat_total_loss(A_mat, x, b).item())
        if return_history:
            history["loss"].append(loss_val)
            history["num_skips"].append(skip_count)
            if step_norms:
                history["mean_step_norm"].append(sum(step_norms) / len(step_norms))
            else:
                history["mean_step_norm"].append(0.0)
            u_all = quat_row_response(A_mat, x)
            amp_all = quat_abs(u_all)
            history["mean_row_mismatch"].append(float(torch.mean(torch.abs(amp_all - b)).item()))
            if x_true is not None:
                history["dist_T"].append(float(quat_dist_right_phase(x, x_true).item()))
                history["dist_raw"].append(float(quat_raw_distance(x, x_true).item()))
        if verbose:
            msg = f"[alg_qrkm] epoch {k + 1}/{T}"
            if x_true is not None and return_history and history["dist_T"]:
                d_t = float(history["dist_T"][-1])
                ln_d_t = math.log(max(d_t, 1e-300))
                msg += f" dist_T={d_t:.6e} ln_dist_T={ln_d_t:.6f}"
            print(msg)

    if not return_history:
        history = {k: [] for k in history}

    return z_path, history


def _check_close(a: torch.Tensor, b: torch.Tensor, tol: float, name: str) -> None:
    diff = torch.sqrt(torch.sum((a - b) * (a - b)))
    if float(diff.item()) > tol:
        raise AssertionError(f"{name} failed: diff={float(diff.item()):.6e} tol={tol}")


def _check_close_scalar(a: float, b: float, tol: float, name: str) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{name} failed: |a-b|={abs(a-b):.6e} tol={tol}")
