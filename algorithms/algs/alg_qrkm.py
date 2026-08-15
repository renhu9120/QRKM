from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Ensure direct execution of this file can import project packages.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from algorithms.gradient.grad_qrkm import grad_qrkm
from algorithms.initialization.quat_spectral_init import quat_spectral_init
from core.quat_metrics import quat_dist_right_phase, quat_right_phase_align, success_under_right_phase
from core.quat_ops import quat_abs, quat_conj, quat_mul
from core.quat_sampling import make_noiseless_measurements, make_quaternion_gaussian_matrix, make_random_quaternion_signal
from utils.common import assert_float64, pick_device, to_float64
from utils.seed import set_seed
from utils.utils_plt import plot_conv_curvs


def alg_qrkm(
        A_mat: torch.Tensor,
        y: torch.Tensor,
        T: int = 200,
        x_true: Optional[torch.Tensor] = None,
        stop_err: float = 0.0,
        use_pfe: bool = False,
        z0: Optional[torch.Tensor] = None,
        init_num_power_iters: int = 50,
        seed: int = 1234,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,  # must be float64; kept for call-site parity with other algs
        eps: float = 1e-12,
        eps_beta: float = 1e-12,
        return_history: bool = True,
        verbose: bool = True,
        diagnostic_mode: bool = False,
        diag_tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Quaternion randomized Kaczmarz method (QRKM).

    Pipeline
    --------
    1) Initialization (`quat_spectral_init`):
       Spectral / power-iteration style quaternion initializer from (A, y).

    2) Randomized sweeps (`grad_qrkm`):
       ``T`` outer epochs; each epoch performs ``n`` row Kaczmarz updates along
       a random row ordering (with replacement), using row energies for the
       sampling weights.

    Parameters
    ----------
    A_mat : torch.Tensor
        Measurement matrix of shape (n, d, 4).
    y : torch.Tensor
        Intensity measurements of shape (n,) with y_k = |a_k^* x|^2.
    T : int, default=200
        Number of outer sweeps (epochs).
    x_true : Optional[torch.Tensor], default=None
        Ground-truth signal of shape (d, 4), used only for distance logging in
        the returned history.
    use_pfe : bool, default=False
        If True, apply phase factor estimate (pure-quaternion projection) after
        each outer QRKM sweep.
    init_num_power_iters : int, default=50
        Power iterations inside the spectral initializer.
    seed : int, default=1234
        RNG seed (global state and initializer).
    device : str or torch.device, default="cuda"
        Device for all tensors.
    dtype : torch.dtype, default=torch.float64
        Must be float64 (enforced).
    eps, eps_beta : float
        Numerical guards for amplitudes and row energies.
    return_history : bool, default=True
        If True, populate per-epoch diagnostics in ``history``.
    verbose : bool, default=True
        Print one line per epoch when True.
    diagnostic_mode : bool, default=False
        If True, run consistency checks against explicit row gradients (slow).
    diag_tol : float
        Tolerance for diagnostic_mode assertions.

    Returns
    -------
    dict
        ``x_hat`` — final estimate (d, 4); ``history`` — epoch-wise lists;
        ``config`` — serialized hyperparameters.
    """
    if dtype != torch.float64:
        raise TypeError("alg_qrkm requires torch.float64")
    if A_mat.dtype != torch.float64 or y.dtype != torch.float64:
        raise TypeError("alg_qrkm requires float64 tensors")
    set_seed(int(seed))
    dev = pick_device(device)
    A_mat = to_float64(A_mat, name="A_mat").to(dev)
    y = to_float64(y, name="y").to(dev)
    if y.dim() == 2 and y.shape[1] == 1:
        y = y.squeeze(1)
    assert_float64(A_mat, name="A_mat")
    assert_float64(y, name="y")
    if x_true is not None:
        x_true = to_float64(x_true, name="x_true").to(dev)
        assert_float64(x_true, name="x_true")
        # Accept both (d,4) and (d,1,4) for compatibility with other alg wrappers.
        if x_true.dim() == 3 and x_true.shape[1] == 1 and x_true.shape[2] == 4:
            x_true = x_true.squeeze(1)
        if x_true.dim() != 2 or x_true.shape[1] != 4:
            raise ValueError(f"alg_qrkm expects x_true shape (d,4) or (d,1,4), got {tuple(x_true.shape)}")

    # --- (1) Initialization ---
    if z0 is None:
        z0 = quat_spectral_init(
            A_mat,
            y,
            num_power_iters=int(init_num_power_iters),
            init_mode="random",
            seed=int(seed),
            eps=float(eps),
            device=dev,
            dtype=torch.float64,
            verbose=verbose,
        )
    else:
        z0 = to_float64(z0, name="z0").to(dev)
        if z0.dim() == 3 and z0.shape[1] == 1 and z0.shape[2] == 4:
            z0 = z0.squeeze(1)
        if z0.dim() != 2 or z0.shape[1] != 4:
            raise ValueError(f"alg_qrkm expects z0 shape (d,4) or (d,1,4), got {tuple(z0.shape)}")

    # --- (2) Randomized Kaczmarz refinement ---
    z_path, history = grad_qrkm(
        A_mat,
        y,
        z0,
        T=T,
        eps=float(eps),
        eps_beta=float(eps_beta),
        return_history=return_history,
        verbose=verbose,
        diagnostic_mode=diagnostic_mode,
        diag_tol=float(diag_tol),
        x_true=x_true,
        use_pfe=bool(use_pfe),
    )

    if x_true is not None and float(stop_err) > 0.0 and return_history and isinstance(history, dict):
        dist_hist = history.get("dist_T", [])
        if dist_hist:
            for i, dval in enumerate(dist_hist, start=1):
                if float(dval) <= float(stop_err):
                    z_path = z_path[: i + 1]
                    for k, v in list(history.items()):
                        if isinstance(v, list):
                            history[k] = v[:i]
                    break

    result: Dict[str, Any] = {
        "x_hat": z_path[-1],
        "z_path": z_path,
        "history": history if return_history else {},
        "config": {
            "method": "qrkm",
            "T": int(T),
            "stop_err": float(stop_err),
            "init_num_power_iters": int(init_num_power_iters),
            "seed": int(seed),
            "use_pfe": bool(use_pfe),
            "device": str(dev),
            "dtype": "float64",
            "eps": float(eps),
            "eps_beta": float(eps_beta),
            "diagnostic_mode": bool(diagnostic_mode),
            "diag_tol": float(diag_tol),
        },
    }
    return result


def _smoke_test_main() -> None:
    """Run a single synthetic QRKM trial and save convergence curves."""
    d = 30
    n = 7 * d
    T = 200
    init_num_power_iters = 50
    seed = 1234
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stop_tol = 1e-5
    eps = 1e-12
    eps_beta = 1e-12

    output_dir = os.path.join(_ROOT, "output")
    convergence_csv_name = "qrkm_smoke_from_alg_convergence.csv"
    curve_pt_name = "qrkm_smoke_from_alg_curve.pt"
    fig_base_name_ln_dist = "qrkm_smoke_from_alg_ln_dist_T"

    dev = pick_device(device)
    A = make_quaternion_gaussian_matrix(n, d, device=dev, dtype=dtype)
    x_true = make_random_quaternion_signal(d, normalize=True, device=dev, dtype=dtype)
    y = make_noiseless_measurements(A, x_true)

    t0 = time.perf_counter()
    out = alg_qrkm(
        A,
        y,
        x_true=x_true,
        T=T,
        init_num_power_iters=init_num_power_iters,
        seed=seed,
        device=dev,
        dtype=dtype,
        eps=eps,
        eps_beta=eps_beta,
        return_history=True,
        verbose=True,
    )
    dt = time.perf_counter() - t0

    x_hat = out["x_hat"]
    hist = out["history"]
    dist_T_list = hist["dist_T"]
    x_steps = np.arange(1, len(dist_T_list) + 1, dtype=float)
    dist_T_arr = np.asarray(dist_T_list, dtype=float)
    ln_vals = np.log(np.maximum(dist_T_arr, 1e-300))

    loss_final = hist["loss"][-1]
    dist_T_final = hist["dist_T"][-1]
    dist_raw_final = hist["dist_raw"][-1]
    ok = success_under_right_phase(x_hat, x_true, tol=stop_tol)
    q_opt = quat_right_phase_align(x_hat, x_true, eps=eps)
    q_opt_abs = float(quat_abs(q_opt.unsqueeze(0)).item())

    print("--- QRKM smoke test summary ---")
    print(f"final_loss={loss_final:.6e}")
    print(f"final_dist_T={dist_T_final:.6e}")
    print(f"final_ln_dist_T={float(ln_vals[-1]):.6f}")
    print(f"final_dist_raw={dist_raw_final:.6e}")
    print(f"success_right_phase={ok}")
    print(f"final_q_opt_abs={q_opt_abs:.16e}")
    print(f"elapsed_sec={dt:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, convergence_csv_name)
    fieldnames = (
        "iter",
        "loss",
        "dist_T",
        "dist_raw",
        "ln_dist_T",
        "num_skips",
        "mean_step_norm",
        "mean_row_mismatch",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for k in range(len(dist_T_list)):
            writer.writerow(
                {
                    "iter": k + 1,
                    "loss": hist["loss"][k],
                    "dist_T": hist["dist_T"][k],
                    "dist_raw": hist["dist_raw"][k],
                    "ln_dist_T": float(ln_vals[k]),
                    "num_skips": hist["num_skips"][k],
                    "mean_step_norm": hist["mean_step_norm"][k],
                    "mean_row_mismatch": hist["mean_row_mismatch"][k],
                }
            )
    print(f"saved_convergence_csv={csv_path}")

    curve_path = os.path.join(output_dir, curve_pt_name)
    torch.save(
        {
            "loss": hist["loss"],
            "dist_T": hist["dist_T"],
            "dist_raw": hist["dist_raw"],
            "ln_dist_T": ln_vals.tolist(),
            "config": out["config"],
        },
        curve_path,
    )
    print(f"saved_curve_pt={curve_path}")

    if len(x_steps) > 0 and len(ln_vals) > 0:
        fig_path = os.path.join(output_dir, fig_base_name_ln_dist)
        plot_conv_curvs(
            x=x_steps,
            data_series=[
                {"y": np.asarray(ln_vals, dtype=float), "label": "QRKM", "style": "-", "color": "darkorange"},
            ],
            title="QRKM smoke test: ln(dist_T) vs epochs",
            xlabel=f"epoch k (T={T}), d={d}, n={n}",
            ylabel="ln(dist_T)",
            figsize=(8, 4),
            filename=fig_path,
            show_preview=True,
        )


if __name__ == "__main__":
    _smoke_test_main()
