from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, Optional, List

import numpy as np
import torch

# Ensure direct execution of this file can import project packages.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from algorithms.gradient.grad_qwf import grad_qwf
from algorithms.initialization.init_si import init_si
from algorithms.initialization.quat_spectral_init import quat_spectral_init
from utils.utils_verbose import ensure_signal_shape, print_iter_log
from core.q_funcs import q_arr_dist, q_conj, phase_factor_estimate_batched
from core.quat_metrics import quat_dist_right_phase, quat_right_phase_align, success_under_right_phase
from core.quat_ops import quat_conj, quat_mul
from core.quat_sampling import make_noiseless_measurements, make_quaternion_gaussian_matrix, make_random_quaternion_signal
from core.hyperparams import QWF as HP_QWF
from utils.common import pick_device
from utils.utils_plt import plot_conv_curvs


def alg_qwf(
        A_mat: torch.Tensor,
        y: torch.Tensor,
        T: int = 200,
        x_true: Optional[torch.Tensor] = None,
        stop_err: float = 0.0,
        verbose: bool = False,
        use_pfe: bool = False,
        z0: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Run QWF with unified float64 quaternion interface.

    Returns a dict with `x_hat`, `z_path`, `history["dist_T"]`, and `config`.
    """

    # --- device / dtype harmonization ---
    device = A_mat.device
    if A_mat.dtype != torch.float64:
        raise TypeError("alg_qwf requires float64 tensors")
    if y.dtype != torch.float64:
        raise TypeError("alg_qwf requires float64 tensors")
    A_mat = A_mat.to(device)
    y = y.to(device)

    A_model = q_conj(A_mat)

    # --- (1) Initialization ---
    if z0 is None:
        z0 = init_si(A_model, y)  # your init returns (d,4) or (d,1,4)
    else:
        z0 = z0.to(device)
        if z0.dim() == 2:
            z0 = z0.unsqueeze(1)

    # --- (2) Gradient refinement ---
    x_true = ensure_signal_shape(x_true, z0)
    z_path: List[torch.Tensor] = grad_qwf(
        A_model,
        y,  # pass canonical (n,)
        z0,
        T=T,
        x_true=x_true,
        stop_err=stop_err,
        use_pfe=use_pfe,
    )

    # --- (3) Optional diagnostics: distance and log-distance ---
    dist_log: List[float] = []
    if x_true is not None:
        # compute per-iterate distance and take log; keep as CPU 0-D tensors
        for z in z_path:
            dval = float(q_arr_dist(z, x_true).item())
            dist_log.append(dval)
    if verbose:
        total_epochs = max(1, len(z_path) - 1)
        z_iter = z_path[1:] if len(z_path) > 1 else z_path
        for i, z in enumerate(z_iter, start=1):
            log_name = "alg_qwf_pfe" if use_pfe else "alg_qwf"
            print_iter_log(log_name, i, total_epochs, A_model, y, z, x_true)

    x_hat = z_path[-1]
    if x_hat.dim() == 3 and x_hat.shape[1] == 1 and x_hat.shape[2] == 4:
        x_hat = x_hat.squeeze(1)

    return {
        "x_hat": x_hat,
        "z_path": z_path,
        "history": {
            "dist_T": dist_log,
        },
        "config": {
            "method": "qwf",
            "T": int(T),
            "stop_err": float(stop_err),
            "use_pfe": bool(use_pfe),
        },
    }


def _smoke_test_main() -> None:
    """Run a single synthetic QWF trial and save convergence curves."""
    d = 50
    n = 10 * d
    T = 200
    init_num_power_iters = 50
    seed = 1234
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stop_tol = 1e-5
    eps = 1e-12

    output_dir = os.path.join(_ROOT, "output")
    convergence_csv_name = "qwf_smoke_from_alg_convergence.csv"
    curve_pt_name = "qwf_smoke_from_alg_curve.pt"
    fig_base_name_ln_dist = "qwf_smoke_from_alg_ln_dist_T"

    dev = pick_device(device)
    A = make_quaternion_gaussian_matrix(n, d, device=dev, dtype=dtype)
    x_true = make_random_quaternion_signal(d, normalize=True, device=dev, dtype=dtype)
    y_intensity = make_noiseless_measurements(A, x_true)
    y_amp = torch.sqrt(torch.clamp(y_intensity, min=0.0))
    z0 = quat_spectral_init(
        A,
        y_intensity,
        num_power_iters=init_num_power_iters,
        init_mode="random",
        seed=seed,
        eps=eps,
        device=dev,
        dtype=dtype,
        verbose=True,
    )

    t0 = time.perf_counter()
    out = alg_qwf(
        A,
        y_amp,
        x_true=x_true,
        T=T,
        stop_err=0.0,
        verbose=False,
        use_pfe=False,
        z0=z0,
    )
    dt = time.perf_counter() - t0

    x_hat = out["x_hat"]
    hist = out["history"]
    dist_T_list = hist["dist_T"]
    x_steps = np.arange(1, len(dist_T_list) + 1, dtype=float)
    dist_T_arr = np.asarray(dist_T_list, dtype=float)
    ln_vals = np.log(np.maximum(dist_T_arr, 1e-300)) if dist_T_arr.size > 0 else np.asarray([], dtype=float)

    ok = success_under_right_phase(x_hat, x_true, tol=stop_tol)
    q_opt = quat_right_phase_align(x_hat, x_true, eps=eps)
    q_opt_abs = float(torch.sqrt(torch.sum(q_opt * q_opt)).item())
    dist_T_final = float(dist_T_arr[-1]) if dist_T_arr.size > 0 else float("nan")

    print("--- QWF smoke test summary ---")
    print(f"final_dist_T={dist_T_final:.6e}")
    if ln_vals.size > 0:
        print(f"final_ln_dist_T={float(ln_vals[-1]):.6f}")
    print(f"success_right_phase={ok}")
    print(f"final_q_opt_abs={q_opt_abs:.16e}")
    print(f"elapsed_sec={dt:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, convergence_csv_name)
    fieldnames = ("iter", "dist_T", "ln_dist_T")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for k in range(len(dist_T_list)):
            writer.writerow(
                {
                    "iter": k + 1,
                    "dist_T": dist_T_list[k],
                    "ln_dist_T": float(ln_vals[k]),
                }
            )
    print(f"saved_convergence_csv={csv_path}")

    curve_path = os.path.join(output_dir, curve_pt_name)
    torch.save(
        {
            "dist_T": hist["dist_T"],
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
                {"y": np.asarray(ln_vals, dtype=float), "label": "QWF", "style": "-", "color": "steelblue"},
            ],
            title="QWF smoke test: ln(dist_T) vs iterations",
            xlabel=f"iteration k (T={T}), d={d}, n={n}",
            ylabel="ln(dist_T)",
            figsize=(8, 4),
            filename=fig_path,
            show_preview=True,
        )


if __name__ == "__main__":
    _smoke_test_main()
