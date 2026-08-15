"""Quaternion ADMM (QADMM) wrapper — baseline phase retrieval driver."""

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

from algorithms.gradient.grad_qadmm import grad_qadmm
from algorithms.initialization.quat_spectral_init import quat_spectral_init
from core.q_funcs import phase_factor_estimate_batched
from core.quat_metrics import quat_dist_right_phase
from core.quat_ops import quat_abs, quat_conj, quat_mul
from core.quat_metrics import quat_right_phase_align, success_under_right_phase
from core.quat_sampling import make_noiseless_measurements, make_quaternion_gaussian_matrix, make_random_quaternion_signal
from core.hyperparams import QADMM as HP_QADMM
from utils.common import assert_float64, pick_device, to_float64
from utils.seed import set_seed
from utils.utils_plt import plot_conv_curvs


def alg_qadmm(
    A_mat: torch.Tensor,
    y: torch.Tensor,
    T: int = 200,
    x_true: Optional[torch.Tensor] = None,
    stop_err: float = 0.0,
    rho: float = HP_QADMM["rho"],
    init_num_power_iters: int = 50,
    seed: int = 1234,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
    return_history: bool = True,
    verbose: bool = True,
    use_pfe: bool = False,
    z0: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Run QADMM with unified float64 quaternion interface.

    Returns a dict with `x_hat`, `z_path`, `history`, and `config`.
    """
    if dtype != torch.float64:
        raise TypeError("alg_qadmm requires torch.float64")
    if A_mat.dtype != torch.float64 or y.dtype != torch.float64:
        raise TypeError("alg_qadmm requires float64 tensors")
    set_seed(int(seed))
    dev = pick_device(device)
    A_mat = to_float64(A_mat, name="A_mat").to(dev)
    y = to_float64(y, name="y").to(dev)
    if y.dim() == 2 and y.shape[1] == 1:
        y = y.squeeze(1)
    assert_float64(A_mat, name="A_mat")
    assert_float64(y, name="y")
    y_intensity = y

    x_true_grad: Optional[torch.Tensor] = None
    if x_true is not None:
        x_true = to_float64(x_true, name="x_true").to(dev)
        assert_float64(x_true, name="x_true")
        if x_true.dim() == 3 and x_true.shape[1] == 1 and x_true.shape[2] == 4:
            x_true = x_true.squeeze(1)
        if x_true.dim() != 2 or x_true.shape[1] != 4:
            raise ValueError(f"alg_qadmm expects x_true shape (d,4) or (d,1,4), got {tuple(x_true.shape)}")
        x_true_grad = x_true

    if z0 is not None:
        q0 = z0.to(dev)
        if q0.dim() == 3 and q0.shape[1] == 1 and q0.shape[2] == 4:
            q0 = q0.squeeze(1)
        if q0.dim() != 2 or q0.shape[1] != 4:
            raise ValueError(f"z0 expects (d,4) or (d,1,4), got {tuple(q0.shape)}")
        q0 = to_float64(q0, name="z0")
        assert_float64(q0, name="z0")
    else:
        q0 = quat_spectral_init(
            A_mat,
            y_intensity,
            num_power_iters=int(init_num_power_iters),
            init_mode="random",
            seed=int(seed),
            eps=float(eps),
            device=dev,
            dtype=torch.float64,
            verbose=verbose,
        )

    q_path, history = grad_qadmm(
        A_mat,
        y_intensity,
        q0,
        T=int(T),
        rho=float(rho),
        eps=float(eps),
        return_history=return_history,
        verbose=verbose,
        x_true=x_true_grad,
        use_pfe=bool(use_pfe),
    )

    if x_true is not None and stop_err > 0.0:
        dist_hist = history.get("dist_T", []) if isinstance(history, dict) else []
        if dist_hist:
            for i, dval in enumerate(dist_hist, start=1):
                if float(dval) <= float(stop_err):
                    q_path = q_path[: i + 1]
                    if isinstance(history, dict):
                        for k, v in list(history.items()):
                            if isinstance(v, list):
                                history[k] = v[:i]
                    break

    return {
        "x_hat": q_path[-1],
        "z_path": q_path,
        "history": history if return_history else {"dist_T": []},
        "config": {
            "method": "qadmm",
            "T": int(T),
            "stop_err": float(stop_err),
            "rho": float(rho),
            "init_num_power_iters": int(init_num_power_iters),
            "seed": int(seed),
            "device": str(dev),
            "dtype": "float64",
            "eps": float(eps),
            "use_pfe": bool(use_pfe),
        },
    }


def _smoke_test_main() -> None:
    """Run a single synthetic QADMM trial and save convergence curves."""
    # --- experiment configuration ---
    d = 50
    n = 10 * d
    T = 200
    init_num_power_iters = 50
    seed = 1234
    rho = float(HP_QADMM["rho"])
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stop_tol = 0.0
    eps = 1e-12

    output_dir = os.path.join(_ROOT, "output")
    convergence_csv_name = "qadmm_smoke_from_alg_convergence.csv"
    curve_pt_name = "qadmm_smoke_from_alg_curve.pt"
    fig_base_name_ln_dist = "qadmm_smoke_from_alg_ln_dist_T"

    dev = pick_device(device)
    A = make_quaternion_gaussian_matrix(n, d, device=dev, dtype=dtype)
    x_true = make_random_quaternion_signal(d, normalize=True, device=dev, dtype=dtype)
    y = make_noiseless_measurements(A, x_true)

    t0 = time.perf_counter()
    out = alg_qadmm(
        A,
        y,
        x_true=x_true,
        T=T,
        rho=rho,
        init_num_power_iters=init_num_power_iters,
        seed=seed,
        device=dev,
        dtype=dtype,
        eps=eps,
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

    print("--- QADMM smoke test summary ---")
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
        "primal_residual",
        "dual_residual",
        "q_norm",
        "z_norm",
        "lambda_norm",
        "ls_residual_norm",
        "ls_solution_norm",
        "zero_phase_fallbacks",
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
                    "primal_residual": hist["primal_residual"][k],
                    "dual_residual": hist["dual_residual"][k],
                    "q_norm": hist["q_norm"][k],
                    "z_norm": hist["z_norm"][k],
                    "lambda_norm": hist["lambda_norm"][k],
                    "ls_residual_norm": hist["ls_residual_norm"][k],
                    "ls_solution_norm": hist["ls_solution_norm"][k],
                    "zero_phase_fallbacks": hist["zero_phase_fallbacks"][k],
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
            "primal_residual": hist["primal_residual"],
            "dual_residual": hist["dual_residual"],
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
                {"y": np.asarray(ln_vals, dtype=float), "label": "QADMM", "style": "-", "color": "steelblue"},
            ],
            title="QADMM smoke test: ln(dist_T) vs ADMM iterations",
            xlabel=f"iteration k (T={T}), d={d}, n={n}, rho={rho}",
            ylabel="ln(dist_T)",
            figsize=(8, 4),
            filename=fig_path,
            show_preview=True,
        )


if __name__ == "__main__":
    _smoke_test_main()
