"""Fig. 1 of the letter: one synthetic trial, all seven solvers, one shared initialization."""

from __future__ import annotations

import math
import os
import sys
import time
import csv

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from algorithms.algs.alg_qadmm import alg_qadmm
from algorithms.algs.alg_qaraf import alg_qaraf
from algorithms.algs.alg_qpaf import alg_qpaf
from algorithms.algs.alg_qraf import alg_qraf
from algorithms.algs.alg_qrkm import alg_qrkm
from algorithms.algs.alg_qrwf import alg_qrwf
from algorithms.algs.alg_qwf import alg_qwf
import core.hyperparams as hp
from algorithms.initialization.quat_spectral_init import quat_spectral_init_shared
from core.quat_metrics import quat_dist_right_phase
from core.quat_sampling import make_noiseless_measurements, sample_quaternion_gaussian
from utils.common import pick_device
from utils.utils_plt import plot_conv_curvs

# --- experiment configuration (Fig. 1: d = 100, n/d = 8, T = 400) ---
d = 100
n_over_d = 8
T = 400
n = n_over_d * d
init_num_power_iters = hp.POWER_ITERS
# The instance (A, x*) used to be drawn from the *unseeded* global RNG, so this
# figure could not be reproduced. Seeded from the same master seed as every other
# experiment in this round.
seed = hp.BASE_SEED
dtype = torch.float64
device = "cuda" if torch.cuda.is_available() else "cpu"
eps = 1e-12
eps_beta = 1e-12
# Set True locally to pop up the matplotlib window after saving the figure.
SHOW_PREVIEW = False
# Legend placement and size for the comparison figure.
LEGEND_LOC = "lower left"
LEGEND_FONTSIZE: float | str | None = 9

OUTPUT_DIR = os.path.join(_ROOT, "output")
FIG_BASE_NAME = "compare_pr_single_trial_ln_dist_T"
CURVE_PT_NAME = "compare_pr_single_trial_curves.pt"
CURVE_CSV_NAME = "compare_pr_single_trial_curves.csv"


def _ln_dist_right_phase_trajectory(
        z_path: list[torch.Tensor],
        x_true: torch.Tensor,
        *,
        eps: float = 1e-12,
) -> np.ndarray:
    """Log right-phase distance for each iterate; z may be (d,4) or (d,1,4)."""
    x_true = x_true.to(dtype=torch.float64)
    vals: list[float] = []
    for z in z_path:
        zd = z.squeeze(1) if z.dim() == 3 else z
        dval = float(quat_dist_right_phase(zd, x_true, eps=float(eps)).item())
        vals.append(math.log(max(dval, 1e-300)))
    return np.asarray(vals, dtype=float)


def main() -> None:
    dev = pick_device(device)
    gen_data = torch.Generator(device=dev)
    gen_data.manual_seed(int(seed))
    A = sample_quaternion_gaussian((n, d), device=dev, dtype=dtype, generator=gen_data)
    x_true = sample_quaternion_gaussian((d,), device=dev, dtype=dtype, generator=gen_data)
    x_true = x_true / (torch.sqrt(torch.sum(x_true * x_true)) + eps)
    y_intensity = make_noiseless_measurements(A, x_true)
    # Amplitude flows (QRAF / QGN / QWF / QARAF) compare |a^*z| to ψ_k; use ψ = sqrt(y).
    y_amp = torch.sqrt(torch.clamp(y_intensity, min=0.0))
    # Exactly the initializer used by the batched runs behind Table II / Fig. 2, so the
    # single-trial figure and the statistics describe the same protocol.
    gen_init = torch.Generator(device=dev)
    gen_init.manual_seed(int(seed) + 777_777_001)
    z0_shared = quat_spectral_init_shared(
        A, y_intensity,
        power_iters=init_num_power_iters, eps=eps, generator=gen_init,
    )

    timings: dict[str, float] = {}
    curves: dict[str, np.ndarray] = {}

    # --- Baselines: same T, amplitude/intensity per algorithm semantics ---
    def run_baseline(name: str, fn, y_input: torch.Tensor, *, z0: torch.Tensor | None = z0_shared) -> None:
        t1 = time.perf_counter()
        out = fn(A, y_input, T=T, x_true=x_true, stop_err=0.0, verbose=True, z0=z0)
        timings[name] = time.perf_counter() - t1
        if not isinstance(out, dict):
            raise TypeError(f"{name} output must be a dict.")
        if "z_path" not in out:
            raise KeyError(f"{name} output missing required key: z_path")
        z_path = out["z_path"]
        curves[name] = _ln_dist_right_phase_trajectory(z_path, x_true, eps=eps)

    run_baseline("QRKM", alg_qrkm, y_intensity)
    run_baseline("QADMM", alg_qadmm, y_intensity)
    run_baseline("QRAF", alg_qraf, y_amp)
    run_baseline("QARAF", alg_qaraf, y_amp)
    run_baseline("QWF", alg_qwf, y_amp)
    run_baseline("QRWF", alg_qrwf, y_amp)
    run_baseline("QPAF", alg_qpaf, y_amp)  # shared init (fairness, consistent with batched runs)

    for k, sec in timings.items():
        print(f"time_sec[{k}]={sec:.4f}")

    L = min(len(curves[k]) for k in curves)
    if L <= 0:
        raise RuntimeError("Empty trajectories; check T and solvers.")

    # Align lengths so one shared x-axis is unambiguous on the plot.
    for k in list(curves.keys()):
        curves[k] = curves[k][:L]

    # Entry i of every trajectory is the iterate after i updates (entry 0 is the
    # shared initialization), so the epoch axis must start at 0. Labelling it 1..L
    # would report the initialization as "epoch 1" and shift the whole figure by one
    # epoch relative to the epoch counts of Table II.
    x_steps = np.arange(0, L, dtype=float)

    data_series = [
        {"y": curves["QRKM"], "label": "QRKM", "style": "-", "color": "red", "markevery": 20},
        {"y": curves["QADMM"], "label": "QADMM", "style": "--", "color": "teal", "markevery": 20},
        {"y": curves["QRAF"], "label": "QRAF", "style": "-x", "color": "blue", "markevery": 20},
        {"y": curves["QARAF"], "label": "QARAF", "style": "-.", "color": "orange", "markevery": 20},
        {"y": curves["QWF"], "label": "QWF", "style": "-o", "color": "black", "markevery": 20},
        {"y": curves["QRWF"], "label": "QRWF", "style": ":", "color": "brown", "markevery": 20},
        {"y": curves["QPAF"], "label": "QPAF", "style": "-d", "color": "green", "markevery": 20},
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    curve_path = os.path.join(OUTPUT_DIR, CURVE_PT_NAME)
    torch.save(
        {
            "curves": {k: v.tolist() for k, v in curves.items()},
            "x_steps": x_steps.tolist(),
            "timings_sec": timings,
            "config": {"d": d, "n": n, "T": T, "seed": seed, "init_num_power_iters": init_num_power_iters},
        },
        curve_path,
    )
    print(f"saved_curve_pt={curve_path}")

    curve_csv_path = os.path.join(OUTPUT_DIR, CURVE_CSV_NAME)
    curve_order = [series["label"] for series in data_series]
    with open(curve_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *curve_order])
        for i in range(L):
            writer.writerow([int(x_steps[i]), *[float(curves[name][i]) for name in curve_order]])
    print(f"saved_curve_csv={curve_csv_path}")


    fig_path = os.path.join(OUTPUT_DIR, FIG_BASE_NAME)
    plot_conv_curvs(
        x=x_steps,
        data_series=data_series,
        xlabel=f"epoch",
        ylabel="ln(dist_T)",
        figsize=(8, 4),
        filename=fig_path,
        show_preview=SHOW_PREVIEW,
        legend_loc=LEGEND_LOC,
        legend_fontsize=LEGEND_FONTSIZE,
    )


if __name__ == "__main__":
    main()
