"""Verify QRKM driver on a small synthetic instance."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from algorithms.algs.alg_qrkm import alg_qrkm
from core.quat_sampling import make_noiseless_measurements, make_quaternion_gaussian_matrix, make_random_quaternion_signal
from utils.common import pick_device


def run_all() -> None:
    dev = pick_device("cuda" if torch.cuda.is_available() else "cpu")
    d = 10
    n = 80
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    A = make_quaternion_gaussian_matrix(n, d, device=dev, dtype=torch.float64)
    x_true = make_random_quaternion_signal(d, normalize=True, device=dev, dtype=torch.float64)
    y = make_noiseless_measurements(A, x_true)
    out = alg_qrkm(
        A,
        y,
        x_true=x_true,
        T=25,
        init_num_power_iters=30,
        seed=2026,
        device=dev,
        dtype=torch.float64,
        verbose=False,
        diagnostic_mode=False,
    )
    hist = out["history"]
    required = ("loss", "dist_T", "dist_raw", "num_skips", "mean_step_norm", "mean_row_mismatch")
    for key in required:
        if key not in hist:
            raise AssertionError(f"Missing history key: {key}")
    if len(hist["loss"]) != 25:
        raise AssertionError("History length mismatch.")
    d0 = hist["dist_T"][0]
    d_last = hist["dist_T"][-1]
    if not (d_last < d0):
        raise AssertionError("dist_T did not decrease over this synthetic run.")
    out_diag = alg_qrkm(
        A,
        y,
        x_true=None,
        T=1,
        init_num_power_iters=10,
        seed=99,
        device=dev,
        dtype=torch.float64,
        verbose=False,
        diagnostic_mode=True,
        diag_tol=1e-8,
    )
    _ = out_diag
    print("[verify_alg_qrkm] all checks passed.")


if __name__ == "__main__":
    run_all()
