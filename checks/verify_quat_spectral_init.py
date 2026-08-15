"""Verify spectral initializer shapes and typical error vs random init."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from core.quat_metrics import quat_dist_right_phase
from core.quat_sampling import make_noiseless_measurements, make_quaternion_gaussian_matrix, make_random_quaternion_signal
from algorithms.initialization.quat_spectral_init import quat_spectral_init
from utils.common import pick_device


def run_all() -> None:
    dev = pick_device("cuda" if torch.cuda.is_available() else "cpu")
    d = 20
    n = 80
    g = torch.Generator(device=dev)
    g.manual_seed(7)
    A = make_quaternion_gaussian_matrix(n, d, device=dev, dtype=torch.float64)
    x_true = make_random_quaternion_signal(d, normalize=True, device=dev, dtype=torch.float64)
    y = make_noiseless_measurements(A, x_true)
    x0 = quat_spectral_init(A, y, num_power_iters=60, seed=999, eps=1e-12, device=dev, dtype=torch.float64, verbose=False)
    if x0.shape != (d, 4) or x0.dtype != torch.float64 or x0.device != dev:
        raise AssertionError("Bad spectral init tensor metadata.")
    x_rand = torch.randn(d, 4, dtype=torch.float64, device=dev, generator=g)
    x_rand = x_rand / (torch.sqrt(torch.sum(x_rand * x_rand)) + 1e-12)
    wins = 0
    trials = 8
    for t in range(trials):
        g.manual_seed(100 + t)
        x_rand = torch.randn(d, 4, dtype=torch.float64, device=dev, generator=g)
        x_rand = x_rand / (torch.sqrt(torch.sum(x_rand * x_rand)) + 1e-12)
        d_spec = float(quat_dist_right_phase(x0, x_true).item())
        d_rand = float(quat_dist_right_phase(x_rand, x_true).item())
        if d_spec < d_rand:
            wins += 1
    if wins < 5:
        raise AssertionError("Spectral init did not beat random majority in this configuration.")
    print("[verify_quat_spectral_init] all checks passed.")


if __name__ == "__main__":
    run_all()
