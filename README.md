# QRKM — A Randomized Quaternionic Kaczmarz Method for Phase Retrieval

Reference implementation and synthetic experiments for the letter

> R. Hu, P. Lian, Z. Liu, *A Randomized Quaternionic Kaczmarz Method for Phase
> Retrieval*.

QRKM recovers a quaternion signal `x ∈ H^d` from phaseless measurements
`y_k = |a_k^* x|^2` by sweeping the rows of `A` in a random, energy-weighted order
and applying one normalized Kaczmarz update per row,

```
z ← z + (1 / β_ℓ) · a_ℓ (ψ_ℓ φ_ℓ − u_ℓ),   u_ℓ = a_ℓ^* z,  φ_ℓ = u_ℓ/|u_ℓ|,  β_ℓ = ‖a_ℓ‖²,
```

so the step size is fixed by the row geometry and no parameter has to be tuned.

## Contents

```
algorithms/
  algs/            QRKM and the six baselines (QADMM, QRAF, QARAF, QWF, QRWF, QPAF)
  gradient/        the per-method update rules
  initialization/  quaternion spectral initializers, including the shared one of Eq. (3)
core/              quaternion arithmetic, sampling, metrics, and the hyperparameter table
utils/             device/dtype helpers, seeding, logging, plotting
checks/            standalone correctness checks for the quaternion kernels
experiments/       drivers for the synthetic figures
results/           the measured data behind the figures and tables of the letter
```

All tensors are Hamilton quaternions stored as real arrays of shape `(..., 4)`, and
everything runs in `torch.float64`.

## Requirements

```
python >= 3.10
torch >= 2.1        # CUDA optional; every script also runs on CPU
numpy >= 1.24
matplotlib >= 3.7
```

```bash
pip install -r requirements.txt
```

## Reproducing the synthetic results

Run from the repository root.

**Fig. 1 — convergence on one instance** (`d = 100`, `n/d = 8`, `T = 400`):

```bash
python experiments/run_convergence.py
```

Writes `output/compare_pr_single_trial_curves.csv` and the figure. The measured
curves reported in the letter are in `results/fig1_convergence_curves.csv`.

**Fig. 2 — success rate and mean epochs against `n/d`:**

```bash
python experiments/plot_success_rate.py
```

Redraws both panels from the measured sweep in `results/success_rate_d100/`.

**Correctness checks** for the quaternion kernels and for QRKM itself:

```bash
python checks/verify_quat_ops.py
python checks/verify_quat_real_embed.py
python checks/verify_quat_pr_loss.py
python checks/verify_quat_spectral_init.py
python checks/verify_alg_qrkm.py
```

## Experimental protocol

Fixed in `core/hyperparams.py`, which is the single source of the values in Table I:

| | |
|---|---|
| shared initialization | intensity-based quaternion spectral estimator, Eq. (3), `K_init = 50` power iterations |
| success criterion | quotient distance `d_T(z, x*) ≤ 1e-5` |
| epoch budget | `1500` (synthetic) |
| master seed | `20260812` |

Every compared method is started from the **same** initialization on the **same**
measurement instance, computed once per instance and passed to each solver, so the
comparison isolates the iterative update. The data stream and the initializer stream
are seeded separately, so nothing a solver draws can shift the data another solver
sees.

## Measured data

`results/` holds the numbers behind the figures and tables, exactly as produced:

| file | used for |
|---|---|
| `fig1_convergence_curves.csv` | Fig. 1 |
| `success_rate_d100/sr_*.csv` | Fig. 2 and the `d = 100` block of Table II |
| `table2_d64/sr_*.csv` | the `d = 64` block of Table II |
| `*/provenance.json` | seed, epoch budget, hyperparameters, package versions, device |
| `success_rate_d100/fingerprints.json` | SHA-256 digests of `A`, `x*`, `y` and `z0` for every batch |
| `*_qaraf_resweep.json` | provenance and fingerprints of the QARAF sweep, which was run separately |

The fingerprints certify that all seven methods were run on byte-identical inputs at
every node of the sweep, so the comparison is paired rather than merely
same-distribution.

The sweep is evaluated in decreasing `n/d` and exits at the first ratio with zero
successes; ratios below that point were not evaluated and no row is written for them.

## Scope of this release

This repository contains the reference implementation of QRKM and the synthetic
experiments. Two parts of the study are not included: the batched GPU runner used to
collect the sweep statistics and the wall-clock timings, and the image-reconstruction
pipeline of Table III. The measured outputs of the sweep are provided in `results/`,
together with the seeds, hyperparameters and input fingerprints needed to check them.

## License

MIT — see `LICENSE`.
