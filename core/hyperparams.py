"""Single source of truth for algorithm hyperparameters.

Every code path reads its constants from here, so that the values reported in
Table I of the letter cannot drift apart from the values the experiments actually
use.

Values follow the manuscript table ``\\label{Table_parameters}``.  Parameters that
are only used by a method's *native* initializer (and are therefore inactive when
a shared spectral initialization is supplied) are listed separately in
``INACTIVE_INIT_PARAMS`` for documentation purposes only.
"""

from __future__ import annotations

from typing import Any, Dict

# --- active iteration parameters (manuscript Table I) ----------------------------

QRKM: Dict[str, Any] = {
    # Parameter-free geometric step 1/beta_l with beta_l = ||a_l||_2^2.
    "eps": 1e-12,
    "eps_beta": 1e-12,
}

QADMM: Dict[str, Any] = {
    "rho": 0.6,
    # Tikhonov term added to the normal matrix before the linear solve.
    "damping": 1e-10,
}

QRAF: Dict[str, Any] = {
    "beta": 5.0,
    "step_size": 6.0,  # mu in Table I
}

QARAF: Dict[str, Any] = {
    # Step size used throughout this letter; stable across every experiment reported.
    "eta": 1.6,
    "beta": 0.8,     # momentum weight (mu in the source table)
    "beta_i": 5.0,   # reweighting parameter inside the gradient (beta in the source table)
}

QWF: Dict[str, Any] = {
    # eta = eta_coeff * n / sum_k y_k
    "eta_coeff": 0.2,
}

QRWF: Dict[str, Any] = {
    "eta": 0.8,  # mu in Table I
}

QPAF: Dict[str, Any] = {
    "gamma": 0.5,
    "alpha": 2.0,
    "eta": 2.5,
}

# --- parameters that only affect a method's native initializer -------------------
# They are inactive in every comparison of this paper because all methods are given
# the same shared quaternion spectral initialization.
INACTIVE_INIT_PARAMS: Dict[str, Dict[str, Any]] = {
    "QRAF": {"|S|": "floor(3n/13)", "gamma": 0.5},
    "QPAF": {"gamma": 0.5},
    # Lower/upper thresholds of QRWF's own truncated spectral initializer,
    # S_in = (1/n) sum_k y_k a_k a_k^* 1{alpha_l * lambda_0 < y_k < alpha_u * lambda_0}.
    # QRWF itself uses no gradient truncation -- that is the defining property of
    # reshaped Wirtinger flow -- so these act only on the initialization.
    "QRWF": {"alpha_l": 1.0, "alpha_u": 5.0},
}

# --- shared experiment protocol constants ---------------------------------------

POWER_ITERS = 50          # K_init for the shared quaternion spectral initialization
STOP_ERR = 1e-5           # quotient-distance success threshold
EPOCH_BUDGET = 1500       # synthetic sweep budget T
IMAGE_EPOCH_BUDGET = 300  # image experiment budget T
BASE_SEED = 20260812      # master seed for every re-run in this round


def as_dict() -> Dict[str, Dict[str, Any]]:
    """Return every method block, for provenance dumps."""
    return {
        "QRKM": dict(QRKM),
        "QADMM": dict(QADMM),
        "QRAF": dict(QRAF),
        "QARAF": dict(QARAF),
        "QWF": dict(QWF),
        "QRWF": dict(QRWF),
        "QPAF": dict(QPAF),
        "_protocol": {
            "power_iters": POWER_ITERS,
            "stop_err": STOP_ERR,
            "epoch_budget": EPOCH_BUDGET,
            "image_epoch_budget": IMAGE_EPOCH_BUDGET,
            "base_seed": BASE_SEED,
        },
        "_inactive_init_params": INACTIVE_INIT_PARAMS,
    }
