from typing import Optional, List

import torch

from core.q_funcs import q_sep_norm, q_star_conj, q_arr_dist, q_mat_mul, q_pfe, phase_factor_estimate


def grad_qaraf(
        A: torch.Tensor,
        psi_meas: torch.Tensor,  # (n,) or (n,1)
        z0: torch.Tensor,
        T: int,
        *,
        eta: float = 6.0,
        beta_i: float = 5.0,
        beta: float = 0.8,
        device: Optional[str] = None,
        x_true: Optional[torch.Tensor] = None,
        stop_err: float = 0.0,
        eps: float = 1e-12,
        use_pfe: bool = False,
) -> List[torch.Tensor]:
    """
    Quaternion Accelerated Reweighted Amplitude Flow (QARAF)

    Update (gradient part):
        ratio_k = |a_k^* ψ_t| / y_k
        ω_k = ratio_k / (ratio_k + β_i)
        coef_k = ω_k * (1 - y_k / |a_k^* ψ_t|)

        grad_t = (1/n) Σ coef_k * (a_k a_k^*) ψ_t

        z_{t+1}   = ψ_t - η * grad_t
        ψ_{t+1}   = z_{t+1} + β (z_{t+1} - z_t)

    Parameters
    ----------
    A           : (n, d, 4)
    psi_meas    : (n,) or (n,1)    measurement magnitudes y_k
    z0          : (d,1,4)
    T           : total iterations
    eta         : step size for gradient
    beta_i      : reweighting parameter
    beta        : acceleration coefficient
    x_true      : optional ground truth for early stopping
    stop_err    : distance threshold

    Returns
    -------
    traj : list of z_t for all iterations
    """

    if device is None:
        device = A.device

    A = A.to(device)
    psi_meas = psi_meas.to(device)
    z = z0.to(device)

    n, d, _ = A.shape

    # normalize psi → (n,)
    if psi_meas.dim() == 2 and psi_meas.shape[1] == 1:
        y_vec = psi_meas.squeeze(1)
    elif psi_meas.dim() == 1:
        y_vec = psi_meas
    else:
        raise ValueError("`psi_meas` must be (n,) or (n,1).")

    # Precompute A*
    A_star = q_star_conj(A)

    # store all z_t
    traj: List[torch.Tensor] = [z.clone()]

    # acceleration variable ψ_t
    psi_t = z.clone()

    for _ in range(T - 1):

        # 1) α_k^* ψ_t  → (n,1,4)
        alpha_star_psi = q_mat_mul(A, psi_t)

        # |α_k^* ψ_t|
        alpha_norm = q_sep_norm(alpha_star_psi).squeeze(-1).clamp_min(eps)

        # 2) reweighting coefficient ω_t
        ratio = alpha_norm / y_vec.clamp_min(eps)
        omega_t = ratio / (ratio + beta_i)

        # 3) coef = ω_k (1 - y_k / |α|)
        coef = omega_t * (1.0 - y_vec / alpha_norm)  # (n,)

        # broadcast coef onto quaternion tensor
        coef_q = coef.view(n, 1, 1) * alpha_star_psi  # (n,1,4)

        # 4) gradient = (1/n) Σ coef_k (a_k a_k^*) ψ_t
        Sf_in = q_mat_mul(A_star, coef_q)  # (d,1,4)
        grad = (1.0 / n) * Sf_in

        # 5) update z_{t+1}
        z_next = psi_t - eta * grad
        if use_pfe:
            z_next = phase_factor_estimate(z_next)
        traj.append(z_next)

        # 6) acceleration: ψ_{t+1} = z_{t+1} + β (z_{t+1} - z_t)
        psi_t = z_next + beta * (z_next - z)

        # shift z
        z = z_next

        # 7) optional early stopping
        if stop_err != 0 and x_true is not None:
            if q_arr_dist(z_next, x_true) < stop_err:
                break

    return traj
