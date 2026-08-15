from typing import List, Optional

import torch

from core.q_funcs import q_sep_norm, q_star_conj, q_arr_dist, q_mat_mul, q_pfe, phase_factor_estimate


def grad_qrwf(
        A: torch.Tensor,
        psi: torch.Tensor,  # (n,1) or (n,)
        z0: torch.Tensor,
        T: int,
        *,
        eta: float = 0.8,
        device: Optional[str] = None,
        x_true: Optional[torch.Tensor] = None,
        stop_err: float = 0.0,
        eps: float = 1e-12,
        use_pfe: bool = False,
) -> List[torch.Tensor]:
    """
    Quaternion Reweighted Wirtinger Flow (QRWF) — gradient part.
    ------------------------------------------------------------
    Iterative update:
        ∇ℓ_rw(z) = (1/n) Σ_k (1 - ψ_k / |a_k^* z|) * (a_k a_k^*) z,
        z_{t+1} = z_t - η ∇ℓ_rw(z_t)

    Parameters
    ----------
    A       : (n, d, 4) quaternion measurement matrix
    psi     : (n,) or (n,1) magnitude measurements
    z0      : (d, 1, 4) initialization
    T       : number of iterations
    eta     : step size (default = 0.8)
    device  : optional CUDA / CPU device
    x_true  : optional ground truth (for early stop)
    stop_err: early stop threshold
    eps     : numerical stability constant

    Returns
    -------
    traj : list of (d, 1, 4) tensors (iteration trajectory)
    """
    if device is None:
        device = A.device

    A = A.to(device)
    psi = psi.to(device)
    z = z0.to(device)

    n, d, _ = A.shape
    if psi.dim() == 2 and psi.shape[1] == 1:
        psi_vec = psi.squeeze(1)
    elif psi.dim() == 1:
        psi_vec = psi
    else:
        raise ValueError(f"`psi` must be (n,) or (n,1), got {psi.shape}")

    A_star = q_star_conj(A)  # (n, d, 4)
    traj: List[torch.Tensor] = [z.clone()]

    for _ in range(T - 1):
        # 1) forward projection α_k^* z_t
        alpha_star_z = q_mat_mul(A, z)  # (n,1,4)

        # 2) compute |α_k^* z_t|
        alpha_norm = q_sep_norm(alpha_star_z).squeeze(-1).clamp_min(eps)  # (n,)

        # 3) compute coef = 1 - ψ_k / |α_k^* z|
        coef = 1.0 - psi_vec / alpha_norm  # (n,)

        # 4) broadcast coef onto quaternion residuals
        coef_q = coef.view(n, 1, 1) * alpha_star_z  # (n,1,4)

        # 5) accumulate gradient: (1/n) Σ_k coef_k * (a_k a_k^*) z
        Sf_in = q_mat_mul(A_star, coef_q)  # (d,1,4)
        grad = (1.0 / n) * Sf_in

        # 6) gradient descent update
        z_next = z - eta * grad
        if use_pfe:
            z_next = phase_factor_estimate(z_next)
        traj.append(z_next)

        # 7) optional early stop
        if stop_err != 0 and x_true is not None:
            if q_arr_dist(z_next, x_true) < stop_err:
                break

        z = z_next

    return traj
