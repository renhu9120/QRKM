from typing import List, Optional

import torch

from core.q_funcs import q_arr_dist, q_mat_mul, q_sep_norm, q_star_conj, phase_factor_estimate


def grad_qpaf(
    A: torch.Tensor,
    psi: torch.Tensor,
    z0: torch.Tensor,
    T: int,
    *,
    alpha: float = 2.0,
    eta: float = 2.5,
    device: Optional[str] = None,
    x_true: Optional[torch.Tensor] = None,
    stop_err: float = 0.0,
    eps: float = 1e-12,
    use_pfe: bool = False,
) -> List[torch.Tensor]:
    """
    Quaternion Phase Amplitude Flow (QPAF) — gradient refinement.

    Performs T steps of smoothed amplitude flow:

        ε_k = sqrt(α) ψ_k,
        c_k = 1 - sqrt(ψ_k^2 + ε_k^2) / sqrt(|a_k^* z|^2 + ε_k^2),
        ∇ℓ(z) = (1/n) Σ_k c_k (a_k a_k^*) z,
        z_{t+1} = z_t - η ∇ℓ(z_t).

    Parameters
    ----------
    A : torch.Tensor, shape (n, d, 4)
        Measurement matrix in the ``A_model`` convention.
    psi : torch.Tensor, shape (n,) or (n, 1)
        Amplitude measurements |a_k^* x|.
    z0 : torch.Tensor, shape (d, 1, 4)
        Initial iterate.
    T : int
        Number of iterates recorded (including z0); runs T-1 update steps.
    alpha : float, default=2.0
        Smoothing parameter (theoretical range roughly (0.37, 29)).
    eta : float, default=2.5
        Gradient step size multiplier (gradient already includes 1/n).

    Returns
    -------
    traj : list of (d, 1, 4) tensors
    """
    if device is None:
        device = A.device

    A = A.to(device)
    psi = psi.to(device)
    z = z0.to(device)

    n, _, _ = A.shape
    if psi.dim() == 2 and psi.shape[1] == 1:
        psi_vec = psi.squeeze(1)
    elif psi.dim() == 1 and psi.shape[0] == n:
        psi_vec = psi
    else:
        raise ValueError(f"`psi` must be (n,1) or (n,), got {tuple(psi.shape)}")

    sqrt_alpha = float(alpha) ** 0.5
    epsilon = sqrt_alpha * psi_vec
    eps_sq = epsilon * epsilon

    A_star = q_star_conj(A)
    traj: List[torch.Tensor] = [z.clone()]

    for _ in range(T - 1):
        alpha_star_z = q_mat_mul(A, z)
        alpha_norm = q_sep_norm(alpha_star_z).squeeze(-1).clamp_min(eps)

        num = torch.sqrt(psi_vec.pow(2) + eps_sq)
        den = torch.sqrt(alpha_norm.pow(2) + eps_sq)
        coef = 1.0 - num / den

        coef_q = coef.view(n, 1, 1) * alpha_star_z
        grad = q_mat_mul(A_star, coef_q)
        grad = (1.0 / n) * grad

        z_next = z - float(eta) * grad
        if use_pfe:
            z_next = phase_factor_estimate(z_next)
        traj.append(z_next)

        if stop_err != 0 and x_true is not None:
            if q_arr_dist(z_next, x_true) < stop_err:
                break

        z = z_next

    return traj
