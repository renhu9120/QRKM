from typing import List, Optional

import torch

from core.q_funcs import q_sep_norm, q_star_conj, q_arr_dist, q_pfe, q_mat_mul, q_proj, phase_factor_estimate



def grad_qraf(
        A: torch.Tensor,
        psi: torch.Tensor,  # now allowed: (n, 1) or (n,)
        z0: torch.Tensor,
        T: int,
        *,
        beta: float = 5.0,
        step_size: float = 6.0,
        device: Optional[str] = None,
        # optional early stopping vs ground-truth
        x_true: Optional[torch.Tensor] = None,
        stop_err: float = 0.0,
        eps: float = 1e-12,
        use_pfe: bool = False,
) -> List[torch.Tensor]:
    """
    Quaternion Reweighted Amplitude Flow (QRAF) — gradient part.

    Performs T steps of:
        ∇ℓ_rw(z) = (1/n) Σ_k ω_k^(t) * (1 - ψ_k / |a_k^* z|) * (a_k a_k^*) z,
        z_{t+1}  = z_t - η ∇ℓ_rw(z_t),

    with
        ratio_k = |a_k^* z| / ψ_k,
        ω_k^(t) = ratio_k / (ratio_k + β).

    Shapes
    ------
    A   : (n, d, 4)
    psi : (n, 1) or (n,)    <-- accepted; internally converted to (n,)
    z0  : (d, 1, 4)
    Returns a list of (d, 1, 4) tensors.
    """
    # device / dtype
    if device is None:
        device = A.device
    A = A.to(device)
    psi = psi.to(device)
    z = z0.to(device)

    n, d, _ = A.shape

    # --- normalize psi shape to (n,) ---
    if psi.dim() == 2 and psi.shape[1] == 1:
        psi_vec = psi.squeeze(1)  # (n,)
    elif psi.dim() == 1 and psi.shape[0] == n:
        psi_vec = psi  # (n,)
    else:
        raise ValueError(f"`psi` must be (n,1) or (n,), got {tuple(psi.shape)}")

    # Precompute A* once
    A_star = q_star_conj(A)  # (n, d, 4)

    traj: List[torch.Tensor] = [z.clone()]

    for _ in range(T - 1):
        # 1) forward projection: α_k^* z_t -> (n,1,4)
        alpha_star_z = q_mat_mul(A, z)

        # 2) magnitudes |α_k^* z_t| -> (n,)
        alpha_norm = q_sep_norm(alpha_star_z).squeeze(-1).clamp_min(eps)

        # 3) weights and mismatch factor
        ratio = alpha_norm / psi_vec.clamp_min(eps)  # (n,)
        omega_t = ratio / (ratio + beta)  # (n,)
        coef = omega_t * (1.0 - psi_vec / alpha_norm)  # (n,)

        # broadcast onto quaternion residuals: (n,1,4)
        coef_q = coef.view(n, 1, 1) * alpha_star_z

        # 4) gradient: (1/n) Σ coef_k * (a_k a_k^*) z
        grad = q_mat_mul(A_star, coef_q)  # (d,1,4)
        grad = (1.0 / n) * grad

        # 5) update
        z_next = z - step_size * grad
        if use_pfe:
            z_next = phase_factor_estimate(z_next)
        traj.append(z_next)

        # 6) optional early stop w.r.t. ground-truth
        if stop_err != 0 and x_true is not None:
            if q_arr_dist(z_next, x_true) < stop_err:
                break

        z = z_next

    return traj


