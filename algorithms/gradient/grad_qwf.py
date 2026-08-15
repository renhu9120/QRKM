from typing import List, Optional

import torch

from core.q_funcs import q_sep_norm, q_star_conj, q_arr_dist, q_pfe, q_mat_mul, q_proj, q_scaarr_mul, phase_factor_estimate
from core.hyperparams import QWF as HP_QWF


# def grad_qraf(
#         A_mat: torch.Tensor,
#         y_k: torch.Tensor,
#         z_0: torch.Tensor,
#         T: int,
#         x_arr: Optional[torch.Tensor] = None,
#         err: float = 0.0,
#         term_flag: bool = False
# ) -> List[torch.Tensor]:
#     """
#     Mini-batch QIRAF gradient descent algs for quaternionic phase retrieval.
#
#     This function performs T-step gradient descent on quaternion measurements using
#     reweighted amplitude flow with incremental mini-batches.
#
#     Parameters:
#         A_mat (Tensor): Measurement matrix of shape (n, d, 4), quaternion-valued
#         y_k   (Tensor): Amplitude measurements of shape (n,)
#         z_0   (Tensor): Initial estimate vector of shape (d, 1, 4)
#         T     (int): Number of total iterations
#         x_arr (Tensor, optional): Ground truth signal for optional early stopping
#         err   (float): Error tolerance for early stopping
#         term_flag (bool): Whether to enable early stopping based on distance to x_arr
#
#     Returns:
#         z_arr (List[Tensor]): List of estimates at each iteration, length <= T
#     """
#     device = A_mat.device
#     n = A_mat.shape[0]
#
#     # Hyperparameters for update rule
#     beta_i = 5.0
#     eta = 6.0
#
#     A_star = q_star_conj(A_mat)  # (b, d, 4)
#
#     # Initialize solution path list
#     z_arr = [z_0.clone()]
#     for t in range(T - 1):
#
#         # === 2. Forward projection: alpha_t = A_mat @ z ===
#         alpha_star_z = q_mat_mul(A_mat, z_arr[-1])  # (b, 1, 4)
#         alpha_norm = q_sep_norm(alpha_star_z).squeeze(-1)  # (b,)
#
#         # === 3. Compute reweighting factors ===
#         ratio = alpha_norm / y_k.view(-1)  # (b,)
#         omega_t = ratio / (ratio + beta_i)  # (b,)
#         coef = omega_t * (1 - y_k.view(-1) / alpha_norm)  # (b,)
#         coef_expand = coef.unsqueeze(1)  # (b, 1)
#
#         # === 4. Scale quaternion residuals ===
#         coef_q = coef_expand.unsqueeze(-1) * alpha_star_z  # (b, 1, 4)
#
#         # === 5. Compute gradient direction ===
#         Sf_in = q_mat_mul(A_star, coef_q)  # (d, 1, 4)
#
#         # === 6. Gradient step ===
#         delta = (-eta / n) * Sf_in
#         z_next = z_arr[-1] + delta
#         z_arr.append(z_next)
#
#         # === 7. Early stopping (if enabled) ===
#         if term_flag and x_arr is not None:
#             if q_arr_dist(z_next, x_arr) < err:
#                 break
#
#     return z_arr
#
#
# def grad_pqraf(
#         A_mat: torch.Tensor,
#         y_k: torch.Tensor,
#         z_0: torch.Tensor,
#         T: int,
#         x_arr: Optional[torch.Tensor] = None,
#         err: float = 0.0,
#         term_flag: bool = False
# ) -> List[torch.Tensor]:
#     """
#     Mini-batch QIRAF gradient descent algs for quaternionic phase retrieval.
#
#     This function performs T-step gradient descent on quaternion measurements using
#     reweighted amplitude flow with incremental mini-batches.
#
#     Parameters:
#         A_mat (Tensor): Measurement matrix of shape (n, d, 4), quaternion-valued
#         y_k   (Tensor): Amplitude measurements of shape (n,)
#         z_0   (Tensor): Initial estimate vector of shape (d, 1, 4)
#         T     (int): Number of total iterations
#         batchsize_r (float): Ratio of measurements to use in each mini-batch (default: 0.3)
#         x_arr (Tensor, optional): Ground truth signal for optional early stopping
#         err   (float): Error tolerance for early stopping
#         term_flag (bool): Whether to enable early stopping based on distance to x_arr
#
#     Returns:
#         z_arr (List[Tensor]): List of estimates at each iteration, length <= T
#     """
#     device = A_mat.device
#     n = A_mat.shape[0]
#
#     # Hyperparameters for update rule
#     beta_i = 5.0
#     eta = 6.0
#
#     # Initialize solution path list
#     z_arr = [z_0.clone()]
#
#     for t in range(T - 1):
#
#         # === 2. Forward projection: alpha_t = A_mat @ z ===
#         alpha_star_z = q_mat_mul(A_mat, z_arr[-1])  # (b, 1, 4)
#         alpha_norm = q_sep_norm(alpha_star_z).squeeze(-1)  # (b,)
#
#         # === 3. Compute reweighting factors ===
#         ratio = alpha_norm / y_k.view(-1)  # (b,)
#         omega_t = ratio / (ratio + beta_i)  # (b,)
#         coef = omega_t * (1 - y_k.view(-1) / alpha_norm)  # (b,)
#         coef_expand = coef.unsqueeze(1)  # (b, 1)
#
#         # === 4. Scale quaternion residuals ===
#         coef_q = coef_expand.unsqueeze(-1) * alpha_star_z  # (b, 1, 4)
#
#         # === 5. Compute gradient direction ===
#         A_star = q_star_conj(A_mat)  # (b, d, 4)
#         Sf_in = q_mat_mul(A_star, coef_q)  # (d, 1, 4)
#
#         # === 6. Gradient step ===
#         delta = (-eta / n) * Sf_in
#         z_next = z_arr[-1] + delta
#
#         if (t + 1) % 5 == 0:
#             z_next = q_pfe(z_next)
#
#         z_arr.append(z_next)
#
#         if term_flag and x_arr is not None:
#             if q_arr_dist(z_next, x_arr) < err:
#                 break
#
#     z_arr[-1] = q_pfe(z_arr[-1])
#     return z_arr
#
#
# def grad_pqraf_proj(
#         A_mat: torch.Tensor,
#         y_k: torch.Tensor,
#         z_0: torch.Tensor,
#         T: int,
#         x_arr: Optional[torch.Tensor] = None,
#         err: float = 0.0,
#         term_flag: bool = False
# ) -> List[torch.Tensor]:
#     """
#     Mini-batch QIRAF gradient descent algs for quaternionic phase retrieval.
#
#     This function performs T-step gradient descent on quaternion measurements using
#     reweighted amplitude flow with incremental mini-batches.
#
#     Parameters:
#         A_mat (Tensor): Measurement matrix of shape (n, d, 4), quaternion-valued
#         y_k   (Tensor): Amplitude measurements of shape (n,)
#         z_0   (Tensor): Initial estimate vector of shape (d, 1, 4)
#         T     (int): Number of total iterations
#         batchsize_r (float): Ratio of measurements to use in each mini-batch (default: 0.3)
#         x_arr (Tensor, optional): Ground truth signal for optional early stopping
#         err   (float): Error tolerance for early stopping
#         term_flag (bool): Whether to enable early stopping based on distance to x_arr
#
#     Returns:
#         z_arr (List[Tensor]): List of estimates at each iteration, length <= T
#     """
#     device = A_mat.device
#     n = A_mat.shape[0]
#
#     # Hyperparameters for update rule
#     beta_i = 5.0
#     eta = 6.0
#
#     # Initialize solution path list
#     z_arr = [z_0.clone()]
#
#     for t in range(T - 1):
#
#         # === 2. Forward projection: alpha_t = A_mat @ z ===
#         alpha_star_z = q_mat_mul(A_mat, z_arr[-1])  # (b, 1, 4)
#         alpha_norm = q_sep_norm(alpha_star_z).squeeze(-1)  # (b,)
#
#         # === 3. Compute reweighting factors ===
#         ratio = alpha_norm / y_k.view(-1)  # (b,)
#         omega_t = ratio / (ratio + beta_i)  # (b,)
#         coef = omega_t * (1 - y_k.view(-1) / alpha_norm)  # (b,)
#         coef_expand = coef.unsqueeze(1)  # (b, 1)
#
#         # === 4. Scale quaternion residuals ===
#         coef_q = coef_expand.unsqueeze(-1) * alpha_star_z  # (b, 1, 4)
#
#         # === 5. Compute gradient direction ===
#         A_star = q_star_conj(A_mat)  # (b, d, 4)
#         Sf_in = q_mat_mul(A_star, coef_q)  # (d, 1, 4)
#
#         # === 6. Gradient step ===
#         delta = (-eta / n) * Sf_in
#         z_next = z_arr[-1] + delta
#
#         if (t + 1) % 5 == 0:
#             z_next = q_proj(z_next)
#
#         z_arr.append(z_next)
#
#         if term_flag and x_arr is not None:
#             if q_arr_dist(z_next, x_arr) < err:
#                 break
#
#     z_arr[-1] = q_proj(z_arr[-1])
#     return z_arr


def grad_qwf(
        A: torch.Tensor,
        psi: torch.Tensor,  # (n,1) or (n,)
        z0: torch.Tensor,
        T: int,
        *,
        device: Optional[str] = None,
        x_true: Optional[torch.Tensor] = None,
        stop_err: float = 0.0,
        eps: float = 1e-12,
        use_pfe: bool = False,
) -> List[torch.Tensor]:
    """
    Quaternion Wirtinger Flow (QWF) — gradient part.
    ------------------------------------------------
    Performs T steps of:
        ∇ℓ(z) = (1/n) Σ_k (|a_k^* z|^2 - ψ_k^2) * (a_k a_k^*) z,
        z_{t+1} = z_t - η ∇ℓ(z_t),

    with η = 0.2 * n / Σ ψ_k^2  (from WF paper, p.2873)

    Parameters
    ----------
    A       : (n, d, 4) quaternion measurement matrix
    psi     : (n,) or (n,1) magnitude measurements
    z0      : (d, 1, 4) initialization
    T       : number of iterations
    device  : optional device placement
    x_true  : optional ground truth for early stopping
    stop_err: distance threshold for early termination
    eps     : small constant for numerical stability

    Returns
    -------
    traj : list of (d, 1, 4) tensors representing iteration trajectory
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

    # learning rate
    eta = float(HP_QWF["eta_coeff"]) * n / torch.sum(psi_vec ** 2).clamp_min(eps)

    # precompute A*
    A_star = q_star_conj(A)

    traj: List[torch.Tensor] = [z.clone()]

    for _ in range(T - 1):
        # 1) α_k^* z_t  → (n,1,4)
        alpha_star_z = q_mat_mul(A, z)

        # 2) compute coef = |α_k^* z|^2 - ψ_k^2
        alpha_norm_sq = q_sep_norm(alpha_star_z).squeeze(-1) ** 2
        coef = alpha_norm_sq - psi_vec ** 2  # (n,)

        # 3) broadcast coef over quaternion structure
        coef_q = coef.view(n, 1, 1) * alpha_star_z  # (n,1,4)

        # 4) accumulate gradient (1/n) Σ_k coef_k * (a_k a_k^*) z
        Sf_in = q_mat_mul(A_star, coef_q)  # (d,1,4)
        grad = (1.0 / n) * Sf_in

        # 5) gradient descent update
        z_next = z - eta * grad
        if use_pfe:
            z_next = phase_factor_estimate(z_next)
        traj.append(z_next)

        # 6) optional early stop
        if stop_err != 0 and x_true is not None:
            if q_arr_dist(z_next, x_true) < stop_err:
                break

        z = z_next

    return traj


def grad_pqwf(
        A: torch.Tensor,
        psi: torch.Tensor,  # (n,1) or (n,)
        z0: torch.Tensor,
        T: int,
        *,
        device: Optional[str] = None,
        x_true: Optional[torch.Tensor] = None,
        stop_err: float = 0.0,
        eps: float = 1e-12,
) -> List[torch.Tensor]:
    """
    Quaternion Wirtinger Flow (QWF) — gradient part.
    ------------------------------------------------
    Performs T steps of:
        ∇ℓ(z) = (1/n) Σ_k (|a_k^* z|^2 - ψ_k^2) * (a_k a_k^*) z,
        z_{t+1} = z_t - η ∇ℓ(z_t),

    with η = 0.2 * n / Σ ψ_k^2  (from WF paper, p.2873)

    Parameters
    ----------
    A       : (n, d, 4) quaternion measurement matrix
    psi     : (n,) or (n,1) magnitude measurements
    z0      : (d, 1, 4) initialization
    T       : number of iterations
    device  : optional device placement
    x_true  : optional ground truth for early stopping
    stop_err: distance threshold for early termination
    eps     : small constant for numerical stability

    Returns
    -------
    traj : list of (d, 1, 4) tensors representing iteration trajectory
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

    # learning rate
    eta = float(HP_QWF["eta_coeff"]) * n / torch.sum(psi_vec ** 2).clamp_min(eps)

    # precompute A*
    A_star = q_star_conj(A)

    traj: List[torch.Tensor] = [z.clone()]

    for t in range(T - 1):
        # 1) α_k^* z_t  → (n,1,4)
        alpha_star_z = q_mat_mul(A, z)

        # 2) compute coef = |α_k^* z|^2 - ψ_k^2
        alpha_norm_sq = q_sep_norm(alpha_star_z).squeeze(-1) ** 2
        coef = alpha_norm_sq - psi_vec ** 2  # (n,)

        # 3) broadcast coef over quaternion structure
        coef_q = coef.view(n, 1, 1) * alpha_star_z  # (n,1,4)

        # 4) accumulate gradient (1/n) Σ_k coef_k * (a_k a_k^*) z
        Sf_in = q_mat_mul(A_star, coef_q)  # (d,1,4)
        grad = (1.0 / n) * Sf_in

        # 5) update
        z_next = z - eta * grad

        # to pure quaternion
        if (t + 1) % 5 == 0:
            z_next = q_pfe(z_next)

        traj.append(z_next)

        # 6) optional early stop w.r.t. ground-truth
        if stop_err != 0 and x_true is not None:
            if q_arr_dist(z_next, x_true) < stop_err:
                break

        z = z_next

        # make the final to pure quaternion
    traj[-1] = q_pfe(traj[-1])
    return traj


def grad_pjqwf(
        A: torch.Tensor,
        psi: torch.Tensor,  # (n,1) or (n,)
        z0: torch.Tensor,
        T: int,
        *,
        device: Optional[str] = None,
        x_true: Optional[torch.Tensor] = None,
        stop_err: float = 0.0,
        eps: float = 1e-12,
) -> List[torch.Tensor]:
    """
    Quaternion Wirtinger Flow (QWF) — gradient part.
    ------------------------------------------------
    Performs T steps of:
        ∇ℓ(z) = (1/n) Σ_k (|a_k^* z|^2 - ψ_k^2) * (a_k a_k^*) z,
        z_{t+1} = z_t - η ∇ℓ(z_t),

    with η = 0.2 * n / Σ ψ_k^2  (from WF paper, p.2873)

    Parameters
    ----------
    A       : (n, d, 4) quaternion measurement matrix
    psi     : (n,) or (n,1) magnitude measurements
    z0      : (d, 1, 4) initialization
    T       : number of iterations
    device  : optional device placement
    x_true  : optional ground truth for early stopping
    stop_err: distance threshold for early termination
    eps     : small constant for numerical stability

    Returns
    -------
    traj : list of (d, 1, 4) tensors representing iteration trajectory
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

    # learning rate
    eta = float(HP_QWF["eta_coeff"]) * n / torch.sum(psi_vec ** 2).clamp_min(eps)

    # precompute A*
    A_star = q_star_conj(A)

    traj: List[torch.Tensor] = [z.clone()]

    for _ in range(T - 1):
        # 1) α_k^* z_t  → (n,1,4)
        alpha_star_z = q_mat_mul(A, z)

        # 2) compute coef = |α_k^* z|^2 - ψ_k^2
        alpha_norm_sq = q_sep_norm(alpha_star_z).squeeze(-1) ** 2
        coef = alpha_norm_sq - psi_vec ** 2  # (n,)

        # 3) broadcast coef over quaternion structure
        coef_q = coef.view(n, 1, 1) * alpha_star_z  # (n,1,4)

        # 4) accumulate gradient (1/n) Σ_k coef_k * (a_k a_k^*) z
        Sf_in = q_mat_mul(A_star, coef_q)  # (d,1,4)
        grad = (1.0 / n) * Sf_in

        # 5) update
        z_next = z - eta * grad

        # to pure quaternion using projection method
        z_next = q_proj(z_next)

        traj.append(z_next)

        # 6) optional early stop w.r.t. ground-truth
        if stop_err != 0 and x_true is not None:
            if q_arr_dist(z_next, x_true) < stop_err:
                break

        z = z_next

    return traj
