import torch

from core.q_funcs import q_max_eign, q_mat_mul, q_star_conj


def init_qpaf(
    A: torch.Tensor,
    y: torch.Tensor,
    *,
    device: str = "cuda",
    max_iter: int = 50,
    tol: float = 1e-5,
    gamma: float = 0.5,
) -> torch.Tensor:
    """
    QPAF spectral initialization (Quaternion Phase Amplitude Flow).

    Implements ``Init_QPAF`` from the QPAF reference:

        λ0 = sqrt(mean(y_k^2)),
        w_k = γ - exp(-y_k^2 / λ0^2),
        S_in = (1/n) Σ_k w_k (a_k a_k^*),
        z0 = λ0 * v_max(S_in),

    where v_max is the leading eigenvector of S_in obtained by the quaternion
    power method. Weights w_k may be negative and are not clipped.

    Parameters
    ----------
    A : torch.Tensor, shape (n, d, 4)
        Measurement vectors in the ``A_model = q_conj(A_raw)`` convention used
        by ``alg_qpaf`` (rows store conjugated measurement vectors).
    y : torch.Tensor, shape (n,)
        Amplitude observations |a_k^* x|, real and nonnegative.
    gamma : float, default=0.5
        Weight offset in w_k = γ - exp(-y_k^2 / λ0^2).
    max_iter : int, default=50
        Power-method iterations for the leading eigenvector.
    tol : float, default=1e-5
        Convergence tolerance for the eigenvector computation.

    Returns
    -------
    z0 : torch.Tensor, shape (d, 4)
        Initial estimate in quaternion form.
    """
    A = A.to(device)
    y = y.to(device)
    n, d, _ = A.shape

    lam0 = torch.sqrt(y.pow(2).mean())
    lam0_sq = lam0 * lam0
    weights = gamma - torch.exp(-y.pow(2) / lam0_sq)

    S_in = torch.zeros((d, d, 4), dtype=A.dtype, device=device)
    for k in range(n):
        ak_row = A[k].unsqueeze(0)
        ak_col = q_star_conj(ak_row)
        outer = q_mat_mul(ak_col, ak_row)
        S_in = S_in + weights[k] * outer

    S_in = (1.0 / n) * S_in
    _, v_in = q_max_eign(S_in, N=max_iter, err=tol)
    z0 = lam0 * v_in
    if z0.dim() == 3 and z0.shape[1] == 1:
        z0 = z0.squeeze(1)
    return z0
