import torch
from core.q_funcs import q_star_conj, q_max_eign, q_mat_mul, q_scaarr_mul


def init_si(A: torch.Tensor,
            y: torch.Tensor,
            device: str = "cuda",
            max_iter: int = 50,
            tol: float = 1e-5) -> torch.Tensor:
    """
    Quaternion Spectral Initialization for Phase Retrieval (Original Algorithm).

    Implements Algorithm 1 (Spectral Initialization) in the quaternion setting:

        1. Construct S_in = (1/n) Σ_{k=1}^n y_k * (α_k α_k^*).
        2. Compute the leading eigenvector v_in of S_in.
        3. Compute λ0 = sqrt((1/n) Σ y_k).
        4. Set z0 = λ0 * v_in.

    Parameters
    ----------
    A : torch.Tensor
        Quaternion measurement matrix of shape (n, d, 4),
        where A[k] is the k-th measurement vector α_k ∈ ℍ^d.
        - n : number of measurements
        - d : signal dimension
    y : torch.Tensor
        Measurement magnitudes of shape (n,), real and non-negative.
    device : str, optional
        Computation device ('cuda' or 'cpu'). Default: 'cuda'.
    max_iter : int, optional
        Maximum iterations for the eigenvector power method. Default: 50.
    tol : float, optional
        Convergence tolerance for eigenvector computation. Default: 1e-5.

    Returns
    -------
    z0 : torch.Tensor
        Spectral initialization vector of shape (d, 4), quaternion form.
    """

    # Move to device
    A = A.to(device)
    y = y.to(device)
    n, d, _ = A.shape

    # --- Step 1: Construct S_in ---
    S_in = torch.zeros((d, d, 4), dtype=A.dtype, device=device)
    for k in range(n):
        ak_star = A[k].unsqueeze(0)  # (1,d,4)，行向量
        ak = q_star_conj(ak_star)  # (d,1,4)，列向量
        outer = q_mat_mul(ak, ak_star)  # (d,d,4)
        S_in = S_in + y[k] * outer

    # Normalize
    S_in = (1.0 / n) * S_in

    # Step 2: Leading eigenvector of S_in
    _, v_in = q_max_eign(S_in, N=max_iter, err=tol)

    # Step 3: Compute λ0 = sqrt(mean(y_k))
    lambda_0 = torch.sqrt(y.mean())

    # Step 4: Spectral initialization
    z0 = lambda_0 * v_in

    return z0
