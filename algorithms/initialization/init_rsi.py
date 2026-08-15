import torch
from core.q_funcs import q_max_eign, q_star_conj, q_mat_mul


def init_rsi(
    A: torch.Tensor,
    y: torch.Tensor,
    *,
    device: str = "cuda",
    max_iter: int = 50,
    tol: float = 1e-5,
    gamma: float = 0.8,
) -> torch.Tensor:
    """
    Reweighted Spectral Initialization (Quaternion version).

    Implements the initialization step of Reweighted Amplitude Flow (RAF):
        - Select subset S of indices with largest |y_i|
        - Define weights w_i^0 = y_i^γ if i ∈ S else 0
        - Form S_in = (1/m) Σ w_i^0 (a_i a_i^*)
        - z0 = sqrt( (1/m) Σ y_i^2 ) * v_max(S_in)

    Parameters
    ----------
    A : torch.Tensor, shape (m, d, 4)
        Measurement vectors a_i ∈ ℍ^d stored as (d,4).
    y : torch.Tensor, shape (m,)
        Observations (usually |⟨a_i,x⟩|). Must be nonnegative.
    gamma : float, default=0.5
        Exponent in weight definition.

    Returns
    -------
    z0 : torch.Tensor, shape (d, 4)
        Initial guess in quaternion domain.
    """

    # Move data to device
    A = A.to(device)
    y = y.to(device)
    n, d, _ = A.shape

    # Cardinality |S| ≈ 3m/13 (per original RAF paper)
    S_size = int(3 * n / 13)

    # Indices of largest y_i
    top_idx = torch.argsort(y, descending=True)[:S_size]

    # Initialize weights: w_i^0 = y_i^γ if i ∈ S, else 0
    weights = torch.zeros_like(y, dtype=A.dtype, device=device)
    weights[top_idx] = y[top_idx] ** gamma

    # Build matrix S_in = (1/m) Σ w_i^0 a_i a_i^*
    S_in = torch.zeros((d, d, 4), dtype=A.dtype, device=device)
    for i in range(n):
        ai_row = A[i].unsqueeze(0)      # (1,d,4)
        ai_col = q_star_conj(ai_row)    # (d,1,4)
        outer = q_mat_mul(ai_col, ai_row)  # (d,d,4)
        S_in += weights[i] * outer

    S_in = (1.0 / n) * S_in

    # Leading eigenvector v_max(S_in)
    _, v_max = q_max_eign(S_in, N=max_iter, err=tol)

    # Scale: λ0 = sqrt( (1/m) Σ y_i^2 )
    lam0 = torch.sqrt((y.pow(2)).mean())

    # Final initialization
    z0 = lam0 * v_max
    return z0
