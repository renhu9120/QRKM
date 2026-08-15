from typing import Tuple

import torch
from torch import Tensor
from typing import Optional, Tuple

from core.quat_ops import quat_mul


def q_conj(q: torch.Tensor) -> torch.Tensor:
    """四元数共轭，q[...,4] = [r,i,j,k] -> [r,-i,-j,-k]"""
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def q_sep_norm(q: torch.Tensor, eps: float = 0) -> torch.Tensor:
    """
    Point-wise quaternion norm

    Parameters:
        q: a quaternion object that can be a single quaternion number, vector or a matrix

    Returns:
        return the separate norm of each element in q
        :param q:
        :param eps:
    """
    return torch.sqrt(torch.sum(q ** 2, dim=-1) + eps)


def q_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Product of two quaternions.

    Parameters:
        q1: quaternion number
        q2: quaternion number

    Returns:
        product of q1 and q2
    """
    a1, b1, c1, d1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    a2, b2, c2, d2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    return torch.stack([
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 + c1 * a2 + d1 * b2 - b1 * d2,
        a1 * d2 + d1 * a2 + b1 * c2 - c1 * b2
    ], dim=-1)


def q_mul_matrix(Q1: torch.Tensor, Q2: torch.Tensor) -> torch.Tensor:
    """
    Perform element-wise quaternion multiplication between two quaternion matrices.

    Parameters
    ----------
    Q1 : torch.Tensor
        Quaternion matrix of shape (..., 4) where the last dimension is [r, i, j, k].
    Q2 : torch.Tensor
        Quaternion matrix of shape (..., 4), same shape as Q1.

    Returns
    -------
    torch.Tensor
        Element-wise quaternion product, shape (..., 4).
    """
    if Q1.shape != Q2.shape:
        raise ValueError(f"Shape mismatch: Q1.shape={Q1.shape}, Q2.shape={Q2.shape}")
    if Q1.shape[-1] != 4:
        raise ValueError("Last dimension must be 4 (quaternion components).")

    a1, b1, c1, d1 = Q1[..., 0], Q1[..., 1], Q1[..., 2], Q1[..., 3]
    a2, b2, c2, d2 = Q2[..., 0], Q2[..., 1], Q2[..., 2], Q2[..., 3]

    return torch.stack([
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 + c1 * a2 + d1 * b2 - b1 * d2,
        a1 * d2 + d1 * a2 + b1 * c2 - c1 * b2
    ], dim=-1)


def q_scaarr_mul(s: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Multiply a real scalar/vector with a quaternion matrix/tensor.

    Parameters
    ----------
    s : torch.Tensor
        Real tensor, can be:
        - scalar (0-dim),
        - vector of shape (m,), or
        - column vector of shape (m,1).
    Q : torch.Tensor
        Quaternion tensor of shape (m, n, 4).

    Returns
    -------
    R : torch.Tensor
        Result of shape (m, n, 4), where each quaternion row Q[k] is scaled by s[k].
    """
    # Ensure s is (m,)
    if s.dim() == 2 and s.shape[1] == 1:
        s = s.squeeze(1)  # (m,)
    if s.dim() == 0:
        # Scalar, broadcast directly
        return s * Q
    if s.shape[0] != Q.shape[0]:
        raise ValueError(f"Dimension mismatch: s has length {s.shape[0]}, "
                         f"but Q has {Q.shape[0]} rows.")

    # Broadcast: (m, 1, 1) * (m, n, 4)
    R = s[:, None, None] * Q
    return R


def q_mat_mul_q(mat: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Multiply a scalar matrix with a quaternion on the right.

    Parameters:
        mat: quaternion matrix
        q: quaternion

    Returns:
        quaternion matrix of shape (n,m)
    """
    return q_mul(mat, q.unsqueeze(-2).expand_as(mat))


def q_star_conj(q: torch.Tensor) -> torch.Tensor:
    """
    Star conjugation of the given quaternion vector.

    Parameters:
        q: quaternion matrix of (m,n)

    Returns:
        quaternion matrix of shape (n,m)
    """
    q_conj_var = q.clone()
    q_conj_var[..., 1:] *= -1
    return q_conj_var.transpose(0, 1)


def q_mat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Quaternion matrix product via real isomorphism.

    Parameters:
        a: (m, k, 4)
        b: (k, n, 4)

    Returns:
        (m, n, 4)
    """
    A1, A2, A3, A4 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    B1, B2, B3, B4 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    Ta = torch.cat([
        torch.cat([A1, A3, A2, A4], dim=-1),
        torch.cat([-A3, A1, A4, -A2], dim=-1),
        torch.cat([-A2, -A4, A1, A3], dim=-1),
        torch.cat([-A4, A2, -A3, A1], dim=-1)
    ], dim=-2)  # shape (4m, 4k)

    Tb = torch.cat([
        torch.cat([B1, B3, B2, B4], dim=-1),
        torch.cat([-B3, B1, B4, -B2], dim=-1),
        torch.cat([-B2, -B4, B1, B3], dim=-1),
        torch.cat([-B4, B2, -B3, B1], dim=-1)
    ], dim=-2)  # shape (4k, 4n)

    Tab = Ta @ Tb  # shape (4m, 4n)
    m, n = a.shape[0], b.shape[1]

    M1 = Tab[0:m, 0:n]
    M3 = Tab[0:m, n:2 * n]
    M2 = Tab[0:m, 2 * n:3 * n]
    M4 = Tab[0:m, 3 * n:4 * n]

    return torch.stack([M1, M2, M3, M4], dim=-1)


def matA_R_torch(A1, A2, A3, A4):
    # used in initialization eigen vector computation

    row1 = torch.cat([A1, -A2, -A3, -A4], dim=1)
    row2 = torch.cat([A2, A1, -A4, A3], dim=1)
    row3 = torch.cat([A3, A4, A1, -A2], dim=1)
    row4 = torch.cat([A4, -A3, A2, A1], dim=1)
    return torch.cat([row1, row2, row3, row4], dim=0)


def q_max_eign(S_in: torch.Tensor, M=300, N=10, err=1e-5):
    """
    Compute the maximum eigenvalue of a quaternion matrix and obtain the corresponding eigenvector

    Parameters:
        S_in: matrix to be considered
        M: iteration count for power method
        N: loop count for power method
        err: error threshold for eigensolver

    Returns:
        lambda_old: largest eigenvalue,
        u_arr: eigenvector corresponding to the largest eigenvalue
    """

    n = S_in.shape[1]
    device = S_in.device  # <<=== 关键

    A1, A2, A3, A4 = S_in[..., 0], S_in[..., 1], S_in[..., 2], S_in[..., 3]
    A_R = matA_R_torch(A1, A2, A3, A4).to(device)  # 确保在同一 device 上

    u = torch.zeros(4 * n, dtype=torch.float64, device=device)  # <<=== 修复
    u[0] = 1.0
    lambda_old = 1.0

    for _ in range(M):
        for _ in range(N):
            y = A_R @ u
            u = y / torch.norm(y)

        y = A_R @ u
        lambda_new = torch.dot(u, y).item()

        if (lambda_new - lambda_old) ** 2 < err:
            lambda_old = lambda_new
            break
        lambda_old = lambda_new

        u_mat = u.reshape(4, n).T
        norm2 = (u_mat ** 2).sum(dim=1)
        i1 = torch.argmax(norm2)
        mx = norm2[i1]

        q = torch.tensor([
            u[i1],
            -u[i1 + n],
            -u[i1 + 2 * n],
            -u[i1 + 3 * n]
        ], dtype=torch.float64, device=device)  # <<=== 修复

        u1 = u[0:n].unsqueeze(1)
        u2 = u[n:2 * n].unsqueeze(1)
        u3 = u[2 * n:3 * n].unsqueeze(1)
        u4 = u[3 * n:4 * n].unsqueeze(1)
        u_R = matA_R_torch(u1, u2, u3, u4).to(device)  # <<=== 修复
        u = (u_R @ q) / torch.sqrt(mx)

    u_arr = u.reshape(n, 4).unsqueeze(1)  # (n, 1, 4)
    return lambda_old, u_arr


def q_combine_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Combine two complex numbers to a quaternion
    """
    ar, ai = a.real, a.imag
    br, bi = b.real, b.imag
    q = torch.stack((ar, ai, br, bi), dim=-1)
    return q


def q_arg(
        q: torch.Tensor,
        eps: float = 1e-12
) -> Tensor:
    """
    计算四元数张量 q 的“辐角/相位”θ 及其轴向单位向量 n，并返回模长 r。
    极坐标: q = r * (cos θ + n sin θ), 其中 θ ∈ [0, π], n^2 = -1。

    参数
    ----
    q   : (..., 4)  最后一维按 [r, i, j, k]
    eps : float     数值稳定常数，用于处理 v≈0 的情形

    返回
    ----
    theta : (...,)      辐角 θ
    n     : (..., 3)    轴向单位向量（当 ||v||<=eps 时置为 0，方向不唯一）
    r     : (...,)      模长 ||q||

    说明
    ----
    - 对纯实正数：θ=0, n=0；纯实负数：θ=π, n=0（方向任意，这里置 0）。
    - 对 q=0：θ=0, n=0, r=0。
    - 若需要某个固定方向来代表“负实数”的 n，可在外层根据 (r>0 & ||v||<=eps & a<0)
      的掩码自行替换 n 为你指定的单位纯虚方向。
    """
    assert q.size(-1) == 4, "q must have shape (..., 4) as [r,i,j,k]"

    # ---- 关键：确保是浮点 ----
    if not torch.is_floating_point(q):
        q = q.to(torch.get_default_dtype())  # 或 q = q.to(torch.float64)

    a = q[..., 0]  # 标量部
    v = q[..., 1:]  # 向量部 (...,3)

    vnorm = torch.linalg.norm(v, dim=-1)  # (...,)

    # atan2 更稳健：θ = atan2(||v||, a) ∈ [0, π]
    theta = torch.atan2(vnorm, a)

    return theta


def q_sqrt(
        q: torch.Tensor,
        axis: Optional[torch.Tensor] = None,
        eps: float = 1e-12,
        return_both: bool = False,
) -> Tensor:
    """
    Quaternion square root (vectorized, numerically stable).

    Args:
        q: (..., 4) tensor, quaternion as [a, b, c, d] = [scalar, i, j, k].
        axis: (3,) optional unit pure-imag direction used ONLY when q is a
              negative real number (a<0, v=0). If None, uses [1,0,0].
        eps: small positive for numerical safety.
        return_both: if True, returns (root, -root).

    Returns:
        root, neg_root: both of shape (..., 4).
                        If return_both=False, neg_root equals -root anyway.

    Notes:
        - For q=0: returns (0, 0).
        - For negative reals: infinite many roots; this picks the one along `axis`.
        - Uses the closed-form: sqrt(q) = s + n * t with
              s = sqrt((|q| + a)/2),  t = sqrt((|q| - a)/2),  n = v/||v||
          and a safe fallback n := axis when v=0 & a<0.
    """
    assert q.size(-1) == 4, "q must be (..., 4) = [a, b, c, d]"
    if not torch.is_floating_point(q):
        q = q.to(torch.get_default_dtype())

    a = q[..., 0]  # scalar part (...,)
    v = q[..., 1:]  # vector part (...,3)
    vnorm = torch.linalg.norm(v, dim=-1)  # (...,)
    r = torch.sqrt(a * a + vnorm * vnorm)  # |q| (...,)

    # s = sqrt((|q| + a)/2), t = sqrt((|q| - a)/2)  (clamped to avoid -0)
    s = torch.sqrt(torch.clamp(0.5 * (r + a), min=0.0))
    t = torch.sqrt(torch.clamp(0.5 * (r - a), min=0.0))

    # unit direction n = v / ||v||  (undefined only when v=0)
    denom = torch.clamp(vnorm, min=eps).unsqueeze(-1)  # (...,1)
    n = v / denom  # (...,3)
    n = torch.where((vnorm > eps).unsqueeze(-1), n, torch.zeros_like(n))

    # handle negative real numbers: v=0 and a<0  -> choose an axis
    neg_real_mask = ((vnorm <= eps) & (a < 0)).unsqueeze(-1)  # (...,1)
    if axis is None:
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=q.dtype, device=q.device)
    axis = axis / torch.clamp(torch.linalg.norm(axis), min=eps)
    axis_broadcast = axis.view(*([1] * (q.ndim - 1)), 3).expand_as(n)
    n = torch.where(neg_real_mask, axis_broadcast, n)  # replace n only on negative reals

    # Compose root = s + n * t  => [s, n*t]
    vec = n * t.unsqueeze(-1)  # (...,3)
    root = torch.cat([s.unsqueeze(-1), vec], dim=-1)  # (...,4)

    return root


def q_split_complex(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a quaternion to two complex numbers
    """
    r, i, j, k = q.unbind(-1)
    ga = torch.complex(r, i)
    gb = torch.complex(j, k)
    return ga, gb


def q_sign(q: torch.Tensor) -> torch.Tensor:
    """
    Extract the sign of a quaternion: q / ||q||

    Parameters:
            q: (..., 4) — a quaternion tensor

    Returns:
            return: (..., 4) — normalized quaternion with unit norm
    """
    norm = torch.norm(q, dim=-1, keepdim=True)  # (..., 1)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)  # 避免除以0
    return q / norm


def q_split(Q: torch.Tensor):
    """
    Split a quaternion matrix to four real parts

    Parameters:
            Q: (..., 4) — a quaternion matrix

    Returns:
            return: four real parts
    """
    return Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]


def q_combine(M1: torch.Tensor, M2: torch.Tensor, M3: torch.Tensor, M4: torch.Tensor) -> torch.Tensor:
    """
    Extract the sign of a quaternion: q / ||q||

    Parameters:
            M1: scalar part of a quaternion matrix
            M2: i part
            M3: j part
            M4: k part


    Returns:
            return: a entire quaternion matrix
    """
    return torch.stack((M1, M2, M3, M4), dim=-1)


def q_arr_dist(z: torch.Tensor, x_arr: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternionic distance between z and x_arr, accounting for
    global quaternion sign ambiguity.

    The distance is defined as:
        || z - x_arr * sign(x_arr^* z) ||

    Parameters:
        z (Tensor): Estimated quaternion vector, shape (d, 1, 4)
        x_arr (Tensor): Ground truth quaternion vector, shape (d, 1, 4)

    Returns:
        dist (Tensor): Scalar tensor representing the Euclidean distance.
    """
    # 1. Compute quaternion conjugate of x_arr
    x_star = q_star_conj(x_arr)  # shape: (d, 1, 4)

    # 2. Inner product x_arr^* z (quaternion multiplication)
    prod = q_mul(x_star, z)  # shape: (d, 1, 4)

    # 3. Extract global sign from the first quaternion element
    sign = q_sign(prod[0, 0])  # shape: (4,)

    # 4. Align x_arr with z by multiplying with sign
    x_signed = q_mat_mul_q(x_arr, sign)  # shape: (d, 1, 4)

    # 5. Compute Euclidean distance
    diff = z - x_signed
    dist = torch.norm(diff)

    return dist


def phase_factor_estimate(z: torch.Tensor) -> torch.Tensor:
    """
    Estimate a right quaternion phase factor and project onto pure quaternion space.

    This implements the pure-quaternion phase factor estimate in Chen & Ng (2023),
    i.e., solve
        q_hat = arg min_{q in T_Q} ||Re(z q)||
    and return Im(z q_hat), where T_Q is the unit quaternion set.

    Parameters
    ----------
    z : torch.Tensor
        Full quaternion vector with shape (d, 1, 4) or (d, 4). The last axis is
        ordered as [r, i, j, k].

    Returns
    -------
    torch.Tensor
        Pure quaternion tensor with the same shape as input `z` and dtype float64.
    """
    if z.shape[-1] != 4:
        raise ValueError(f"`z` must end with quaternion axis size 4, got {tuple(z.shape)}")
    if z.ndim not in (2, 3):
        raise ValueError(f"`z` must have shape (d,4) or (d,1,4), got {tuple(z.shape)}")

    z64 = z.to(dtype=torch.float64)
    original_dim = z64.ndim
    if original_dim == 2:
        z64 = z64.unsqueeze(1)  # (d,1,4)

    # Eliminate invalid values to avoid failing eigensolver.
    if torch.isnan(z64).any() or torch.isinf(z64).any():
        z_pure = z64.clone()
        z_pure[..., 0] = 0.0
        return z_pure.squeeze(1) if original_dim == 2 else z_pure

    # V(z) in paper notation corresponds to (d,4) real matrix.
    z_v = z64.squeeze(1)  # (d,4)
    gram = z_v.T @ z_v  # (4,4)

    _, eigvecs = torch.linalg.eigh(gram)
    q_t = eigvecs[:, 0]  # eigenvector for smallest eigenvalue

    q_t_conj = torch.stack((q_t[0], -q_t[1], -q_t[2], -q_t[3]))
    z_righted = q_mat_mul_q(z64, q_t_conj)
    z_pure = z_righted.clone()
    z_pure[..., 0] = 0.0

    return z_pure.squeeze(1) if original_dim == 2 else z_pure


def phase_factor_estimate_batched(z: torch.Tensor) -> torch.Tensor:
    # z: (B,d,4), output pure quaternion (B,d,4)
    z64 = z.to(dtype=torch.float64)
    if torch.isnan(z64).any() or torch.isinf(z64).any():
        out = z64.clone()
        out[..., 0] = 0.0
        return out

    gram = z64.transpose(1, 2) @ z64  # (B,4,4)
    _, eigvecs = torch.linalg.eigh(gram)
    q_t = eigvecs[:, :, 0]  # (B,4), smallest eigenvector
    q_t_conj = q_t.clone()
    q_t_conj[:, 1:] = -q_t_conj[:, 1:]

    z_righted = quat_mul(z64, q_t_conj.unsqueeze(1).expand_as(z64))
    z_pure = z_righted.clone()
    z_pure[..., 0] = 0.0
    return z_pure


def q_pfe(z: torch.Tensor) -> torch.Tensor:
    """
    Find the closest pure quaternion vector from the given full quaternion vector.

    Method from: J. CHEN, M. K. NG. Phase Retrieval of Quaternion Signal via Wirtinger Flow[J]. IEEE Transactions on
    Signal Processing, 2023, 71: 2863-2878.

    Parameters:
        z: a full quaternion
    Returns:
        z_pure: a pure quaternion vector
    """
    device = z.device
    dtype = torch.float64
    z = z.to(dtype=dtype)

    # eliminate NaN and Inf
    if torch.isnan(z).any() or torch.isinf(z).any():
        z_pure = z.clone()
        z_pure[..., 0] = 0.0
        return z_pure

    z_V = z.squeeze(1)  # (n, 3)
    z_V_VT = z_V.T @ z_V  # (3, 3)

    eigvals, eigvecs = torch.linalg.eigh(z_V_VT)
    min_idx = torch.argmin(eigvals)
    q_t_cpu = eigvecs[:, min_idx]  # shape (3,)

    q_t = q_t_cpu.to(device=device)
    q_t_conj = torch.tensor([q_t[0], -q_t[1], -q_t[2], -q_t[3]],
                            dtype=q_t.dtype, device=q_t.device)

    z_righted = q_mat_mul_q(z, q_t_conj)
    z_pure = z_righted.clone()
    z_pure[..., 0] = 0.0
    return z_pure


def q_normalize_vector(vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize a patch vector (flattened quaternion patch) to have unit norm.

    Parameters:
        vec: Tensor of shape (N, 4)

    Returns:
        vec_normalized: Tensor of shape (N, 4), with total L2 norm = 1
        original_norm: Tensor scalar, the original L2 norm (used for un-normalization)
    """
    if vec.ndim != 3 or vec.shape[-1] != 4:
        raise ValueError("Input must be of shape (N, 4)")

    # Compute total norm across all elements and channels
    original_norm = torch.norm(vec)  # scalar

    if original_norm.item() == 0:
        # Prevent division by zero
        return vec.clone(), torch.tensor(1.0, dtype=vec.dtype, device=vec.device)

    vec_normalized = vec / original_norm

    return vec_normalized, original_norm


def q_unnormalize_vector(vec_normalized: torch.Tensor, original_norm: torch.Tensor) -> torch.Tensor:
    """
    Un-normalize a patch vector by restoring its original scale.

    Parameters:
        vec_normalized: Tensor of shape (N, 4), assumed to be unit norm
        original_norm: Scalar tensor, original norm value

    Returns:
        vec: Tensor of shape (N, 4), restored to original norm
    """
    return vec_normalized * original_norm


def q_exp(mu: torch.Tensor, theta: torch.Tensor, *, normalize_mu: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """
    Quaternion exponential for a pure-unit direction.

    Computes the quaternion-valued field
        exp(mu * theta) = cos(theta) + mu * sin(theta),
    where mu is a *pure* unit quaternion direction (zero real part), and theta is a real tensor.

    Parameters
    ----------
    mu : Tensor
        Shape (4,), representing the quaternion [r, i, j, k]. Expected to be pure, i.e., r == 0.
        If `normalize_mu` is True, the vector part is normalized to unit length.
    theta : Tensor
        Real-valued tensor of shape (...,). The function returns a quaternion for each entry.
        Device and dtype are inherited for the output.
    normalize_mu : bool, optional (default: True)
        If True, normalize the vector part (i, j, k) to unit length before computation.
        If False, raises if ||(i, j, k)|| is not within tolerance of 1.
    eps : float, optional (default: 1e-12)
        Numerical tolerance used in purity and unit-norm checks.

    Returns
    -------
    Tensor
        Quaternion tensor of shape (..., 4), with components [r, i, j, k].

    Notes
    -----
    - The formula assumes mu is a pure unit quaternion (real part zero, vector part unit norm).
    - Broadcasting: theta can be any shape; the result appends a last dimension of size 4.
    - For GPU acceleration, ensure `theta` (and optionally `mu`) are on CUDA before calling.
    """
    if mu.ndim != 1 or mu.shape[0] != 4:
        raise ValueError("`mu` must have shape (4,) representing [r, i, j, k].")

    r_mu = mu[0]
    v_mu = mu[1:]  # (3,)

    # Purity check: real part should be (numerically) zero.
    if torch.abs(r_mu) > eps:
        raise ValueError("`mu` must be pure: expected real part 0 (|r| <= eps).")

    # Unit-norm handling for the vector part.
    v_norm = torch.linalg.norm(v_mu)
    if normalize_mu:
        if v_norm <= eps:
            raise ValueError("`mu` vector part has near-zero norm and cannot be normalized.")
        v_mu = v_mu / v_norm
    else:
        if torch.abs(v_norm - 1.0) > 1e-6:
            raise ValueError("`mu` vector part must have unit norm when `normalize_mu=False`.")

    # Compute scalar trigonometric fields (elementwise, broadcast over theta's shape).
    ct = torch.cos(theta)  # (...)
    st = torch.sin(theta)  # (...)

    # Build quaternion components: q = [ct, (v_mu * st)]
    # Expand st to (..., 1) to multiply by (3,) and broadcast to (..., 3).
    st_expanded = st.unsqueeze(-1)  # (..., 1)
    vec = st_expanded * v_mu.view(1, 1, -1) if st_expanded.ndim == 2 else st_expanded * v_mu  # (..., 3)

    # Concatenate real and vector parts to (..., 4).
    q = torch.cat((ct.unsqueeze(-1), vec), dim=-1)  # (..., 4)
    return q


def q_mask(f: torch.Tensor, mu: torch.Tensor, n: torch.Tensor, *, normalize_mu: bool = True) -> torch.Tensor:
    """
    Apply a quaternionic exponential mask to a quaternion image.

    Computes, elementwise:
        g(x, y) = exp(mu * (-2*pi*n(x, y))) ⊗ f(x, y)
    where:
      - mu is a pure unit quaternion direction [0, a, b, c],
      - n is a real-valued phase map,
      - ⊗ denotes the quaternion full product.

    Parameters
    ----------
    f : Tensor
        Quaternion image with shape (..., 4) where the last dimension is [r, i, j, k].
    mu : Tensor
        Shape (4,), pure quaternion direction [0, a, b, c].
    n : Tensor
        Real-valued tensor with shape matching f's spatial dims (...,).
    normalize_mu : bool, optional (default: True)
        If True, normalize the vector part of `mu` to unit length before use.

    Returns
    -------
    Tensor
        Masked quaternion image with shape (..., 4), same dtype/device as `f`.

    Notes
    -----
    - This function is fully broadcast/vectorized; no Python loops.
    - Ensure `q_mul` supports broadcasting over the last dimension of size 4.
    """
    if f.ndim < 1 or f.shape[-1] != 4:
        raise ValueError("`f` must have shape (..., 4) with quaternion components [r,i,j,k].")
    if mu.shape != (4,):
        raise ValueError("`mu` must have shape (4,), representing [r,i,j,k].")
    if n.shape != f.shape[:-1]:
        raise ValueError("`n` must match the spatial/batch shape of `f` (i.e., n.shape == f.shape[:-1]).")

    # Phase field theta = -2*pi*n
    theta = -2.0 * torch.pi * n  # shape (...,)

    # Quaternion mask field: E = exp(mu * theta) with shape (..., 4)
    E = q_exp(mu, theta, normalize_mu=normalize_mu)

    # Quaternion full product per element: g = E ⊗ f
    g = q_mul(E, f)
    return g


def merge_to_complex(
        a: torch.Tensor,
        b: torch.Tensor,
        dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Merge two real tensors a, b into a complex tensor a + 1j * b.

    parameters:
        a, b : real-valued tensors (any shape, broadcastable)
        dtype: optional output complex dtype (torch.complex64 or torch.complex128).
               If None, inferred from a/b: float32→complex64, float64→complex128,
               integers default to complex64 unless default dtype is float64.

    Returns:
        torch.Tensor (complex), shape = broadcast(a, b).shape

    Notes:
        - a, b may be integers; they will be safely cast to a floating type.
        - If a and b are on different devices, b is moved to a.device.
        - Supports broadcasting just like normal PyTorch ops.
    """
    if a.device != b.device:
        b = b.to(a.device)

    # Decide target complex dtype
    if dtype is None:
        # Infer a floating dtype first
        rt = torch.result_type(a, b)
        if rt.is_floating_point:
            float_dtype = rt
        else:
            # If both are integers, fall back to default float
            float_dtype = torch.get_default_dtype()
        dtype = torch.complex64 if float_dtype == torch.float32 else torch.complex128
    else:
        assert dtype in (torch.complex64, torch.complex128), "dtype must be complex64 or complex128"
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

    a_f = a.to(float_dtype)
    b_f = b.to(float_dtype)

    # Broadcast to common shape
    a_f, b_f = torch.broadcast_tensors(a_f, b_f)

    # Build complex tensor
    return torch.complex(a_f, b_f).to(dtype)


def q_proj(z: torch.Tensor) -> torch.Tensor:
    """
    Find the closest pure quaternion vector from the given full quaternion vector.

    Method from: J. CHEN, M. K. NG. Phase Retrieval of Quaternion Signal via Wirtinger Flow[J]. IEEE Transactions on
    Signal Processing, 2023, 71: 2863-2878.

    Parameters:
        z: a full quaternion
    Returns:
        z_pure: a pure quaternion vector
    """
    device = z.device
    dtype = torch.float64
    z = z.to(dtype=dtype)

    # eliminate NaN and Inf
    if torch.isnan(z).any() or torch.isinf(z).any():
        z_pure = z.clone()
        z_pure[..., 0] = 0.0
        return z_pure

    z_pure = z.clone()
    z_pure[..., 0] = 0.0
    return z_pure
