"""Shared batch helpers for correlation / reference-model metrics.

These helpers build the centered reference model, compute reference
support, and provide normalization/orthonormalization utilities.
They operate on ``(N, M)`` signal batches.
"""

from __future__ import annotations

import numpy as np

from quality_tool.core.analysis_context import AnalysisContext


def resolve_reference_scales(
    ctx: AnalysisContext,
    z_axis: np.ndarray | None,
    m: int,
) -> tuple[float | None, float | None, np.ndarray]:
    """Derive reference carrier period and envelope scale in ROI-axis units.

    Also returns the centered coordinate ``u[n]``.

    Parameters
    ----------
    ctx : AnalysisContext
        Must contain ``reference_carrier_period_nm`` and
        ``reference_envelope_scale_nm`` (derived from metadata).
    z_axis : np.ndarray | None
        Physical z-axis of length *m*, or ``None``.
    m : int
        Signal / ROI length.

    Returns
    -------
    T_ref : float | None
        Carrier period in ROI-axis units (or None if unavailable).
    L_ref : float | None
        Gaussian 1/e scale in ROI-axis units (or None if unavailable).
    u : np.ndarray
        Centered coordinate, shape ``(m,)``.
    """
    n_c = (m - 1) / 2.0

    if z_axis is not None and z_axis.shape == (m,):
        z = z_axis.astype(float)
        u = z - z[int(round(n_c))]
    elif ctx.z_step_nm is not None and ctx.z_step_nm > 0:
        u = (np.arange(m, dtype=float) - n_c) * ctx.z_step_nm
    else:
        # No physical axis information — cannot build model.
        u = (np.arange(m, dtype=float) - n_c)
        return None, None, u

    T_ref = ctx.reference_carrier_period_nm
    L_ref = ctx.reference_envelope_scale_nm
    return T_ref, L_ref, u


def build_reference_model(
    u: np.ndarray,
    T_ref: float,
    L_ref: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build centered reference envelope and quadrature pair.

    Parameters
    ----------
    u : np.ndarray
        Centered coordinate, shape ``(M,)``.
    T_ref : float
        Carrier period in axis units.
    L_ref : float
        Gaussian 1/e envelope scale in axis units.

    Returns
    -------
    g_ref : np.ndarray
        Reference envelope, shape ``(M,)``.
    r_c : np.ndarray
        Cosine reference, shape ``(M,)``.
    r_s : np.ndarray
        Sine reference, shape ``(M,)``.
    """
    g_ref = np.exp(-(u / L_ref) ** 2)
    phase = 2.0 * np.pi * u / T_ref
    r_c = g_ref * np.cos(phase)
    r_s = g_ref * np.sin(phase)
    return g_ref, r_c, r_s


def build_reference_support(
    g_ref: np.ndarray,
    alpha_ref: float,
) -> np.ndarray:
    """Build boolean support mask from reference envelope.

    Parameters
    ----------
    g_ref : np.ndarray
        Reference envelope, shape ``(M,)``.
    alpha_ref : float
        Threshold fraction.

    Returns
    -------
    np.ndarray
        Bool mask of shape ``(M,)``.
    """
    return g_ref >= alpha_ref


def normalize_on_support(
    v: np.ndarray,
    support: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Zero-mean, unit-norm normalization on support indices.

    Parameters
    ----------
    v : np.ndarray
        Shape ``(N, M)`` or ``(M,)``.
    support : np.ndarray
        Bool mask of shape ``(M,)``.
    epsilon : float
        Division guard.

    Returns
    -------
    np.ndarray
        Normalized array with same shape as *v*, values outside
        support are zero.
    """
    if v.ndim == 1:
        vs = v[support]
        mean_v = vs.mean()
        centered = vs - mean_v
        norm = np.sqrt(np.sum(centered ** 2)) + epsilon
        out = np.zeros_like(v)
        out[support] = centered / norm
        return out

    # Batch: (N, M)
    vs = v[:, support]  # (N, K)
    mean_v = vs.mean(axis=1, keepdims=True)  # (N, 1)
    centered = vs - mean_v  # (N, K)
    norm = np.sqrt(np.sum(centered ** 2, axis=1, keepdims=True)) + epsilon
    out = np.zeros_like(v)
    out[:, support] = centered / norm
    return out


def orthonormalize_basis(
    c_norm: np.ndarray,
    s_norm: np.ndarray,
    support: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build orthonormal {q1, q2} basis from normalized cos/sin refs.

    Works for 1-D vectors (single reference, shared across batch).

    Parameters
    ----------
    c_norm : np.ndarray
        Normalized cosine reference, shape ``(M,)``.
    s_norm : np.ndarray
        Normalized sine reference, shape ``(M,)``.
    support : np.ndarray
        Bool mask of shape ``(M,)``.
    epsilon : float
        Division guard.

    Returns
    -------
    q1 : np.ndarray
        Shape ``(M,)``.
    q2 : np.ndarray
        Shape ``(M,)``.
    q2_norm : float
        Norm of q2 before normalization (for stability check).
    """
    q1 = c_norm
    # Project s_norm onto q1.
    proj = np.sum(s_norm[support] * q1[support])
    q2_raw = np.zeros_like(s_norm)
    q2_raw[support] = s_norm[support] - proj * q1[support]
    q2_raw_norm = np.sqrt(np.sum(q2_raw[support] ** 2))
    q2 = np.zeros_like(s_norm)
    if q2_raw_norm > epsilon:
        q2[support] = q2_raw[support] / (q2_raw_norm + epsilon)
    return q1, q2, q2_raw_norm
