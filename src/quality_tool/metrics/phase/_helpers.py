"""Shared batch helpers for phase metrics.

These helpers compute the analytic signal, unwrapped phase, phase
support, and local phase slopes that all phase metrics consume.
They operate on ``(N, M)`` signal batches and return per-signal
arrays with support masks.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from quality_tool.core.analysis_context import AnalysisContext


def compute_analytic_batch(
    signals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute envelope and unwrapped phase for a batch of signals.

    Parameters
    ----------
    signals : np.ndarray
        Shape ``(N, M)`` prepared signals.

    Returns
    -------
    envelope : np.ndarray
        Shape ``(N, M)``.
    phase : np.ndarray
        Shape ``(N, M)`` unwrapped phase.
    """
    analytic = hilbert(signals, axis=1)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic), axis=1)
    return envelope, phase


def compute_phase_support(
    envelope: np.ndarray,
    ctx: AnalysisContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determine phase support masks for a batch.

    Parameters
    ----------
    envelope : np.ndarray
        Shape ``(N, M)``.
    ctx : AnalysisContext
        Shared constants.

    Returns
    -------
    support : np.ndarray
        Bool ``(N, M)`` — final trimmed support mask.
    n0 : np.ndarray
        Int ``(N,)`` — index of envelope peak per signal.
    e_peak : np.ndarray
        Float ``(N,)`` — envelope peak value per signal.
    """
    n, m = envelope.shape
    alpha = ctx.phase_support_threshold_fraction
    guard = ctx.phase_guard_samples

    n0 = np.argmax(envelope, axis=1)  # (N,)
    e_peak = envelope[np.arange(n), n0]  # (N,)
    threshold = alpha * e_peak  # (N,)

    # Candidate support: envelope >= alpha * peak
    candidates = envelope >= threshold[:, None]  # (N, M)

    # Keep only the largest connected component containing n0.
    support = _largest_connected_containing(candidates, n0)

    # Trim guard samples from edges.
    if guard > 0:
        support = _trim_guard(support, guard)

    return support, n0, e_peak


def _largest_connected_containing(
    mask: np.ndarray, center: np.ndarray,
) -> np.ndarray:
    """Keep only the connected run in each row that contains *center*.

    Parameters
    ----------
    mask : np.ndarray
        Bool ``(N, M)``.
    center : np.ndarray
        Int ``(N,)`` — required index per row.

    Returns
    -------
    np.ndarray
        Bool ``(N, M)`` with only the connected component kept.
    """
    n, m = mask.shape
    result = np.zeros_like(mask)
    for i in range(n):
        row = mask[i]
        c = center[i]
        if not row[c]:
            continue
        # Walk left from center.
        left = c
        while left > 0 and row[left - 1]:
            left -= 1
        # Walk right from center.
        right = c
        while right < m - 1 and row[right + 1]:
            right += 1
        result[i, left: right + 1] = True
    return result


def _trim_guard(support: np.ndarray, guard: int) -> np.ndarray:
    """Trim *guard* samples from both edges of each row's support."""
    n, m = support.shape
    result = support.copy()
    for i in range(n):
        row = result[i]
        indices = np.nonzero(row)[0]
        if indices.size == 0:
            continue
        lo, hi = indices[0], indices[-1]
        new_lo = lo + guard
        new_hi = hi - guard
        if new_lo > new_hi:
            result[i, :] = False
        else:
            result[i, lo:new_lo] = False
            result[i, new_hi + 1: hi + 1] = False
    return result


def compute_local_coordinate(
    m: int,
    z_axis: np.ndarray | None,
    ctx: AnalysisContext,
) -> np.ndarray:
    """Build the local coordinate ``u`` for phase derivatives.

    Returns the physical z-axis if available, otherwise sample index.

    Parameters
    ----------
    m : int
        Signal length.
    z_axis : np.ndarray | None
        Physical z-axis of length *m*, or ``None``.
    ctx : AnalysisContext
        Shared constants (unused currently, reserved).

    Returns
    -------
    np.ndarray
        1-D array of length *m*.
    """
    if z_axis is not None and z_axis.shape == (m,):
        return z_axis.astype(float)
    return np.arange(m, dtype=float)


def compute_local_slopes(
    phase: np.ndarray,
    support: np.ndarray,
    u: np.ndarray,
    epsilon: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute local phase slopes on support for each signal.

    Returns per-signal arrays because support lengths vary.

    Parameters
    ----------
    phase : np.ndarray
        Shape ``(N, M)`` unwrapped phase.
    support : np.ndarray
        Bool ``(N, M)`` support mask.
    u : np.ndarray
        1-D local coordinate of length *M*.
    epsilon : float
        Division guard.

    Returns
    -------
    slopes_list : list[np.ndarray]
        Per-signal 1-D arrays of local phase slopes.
    pair_indices_list : list[np.ndarray]
        Per-signal 1-D arrays of support indices for slope pairs.
    """
    n_signals = phase.shape[0]
    slopes_list: list[np.ndarray] = []
    pair_indices_list: list[np.ndarray] = []
    for i in range(n_signals):
        idx = np.nonzero(support[i])[0]
        if idx.size < 2:
            slopes_list.append(np.array([]))
            pair_indices_list.append(np.array([], dtype=int))
            continue
        dphi = np.diff(phase[i, idx])
        du = np.diff(u[idx]) + epsilon
        slopes_list.append(dphi / du)
        pair_indices_list.append(idx[:-1])
    return slopes_list, pair_indices_list


def validate_phase_support(
    support: np.ndarray,
    slopes_list: list[np.ndarray],
    u: np.ndarray,
    ctx: AnalysisContext,
) -> np.ndarray:
    """Check minimum-support and minimum-periods validity per signal.

    Returns
    -------
    np.ndarray
        Bool ``(N,)`` — True where support is sufficient.
    """
    n = support.shape[0]
    valid = np.ones(n, dtype=bool)
    min_samples = ctx.minimum_phase_support_samples
    min_periods = ctx.minimum_phase_support_periods
    eps = ctx.epsilon

    for i in range(n):
        idx = np.nonzero(support[i])[0]
        if idx.size < min_samples:
            valid[i] = False
            continue
        slopes = slopes_list[i]
        if slopes.size == 0:
            valid[i] = False
            continue
        d_med = float(np.median(slopes))
        if abs(d_med) < eps:
            valid[i] = False
            continue
        # Estimate number of carrier periods from median slope.
        span = u[idx[-1]] - u[idx[0]]
        n_periods = abs(d_med) * span / (2.0 * np.pi)
        if n_periods < min_periods:
            valid[i] = False
    return valid
