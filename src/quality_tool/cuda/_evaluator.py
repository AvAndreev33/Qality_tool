"""GPU-accelerated metric evaluator for Quality_tool.

Mirrors the CPU ``evaluate_metric_maps`` interface but runs all
computation on GPU via CuPy.  Signals are uploaded once, all
preprocessing / representations / metrics run on device, and only
the final (H, W) score maps are downloaded back to host.

All 42 metrics are supported.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from quality_tool.core.analysis_context import AnalysisContext, build_analysis_context
from quality_tool.core.models import MetricMapResult, SignalSet
from quality_tool.evaluation.planner import build_plan
from quality_tool.evaluation.recipe import RAW, SignalRecipe
from quality_tool.spectral.priors import SpectralPriors, compute_spectral_priors

from quality_tool.cuda._backend import GPU_METRIC_NAMES

_DEFAULT_CHUNK = 50_000


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def evaluate_metric_maps_gpu(
    signal_set: SignalSet,
    metrics: Sequence,
    *,
    active_recipe: SignalRecipe | None = None,
    envelope_method=None,
    analysis_context: AnalysisContext | None = None,
    chunk_size: int = _DEFAULT_CHUNK,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, MetricMapResult]:
    """Evaluate metrics on GPU, returning the same result type as CPU."""
    import cupy as cp

    if analysis_context is None:
        analysis_context = build_analysis_context(signal_set)

    h, w, m = signal_set.signals.shape
    n_total = h * w
    z_axis = signal_set.z_axis

    gpu_metrics = [mt for mt in metrics if mt.name in GPU_METRIC_NAMES]
    if not gpu_metrics:
        return {}

    plan = build_plan(
        gpu_metrics,
        active_recipe=active_recipe,
        has_envelope=True,
    )

    signals_2d = signal_set.signals.reshape(n_total, m)
    results: dict[str, MetricMapResult] = {}

    for group in plan.groups:
        recipe = group.recipe

        score_accum = {mt.name: np.full(n_total, np.nan) for mt in group.metrics}
        valid_accum = {mt.name: np.zeros(n_total, dtype=bool) for mt in group.metrics}
        feat_accum: dict[str, dict[str, list]] = {mt.name: {} for mt in group.metrics}

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)

            chunk = cp.asarray(signals_2d[start:end], dtype=cp.float64)
            chunk = _preprocess_gpu(chunk, recipe, analysis_context, cp)

            chunk_m = chunk.shape[1]
            priors = compute_spectral_priors(chunk_m, analysis_context)

            envelope = None
            power = None
            amplitude = None
            frequencies = None

            if group.needs_envelope:
                envelope = _hilbert_envelope_gpu(chunk, cp)

            if group.needs.needs_spectral or group.needs.power or group.needs.amplitude:
                fft_result = _spectral_gpu(chunk, z_axis, cp)
                frequencies = fft_result["frequencies"]
                amplitude = fft_result["amplitude"]
                power = fft_result["power"]

            ctx_bundle = _ContextBundle(
                ctx=analysis_context, priors=priors,
                z_axis_host=z_axis, chunk_m=chunk_m, cp=cp,
            )

            for mt in group.metrics:
                s, v, f = _dispatch_metric_gpu(
                    mt.name, chunk, envelope, amplitude, power,
                    frequencies, ctx_bundle,
                )
                score_accum[mt.name][start:end] = cp.asnumpy(s)
                valid_accum[mt.name][start:end] = cp.asnumpy(v)
                for fk, fv in f.items():
                    feat_accum[mt.name].setdefault(fk, []).append(
                        (start, end, cp.asnumpy(fv))
                    )

            if progress_callback is not None:
                progress_callback(end, n_total)

        for mt in group.metrics:
            score_map = score_accum[mt.name].reshape(h, w)
            valid_map = valid_accum[mt.name].reshape(h, w)

            feature_maps: dict[str, np.ndarray] = {}
            for fk, chunks_list in feat_accum[mt.name].items():
                flat = np.full(n_total, np.nan)
                for s, e, arr in chunks_list:
                    flat[s:e] = arr
                feature_maps[fk] = flat.reshape(h, w)

            results[mt.name] = MetricMapResult(
                metric_name=mt.name,
                score_map=score_map,
                valid_map=valid_map,
                feature_maps=feature_maps,
                metadata={
                    "metric_name": mt.name,
                    "backend": "cuda",
                    "image_shape": (h, w),
                },
            )

    return results


# ------------------------------------------------------------------
# Helper: bundled context passed to metric functions
# ------------------------------------------------------------------

class _ContextBundle:
    __slots__ = ("ctx", "priors", "z_axis_host", "chunk_m", "cp")

    def __init__(self, ctx, priors, z_axis_host, chunk_m, cp):
        self.ctx = ctx
        self.priors = priors
        self.z_axis_host = z_axis_host
        self.chunk_m = chunk_m
        self.cp = cp


# ------------------------------------------------------------------
# GPU preprocessing
# ------------------------------------------------------------------

def _preprocess_gpu(signals, recipe: SignalRecipe, ctx: AnalysisContext, cp):
    if recipe == RAW:
        return signals

    if recipe.baseline:
        signals = signals - cp.mean(signals, axis=1, keepdims=True)

    if recipe.detrend:
        signals = _detrend_gpu(signals, cp)

    if recipe.normalize:
        lo = cp.min(signals, axis=1, keepdims=True)
        hi = cp.max(signals, axis=1, keepdims=True)
        span = hi - lo
        safe_span = cp.where(span == 0.0, 1.0, span)
        signals = (signals - lo) / safe_span
        signals[span.squeeze(axis=1) == 0.0] = 0.0

    if recipe.roi_enabled:
        seg_size = recipe.segment_size
        if seg_size is None:
            seg_size = ctx.default_segment_size
        signals = _extract_roi_gpu(signals, seg_size, cp)

    return signals


def _detrend_gpu(signals, cp):
    n, m = signals.shape
    x = cp.arange(m, dtype=cp.float64)
    x_mean = (m - 1) / 2.0
    x_centered = x - x_mean
    ss_x = float(cp.sum(x_centered ** 2))
    if ss_x == 0:
        return signals - cp.mean(signals, axis=1, keepdims=True)
    y_mean = cp.mean(signals, axis=1, keepdims=True)
    slope = (signals @ x_centered) / ss_x
    trend = slope[:, cp.newaxis] * x_centered[cp.newaxis, :]
    return signals - y_mean - trend


def _extract_roi_gpu(signals, segment_size: int, cp):
    n, m = signals.shape
    if segment_size > m:
        return signals
    centers = cp.argmax(signals, axis=1)
    half = segment_size // 2
    starts = cp.clip(centers - half, 0, m - segment_size)
    offsets = cp.arange(segment_size)
    indices = starts[:, cp.newaxis] + offsets[cp.newaxis, :]
    row_idx = cp.arange(n)[:, cp.newaxis]
    return signals[row_idx, indices]


# ------------------------------------------------------------------
# GPU representations
# ------------------------------------------------------------------

def _hilbert_envelope_gpu(signals, cp):
    n, m = signals.shape
    fft = cp.fft.fft(signals, axis=1)
    h = cp.zeros(m, dtype=cp.float64)
    if m % 2 == 0:
        h[0] = 1.0
        h[1:m // 2] = 2.0
        h[m // 2] = 1.0
    else:
        h[0] = 1.0
        h[1:(m + 1) // 2] = 2.0
    analytic = cp.fft.ifft(fft * h[cp.newaxis, :], axis=1)
    return cp.abs(analytic)


def _hilbert_analytic_gpu(signals, cp):
    """Return (envelope, unwrapped_phase) on GPU."""
    n, m = signals.shape
    fft = cp.fft.fft(signals, axis=1)
    h = cp.zeros(m, dtype=cp.float64)
    if m % 2 == 0:
        h[0] = 1.0
        h[1:m // 2] = 2.0
        h[m // 2] = 1.0
    else:
        h[0] = 1.0
        h[1:(m + 1) // 2] = 2.0
    analytic = cp.fft.ifft(fft * h[cp.newaxis, :], axis=1)
    envelope = cp.abs(analytic)
    phase = cp.unwrap(cp.angle(analytic), axis=1)
    return envelope, phase


def _spectral_gpu(signals, z_axis_host, cp) -> dict:
    n, m = signals.shape
    if z_axis_host is not None and len(z_axis_host) >= 2:
        spacing = float(np.mean(np.diff(z_axis_host)))
        if spacing <= 0:
            spacing = 1.0
    else:
        spacing = 1.0
    fft_coeffs = cp.fft.rfft(signals, axis=1)
    amplitude = cp.abs(fft_coeffs)
    power = amplitude ** 2
    freqs_host = np.fft.rfftfreq(m, d=spacing)
    frequencies = cp.asarray(freqs_host)
    return {"frequencies": frequencies, "amplitude": amplitude, "power": power}


def _hann_windowed_power_gpu(signals, cp):
    """Return (power (N,F), dc_index=0)."""
    n, m = signals.shape
    window = cp.hanning(m).astype(signals.dtype)
    windowed = signals * window[cp.newaxis, :]
    fft_coeffs = cp.fft.rfft(windowed, axis=1)
    power = cp.abs(fft_coeffs) ** 2
    return power, 0


def _find_carrier_and_band_gpu(power, bw, dc_index, cp):
    """Find carrier bin and band masks on GPU. Returns (k_c, in_band, out_band)."""
    n, f = power.shape
    search = power.copy()
    search[:, dc_index] = -cp.inf
    k_c = cp.argmax(search, axis=1)
    bin_idx = cp.arange(f)[cp.newaxis, :]
    k_c_2d = k_c[:, cp.newaxis]
    in_band = cp.abs(bin_idx - k_c_2d) <= bw
    dc_mask = cp.zeros(f, dtype=bool)
    dc_mask[dc_index] = True
    out_band = ~in_band & ~dc_mask[cp.newaxis, :]
    return k_c, in_band, out_band


# ------------------------------------------------------------------
# GPU metric dispatch
# ------------------------------------------------------------------

def _dispatch_metric_gpu(name, signals, envelope, amplitude, power,
                         frequencies, cb) -> tuple:
    fn = _DISPATCH[name]
    return fn(signals, envelope, amplitude, power, frequencies, cb)


# ------------------------------------------------------------------
# Baseline metrics
# ------------------------------------------------------------------

def _metric_snr(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    n, m = signals.shape
    quarter = max(m // 4, 1)
    noise = cp.concatenate([signals[:, :quarter], signals[:, -quarter:]], axis=1)
    noise_std = cp.std(noise, axis=1, ddof=0)
    peak_to_peak = cp.max(signals, axis=1) - cp.min(signals, axis=1)
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    valid = noise_std >= 1e-12
    scores[valid] = peak_to_peak[valid] / noise_std[valid]
    return scores, valid, {"peak_to_peak": peak_to_peak, "noise_std": noise_std}


def _metric_fringe_visibility(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    n = signals.shape[0]
    i_max = cp.max(signals, axis=1)
    i_min = cp.min(signals, axis=1)
    denom = i_max + i_min
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    valid = (i_min >= 0.0) & (denom > 0.0)
    scores[valid] = (i_max[valid] - i_min[valid]) / denom[valid]
    return scores, valid, {"i_max": i_max, "i_min": i_min}


def _metric_power_band_ratio(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    n = signals.shape[0]
    if power is None or frequencies is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}
    total_power = cp.sum(power[:, 1:], axis=1)
    signal_mask = (frequencies >= ctx.default_low_freq) & (frequencies <= ctx.default_high_freq)
    signal_power = cp.sum(power[:, signal_mask], axis=1)
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    valid = total_power >= 1e-20
    scores[valid] = signal_power[valid] / total_power[valid]
    return scores, valid, {"signal_power": signal_power, "total_power": total_power}


# ------------------------------------------------------------------
# Envelope metrics
# ------------------------------------------------------------------

def _metric_envelope_height(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    n = signals.shape[0]
    if envelope is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}
    e_peak = cp.max(envelope, axis=1)
    valid = cp.all(cp.isfinite(envelope), axis=1)
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    scores[valid] = e_peak[valid]
    return scores, valid, {"e_peak": e_peak}


def _metric_envelope_area(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    n = signals.shape[0]
    if envelope is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}
    ea = cp.sum(envelope, axis=1)
    valid = cp.all(cp.isfinite(envelope), axis=1)
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    scores[valid] = ea[valid]
    return scores, valid, {"envelope_area": ea}


def _metric_envelope_width(signals, envelope, amplitude, power, frequencies, cb):
    """FWHM via half-max crossings — per-signal loop on GPU arrays."""
    cp = cb.cp
    n, m = signals.shape
    if envelope is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    n0 = cp.argmax(envelope, axis=1)
    e_peak = cp.max(envelope, axis=1)
    half = 0.5 * e_peak

    # Download to host for the crossing scan (irregular per-signal logic)
    env_h = cp.asnumpy(envelope)
    n0_h = cp.asnumpy(n0)
    half_h = cp.asnumpy(half)

    z_l = np.full(n, np.nan)
    z_r = np.full(n, np.nan)
    for i in range(n):
        hv = half_h[i]
        pi = n0_h[i]
        ev = env_h[i]
        for j in range(pi, 0, -1):
            if ev[j - 1] <= hv <= ev[j]:
                d = ev[j] - ev[j - 1]
                z_l[i] = (j - 1) + (hv - ev[j - 1]) / d if d > 0 else float(j - 1)
                break
        for j in range(pi, m - 1):
            if ev[j + 1] <= hv <= ev[j]:
                d = ev[j] - ev[j + 1]
                z_r[i] = j + (ev[j] - hv) / d if d > 0 else float(j + 1)
                break

    crossing_valid = np.isfinite(z_l) & np.isfinite(z_r)
    fwhm = np.where(crossing_valid, z_r - z_l, np.nan)
    scores = cp.asarray(fwhm)
    valid = cp.asarray(crossing_valid) & cp.all(cp.isfinite(envelope), axis=1)
    return scores, valid, {}


def _metric_envelope_sharpness(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n = signals.shape[0]
    if envelope is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    e_peak = cp.max(envelope, axis=1)
    # Reuse width logic
    s_w, v_w, _ = _metric_envelope_width(signals, envelope, amplitude, power, frequencies, cb)
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    valid = v_w
    fwhm = s_w
    ok = valid & (cp.isfinite(fwhm))
    scores[ok] = e_peak[ok] / (fwhm[ok] + eps)
    return scores, valid, {}


def _metric_envelope_symmetry(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n, m = signals.shape
    if envelope is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    n0 = cp.argmax(envelope, axis=1)
    finite = cp.all(cp.isfinite(envelope), axis=1)

    # Compute symmetry — download for per-signal mirror logic
    env_h = cp.asnumpy(envelope)
    n0_h = cp.asnumpy(n0)

    scores_h = np.full(n, np.nan)
    valid_h = np.zeros(n, dtype=bool)
    for i in range(n):
        pk = n0_h[i]
        h_range = min(pk, m - 1 - pk)
        if h_range < 1:
            continue
        left = env_h[i, pk - h_range:pk][::-1]
        right = env_h[i, pk + 1:pk + 1 + h_range]
        d = float(np.sum(np.abs(left - right)))
        s = float(np.sum(left + right))
        if s > eps:
            scores_h[i] = 1.0 - d / (s + eps)
            valid_h[i] = True

    scores = cp.asarray(scores_h)
    valid = cp.asarray(valid_h) & finite
    return scores, valid, {}


def _metric_single_peakness(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    alpha = ctx.alpha_main_support
    n = signals.shape[0]
    if envelope is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    e_peak = cp.max(envelope, axis=1)
    w_main = envelope >= alpha * e_peak[:, cp.newaxis]
    main_mass = cp.sum(envelope * w_main, axis=1)
    total = cp.sum(envelope, axis=1)
    valid = cp.all(cp.isfinite(envelope), axis=1) & (total > eps)
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    scores[valid] = main_mass[valid] / (total[valid] + eps)
    return scores, valid, {}


def _metric_main_peak_to_sidelobe(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    alpha = ctx.alpha_main_support
    min_dist = ctx.secondary_peak_min_distance
    n = signals.shape[0]
    if envelope is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    e_peak_g = cp.max(envelope, axis=1)
    w_main = envelope >= alpha * e_peak_g[:, cp.newaxis]

    # Per-signal secondary peak detection — host side
    env_h = cp.asnumpy(envelope)
    w_main_h = cp.asnumpy(w_main)
    e_peak_h = cp.asnumpy(e_peak_g)

    scores_h = np.full(n, np.nan)
    valid_h = np.zeros(n, dtype=bool)
    from scipy.signal import find_peaks

    for i in range(n):
        peaks, _ = find_peaks(env_h[i], distance=max(1, min_dist))
        if len(peaks) == 0:
            continue
        outside = peaks[~w_main_h[i, peaks]]
        if len(outside) == 0:
            continue
        e_side = float(np.max(env_h[i, outside]))
        if e_side > 0:
            scores_h[i] = e_peak_h[i] / (e_side + eps)
            valid_h[i] = True

    return cp.asarray(scores_h), cp.asarray(valid_h), {}


# ------------------------------------------------------------------
# Spectral metrics
# ------------------------------------------------------------------

def _metric_spectral_entropy(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n = signals.shape[0]
    if power is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}
    f = power.shape[1]
    pos = cp.ones(f, dtype=bool)
    if ctx.dc_exclude:
        pos[0] = False
    k = int(cp.sum(pos))
    if k <= 1:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}
    p_pos = power[:, pos]
    total = cp.sum(p_pos, axis=1, keepdims=True) + eps
    p_norm = p_pos / total
    entropy = -cp.sum(p_norm * cp.log(p_norm + eps), axis=1) / cp.log(cp.float64(k))
    return entropy, cp.ones(n, dtype=bool), {}


def _metric_spectral_centroid_offset(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    priors = cb.priors
    n = signals.shape[0]
    if power is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    f = power.shape[1]
    pos = cp.ones(f, dtype=bool)
    if ctx.dc_exclude:
        pos[0] = False

    bins = cp.arange(f, dtype=cp.float64)
    p_pos = power[:, pos]
    b_pos = bins[pos]
    total = cp.sum(p_pos, axis=1, keepdims=True) + eps
    p_norm = p_pos / total
    mu = cp.sum(p_norm * b_pos[cp.newaxis, :], axis=1)
    k_exp = float(priors.expected_carrier_bin)
    scores = cp.abs(mu - k_exp) / (k_exp + eps)
    return scores, cp.ones(n, dtype=bool), {}


def _metric_dominant_spectral_peak_prominence(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n = signals.shape[0]
    if power is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    f = power.shape[1]
    pw = ctx.prominence_window_bins
    ex = ctx.prominence_exclusion_half_width_bins

    search = power.copy()
    if ctx.dc_exclude:
        search[:, 0] = -cp.inf
    k_0 = cp.argmax(search, axis=1)
    p_peak = power[cp.arange(n), k_0]

    bins = cp.arange(f)[cp.newaxis, :]
    k_0_2d = k_0[:, cp.newaxis]
    in_window = cp.abs(bins - k_0_2d) <= pw
    in_exclusion = cp.abs(bins - k_0_2d) <= ex
    loc_mask = in_window & ~in_exclusion

    masked = cp.where(loc_mask, power, cp.nan)
    # nanmedian — download to host for reliable median
    masked_h = cp.asnumpy(masked)
    med_h = np.nanmedian(masked_h, axis=1)
    med = cp.asarray(med_h)

    has_loc = cp.sum(loc_mask, axis=1) > 0
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    valid = has_loc & (med > 0)
    scores[valid] = p_peak[valid] / (med[valid] + eps)
    return scores, valid, {}


def _metric_energy_concentration(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    priors = cb.priors
    n = signals.shape[0]
    if power is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    f = power.shape[1]
    band = cp.zeros(f, dtype=bool)
    lo = max(0, priors.expected_band_low_bin)
    hi = min(f - 1, priors.expected_band_high_bin)
    if lo <= hi:
        band[lo:hi + 1] = True

    e_band = cp.sum(power[:, band], axis=1)
    e_total = cp.sum(power, axis=1) + eps
    scores = e_band / e_total
    return scores, cp.ones(n, dtype=bool), {}


def _metric_carrier_to_background(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    priors = cb.priors
    n = signals.shape[0]
    if power is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    f = power.shape[1]
    in_band = cp.zeros(f, dtype=bool)
    lo = max(0, priors.expected_band_low_bin)
    hi = min(f - 1, priors.expected_band_high_bin)
    if lo <= hi:
        in_band[lo:hi + 1] = True
    out_band = ~in_band
    if ctx.dc_exclude:
        out_band[0] = False

    n_in = int(cp.sum(in_band))
    n_out = int(cp.sum(out_band))
    if n_in == 0 or n_out == 0:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    p_car = cp.sum(power[:, in_band], axis=1) / n_in
    # median out-of-band — host
    out_h = cp.asnumpy(power[:, out_band])
    p_bg_h = np.median(out_h, axis=1)
    p_bg = cp.asarray(p_bg_h)

    scores = p_car / (p_bg + eps)
    return scores, cp.ones(n, dtype=bool), {}


def _metric_presence_expected_carrier(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    priors = cb.priors
    n = signals.shape[0]
    if power is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    f = power.shape[1]
    # Global peak (exclude DC)
    search = power.copy()
    if ctx.dc_exclude:
        search[:, 0] = -cp.inf
    k_0 = cp.argmax(search, axis=1)
    p_global = power[cp.arange(n), k_0]

    # Best in-band
    band = cp.zeros(f, dtype=bool)
    lo = max(0, priors.expected_band_low_bin)
    hi = min(f - 1, priors.expected_band_high_bin)
    if lo <= hi:
        band[lo:hi + 1] = True
    masked = cp.where(band[cp.newaxis, :], power, -cp.inf)
    k_star = cp.argmax(masked, axis=1)
    p_band = power[cp.arange(n), k_star]

    scores = p_band / (p_global + eps)
    valid = p_global > eps
    return scores, valid, {}


def _metric_spectral_peak_sharpness(signals, envelope, amplitude, power, frequencies, cb):
    """SPeakS = P[k0] / mean(P[k0-w:k0+w] excl k0)."""
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    priors = cb.priors
    n = signals.shape[0]
    if power is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    f = power.shape[1]
    bw = priors.expected_band_half_width_bins

    search = power.copy()
    if ctx.dc_exclude:
        search[:, 0] = -cp.inf
    k_0 = cp.argmax(search, axis=1)
    p_peak = power[cp.arange(n), k_0]

    bins = cp.arange(f)[cp.newaxis, :]
    k_0_2d = k_0[:, cp.newaxis]
    in_window = (cp.abs(bins - k_0_2d) <= bw) & (bins != k_0_2d)
    cnt = cp.sum(in_window, axis=1).astype(cp.float64)
    neigh_sum = cp.where(in_window, power, 0.0).sum(axis=1)
    neigh_mean = neigh_sum / (cnt + eps)

    scores = p_peak / (neigh_mean + eps)
    valid = cnt > 0
    return scores, valid, {}


def _metric_spectral_kurtosis(signals, envelope, amplitude, power, frequencies, cb):
    """SK = E[(f - mu)^4] / (sigma^2)^2 — excess kurtosis of spectral distribution."""
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n = signals.shape[0]
    if power is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    f = power.shape[1]
    pos = cp.ones(f, dtype=bool)
    if ctx.dc_exclude:
        pos[0] = False
    bins = cp.arange(f, dtype=cp.float64)

    p_pos = power[:, pos]
    b_pos = bins[pos]
    total = cp.sum(p_pos, axis=1, keepdims=True) + eps
    p_norm = p_pos / total

    mu = cp.sum(p_norm * b_pos[cp.newaxis, :], axis=1, keepdims=True)
    diff = b_pos[cp.newaxis, :] - mu
    var = cp.sum(p_norm * diff ** 2, axis=1)
    m4 = cp.sum(p_norm * diff ** 4, axis=1)

    scores = cp.full(n, cp.nan, dtype=cp.float64)
    valid = var > eps
    scores[valid] = m4[valid] / (var[valid] ** 2 + eps) - 3.0
    return scores, valid, {}


def _metric_spectral_spread(signals, envelope, amplitude, power, frequencies, cb):
    """Spectral spread = sqrt(variance of spectral distribution)."""
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n = signals.shape[0]
    if power is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    f = power.shape[1]
    pos = cp.ones(f, dtype=bool)
    if ctx.dc_exclude:
        pos[0] = False
    bins = cp.arange(f, dtype=cp.float64)

    p_pos = power[:, pos]
    b_pos = bins[pos]
    total = cp.sum(p_pos, axis=1, keepdims=True) + eps
    p_norm = p_pos / total

    mu = cp.sum(p_norm * b_pos[cp.newaxis, :], axis=1, keepdims=True)
    var = cp.sum(p_norm * (b_pos[cp.newaxis, :] - mu) ** 2, axis=1)
    scores = cp.sqrt(var)
    return scores, cp.ones(n, dtype=bool), {}


def _metric_envelope_spectrum_consistency(signals, envelope, amplitude, power, frequencies, cb):
    """ESC — requires metadata. Falls back to invalid if unavailable."""
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    priors = cb.priors
    n = signals.shape[0]

    if envelope is None or power is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    # Reference scales from metadata
    cl = ctx.coherence_length_nm
    zs = ctx.z_step_nm
    if cl is None or zs is None or cl <= 0 or zs <= 0:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    ew_ref = cl / zs  # reference envelope width in samples
    f = power.shape[1]
    bins = cp.arange(f, dtype=cp.float64)
    pos = cp.ones(f, dtype=bool)
    if ctx.dc_exclude:
        pos[0] = False
    p_pos = power[:, pos]
    b_pos = bins[pos]
    total = cp.sum(p_pos, axis=1, keepdims=True) + eps
    p_norm = p_pos / total
    mu = cp.sum(p_norm * b_pos[cp.newaxis, :], axis=1, keepdims=True)
    var = cp.sum(p_norm * (b_pos[cp.newaxis, :] - mu) ** 2, axis=1)
    ss_obs = cp.sqrt(var)

    # Envelope FWHM — download for per-signal scan
    env_h = cp.asnumpy(envelope)
    ew_obs_h = np.full(n, np.nan)
    for i in range(n):
        ev = env_h[i]
        pk = int(np.argmax(ev))
        hv = 0.5 * ev[pk]
        zl = zr = np.nan
        for j in range(pk, 0, -1):
            if ev[j - 1] <= hv <= ev[j]:
                d = ev[j] - ev[j - 1]
                zl = (j - 1) + (hv - ev[j - 1]) / d if d > 0 else float(j - 1)
                break
        for j in range(pk, len(ev) - 1):
            if ev[j + 1] <= hv <= ev[j]:
                d = ev[j] - ev[j + 1]
                zr = j + (ev[j] - hv) / d if d > 0 else float(j + 1)
                break
        if np.isfinite(zl) and np.isfinite(zr):
            ew_obs_h[i] = zr - zl

    ew_obs = cp.asarray(ew_obs_h)
    ss_ref = 1.0 / (ew_ref + eps)
    c_ref = ew_ref * ss_ref
    c_obs = ew_obs * ss_obs

    good = cp.isfinite(ew_obs) & (ss_obs > eps)
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    scores[good] = cp.abs(c_obs[good] - c_ref) / (c_ref + eps)
    return scores, good, {}


# ------------------------------------------------------------------
# Noise metrics
# ------------------------------------------------------------------

def _metric_spectral_snr(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    bw = ctx.band_half_width_bins
    n, m = signals.shape

    hw_power, dc = _hann_windowed_power_gpu(signals, cp)
    k_c, in_band, out_band = _find_carrier_and_band_gpu(hw_power, bw, dc, cp)

    p_sig = cp.where(in_band, hw_power, 0.0).sum(axis=1)
    p_noise = cp.where(out_band, hw_power, 0.0).sum(axis=1)

    valid = out_band.any(axis=1)
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    scores[valid] = 10.0 * cp.log10((p_sig[valid] + eps) / (p_noise[valid] + eps))
    return scores, valid, {"p_signal": p_sig, "p_noise": p_noise}


def _metric_local_snr(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n, m = signals.shape
    if envelope is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    e_max = cp.max(envelope, axis=1, keepdims=True)
    w_sig = envelope >= 0.5 * e_max
    w_noise = ~w_sig
    sig_sq = signals ** 2
    e_signal = cp.where(w_sig, sig_sq, 0.0).sum(axis=1)
    e_noise = cp.where(w_noise, sig_sq, 0.0).sum(axis=1)

    valid = (e_noise > eps) & (w_sig.any(axis=1)) & (w_noise.any(axis=1))
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    scores[valid] = 10.0 * cp.log10((e_signal[valid] + eps) / (e_noise[valid] + eps))
    return scores, valid, {}


def _metric_low_frequency_drift(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    drift_w = ctx.drift_window
    n, m = signals.shape

    # Moving average on host (per-signal convolve)
    sig_h = cp.asnumpy(signals)
    kernel = np.ones(drift_w, dtype=np.float64) / drift_w
    trend_h = np.empty_like(sig_h)
    for i in range(n):
        trend_h[i] = np.convolve(sig_h[i], kernel, mode="same")

    trend = cp.asarray(trend_h)
    total_energy = cp.sum(signals ** 2, axis=1)
    trend_energy = cp.sum(trend ** 2, axis=1)

    valid = total_energy > eps
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    scores[valid] = trend_energy[valid] / (total_energy[valid] + eps)
    return scores, valid, {}


def _metric_high_frequency_noise(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    bw = ctx.band_half_width_bins
    n, m = signals.shape

    hw_power, dc = _hann_windowed_power_gpu(signals, cp)
    f = hw_power.shape[1]

    search = hw_power.copy()
    search[:, dc] = -cp.inf
    k_c = cp.argmax(search, axis=1)

    bin_idx = cp.arange(f)[cp.newaxis, :]
    hf_mask = bin_idx > (k_c[:, cp.newaxis] + bw)
    hf_power = cp.where(hf_mask, hw_power, 0.0).sum(axis=1)
    total_power = hw_power[:, 1:].sum(axis=1)

    valid = total_power > eps
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    scores[valid] = hf_power[valid] / (total_power[valid] + eps)
    return scores, valid, {}


def _metric_residual_noise_energy(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    bw = ctx.band_half_width_bins
    n, m = signals.shape

    window = cp.hanning(m).astype(signals.dtype)
    x_w = signals * window[cp.newaxis, :]
    X = cp.fft.fft(x_w, axis=1)
    power_full = cp.abs(X) ** 2

    # Carrier on positive freqs (exclude DC)
    half = m // 2
    power_pos = power_full[:, 1:half + 1]
    k_c_pos = cp.argmax(power_pos, axis=1) + 1  # shift for DC offset

    # Build symmetric band mask
    bin_idx = cp.arange(m)[cp.newaxis, :]
    k_c_2d = k_c_pos[:, cp.newaxis]
    k_c_neg = (m - k_c_pos)[:, cp.newaxis]
    in_pos = cp.abs(bin_idx - k_c_2d) <= bw
    in_neg = cp.abs(bin_idx - k_c_neg) <= bw
    band_mask = in_pos | in_neg

    X_b = cp.where(band_mask, X, 0.0 + 0.0j)
    s_hat = cp.real(cp.fft.ifft(X_b, axis=1))
    residual = x_w - s_hat

    total_energy = cp.sum(x_w ** 2, axis=1)
    res_energy = cp.sum(residual ** 2, axis=1)

    valid = total_energy > eps
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    scores[valid] = res_energy[valid] / (total_energy[valid] + eps)
    return scores, valid, {}


def _metric_envelope_peak_to_background(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n = signals.shape[0]
    if envelope is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    e_peak = cp.max(envelope, axis=1)
    w_main = envelope >= 0.5 * e_peak[:, cp.newaxis]

    # Per-signal median background — host
    env_h = cp.asnumpy(envelope)
    w_h = cp.asnumpy(w_main)
    e_peak_h = cp.asnumpy(e_peak)

    scores_h = np.full(n, np.nan)
    valid_h = np.zeros(n, dtype=bool)
    for i in range(n):
        bg = env_h[i, ~w_h[i]]
        if len(bg) == 0:
            continue
        e_bg = float(np.median(bg))
        scores_h[i] = (e_peak_h[i] + eps) / (e_bg + eps)
        valid_h[i] = True

    return cp.asarray(scores_h), cp.asarray(valid_h), {}


# ------------------------------------------------------------------
# Phase metrics
# ------------------------------------------------------------------

def _phase_common(signals, cb):
    """Shared phase computation — returns analytic data on host."""
    cp = cb.cp
    ctx = cb.ctx
    envelope, phase = _hilbert_analytic_gpu(signals, cp)
    env_h = cp.asnumpy(envelope)
    phase_h = cp.asnumpy(phase)

    n, m = signals.shape
    z_axis = cb.z_axis_host
    if z_axis is not None and z_axis.shape == (m,):
        u = z_axis.astype(float)
    else:
        u = np.arange(m, dtype=float)

    alpha = ctx.phase_support_threshold_fraction
    guard = ctx.phase_guard_samples
    eps = ctx.epsilon

    n0 = np.argmax(env_h, axis=1)
    e_peak = env_h[np.arange(n), n0]
    threshold = alpha * e_peak
    candidates = env_h >= threshold[:, None]

    # Largest connected component containing peak
    support = np.zeros_like(candidates)
    for i in range(n):
        row = candidates[i]
        c = n0[i]
        if not row[c]:
            continue
        left = c
        while left > 0 and row[left - 1]:
            left -= 1
        right = c
        while right < m - 1 and row[right + 1]:
            right += 1
        support[i, left:right + 1] = True

    # Trim guard
    if guard > 0:
        for i in range(n):
            idx = np.nonzero(support[i])[0]
            if idx.size == 0:
                continue
            lo_i, hi_i = idx[0], idx[-1]
            new_lo = lo_i + guard
            new_hi = hi_i - guard
            if new_lo > new_hi:
                support[i, :] = False
            else:
                support[i, lo_i:new_lo] = False
                support[i, new_hi + 1:hi_i + 1] = False

    # Local slopes
    slopes_list = []
    pair_idx_list = []
    for i in range(n):
        idx = np.nonzero(support[i])[0]
        if idx.size < 2:
            slopes_list.append(np.array([]))
            pair_idx_list.append(np.array([], dtype=int))
            continue
        dphi = np.diff(phase_h[i, idx])
        du = np.diff(u[idx]) + eps
        slopes_list.append(dphi / du)
        pair_idx_list.append(idx[:-1])

    # Validate support
    min_samples = ctx.minimum_phase_support_samples
    min_periods = ctx.minimum_phase_support_periods
    valid_support = np.ones(n, dtype=bool)
    for i in range(n):
        idx = np.nonzero(support[i])[0]
        if idx.size < min_samples:
            valid_support[i] = False
            continue
        slopes = slopes_list[i]
        if slopes.size == 0:
            valid_support[i] = False
            continue
        d_med = float(np.median(slopes))
        if abs(d_med) < eps:
            valid_support[i] = False
            continue
        span = u[idx[-1]] - u[idx[0]]
        n_periods = abs(d_med) * span / (2.0 * np.pi)
        if n_periods < min_periods:
            valid_support[i] = False

    return env_h, phase_h, u, support, slopes_list, pair_idx_list, valid_support, e_peak, n0


def _metric_phase_monotonicity(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    tau_mon = ctx.phase_monotonicity_tolerance_fraction
    n = signals.shape[0]

    env_h, phase_h, u, support, slopes_list, pair_idx_list, valid_support, e_peak, _ = \
        _phase_common(signals, cb)

    scores_h = np.full(n, np.nan)
    for i in range(n):
        if not valid_support[i]:
            continue
        d = slopes_list[i]
        if d.size == 0:
            continue
        pidx = pair_idx_list[i]
        d_med = float(np.median(d))
        s_ref = np.sign(d_med)
        inlier = (s_ref * d > 0) & (np.abs(d - d_med) <= tau_mon * abs(d_med))
        e_i = env_h[i]
        ep = e_peak[i]
        w = np.minimum(e_i[pidx], e_i[pidx + 1]) / (ep + eps)
        w_sum = float(np.sum(w))
        if w_sum > eps:
            scores_h[i] = float(np.sum(w[inlier])) / (w_sum + eps)

    scores = cp.asarray(scores_h)
    valid = cp.asarray(valid_support) & cp.isfinite(scores)
    return scores, valid, {}


def _metric_phase_linear_fit_residual(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    p_weight = ctx.phase_weight_power
    n = signals.shape[0]

    env_h, phase_h, u, support, slopes_list, _, valid_support, e_peak, _ = \
        _phase_common(signals, cb)

    scores_h = np.full(n, np.nan)
    for i in range(n):
        if not valid_support[i]:
            continue
        idx = np.nonzero(support[i])[0]
        if idx.size < 2:
            continue
        phi = phase_h[i, idx]
        ui = u[idx]
        w = ((env_h[i, idx] / (e_peak[i] + eps)) + eps) ** p_weight
        # Weighted linear fit (normal equations)
        sw = np.sum(w)
        swu = np.sum(w * ui)
        swuu = np.sum(w * ui ** 2)
        swp = np.sum(w * phi)
        swup = np.sum(w * ui * phi)
        det = sw * swuu - swu ** 2
        if abs(det) < eps:
            continue
        beta1 = (sw * swup - swu * swp) / det
        beta0 = (swp - beta1 * swu) / sw
        r = phi - (beta0 + beta1 * ui)
        scores_h[i] = float(np.sqrt(np.sum(w * r ** 2) / (sw + eps))) / np.pi

    scores = cp.asarray(scores_h)
    valid = cp.asarray(valid_support) & cp.isfinite(scores)
    return scores, valid, {}


def _metric_phase_slope_stability(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n = signals.shape[0]

    _, _, _, _, slopes_list, _, valid_support, _, _ = _phase_common(signals, cb)

    scores_h = np.full(n, np.nan)
    for i in range(n):
        if not valid_support[i]:
            continue
        d = slopes_list[i]
        if d.size == 0:
            continue
        d_med = float(np.median(d))
        mad = float(np.median(np.abs(d - d_med)))
        scores_h[i] = mad / (abs(d_med) + eps)

    scores = cp.asarray(scores_h)
    valid = cp.asarray(valid_support) & cp.isfinite(scores)
    return scores, valid, {}


def _metric_phase_curvature_index(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    p_weight = ctx.phase_weight_power
    n = signals.shape[0]

    env_h, phase_h, u, support, _, _, valid_support, e_peak, _ = \
        _phase_common(signals, cb)

    scores_h = np.full(n, np.nan)
    for i in range(n):
        if not valid_support[i]:
            continue
        idx = np.nonzero(support[i])[0]
        if idx.size < 3:
            continue
        phi = phase_h[i, idx]
        ui = u[idx]
        w = ((env_h[i, idx] / (e_peak[i] + eps)) + eps) ** p_weight
        # Weighted quadratic fit (3x3 normal equations)
        u1 = ui
        u2 = ui ** 2
        A = np.column_stack([np.ones_like(ui), u1, u2])
        W = np.diag(w)
        AtW = A.T @ W
        AtWA = AtW @ A
        AtWp = AtW @ phi
        try:
            coeffs = np.linalg.solve(AtWA, AtWp)
        except np.linalg.LinAlgError:
            continue
        gamma1 = coeffs[1]
        gamma2 = coeffs[2]
        span = ui[-1] - ui[0]
        scores_h[i] = abs(gamma2) * span / (abs(gamma1) + eps)

    scores = cp.asarray(scores_h)
    valid = cp.asarray(valid_support) & cp.isfinite(scores)
    return scores, valid, {}


def _metric_phase_jump_fraction(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    tau_jump = ctx.phase_jump_tolerance_fraction
    n = signals.shape[0]

    _, _, _, _, slopes_list, _, valid_support, _, _ = _phase_common(signals, cb)

    scores_h = np.full(n, np.nan)
    for i in range(n):
        if not valid_support[i]:
            continue
        d = slopes_list[i]
        if d.size == 0:
            continue
        d_med = float(np.median(d))
        is_jump = (np.sign(d) != np.sign(d_med)) | \
                  (np.abs(d - d_med) > tau_jump * abs(d_med))
        scores_h[i] = float(np.sum(is_jump)) / (len(d) + eps)

    scores = cp.asarray(scores_h)
    valid = cp.asarray(valid_support) & cp.isfinite(scores)
    return scores, valid, {}


# ------------------------------------------------------------------
# Correlation metrics
# ------------------------------------------------------------------

def _correlation_common(signals, envelope, cb):
    """Shared correlation setup — reference model, support, normalization."""
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n, m = signals.shape

    z_axis = cb.z_axis_host
    # Resolve reference scales
    n_c = (m - 1) / 2.0
    if z_axis is not None and z_axis.shape == (m,):
        z = z_axis.astype(float)
        u = z - z[int(round(n_c))]
    elif ctx.z_step_nm is not None and ctx.z_step_nm > 0:
        u = (np.arange(m, dtype=float) - n_c) * ctx.z_step_nm
    else:
        u = np.arange(m, dtype=float) - n_c
        return None  # Cannot build model

    T_ref = ctx.reference_carrier_period_nm
    L_ref = ctx.reference_envelope_scale_nm
    if T_ref is None or L_ref is None or T_ref <= 0 or L_ref <= 0:
        return None

    # Build reference model
    g_ref = np.exp(-(u / L_ref) ** 2)
    phase_r = 2.0 * np.pi * u / T_ref
    r_c = g_ref * np.cos(phase_r)
    r_s = g_ref * np.sin(phase_r)

    alpha_ref = ctx.reference_support_threshold_fraction
    support = g_ref >= alpha_ref
    min_supp = ctx.minimum_reference_support_samples
    if int(np.sum(support)) < min_supp:
        return None

    # Normalize references on support
    def _norm_1d(v):
        vs = v[support]
        mean_v = vs.mean()
        centered = vs - mean_v
        norm = np.sqrt(np.sum(centered ** 2)) + eps
        out = np.zeros_like(v)
        out[support] = centered / norm
        return out

    c_norm = _norm_1d(r_c)
    s_norm = _norm_1d(r_s)
    g_norm = _norm_1d(g_ref)

    # Orthonormalize for best-phase
    q1 = c_norm
    proj = np.sum(s_norm[support] * q1[support])
    q2_raw = np.zeros_like(s_norm)
    q2_raw[support] = s_norm[support] - proj * q1[support]
    q2_norm_val = np.sqrt(np.sum(q2_raw[support] ** 2))
    q2 = np.zeros_like(s_norm)
    if q2_norm_val > eps:
        q2[support] = q2_raw[support] / (q2_norm_val + eps)

    # Normalize signals batch on support (on GPU)
    sig_gpu = signals
    vs_gpu = sig_gpu[:, support]
    mean_s = cp.mean(vs_gpu, axis=1, keepdims=True)
    centered_s = vs_gpu - mean_s
    norm_s = cp.sqrt(cp.sum(centered_s ** 2, axis=1, keepdims=True)) + eps
    x_norm_support = centered_s / norm_s  # (N, K)

    return {
        "u": u, "support": support, "c_norm": c_norm, "s_norm": s_norm,
        "g_norm": g_norm, "q1": q1, "q2": q2,
        "x_norm_support": x_norm_support,
    }


def _metric_centered_reference_correlation(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    n = signals.shape[0]
    common = _correlation_common(signals, envelope, cb)
    if common is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    c_norm_s = cp.asarray(common["c_norm"][common["support"]])  # (K,)
    x_ns = common["x_norm_support"]  # (N, K) GPU
    dot = cp.sum(x_ns * c_norm_s[cp.newaxis, :], axis=1)
    return dot, cp.ones(n, dtype=bool), {}


def _metric_best_phase_reference_correlation(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    n = signals.shape[0]
    common = _correlation_common(signals, envelope, cb)
    if common is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    q1_s = cp.asarray(common["q1"][common["support"]])
    q2_s = cp.asarray(common["q2"][common["support"]])
    x_ns = common["x_norm_support"]

    a1 = cp.sum(x_ns * q1_s[cp.newaxis, :], axis=1)
    a2 = cp.sum(x_ns * q2_s[cp.newaxis, :], axis=1)
    bprc = cp.sqrt(a1 ** 2 + a2 ** 2)
    return bprc, cp.ones(n, dtype=bool), {}


def _metric_reference_envelope_correlation(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    n = signals.shape[0]
    common = _correlation_common(signals, envelope, cb)
    if common is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    support = common["support"]
    g_norm_s = cp.asarray(common["g_norm"][support])  # (K,)

    # Need observed envelopes
    if envelope is None:
        envelope = _hilbert_envelope_gpu(signals, cp)

    # Normalize observed envelopes on support
    e_s = envelope[:, support]  # (N, K)
    mean_e = cp.mean(e_s, axis=1, keepdims=True)
    centered_e = e_s - mean_e
    norm_e = cp.sqrt(cp.sum(centered_e ** 2, axis=1, keepdims=True)) + eps
    e_norm_s = centered_e / norm_e

    dot = cp.sum(e_norm_s * g_norm_s[cp.newaxis, :], axis=1)
    return dot, cp.ones(n, dtype=bool), {}


def _metric_phase_relaxation_gain(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    n = signals.shape[0]
    common = _correlation_common(signals, envelope, cb)
    if common is None:
        return cp.full(n, cp.nan), cp.zeros(n, dtype=bool), {}

    x_ns = common["x_norm_support"]
    c_norm_s = cp.asarray(common["c_norm"][common["support"]])
    q1_s = cp.asarray(common["q1"][common["support"]])
    q2_s = cp.asarray(common["q2"][common["support"]])

    crc = cp.sum(x_ns * c_norm_s[cp.newaxis, :], axis=1)
    a1 = cp.sum(x_ns * q1_s[cp.newaxis, :], axis=1)
    a2 = cp.sum(x_ns * q2_s[cp.newaxis, :], axis=1)
    bprc = cp.sqrt(a1 ** 2 + a2 ** 2)

    prg = bprc - crc
    return prg, cp.ones(n, dtype=bool), {}


# ------------------------------------------------------------------
# Regularity metrics
# ------------------------------------------------------------------

def _metric_autocorrelation_peak_strength(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    t_exp = ctx.expected_period_samples
    delta_t = ctx.period_search_tolerance_fraction
    n, m = signals.shape

    tau_min = max(1, round((1 - delta_t) * t_exp))
    tau_max = min(round((1 + delta_t) * t_exp), m - 1)

    r0 = cp.sum(signals * signals, axis=1)
    best = cp.full(n, -cp.inf, dtype=cp.float64)

    for tau in range(tau_min, tau_max + 1):
        r_tau = cp.sum(signals[:, :m - tau] * signals[:, tau:], axis=1)
        r_norm = r_tau / (r0 + eps)
        best = cp.maximum(best, r_norm)

    valid = r0 > eps
    scores = cp.full(n, cp.nan, dtype=cp.float64)
    scores[valid] = best[valid]
    scores[~valid] = cp.nan
    scores[best == -cp.inf] = cp.nan
    valid = valid & cp.isfinite(scores)
    return scores, valid, {}


def _metric_zero_crossing_stability(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    t_exp = ctx.expected_period_samples
    delta_t = ctx.period_search_tolerance_fraction
    n, m = signals.shape

    lo_range = (1 - delta_t) * t_exp
    hi_range = (1 + delta_t) * t_exp

    # Download to host for per-signal crossing logic
    sig_h = cp.asnumpy(signals)
    scores_h = np.full(n, np.nan)
    valid_h = np.zeros(n, dtype=bool)

    for i in range(n):
        s = sig_h[i]
        crossings = []
        for j in range(m - 1):
            if s[j] < 0 and s[j + 1] >= 0:
                denom = s[j + 1] - s[j]
                if abs(denom) > eps:
                    crossings.append(j + (-s[j]) / denom)
                else:
                    crossings.append(float(j))
        if len(crossings) < 2:
            continue
        dists = np.diff(crossings)
        plausible = dists[(dists >= lo_range) & (dists <= hi_range)]
        if len(plausible) < 2:
            continue
        med = float(np.median(plausible))
        mad = float(np.median(np.abs(plausible - med)))
        scores_h[i] = mad / (abs(med) + eps)
        valid_h[i] = True

    return cp.asarray(scores_h), cp.asarray(valid_h), {}


def _metric_jitter_of_extrema(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    t_exp = ctx.expected_period_samples
    d_frac = ctx.peak_min_distance_fraction
    n, m = signals.shape

    d_min = max(1, int(d_frac * t_exp))

    sig_h = cp.asnumpy(signals)
    scores_h = np.full(n, np.nan)
    valid_h = np.zeros(n, dtype=bool)

    for i in range(n):
        s = sig_h[i]
        # Find local maxima
        cand = []
        for j in range(1, m - 1):
            if s[j] > s[j - 1] and s[j] > s[j + 1]:
                cand.append((j, s[j]))
        if len(cand) < 2:
            continue
        # Greedy selection with min distance
        cand.sort(key=lambda x: -x[1])
        keep = []
        for idx, val in cand:
            if all(abs(idx - k) >= d_min for k in keep):
                keep.append(idx)
        keep.sort()
        if len(keep) < 2:
            continue
        distances = np.diff(keep).astype(float)
        med = float(np.median(distances))
        mad = float(np.median(np.abs(distances - med)))
        scores_h[i] = mad / (med + eps)
        valid_h[i] = True

    return cp.asarray(scores_h), cp.asarray(valid_h), {}


def _metric_local_oscillation_regularity(signals, envelope, amplitude, power, frequencies, cb):
    cp = cb.cp
    ctx = cb.ctx
    eps = ctx.epsilon
    t_exp = ctx.expected_period_samples
    d_frac = ctx.peak_min_distance_fraction
    resample_len = ctx.cycle_resample_length
    n, m = signals.shape

    d_min = max(1, int(d_frac * t_exp))

    sig_h = cp.asnumpy(signals)
    scores_h = np.full(n, np.nan)
    valid_h = np.zeros(n, dtype=bool)

    for i in range(n):
        s = sig_h[i]
        # Find peaks
        cand = []
        for j in range(1, m - 1):
            if s[j] > s[j - 1] and s[j] > s[j + 1]:
                cand.append((j, s[j]))
        if len(cand) < 3:
            continue
        cand.sort(key=lambda x: -x[1])
        keep = []
        for idx, val in cand:
            if all(abs(idx - k) >= d_min for k in keep):
                keep.append(idx)
        keep.sort()
        if len(keep) < 3:
            continue

        # Resample cycles
        cycles = []
        x_new = np.linspace(0, 1, resample_len)
        for ci in range(len(keep) - 1):
            start_i, end_i = keep[ci], keep[ci + 1]
            seg = s[start_i:end_i + 1]
            if len(seg) < 2:
                continue
            x_old = np.linspace(0, 1, len(seg))
            resampled = np.interp(x_new, x_old, seg)
            resampled -= resampled.mean()
            norm_v = np.sqrt(np.dot(resampled, resampled))
            if norm_v > eps:
                resampled /= norm_v
                cycles.append(resampled)

        if len(cycles) < 2:
            continue
        cycle_arr = np.array(cycles)
        sims = np.sum(cycle_arr[:-1] * cycle_arr[1:], axis=1)
        scores_h[i] = float(np.median(sims))
        valid_h[i] = True

    return cp.asarray(scores_h), cp.asarray(valid_h), {}


# ------------------------------------------------------------------
# Dispatch table
# ------------------------------------------------------------------

_DISPATCH = {
    # baseline
    "snr": _metric_snr,
    "fringe_visibility": _metric_fringe_visibility,
    "power_band_ratio": _metric_power_band_ratio,
    # envelope
    "envelope_height": _metric_envelope_height,
    "envelope_area": _metric_envelope_area,
    "envelope_width": _metric_envelope_width,
    "envelope_sharpness": _metric_envelope_sharpness,
    "envelope_symmetry": _metric_envelope_symmetry,
    "single_peakness": _metric_single_peakness,
    "main_peak_to_sidelobe_ratio": _metric_main_peak_to_sidelobe,
    # spectral
    "spectral_entropy": _metric_spectral_entropy,
    "spectral_centroid_offset": _metric_spectral_centroid_offset,
    "dominant_spectral_peak_prominence": _metric_dominant_spectral_peak_prominence,
    "energy_concentration_in_working_band": _metric_energy_concentration,
    "carrier_to_background_spectral_ratio": _metric_carrier_to_background,
    "presence_of_expected_carrier_frequency": _metric_presence_expected_carrier,
    "spectral_peak_sharpness": _metric_spectral_peak_sharpness,
    "spectral_kurtosis": _metric_spectral_kurtosis,
    "spectral_spread": _metric_spectral_spread,
    "envelope_spectrum_consistency": _metric_envelope_spectrum_consistency,
    # noise
    "spectral_snr": _metric_spectral_snr,
    "local_snr": _metric_local_snr,
    "low_frequency_drift_level": _metric_low_frequency_drift,
    "high_frequency_noise_level": _metric_high_frequency_noise,
    "residual_noise_energy": _metric_residual_noise_energy,
    "envelope_peak_to_background_ratio": _metric_envelope_peak_to_background,
    # phase
    "phase_monotonicity_score": _metric_phase_monotonicity,
    "phase_linear_fit_residual": _metric_phase_linear_fit_residual,
    "phase_slope_stability": _metric_phase_slope_stability,
    "phase_curvature_index": _metric_phase_curvature_index,
    "phase_jump_fraction": _metric_phase_jump_fraction,
    # correlation
    "centered_reference_correlation": _metric_centered_reference_correlation,
    "best_phase_reference_correlation": _metric_best_phase_reference_correlation,
    "reference_envelope_correlation": _metric_reference_envelope_correlation,
    "phase_relaxation_gain": _metric_phase_relaxation_gain,
    # regularity
    "autocorrelation_peak_strength": _metric_autocorrelation_peak_strength,
    "zero_crossing_stability": _metric_zero_crossing_stability,
    "jitter_of_extrema": _metric_jitter_of_extrema,
    "local_oscillation_regularity": _metric_local_oscillation_regularity,
}
