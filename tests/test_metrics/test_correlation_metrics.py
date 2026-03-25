"""Tests for correlation / reference-model quality metrics.

Covers:
- valid output on a clean synthetic WLI-like signal with metadata
- invalid when metadata is unavailable
- batch vs single-signal consistency
- score direction sanity
- reference support and normalization helpers
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.analysis_context import AnalysisContext
from quality_tool.metrics.correlation.centered_reference_correlation import (
    CenteredReferenceCorrelation,
)
from quality_tool.metrics.correlation.best_phase_reference_correlation import (
    BestPhaseReferenceCorrelation,
)
from quality_tool.metrics.correlation.reference_envelope_correlation import (
    ReferenceEnvelopeCorrelation,
)
from quality_tool.metrics.correlation.phase_relaxation_gain import PhaseRelaxationGain
from quality_tool.metrics.correlation._helpers import (
    build_reference_model,
    build_reference_support,
    normalize_on_support,
    orthonormalize_basis,
    resolve_reference_scales,
)


# Metadata that provides valid physical scaling.
_WAVELENGTH_NM = 600.0
_COHERENCE_LENGTH_NM = 3000.0
_Z_STEP_NM = 75.0


def _metadata_context() -> AnalysisContext:
    """Context with valid physical scaling for reference model."""
    return AnalysisContext(
        wavelength_nm=_WAVELENGTH_NM,
        coherence_length_nm=_COHERENCE_LENGTH_NM,
        z_step_nm=_Z_STEP_NM,
        reference_carrier_period_nm=_WAVELENGTH_NM,
        reference_envelope_scale_nm=_COHERENCE_LENGTH_NM / 2.0,
    )


def _make_ideal_reference_signal(m: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Build an ideal signal matching the reference model exactly.

    Returns the signal and its z-axis.
    """
    ctx = _metadata_context()
    T_ref = ctx.reference_carrier_period_nm
    L_ref = ctx.reference_envelope_scale_nm
    z_step = ctx.z_step_nm
    n_c = (m - 1) / 2.0
    n = np.arange(m, dtype=float)
    u = (n - n_c) * z_step
    g = np.exp(-(u / L_ref) ** 2)
    signal = g * np.cos(2.0 * np.pi * u / T_ref)
    z_axis = n * z_step
    return signal, z_axis


def _ctx_dict(ctx: AnalysisContext | None = None) -> dict:
    if ctx is None:
        ctx = _metadata_context()
    return {"analysis_context": ctx}


ALL_CORR_METRICS = [
    CenteredReferenceCorrelation(),
    BestPhaseReferenceCorrelation(),
    ReferenceEnvelopeCorrelation(),
    PhaseRelaxationGain(),
]


# ---- Metric attribute tests ----

@pytest.mark.parametrize("metric", ALL_CORR_METRICS, ids=lambda m: m.name)
def test_metric_attributes(metric):
    """All correlation metrics have required attributes."""
    assert metric.category == "correlation"
    assert metric.recipe_binding == "fixed"
    assert hasattr(metric, "display_name")
    assert hasattr(metric, "score_direction")
    assert hasattr(metric, "evaluate")
    assert hasattr(metric, "evaluate_batch")


# ---- Valid on ideal signal ----

@pytest.mark.parametrize("metric", ALL_CORR_METRICS, ids=lambda m: m.name)
def test_valid_on_ideal_signal(metric):
    """Correlation metrics should produce valid results when metadata is present."""
    signal, z_axis = _make_ideal_reference_signal()
    ctx = _ctx_dict()
    result = metric.evaluate(signal, z_axis=z_axis, context=ctx)
    assert result.valid, f"{metric.name} returned invalid on ideal signal"
    assert np.isfinite(result.score)


# ---- Batch consistency ----

@pytest.mark.parametrize("metric", ALL_CORR_METRICS, ids=lambda m: m.name)
def test_batch_consistency(metric):
    """Batch and single-signal evaluation should agree."""
    signal, z_axis = _make_ideal_reference_signal()
    ctx = _ctx_dict()
    single = metric.evaluate(signal, z_axis=z_axis, context=ctx)
    batch = metric.evaluate_batch(signal[np.newaxis, :], z_axis, None, ctx)
    if single.valid:
        assert batch.valid[0]
        np.testing.assert_allclose(batch.scores[0], single.score, rtol=1e-6)


# ---- Invalid without metadata ----

def test_invalid_without_metadata():
    """Correlation metrics should return invalid when metadata is missing."""
    signal, z_axis = _make_ideal_reference_signal()
    # No wavelength/coherence → no reference model.
    ctx = _ctx_dict(AnalysisContext())
    for metric in ALL_CORR_METRICS:
        result = metric.evaluate(signal, z_axis=z_axis, context=ctx)
        assert not result.valid, f"{metric.name} should be invalid without metadata"


def test_no_context_returns_invalid():
    """Missing analysis context should return invalid."""
    signal, _ = _make_ideal_reference_signal()
    for metric in ALL_CORR_METRICS:
        result = metric.evaluate(signal, context=None)
        assert not result.valid


# ---- Score direction sanity ----

def test_ideal_signal_high_crc():
    """Ideal centered signal should have CRC close to 1."""
    signal, z_axis = _make_ideal_reference_signal()
    ctx = _ctx_dict()
    crc = CenteredReferenceCorrelation()
    result = crc.evaluate(signal, z_axis=z_axis, context=ctx)
    assert result.valid
    assert result.score > 0.9


def test_ideal_signal_high_bprc():
    """Ideal signal should have BPRC close to 1."""
    signal, z_axis = _make_ideal_reference_signal()
    ctx = _ctx_dict()
    bprc = BestPhaseReferenceCorrelation()
    result = bprc.evaluate(signal, z_axis=z_axis, context=ctx)
    assert result.valid
    assert result.score > 0.9


def test_ideal_signal_high_rec():
    """Ideal signal should have high reference envelope correlation."""
    signal, z_axis = _make_ideal_reference_signal()
    ctx = _ctx_dict()
    rec = ReferenceEnvelopeCorrelation()
    result = rec.evaluate(signal, z_axis=z_axis, context=ctx)
    assert result.valid
    assert result.score > 0.9


def test_ideal_signal_low_prg():
    """Ideal centered signal should have PRG close to 0."""
    signal, z_axis = _make_ideal_reference_signal()
    ctx = _ctx_dict()
    prg = PhaseRelaxationGain()
    result = prg.evaluate(signal, z_axis=z_axis, context=ctx)
    assert result.valid
    assert result.score < 0.1


def test_phase_shifted_signal_lower_crc():
    """A phase-shifted signal should have lower CRC but similar BPRC."""
    signal, z_axis = _make_ideal_reference_signal(m=256)
    ctx = _ctx_dict()
    # Shift the signal by a fraction of a period.
    shifted = np.roll(signal, 3)
    crc = CenteredReferenceCorrelation()
    bprc = BestPhaseReferenceCorrelation()
    r_orig = crc.evaluate(signal, z_axis=z_axis, context=ctx)
    r_shifted = crc.evaluate(shifted, z_axis=z_axis, context=ctx)
    r_bprc_orig = bprc.evaluate(signal, z_axis=z_axis, context=ctx)
    r_bprc_shifted = bprc.evaluate(shifted, z_axis=z_axis, context=ctx)
    if r_orig.valid and r_shifted.valid:
        assert r_shifted.score < r_orig.score
    if r_bprc_orig.valid and r_bprc_shifted.valid:
        # BPRC should be more tolerant of phase shift.
        assert r_bprc_shifted.score > r_shifted.score


# ---- Helper tests ----

def test_normalize_on_support():
    """Normalized vector on support should be zero-mean unit-norm."""
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    support = np.array([True, True, True, True, True])
    eps = 1e-12
    v_norm = normalize_on_support(v, support, eps)
    np.testing.assert_allclose(v_norm[support].mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(
        np.sqrt(np.sum(v_norm[support] ** 2)), 1.0, atol=1e-6,
    )


def test_orthonormalize_basis():
    """q1 and q2 should be orthonormal on support."""
    m = 50
    support = np.ones(m, dtype=bool)
    eps = 1e-12
    u = np.linspace(-1, 1, m)
    c = np.cos(2 * np.pi * u)
    s = np.sin(2 * np.pi * u)
    c_norm = normalize_on_support(c, support, eps)
    s_norm = normalize_on_support(s, support, eps)
    q1, q2, _ = orthonormalize_basis(c_norm, s_norm, support, eps)
    # Orthogonal.
    dot = np.sum(q1[support] * q2[support])
    np.testing.assert_allclose(dot, 0.0, atol=1e-8)
    # Unit norm.
    np.testing.assert_allclose(np.sum(q1[support] ** 2), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.sum(q2[support] ** 2), 1.0, atol=1e-6)


def test_reference_support_threshold():
    """Reference support should respect the threshold fraction."""
    u = np.linspace(-5, 5, 100)
    g_ref = np.exp(-u ** 2)
    support = build_reference_support(g_ref, 0.1)
    # All points where g_ref >= 0.1 should be in support.
    expected = g_ref >= 0.1
    np.testing.assert_array_equal(support, expected)


# ---- Multi-signal batch test ----

def test_batch_multiple_signals():
    """Batch evaluation on multiple signals."""
    signal, z_axis = _make_ideal_reference_signal()
    rng = np.random.default_rng(42)
    noise_signal = rng.standard_normal(signal.shape) * 0.01
    signals = np.stack([signal, noise_signal])
    ctx = _ctx_dict()
    crc = CenteredReferenceCorrelation()
    result = crc.evaluate_batch(signals, z_axis, None, ctx)
    assert result.scores.shape == (2,)
    assert result.valid.shape == (2,)
    # Ideal signal should score much higher.
    if result.valid[0] and result.valid[1]:
        assert result.scores[0] > result.scores[1]
