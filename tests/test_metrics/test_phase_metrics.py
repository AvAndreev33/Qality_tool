"""Tests for phase quality metrics.

Covers:
- valid output on a clean synthetic WLI-like signal
- invalid output on noise / insufficient support
- batch vs single-signal consistency
- analysis-context parameter usage
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.analysis_context import AnalysisContext
from quality_tool.metrics.phase.phase_slope_stability import PhaseSlopeStability
from quality_tool.metrics.phase.phase_linear_fit_residual import PhaseLinearFitResidual
from quality_tool.metrics.phase.phase_curvature_index import PhaseCurvatureIndex
from quality_tool.metrics.phase.phase_monotonicity_score import PhaseMonotonicityScore
from quality_tool.metrics.phase.phase_jump_fraction import PhaseJumpFraction


def _make_clean_wli_signal(
    m: int = 128,
    period: float = 8.0,
    sigma: float = 25.0,
) -> np.ndarray:
    """Create a clean Gaussian-enveloped cosine (ideal WLI correlogram)."""
    n = np.arange(m, dtype=float)
    center = (m - 1) / 2.0
    envelope = np.exp(-((n - center) / sigma) ** 2)
    carrier = np.cos(2.0 * np.pi * n / period)
    return envelope * carrier


def _make_noisy_signal(m: int = 128, rng: np.random.Generator | None = None) -> np.ndarray:
    """Pure noise signal with no coherent structure."""
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.standard_normal(m) * 0.01


def _default_context() -> dict:
    return {"analysis_context": AnalysisContext()}


# ---- Metric attribute tests ----

ALL_PHASE_METRICS = [
    PhaseSlopeStability(),
    PhaseLinearFitResidual(),
    PhaseCurvatureIndex(),
    PhaseMonotonicityScore(),
    PhaseJumpFraction(),
]


@pytest.mark.parametrize("metric", ALL_PHASE_METRICS, ids=lambda m: m.name)
def test_metric_attributes(metric):
    """All phase metrics have required attributes."""
    assert metric.category == "phase"
    assert metric.recipe_binding == "fixed"
    assert hasattr(metric, "display_name")
    assert hasattr(metric, "score_direction")
    assert hasattr(metric, "evaluate")
    assert hasattr(metric, "evaluate_batch")


# ---- Clean signal tests ----

@pytest.mark.parametrize("metric", ALL_PHASE_METRICS, ids=lambda m: m.name)
def test_valid_on_clean_signal(metric):
    """Phase metrics should produce valid results on a clean WLI signal."""
    signal = _make_clean_wli_signal()
    ctx = _default_context()
    result = metric.evaluate(signal, context=ctx)
    assert result.valid, f"{metric.name} returned invalid on clean signal"
    assert np.isfinite(result.score)


@pytest.mark.parametrize("metric", ALL_PHASE_METRICS, ids=lambda m: m.name)
def test_batch_consistency(metric):
    """Batch and single-signal evaluation should agree."""
    signal = _make_clean_wli_signal()
    ctx = _default_context()
    single = metric.evaluate(signal, context=ctx)
    batch = metric.evaluate_batch(signal[np.newaxis, :], None, None, ctx)
    if single.valid:
        assert batch.valid[0]
        np.testing.assert_allclose(batch.scores[0], single.score, rtol=1e-6)


# ---- Invalid-case tests ----

def test_noise_signal_may_be_invalid():
    """On pure noise, some phase metrics should return invalid."""
    signal = _make_noisy_signal()
    ctx = _default_context()
    # At least one metric should fail on pure noise.
    results = [m.evaluate(signal, context=ctx) for m in ALL_PHASE_METRICS]
    invalid_count = sum(1 for r in results if not r.valid)
    assert invalid_count > 0, "Expected at least one invalid result on pure noise"


def test_short_signal_invalid():
    """Signals too short for support should be invalid."""
    signal = _make_clean_wli_signal(m=8, sigma=2.0)
    ctx = {"analysis_context": AnalysisContext(minimum_phase_support_samples=20)}
    for metric in ALL_PHASE_METRICS:
        result = metric.evaluate(signal, context=ctx)
        assert not result.valid


def test_no_context_returns_invalid():
    """Missing analysis context should return invalid."""
    signal = _make_clean_wli_signal()
    for metric in ALL_PHASE_METRICS:
        result = metric.evaluate(signal, context=None)
        assert not result.valid


# ---- Score direction sanity ----

def test_slope_stability_clean_vs_noisy():
    """Clean signal should have lower (better) PSS than a perturbed one."""
    clean = _make_clean_wli_signal(m=256, sigma=50.0)
    # Add phase noise to create a perturbed signal.
    rng = np.random.default_rng(123)
    noisy = clean + rng.standard_normal(256) * 0.3 * np.max(np.abs(clean))

    ctx = _default_context()
    pss = PhaseSlopeStability()
    r_clean = pss.evaluate(clean, context=ctx)
    r_noisy = pss.evaluate(noisy, context=ctx)
    if r_clean.valid and r_noisy.valid:
        assert r_clean.score < r_noisy.score


def test_monotonicity_clean_is_high():
    """Clean signal should have high monotonicity score."""
    signal = _make_clean_wli_signal(m=256, sigma=50.0)
    ctx = _default_context()
    pms = PhaseMonotonicityScore()
    result = pms.evaluate(signal, context=ctx)
    if result.valid:
        assert result.score > 0.5


def test_jump_fraction_clean_is_low():
    """Clean signal should have low jump fraction."""
    signal = _make_clean_wli_signal(m=256, sigma=50.0)
    ctx = _default_context()
    pjf = PhaseJumpFraction()
    result = pjf.evaluate(signal, context=ctx)
    if result.valid:
        assert result.score < 0.3


# ---- Multi-signal batch test ----

def test_batch_multiple_signals():
    """Batch evaluation on a mix of clean and noisy signals."""
    rng = np.random.default_rng(99)
    clean = _make_clean_wli_signal(m=128)
    noisy = _make_noisy_signal(m=128, rng=rng)
    signals = np.stack([clean, noisy])
    ctx = _default_context()
    pss = PhaseSlopeStability()
    result = pss.evaluate_batch(signals, None, None, ctx)
    assert result.scores.shape == (2,)
    assert result.valid.shape == (2,)
    # Clean signal should be valid.
    assert result.valid[0]
