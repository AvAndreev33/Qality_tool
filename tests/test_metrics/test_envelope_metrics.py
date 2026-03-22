"""Tests for envelope metric implementations.

Covers: metadata, scalar evaluation, batch evaluation, scalar/batch
consistency, invalid-case handling, analysis-context parameter usage,
helper functions, and GUI grouping.
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.analysis_context import AnalysisContext
from quality_tool.core.models import MetricResult
from quality_tool.metrics.envelope.envelope_height import EnvelopeHeight
from quality_tool.metrics.envelope.envelope_area import EnvelopeArea
from quality_tool.metrics.envelope.envelope_width import EnvelopeWidth
from quality_tool.metrics.envelope.envelope_sharpness import EnvelopeSharpness
from quality_tool.metrics.envelope.envelope_symmetry import EnvelopeSymmetry
from quality_tool.metrics.envelope.single_peakness import SinglePeakness
from quality_tool.metrics.envelope.main_peak_to_sidelobe_ratio import (
    MainPeakToSidelobeRatio,
)
from quality_tool.metrics.envelope.num_significant_secondary_peaks import (
    NumSignificantSecondaryPeaks,
)
from quality_tool.metrics.envelope._envelope_helpers import (
    detect_secondary_peaks,
    half_max_crossings_batch,
    main_support_mask_batch,
)


# ---- test signal helpers ----

_M = 128


def _make_context(ctx: AnalysisContext | None = None) -> dict:
    return {"analysis_context": ctx or AnalysisContext()}


def _gaussian_envelope(
    m: int = _M, center: int | None = None, sigma: float = 10.0,
    peak: float = 1.0,
) -> np.ndarray:
    """Symmetric Gaussian envelope centered at *center*."""
    if center is None:
        center = m // 2
    x = np.arange(m, dtype=float)
    return peak * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _double_peak_envelope(
    m: int = 256, sigma: float = 8.0, sep: float = 80.0,
) -> np.ndarray:
    """Envelope with a main peak and a well-separated secondary peak."""
    x = np.arange(m, dtype=float)
    center = m // 2
    main = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    side = 0.3 * np.exp(-0.5 * ((x - center - sep) / sigma) ** 2)
    return main + side


def _flat_envelope(m: int = _M) -> np.ndarray:
    return np.zeros(m)


# ---- all metric instances ----

_ALL_METRICS = [
    EnvelopeHeight(),
    EnvelopeArea(),
    EnvelopeWidth(),
    EnvelopeSharpness(),
    EnvelopeSymmetry(),
    SinglePeakness(),
    MainPeakToSidelobeRatio(),
    NumSignificantSecondaryPeaks(),
]


# ---- metadata tests ----


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_category_is_envelope(metric):
    assert metric.category == "envelope"


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_display_name_is_readable(metric):
    assert metric.display_name != metric.name
    assert len(metric.display_name) > 3


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_recipe_binding_is_fixed(metric):
    assert metric.recipe_binding == "fixed"


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_needs_envelope(metric):
    assert metric.representation_needs.envelope is True


# ---- invalid-case tests ----


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_none_envelope_returns_invalid(metric):
    sig = np.zeros(32)
    result = metric.evaluate(sig, envelope=None, context=_make_context())
    assert result.valid is False


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_empty_envelope_returns_invalid(metric):
    sig = np.zeros(0)
    result = metric.evaluate(sig, envelope=np.array([]), context=_make_context())
    assert result.valid is False


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_nonfinite_envelope_returns_invalid(metric):
    env = np.ones(32)
    env[10] = np.inf
    result = metric.evaluate(np.zeros(32), envelope=env, context=_make_context())
    assert result.valid is False


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_flat_zero_envelope_returns_invalid(metric):
    """A flat zero envelope should be invalid for most metrics."""
    env = _flat_envelope(32)
    result = metric.evaluate(np.zeros(32), envelope=env, context=_make_context())
    # envelope_height and envelope_area give 0.0 as valid score; that's
    # acceptable since max/sum are well-defined on zero envelopes.
    if metric.name in ("envelope_height", "envelope_area"):
        # These still return valid=True with score 0.
        return
    assert result.valid is False


# ---- scalar evaluation on Gaussian envelope ----


def test_envelope_height_gaussian():
    env = _gaussian_envelope(peak=5.0)
    r = EnvelopeHeight().evaluate(np.zeros(_M), envelope=env, context=_make_context())
    assert r.valid is True
    assert r.score == pytest.approx(5.0, abs=0.01)


def test_envelope_area_gaussian():
    env = _gaussian_envelope(peak=1.0, sigma=10.0)
    r = EnvelopeArea().evaluate(np.zeros(_M), envelope=env, context=_make_context())
    assert r.valid is True
    # Gaussian area ≈ sqrt(2*pi)*sigma*peak ≈ 25.07
    assert r.score > 20.0


def test_envelope_width_gaussian():
    sigma = 10.0
    env = _gaussian_envelope(sigma=sigma)
    r = EnvelopeWidth().evaluate(np.zeros(_M), envelope=env, context=_make_context())
    assert r.valid is True
    # FWHM of Gaussian = 2*sqrt(2*ln2)*sigma ≈ 23.55
    expected_fwhm = 2.0 * np.sqrt(2 * np.log(2)) * sigma
    assert r.score == pytest.approx(expected_fwhm, rel=0.05)


def test_envelope_sharpness_gaussian():
    env = _gaussian_envelope(peak=2.0, sigma=10.0)
    r = EnvelopeSharpness().evaluate(np.zeros(_M), envelope=env, context=_make_context())
    assert r.valid is True
    assert r.score > 0.0


def test_envelope_symmetry_gaussian():
    """A symmetric Gaussian should have symmetry near 1.0."""
    env = _gaussian_envelope()
    r = EnvelopeSymmetry().evaluate(np.zeros(_M), envelope=env, context=_make_context())
    assert r.valid is True
    assert r.score > 0.95


def test_single_peakness_gaussian():
    """A clean Gaussian should have single-peakness near 1.0."""
    env = _gaussian_envelope()
    r = SinglePeakness().evaluate(np.zeros(_M), envelope=env, context=_make_context())
    assert r.valid is True
    assert r.score > 0.8


def test_main_peak_sidelobe_no_sidelobes():
    """A clean narrow Gaussian should have a very high peak-to-sidelobe ratio."""
    # Use a narrow Gaussian so tails drop well below alpha*e_peak.
    env = _gaussian_envelope(sigma=3.0)
    r = MainPeakToSidelobeRatio().evaluate(
        np.zeros(_M), envelope=env, context=_make_context(),
    )
    assert r.valid is True
    assert r.score > 100.0  # essentially no sidelobes


def test_num_secondary_peaks_gaussian():
    """A clean Gaussian should have 0 significant secondary peaks."""
    env = _gaussian_envelope()
    r = NumSignificantSecondaryPeaks().evaluate(
        np.zeros(_M), envelope=env, context=_make_context(),
    )
    assert r.valid is True
    assert r.score == 0.0


# ---- double-peak tests ----


def test_single_peakness_double_peak():
    """A double-peak envelope should have lower single-peakness than a clean one."""
    double_env = _double_peak_envelope()
    m = len(double_env)
    clean_env = _gaussian_envelope(m=m)
    ctx = _make_context()
    r_clean = SinglePeakness().evaluate(np.zeros(m), envelope=clean_env, context=ctx)
    r_double = SinglePeakness().evaluate(np.zeros(m), envelope=double_env, context=ctx)
    assert r_clean.valid and r_double.valid
    assert r_double.score < r_clean.score


def test_main_peak_sidelobe_double_peak():
    """A double-peak envelope should have a finite sidelobe ratio."""
    env = _double_peak_envelope()
    m = len(env)
    # Use alpha=0.4 so the 0.3-height secondary peak falls outside support.
    ctx = _make_context(AnalysisContext(alpha_main_support=0.4))
    r = MainPeakToSidelobeRatio().evaluate(
        np.zeros(m), envelope=env, context=ctx,
    )
    assert r.valid is True
    assert r.score < 100.0  # secondary peak is visible


def test_num_secondary_peaks_double_peak():
    """A double-peak envelope should report at least 1 secondary peak."""
    env = _double_peak_envelope()
    m = len(env)
    # Use alpha=0.4 so the 0.3-height secondary peak falls outside support.
    ctx = _make_context(AnalysisContext(alpha_main_support=0.4))
    r = NumSignificantSecondaryPeaks().evaluate(
        np.zeros(m), envelope=env, context=ctx,
    )
    assert r.valid is True
    assert r.score >= 1.0


# ---- asymmetric envelope test ----


def test_symmetry_asymmetric():
    """An asymmetric envelope should score lower than a symmetric one."""
    sym_env = _gaussian_envelope()
    asym_env = sym_env.copy()
    # Attenuate the right side.
    center = _M // 2
    asym_env[center:] *= 0.5

    sym_score = EnvelopeSymmetry().evaluate(
        np.zeros(_M), envelope=sym_env, context=_make_context(),
    ).score
    asym_score = EnvelopeSymmetry().evaluate(
        np.zeros(_M), envelope=asym_env, context=_make_context(),
    ).score

    assert sym_score > asym_score


# ---- batch evaluation tests ----


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_batch_none_envelopes(metric):
    """Batch with None envelopes returns all invalid."""
    sigs = np.zeros((5, 32))
    result = metric.evaluate_batch(sigs, envelopes=None, context=_make_context())
    assert result.valid.shape == (5,)
    assert not np.any(result.valid)


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_batch_scalar_consistency(metric):
    """Batch results should match scalar results."""
    envs = np.stack([
        _gaussian_envelope(64, center=32, sigma=8.0, peak=2.0),
        _gaussian_envelope(64, center=32, sigma=5.0, peak=3.0),
    ])
    sigs = np.zeros_like(envs)
    ctx = _make_context()

    # Scalar.
    scalars = [metric.evaluate(sigs[i], envelope=envs[i], context=ctx)
               for i in range(2)]

    # Batch.
    batch = metric.evaluate_batch(sigs, envelopes=envs, context=ctx)

    for i in range(2):
        if scalars[i].valid:
            assert batch.valid[i], f"signal {i}: scalar valid but batch invalid"
            assert batch.scores[i] == pytest.approx(scalars[i].score, rel=1e-6), \
                f"signal {i}: score mismatch"
        else:
            assert not batch.valid[i], f"signal {i}: scalar invalid but batch valid"


# ---- helper tests ----


class TestHalfMaxCrossings:
    def test_symmetric_gaussian(self):
        env = _gaussian_envelope(64, center=32, sigma=8.0)[np.newaxis, :]
        n0 = np.array([32])
        e_peak = np.array([1.0])
        z_l, z_r, valid = half_max_crossings_batch(env, n0, e_peak)
        assert valid[0]
        fwhm = z_r[0] - z_l[0]
        expected = 2.0 * np.sqrt(2 * np.log(2)) * 8.0
        assert fwhm == pytest.approx(expected, rel=0.05)

    def test_flat_returns_invalid(self):
        env = np.ones((1, 32))
        n0 = np.array([16])
        e_peak = np.array([1.0])
        _, _, valid = half_max_crossings_batch(env, n0, e_peak)
        assert not valid[0]


class TestMainSupportMask:
    def test_basic(self):
        env = np.array([[0.0, 0.5, 1.0, 0.5, 0.0]])
        mask = main_support_mask_batch(env, np.array([1.0]), alpha=0.5)
        assert mask.shape == (1, 5)
        np.testing.assert_array_equal(mask[0], [False, True, True, True, False])


class TestDetectSecondaryPeaks:
    def test_no_secondary(self):
        # Use a narrow Gaussian so tails drop well below alpha threshold.
        env = _gaussian_envelope(64, center=32, sigma=3.0)
        main_mask = env >= 0.1 * np.max(env)
        sec = detect_secondary_peaks(env, main_mask, min_distance=3)
        assert len(sec) == 0

    def test_double_peak(self):
        env = _double_peak_envelope()
        # Use alpha=0.4 so 0.3-height secondary peak is outside support.
        main_mask = env >= 0.4 * np.max(env)
        sec = detect_secondary_peaks(env, main_mask, min_distance=3)
        assert len(sec) >= 1


# ---- analysis-context parameter usage ----


def test_alpha_affects_single_peakness():
    """Different alpha values should change single-peakness scores."""
    env = _double_peak_envelope()
    sig = np.zeros(len(env))

    r_narrow = SinglePeakness().evaluate(
        sig, envelope=env,
        context=_make_context(AnalysisContext(alpha_main_support=0.5)),
    )
    r_wide = SinglePeakness().evaluate(
        sig, envelope=env,
        context=_make_context(AnalysisContext(alpha_main_support=0.01)),
    )
    # Wider support => higher single-peakness.
    assert r_narrow.valid and r_wide.valid
    assert r_wide.score >= r_narrow.score


def test_beta_affects_secondary_count():
    """A higher beta threshold should count fewer secondary peaks."""
    env = _double_peak_envelope()
    sig = np.zeros(len(env))

    r_low = NumSignificantSecondaryPeaks().evaluate(
        sig, envelope=env,
        context=_make_context(AnalysisContext(beta_secondary_peak=0.05)),
    )
    r_high = NumSignificantSecondaryPeaks().evaluate(
        sig, envelope=env,
        context=_make_context(AnalysisContext(beta_secondary_peak=0.5)),
    )
    assert r_low.valid and r_high.valid
    assert r_high.score <= r_low.score


# ---- GUI grouping test ----


def test_envelope_metrics_registered_in_default_registry():
    """Envelope metrics should be in the default registry under 'envelope'."""
    from quality_tool.metrics.envelope import ALL_ENVELOPE_METRICS
    from quality_tool.metrics.registry import MetricRegistry

    registry = MetricRegistry()
    for m in ALL_ENVELOPE_METRICS:
        registry.register(m)

    groups = dict(registry.list_grouped())
    assert "envelope" in groups
    names = [name for name, _ in groups["envelope"]]
    assert len(names) == 8
    assert "envelope_height" in names
    assert "num_significant_secondary_peaks" in names


def test_metrics_dialog_shows_envelope_label():
    """The category label mapping should include envelope."""
    from quality_tool.gui.dialogs.metrics_dialog import _CATEGORY_LABELS

    assert "envelope" in _CATEGORY_LABELS
    assert _CATEGORY_LABELS["envelope"] == "Envelope metrics"
