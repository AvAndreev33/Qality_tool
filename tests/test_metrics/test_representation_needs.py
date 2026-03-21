"""Tests for RepresentationNeeds and resolve_representation_needs."""

from __future__ import annotations

import numpy as np

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.recipe import RAW, SignalRecipe
from quality_tool.metrics.base import (
    NO_EXTRA,
    RepresentationNeeds,
    resolve_representation_needs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubMetric:
    """Minimal metric stub."""

    def __init__(self, name="stub", **kwargs):
        self.name = name
        self.signal_recipe = RAW
        self.recipe_binding = "active"
        for k, v in kwargs.items():
            setattr(self, k, v)

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        return MetricResult(score=0.0)


# ---------------------------------------------------------------------------
# Tests — RepresentationNeeds
# ---------------------------------------------------------------------------

class TestRepresentationNeeds:
    def test_defaults_all_false(self):
        n = RepresentationNeeds()
        assert n.envelope is False
        assert n.amplitude is False
        assert n.power is False
        assert n.complex_fft is False

    def test_needs_spectral_false_by_default(self):
        assert RepresentationNeeds().needs_spectral is False

    def test_needs_spectral_true_for_amplitude(self):
        assert RepresentationNeeds(amplitude=True).needs_spectral is True

    def test_needs_spectral_true_for_power(self):
        assert RepresentationNeeds(power=True).needs_spectral is True

    def test_needs_spectral_true_for_complex(self):
        assert RepresentationNeeds(complex_fft=True).needs_spectral is True

    def test_frozen(self):
        n = RepresentationNeeds()
        try:
            n.amplitude = True  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_no_extra_is_all_false(self):
        assert NO_EXTRA == RepresentationNeeds()


# ---------------------------------------------------------------------------
# Tests — merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_identity(self):
        a = RepresentationNeeds(amplitude=True)
        result = a.merge(NO_EXTRA)
        assert result.amplitude is True
        assert result.power is False

    def test_merge_combines_flags(self):
        a = RepresentationNeeds(amplitude=True)
        b = RepresentationNeeds(power=True, envelope=True)
        result = a.merge(b)
        assert result.amplitude is True
        assert result.power is True
        assert result.envelope is True
        assert result.complex_fft is False

    def test_merge_is_commutative(self):
        a = RepresentationNeeds(amplitude=True)
        b = RepresentationNeeds(power=True)
        assert a.merge(b) == b.merge(a)

    def test_merge_all_true(self):
        all_true = RepresentationNeeds(
            envelope=True, amplitude=True, power=True, complex_fft=True,
        )
        result = all_true.merge(NO_EXTRA)
        assert result == all_true

    def test_merge_two_empty(self):
        assert NO_EXTRA.merge(NO_EXTRA) == NO_EXTRA


# ---------------------------------------------------------------------------
# Tests — resolve_representation_needs
# ---------------------------------------------------------------------------

class TestResolveRepresentationNeeds:
    def test_explicit_representation_needs(self):
        needs = RepresentationNeeds(amplitude=True, power=True)
        metric = _StubMetric(representation_needs=needs)
        assert resolve_representation_needs(metric) == needs

    def test_legacy_needs_spectral_true(self):
        metric = _StubMetric(needs_spectral=True)
        result = resolve_representation_needs(metric)
        assert result.amplitude is True
        assert result.power is False

    def test_legacy_needs_spectral_false(self):
        metric = _StubMetric(needs_spectral=False)
        result = resolve_representation_needs(metric)
        assert result == NO_EXTRA

    def test_no_hint_at_all(self):
        metric = _StubMetric()
        result = resolve_representation_needs(metric)
        assert result == NO_EXTRA

    def test_explicit_takes_precedence_over_legacy(self):
        needs = RepresentationNeeds(power=True)
        metric = _StubMetric(
            representation_needs=needs, needs_spectral=True,
        )
        result = resolve_representation_needs(metric)
        # Explicit representation_needs wins.
        assert result == needs
        assert result.amplitude is False
        assert result.power is True
