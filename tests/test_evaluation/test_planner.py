"""Tests for quality_tool.evaluation.planner — lightweight evaluation planner.

Covers:
- grouping metrics by effective recipe
- fixed vs active binding resolution
- spectral and envelope aggregation per group
- deterministic group ordering
- mixed-binding metric sets
"""

from __future__ import annotations

import numpy as np

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.planner import build_plan
from quality_tool.evaluation.recipe import RAW, SignalRecipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubMetric:
    """Minimal metric stub for planner tests."""

    def __init__(
        self,
        name: str,
        recipe: SignalRecipe = RAW,
        binding: str = "active",
        needs_spectral: bool = False,
    ) -> None:
        self.name = name
        self.signal_recipe = recipe
        self.recipe_binding = binding
        self.needs_spectral = needs_spectral

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        return MetricResult(score=0.0)


_BASELINE_RECIPE = SignalRecipe(baseline=True)
_ROI_RECIPE = SignalRecipe(roi_enabled=True, segment_size=16)


# ---------------------------------------------------------------------------
# Tests — basic grouping
# ---------------------------------------------------------------------------

class TestBasicGrouping:
    def test_single_metric_single_group(self):
        m = _StubMetric("a", binding="active")
        plan = build_plan([m], active_recipe=_BASELINE_RECIPE)
        assert len(plan.groups) == 1
        assert plan.groups[0].recipe == _BASELINE_RECIPE
        assert len(plan.groups[0].metrics) == 1

    def test_two_metrics_same_recipe_one_group(self):
        m1 = _StubMetric("a", binding="active")
        m2 = _StubMetric("b", binding="active")
        plan = build_plan([m1, m2], active_recipe=_BASELINE_RECIPE)
        assert len(plan.groups) == 1
        assert len(plan.groups[0].metrics) == 2

    def test_fixed_and_active_different_groups(self):
        m_fixed = _StubMetric("fixed_raw", recipe=RAW, binding="fixed")
        m_active = _StubMetric("active_baseline", binding="active")
        plan = build_plan(
            [m_fixed, m_active], active_recipe=_BASELINE_RECIPE,
        )
        assert len(plan.groups) == 2
        recipes = {g.recipe for g in plan.groups}
        assert RAW in recipes
        assert _BASELINE_RECIPE in recipes

    def test_two_fixed_metrics_same_recipe_one_group(self):
        m1 = _StubMetric("a", recipe=RAW, binding="fixed")
        m2 = _StubMetric("b", recipe=RAW, binding="fixed")
        plan = build_plan([m1, m2])
        assert len(plan.groups) == 1

    def test_fixed_raw_and_active_raw_same_group(self):
        """When active recipe is RAW, fixed-RAW and active metrics share
        the same effective recipe."""
        m_fixed = _StubMetric("fixed", recipe=RAW, binding="fixed")
        m_active = _StubMetric("active", binding="active")
        plan = build_plan([m_fixed, m_active], active_recipe=RAW)
        assert len(plan.groups) == 1


# ---------------------------------------------------------------------------
# Tests — ordering
# ---------------------------------------------------------------------------

class TestGroupOrdering:
    def test_groups_ordered_by_first_appearance(self):
        m1 = _StubMetric("fixed_raw", recipe=RAW, binding="fixed")
        m2 = _StubMetric("active_baseline", binding="active")
        m3 = _StubMetric("fixed_raw_2", recipe=RAW, binding="fixed")

        plan = build_plan(
            [m1, m2, m3], active_recipe=_BASELINE_RECIPE,
        )
        assert len(plan.groups) == 2
        # First group should be RAW (from m1), second _BASELINE_RECIPE (from m2).
        assert plan.groups[0].recipe == RAW
        assert plan.groups[1].recipe == _BASELINE_RECIPE
        # m3 should be in the first group alongside m1.
        assert len(plan.groups[0].metrics) == 2


# ---------------------------------------------------------------------------
# Tests — spectral aggregation
# ---------------------------------------------------------------------------

class TestSpectralAggregation:
    def test_spectral_flag_propagated(self):
        m1 = _StubMetric("no_fft", binding="active", needs_spectral=False)
        m2 = _StubMetric("fft", binding="active", needs_spectral=True)
        plan = build_plan([m1, m2], active_recipe=_BASELINE_RECIPE)
        assert len(plan.groups) == 1
        assert plan.groups[0].needs_spectral is True

    def test_no_spectral_when_not_needed(self):
        m = _StubMetric("no_fft", binding="active", needs_spectral=False)
        plan = build_plan([m], active_recipe=_BASELINE_RECIPE)
        assert plan.groups[0].needs_spectral is False


# ---------------------------------------------------------------------------
# Tests — envelope aggregation
# ---------------------------------------------------------------------------

class TestEnvelopeAggregation:
    def test_envelope_when_method_available(self):
        m = _StubMetric("a", binding="active")
        plan = build_plan([m], active_recipe=_BASELINE_RECIPE, has_envelope=True)
        assert plan.groups[0].needs_envelope is True

    def test_no_envelope_when_no_method(self):
        m = _StubMetric("a", binding="active")
        plan = build_plan([m], active_recipe=_BASELINE_RECIPE, has_envelope=False)
        assert plan.groups[0].needs_envelope is False


# ---------------------------------------------------------------------------
# Tests — active recipe fallback
# ---------------------------------------------------------------------------

class TestActiveRecipeFallback:
    def test_no_active_recipe_falls_back_to_raw(self):
        m = _StubMetric("a", binding="active")
        plan = build_plan([m], active_recipe=None)
        assert plan.groups[0].recipe == RAW
