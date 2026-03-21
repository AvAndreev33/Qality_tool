"""Tests for planner representation-needs merging.

Covers:
- merging RepresentationNeeds across metrics in a group
- group.needs reflects merged needs
- group.needs_spectral derived correctly
- mixed needs across recipe groups
"""

from __future__ import annotations

import numpy as np

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.planner import build_plan
from quality_tool.evaluation.recipe import RAW, SignalRecipe
from quality_tool.metrics.base import NO_EXTRA, RepresentationNeeds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubMetric:
    def __init__(
        self,
        name: str,
        recipe: SignalRecipe = RAW,
        binding: str = "active",
        needs: RepresentationNeeds | None = None,
        needs_spectral: bool = False,
    ) -> None:
        self.name = name
        self.signal_recipe = recipe
        self.recipe_binding = binding
        if needs is not None:
            self.representation_needs = needs
        self.needs_spectral = needs_spectral

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        return MetricResult(score=0.0)


_BASELINE = SignalRecipe(baseline=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlannerNeedsMerging:
    def test_single_metric_needs_propagated(self):
        needs = RepresentationNeeds(amplitude=True)
        m = _StubMetric("a", binding="active", needs=needs)
        plan = build_plan([m], active_recipe=_BASELINE)
        assert plan.groups[0].needs == needs
        assert plan.groups[0].needs_spectral is True

    def test_no_needs_gives_no_extra(self):
        m = _StubMetric("a", binding="active")
        plan = build_plan([m], active_recipe=_BASELINE)
        assert plan.groups[0].needs == NO_EXTRA
        assert plan.groups[0].needs_spectral is False

    def test_merge_across_metrics_in_group(self):
        m1 = _StubMetric("a", binding="active",
                          needs=RepresentationNeeds(amplitude=True))
        m2 = _StubMetric("b", binding="active",
                          needs=RepresentationNeeds(power=True, envelope=True))
        plan = build_plan([m1, m2], active_recipe=_BASELINE)
        assert len(plan.groups) == 1
        merged = plan.groups[0].needs
        assert merged.amplitude is True
        assert merged.power is True
        assert merged.envelope is True
        assert merged.complex_fft is False

    def test_different_recipes_independent_needs(self):
        m_fixed = _StubMetric(
            "fixed", recipe=RAW, binding="fixed",
            needs=RepresentationNeeds(amplitude=True),
        )
        m_active = _StubMetric(
            "active", binding="active",
            needs=RepresentationNeeds(power=True),
        )
        plan = build_plan([m_fixed, m_active], active_recipe=_BASELINE)
        assert len(plan.groups) == 2

        raw_group = [g for g in plan.groups if g.recipe == RAW][0]
        baseline_group = [g for g in plan.groups if g.recipe == _BASELINE][0]

        assert raw_group.needs.amplitude is True
        assert raw_group.needs.power is False
        assert baseline_group.needs.power is True
        assert baseline_group.needs.amplitude is False

    def test_legacy_needs_spectral_still_works(self):
        m = _StubMetric("legacy", binding="active", needs_spectral=True)
        plan = build_plan([m], active_recipe=_BASELINE)
        assert plan.groups[0].needs.amplitude is True
        assert plan.groups[0].needs_spectral is True

    def test_envelope_need_with_session_envelope(self):
        m = _StubMetric(
            "env_metric", binding="active",
            needs=RepresentationNeeds(envelope=True),
        )
        plan = build_plan([m], active_recipe=_BASELINE, has_envelope=True)
        assert plan.groups[0].needs_envelope is True
        assert plan.groups[0].needs.envelope is True

    def test_envelope_need_without_session_envelope(self):
        m = _StubMetric(
            "env_metric", binding="active",
            needs=RepresentationNeeds(envelope=True),
        )
        plan = build_plan([m], active_recipe=_BASELINE, has_envelope=False)
        # Even without a session envelope method, the planner marks the
        # group as needing envelope — the evaluator provides a fallback.
        assert plan.groups[0].needs_envelope is True
        assert plan.groups[0].needs.envelope is True
