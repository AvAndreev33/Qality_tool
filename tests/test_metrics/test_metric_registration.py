"""Tests for metric registration and GUI grouping.

Verifies that phase and correlation metrics are registered correctly
and appear in the expected groups.
"""

from __future__ import annotations

from quality_tool.metrics.registry import MetricRegistry
from quality_tool.metrics.phase import ALL_PHASE_METRICS
from quality_tool.metrics.correlation import ALL_CORRELATION_METRICS


def _build_test_registry() -> MetricRegistry:
    """Build a registry with only the new metric groups."""
    registry = MetricRegistry()
    for m in ALL_PHASE_METRICS:
        registry.register(m)
    for m in ALL_CORRELATION_METRICS:
        registry.register(m)
    return registry


def test_phase_metrics_registered():
    """All 5 phase metrics should be registered."""
    registry = _build_test_registry()
    names = registry.list_metrics()
    expected = [
        "phase_slope_stability",
        "phase_linear_fit_residual",
        "phase_curvature_index",
        "phase_monotonicity_score",
        "phase_jump_fraction",
    ]
    for name in expected:
        assert name in names, f"{name} not registered"


def test_correlation_metrics_registered():
    """All 4 correlation metrics should be registered."""
    registry = _build_test_registry()
    names = registry.list_metrics()
    expected = [
        "centered_reference_correlation",
        "best_phase_reference_correlation",
        "reference_envelope_correlation",
        "phase_relaxation_gain",
    ]
    for name in expected:
        assert name in names, f"{name} not registered"


def test_grouped_categories():
    """list_grouped() should contain 'phase' and 'correlation' groups."""
    registry = _build_test_registry()
    grouped = registry.list_grouped()
    cats = [cat for cat, _ in grouped]
    assert "phase" in cats
    assert "correlation" in cats


def test_phase_group_has_5_metrics():
    """The phase group should contain exactly 5 metrics."""
    registry = _build_test_registry()
    grouped = dict(registry.list_grouped())
    assert len(grouped["phase"]) == 5


def test_correlation_group_has_4_metrics():
    """The correlation group should contain exactly 4 metrics."""
    registry = _build_test_registry()
    grouped = dict(registry.list_grouped())
    assert len(grouped["correlation"]) == 4


def test_display_names_are_readable():
    """Display names should not be raw snake_case identifiers."""
    registry = _build_test_registry()
    grouped = registry.list_grouped()
    for _, items in grouped:
        for name, display_name in items:
            assert display_name != name or " " in display_name, (
                f"display_name for {name!r} looks like a raw identifier"
            )
