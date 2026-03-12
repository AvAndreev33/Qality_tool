"""Tests for quality_tool.metrics.registry."""

import pytest

from quality_tool.metrics.registry import MetricRegistry


class _DummyMetric:
    """Minimal metric stub for testing."""

    def __init__(self, name: str = "dummy") -> None:
        self.name = name

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        from quality_tool.core.models import MetricResult
        return MetricResult(score=0.0)


class TestMetricRegistry:

    def test_register_and_get(self):
        reg = MetricRegistry()
        metric = _DummyMetric("test")
        reg.register(metric)
        assert reg.get("test") is metric

    def test_list_metrics(self):
        reg = MetricRegistry()
        reg.register(_DummyMetric("alpha"))
        reg.register(_DummyMetric("beta"))
        assert set(reg.list_metrics()) == {"alpha", "beta"}

    def test_empty_registry_list(self):
        reg = MetricRegistry()
        assert reg.list_metrics() == []

    def test_duplicate_registration_raises(self):
        reg = MetricRegistry()
        reg.register(_DummyMetric("dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_DummyMetric("dup"))

    def test_get_missing_raises_key_error(self):
        reg = MetricRegistry()
        with pytest.raises(KeyError, match="no metric named"):
            reg.get("nonexistent")
