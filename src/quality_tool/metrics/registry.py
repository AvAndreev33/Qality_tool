"""Metric registry for Quality_tool.

Provides a simple name-based registry for quality metrics.
"""

from __future__ import annotations

from quality_tool.metrics.base import BaseMetric


class MetricRegistry:
    """Registry of named quality metrics.

    Usage::

        registry = MetricRegistry()
        registry.register(my_metric)
        metric = registry.get("my_metric")
    """

    def __init__(self) -> None:
        self._metrics: dict[str, BaseMetric] = {}

    def register(self, metric: BaseMetric) -> None:
        """Register *metric* under its ``name``.

        Raises
        ------
        ValueError
            If a metric with the same name is already registered.
        """
        if metric.name in self._metrics:
            raise ValueError(
                f"metric {metric.name!r} is already registered"
            )
        self._metrics[metric.name] = metric

    def get(self, name: str) -> BaseMetric:
        """Return the metric registered under *name*.

        Raises
        ------
        KeyError
            If no metric with that name exists.
        """
        try:
            return self._metrics[name]
        except KeyError:
            raise KeyError(
                f"no metric named {name!r}; "
                f"available: {self.list_metrics()}"
            ) from None

    def list_metrics(self) -> list[str]:
        """Return the names of all registered metrics."""
        return list(self._metrics.keys())


default_registry = MetricRegistry()
"""Module-level default registry for convenience."""
