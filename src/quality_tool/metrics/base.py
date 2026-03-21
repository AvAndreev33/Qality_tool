"""Base metric interface for Quality_tool.

Defines the protocol that every quality metric must satisfy.

Signal recipe
-------------
Each metric declares a ``signal_recipe`` (which preparation steps it
needs) and a ``recipe_binding`` that controls how the effective recipe
is determined:

* ``"fixed"``  — the metric always uses its declared recipe.
* ``"active"`` — the metric uses the current session/GUI pipeline.

``raw`` is a recipe (the identity recipe), not a special case outside
the recipe system.

Representation needs
--------------------
Each metric declares a ``representation_needs`` that tells the
evaluator which derived representations to precompute on the prepared
signal.  The planner merges needs across all metrics in a recipe group
so that each representation is computed at most once per recipe per
chunk.

The legacy ``needs_spectral`` boolean is still respected for backward
compatibility but ``representation_needs`` is preferred.

Batch evaluation
----------------
Metrics may optionally implement ``evaluate_batch`` for vectorised
evaluation over a 2-D chunk of signals.  The evaluator will prefer
``evaluate_batch`` when available and fall back to the per-signal
``evaluate`` otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.recipe import RecipeBinding, SignalRecipe


@dataclass(frozen=True)
class RepresentationNeeds:
    """Declares which derived representations a metric requires.

    All fields default to ``False`` (no extra representations needed).
    The planner merges needs across metrics sharing the same recipe by
    OR-ing each field.

    Attributes
    ----------
    envelope : bool
        Metric requires the envelope of the prepared signal.
    amplitude : bool
        Metric requires the one-sided amplitude spectrum.
    power : bool
        Metric requires the one-sided power spectrum.
    complex_fft : bool
        Metric requires raw complex FFT coefficients.
    """

    envelope: bool = False
    amplitude: bool = False
    power: bool = False
    complex_fft: bool = False

    @property
    def needs_spectral(self) -> bool:
        """Whether any spectral representation is required."""
        return self.amplitude or self.power or self.complex_fft

    def merge(self, other: RepresentationNeeds) -> RepresentationNeeds:
        """Merge two needs by OR-ing each field."""
        return RepresentationNeeds(
            envelope=self.envelope or other.envelope,
            amplitude=self.amplitude or other.amplitude,
            power=self.power or other.power,
            complex_fft=self.complex_fft or other.complex_fft,
        )


#: No extra representations needed (the default).
NO_EXTRA = RepresentationNeeds()


@runtime_checkable
class BaseMetric(Protocol):
    """Protocol for quality metric implementations.

    Every metric must expose ``name``, ``signal_recipe``,
    ``recipe_binding``, and ``evaluate``.

    Optional extensions:

    * ``representation_needs`` — structured declaration of required
      derived representations.  Preferred over ``needs_spectral``.
    * ``needs_spectral`` — legacy boolean, still honoured.
    * ``evaluate_batch`` — vectorised evaluation over ``(N, M)``
      signals.
    * ``category`` — grouping label for GUI display (e.g. ``"baseline"``,
      ``"noise"``).
    * ``display_name`` — human-readable label for GUI display.
    """

    name: str

    signal_recipe: SignalRecipe
    """Which signal preparation this metric requires.

    For ``recipe_binding="fixed"`` this is the exact recipe used.
    For ``recipe_binding="active"`` this is ignored in favour of the
    active session pipeline (but may serve as a documentation hint).
    """

    recipe_binding: RecipeBinding
    """How the effective recipe is determined.

    * ``"fixed"``  — always use the metric's declared ``signal_recipe``.
    * ``"active"`` — use the current active processing pipeline.
    """

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        """Evaluate the metric on a single 1-D signal.

        Parameters
        ----------
        signal : np.ndarray
            1-D input signal.
        z_axis : np.ndarray | None
            Optional physical z-axis of the same length as *signal*.
        envelope : np.ndarray | None
            Optional precomputed envelope of the same length as *signal*.
        context : dict | None
            Optional additional context (e.g. precomputed spectral data,
            analysis context).

        Returns
        -------
        MetricResult
            Scalar score, diagnostic features, and validity flag.
        """
        ...


def resolve_category(metric: BaseMetric) -> str:
    """Return the metric's category label, defaulting to ``"other"``."""
    return getattr(metric, "category", "other")


def resolve_display_name(metric: BaseMetric) -> str:
    """Return the metric's human-readable display name.

    Falls back to ``metric.name`` if no ``display_name`` is set.
    """
    return getattr(metric, "display_name", None) or metric.name


def resolve_representation_needs(metric: BaseMetric) -> RepresentationNeeds:
    """Return the effective representation needs for *metric*.

    Checks for ``representation_needs`` first, then falls back to the
    legacy ``needs_spectral`` boolean.
    """
    explicit = getattr(metric, "representation_needs", None)
    if isinstance(explicit, RepresentationNeeds):
        return explicit

    # Legacy path: honour needs_spectral → amplitude.
    if getattr(metric, "needs_spectral", False):
        return RepresentationNeeds(amplitude=True)

    return NO_EXTRA
