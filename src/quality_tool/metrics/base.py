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

Batch evaluation
----------------
Metrics may optionally implement ``evaluate_batch`` for vectorised
evaluation over a 2-D chunk of signals.  The evaluator will prefer
``evaluate_batch`` when available and fall back to the per-signal
``evaluate`` otherwise.

Dependency hints
----------------
``needs_spectral`` — when ``True`` the evaluator precomputes a batch
spectral context and passes it to the metric.  Metrics that do not need
FFT data should leave this as ``False`` (the default) so the evaluator
can skip the FFT entirely.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.recipe import RecipeBinding, SignalRecipe


@runtime_checkable
class BaseMetric(Protocol):
    """Protocol for quality metric implementations.

    Every metric must expose ``name``, ``signal_recipe``,
    ``recipe_binding``, and ``evaluate``.
    Optional extensions: ``needs_spectral``, ``evaluate_batch``.
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
            Optional additional context (e.g. precomputed spectral data).

        Returns
        -------
        MetricResult
            Scalar score, diagnostic features, and validity flag.
        """
        ...
