"""Signal recipe model for Quality_tool.

A signal recipe is a declarative description of how to prepare a signal
before metric evaluation.  It captures which preprocessing steps and ROI
extraction settings are applied to the raw signal.

Key design points:
- ``raw`` is a recipe (all flags False, no ROI), not a special case
  outside the recipe system.
- Recipes are frozen dataclasses: hashable, comparable, suitable for
  dict keys and cache logic.
- Envelope and spectrum are *not* part of the recipe — they are
  derived representations computed on top of a prepared recipe.

Recipe binding
--------------
Each metric declares a *recipe binding* that controls how its effective
recipe is determined:

- ``"fixed"`` — the metric always uses its declared ``signal_recipe``,
  regardless of the active session/GUI pipeline.
- ``"active"`` — the metric uses the current active processing pipeline
  from the session or evaluator context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RecipeBinding = Literal["fixed", "active"]


@dataclass(frozen=True)
class SignalRecipe:
    """Declarative description of signal preparation before evaluation.

    All fields default to the identity (no-op) recipe, which is
    equivalent to using the raw signal with no processing.

    Attributes
    ----------
    baseline : bool
        Apply baseline subtraction.
    normalize : bool
        Apply amplitude normalisation.
    smooth : bool
        Apply smoothing.
    roi_enabled : bool
        Extract a local ROI segment.
    segment_size : int | None
        ROI segment length (only meaningful when ``roi_enabled=True``).
    """

    baseline: bool = False
    normalize: bool = False
    smooth: bool = False
    detrend: bool = False
    roi_enabled: bool = False
    segment_size: int | None = None


#: The identity recipe — raw signal with no processing.
RAW = SignalRecipe()

#: ROI extraction only — no mean subtraction, no detrending.
ROI_ONLY = SignalRecipe(roi_enabled=True)

#: ROI + mean subtraction + linear detrending.
ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED = SignalRecipe(
    baseline=True,
    detrend=True,
    roi_enabled=True,
)


def resolve_effective_recipe(
    signal_recipe: SignalRecipe,
    recipe_binding: RecipeBinding,
    active_recipe: SignalRecipe | None,
) -> SignalRecipe:
    """Determine the effective recipe for a metric.

    Parameters
    ----------
    signal_recipe : SignalRecipe
        The metric's declared recipe.
    recipe_binding : RecipeBinding
        ``"fixed"`` or ``"active"``.
    active_recipe : SignalRecipe | None
        The current session/GUI pipeline recipe.  Used only when
        *recipe_binding* is ``"active"``.

    Returns
    -------
    SignalRecipe
        The concrete recipe that should be used to prepare signals.
    """
    if recipe_binding == "fixed":
        return signal_recipe
    # "active" binding — use the active pipeline, fall back to RAW.
    return active_recipe if active_recipe is not None else RAW


def recipe_from_processing(settings: dict) -> SignalRecipe:
    """Build a :class:`SignalRecipe` from a GUI processing-settings dict.

    This is the single translation point between the GUI settings dict
    and the backend recipe model.

    Parameters
    ----------
    settings : dict
        Dict with keys ``baseline``, ``normalize``, ``smooth``,
        ``roi_enabled``, ``segment_size`` (as used by the processing
        dialog and ``MainWindow._processing``).
    """
    roi_enabled = bool(settings.get("roi_enabled", False))
    return SignalRecipe(
        baseline=bool(settings.get("baseline", False)),
        normalize=bool(settings.get("normalize", False)),
        smooth=bool(settings.get("smooth", False)),
        roi_enabled=roi_enabled,
        segment_size=settings.get("segment_size") if roi_enabled else None,
    )
