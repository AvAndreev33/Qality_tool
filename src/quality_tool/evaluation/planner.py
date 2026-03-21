"""Lightweight evaluation planner for Quality_tool.

Before batch evaluation the planner inspects selected metrics, resolves
their effective recipes, groups metrics that share the same prepared
signal, and determines which derived representations (envelope, spectral)
are needed per group.

This avoids duplicated signal preparation and derived-representation
computation when multiple metrics share the same recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from quality_tool.evaluation.recipe import (
    SignalRecipe,
    resolve_effective_recipe,
)
from quality_tool.metrics.base import (
    BaseMetric,
    RepresentationNeeds,
    NO_EXTRA,
    resolve_representation_needs,
)


@dataclass
class RecipeGroup:
    """A group of metrics that share the same effective recipe.

    Attributes
    ----------
    recipe : SignalRecipe
        The concrete recipe for signal preparation.
    metrics : list[BaseMetric]
        Metrics in this group, in original selection order.
    needs : RepresentationNeeds
        Merged representation needs for the group.
    needs_envelope : bool
        Whether any metric in this group requires envelope computation.
        Derived from ``needs.envelope`` plus session-level envelope
        availability.
    needs_spectral : bool
        Whether any metric in this group requires spectral computation.
        Derived from ``needs.needs_spectral``.
    """

    recipe: SignalRecipe
    metrics: list[BaseMetric] = field(default_factory=list)
    needs: RepresentationNeeds = field(default_factory=lambda: NO_EXTRA)
    needs_envelope: bool = False
    needs_spectral: bool = False


@dataclass
class EvaluationPlan:
    """Execution plan produced by the planner.

    Attributes
    ----------
    groups : list[RecipeGroup]
        Recipe groups in deterministic order (ordered by first
        appearance of the recipe across the metric list).
    """

    groups: list[RecipeGroup] = field(default_factory=list)


def build_plan(
    metrics: Sequence[BaseMetric],
    active_recipe: SignalRecipe | None = None,
    *,
    has_envelope: bool = False,
) -> EvaluationPlan:
    """Build an evaluation plan for a set of metrics.

    Parameters
    ----------
    metrics : sequence of BaseMetric
        The metrics to evaluate (in user-selection order).
    active_recipe : SignalRecipe | None
        The current session/GUI processing pipeline.  Used to resolve
        ``recipe_binding="active"`` metrics.
    has_envelope : bool
        Whether an envelope method is available in the session.  When
        ``False``, ``needs_envelope`` on all groups will be ``False``
        regardless of individual metric hints.

    Returns
    -------
    EvaluationPlan
        Groups of metrics keyed by their effective recipe, with merged
        representation needs per group.
    """
    # Ordered dict preserving first-seen order of recipes.
    groups_by_recipe: dict[SignalRecipe, RecipeGroup] = {}

    for metric in metrics:
        recipe = resolve_effective_recipe(
            getattr(metric, "signal_recipe", SignalRecipe()),
            getattr(metric, "recipe_binding", "active"),
            active_recipe,
        )

        if recipe not in groups_by_recipe:
            groups_by_recipe[recipe] = RecipeGroup(recipe=recipe)

        group = groups_by_recipe[recipe]
        group.metrics.append(metric)

        # Merge structured representation needs.
        metric_needs = resolve_representation_needs(metric)
        group.needs = group.needs.merge(metric_needs)

    # Derive convenience flags from merged needs.
    for group in groups_by_recipe.values():
        group.needs_spectral = group.needs.needs_spectral

        # Envelope: mark the group if *any* metric declares envelope need
        # (regardless of session availability — the evaluator provides a
        # fallback), or if the session has an envelope method enabled
        # (legacy: envelope is passed to all groups when configured).
        if group.needs.envelope:
            group.needs_envelope = True
        elif has_envelope:
            group.needs_envelope = True

    return EvaluationPlan(groups=list(groups_by_recipe.values()))
