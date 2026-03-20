"""Tests for quality_tool.evaluation.recipe — signal recipe model.

Covers:
- recipe equality and hashing
- RAW constant
- resolve_effective_recipe for fixed and active bindings
- recipe_from_processing conversion
"""

from __future__ import annotations

import pytest

from quality_tool.evaluation.recipe import (
    RAW,
    SignalRecipe,
    recipe_from_processing,
    resolve_effective_recipe,
)


# ---------------------------------------------------------------------------
# Tests — SignalRecipe basics
# ---------------------------------------------------------------------------

class TestSignalRecipeBasics:
    def test_raw_is_identity(self):
        assert RAW == SignalRecipe()
        assert RAW.baseline is False
        assert RAW.normalize is False
        assert RAW.smooth is False
        assert RAW.roi_enabled is False
        assert RAW.segment_size is None

    def test_equality(self):
        a = SignalRecipe(baseline=True, normalize=False)
        b = SignalRecipe(baseline=True, normalize=False)
        assert a == b

    def test_inequality(self):
        a = SignalRecipe(baseline=True)
        b = SignalRecipe(baseline=False)
        assert a != b

    def test_hashable(self):
        a = SignalRecipe(baseline=True, roi_enabled=True, segment_size=32)
        b = SignalRecipe(baseline=True, roi_enabled=True, segment_size=32)
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1

    def test_usable_as_dict_key(self):
        recipe = SignalRecipe(smooth=True)
        d = {recipe: "group_1"}
        assert d[SignalRecipe(smooth=True)] == "group_1"

    def test_frozen(self):
        with pytest.raises(AttributeError):
            RAW.baseline = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests — resolve_effective_recipe
# ---------------------------------------------------------------------------

class TestResolveEffectiveRecipe:
    def test_fixed_ignores_active(self):
        fixed_recipe = SignalRecipe(baseline=True)
        active = SignalRecipe(smooth=True, roi_enabled=True, segment_size=16)
        result = resolve_effective_recipe(fixed_recipe, "fixed", active)
        assert result == fixed_recipe

    def test_active_uses_active_recipe(self):
        metric_recipe = RAW
        active = SignalRecipe(normalize=True)
        result = resolve_effective_recipe(metric_recipe, "active", active)
        assert result == active

    def test_active_falls_back_to_raw_when_no_active(self):
        result = resolve_effective_recipe(RAW, "active", None)
        assert result == RAW

    def test_fixed_raw_always_raw(self):
        active = SignalRecipe(baseline=True, smooth=True)
        result = resolve_effective_recipe(RAW, "fixed", active)
        assert result == RAW


# ---------------------------------------------------------------------------
# Tests — recipe_from_processing
# ---------------------------------------------------------------------------

class TestRecipeFromProcessing:
    def test_empty_settings(self):
        recipe = recipe_from_processing({})
        assert recipe == RAW

    def test_baseline_only(self):
        recipe = recipe_from_processing({"baseline": True})
        assert recipe == SignalRecipe(baseline=True)

    def test_full_settings(self):
        settings = {
            "baseline": True,
            "normalize": True,
            "smooth": True,
            "roi_enabled": True,
            "segment_size": 64,
        }
        recipe = recipe_from_processing(settings)
        assert recipe == SignalRecipe(
            baseline=True,
            normalize=True,
            smooth=True,
            roi_enabled=True,
            segment_size=64,
        )

    def test_roi_disabled_clears_segment_size(self):
        settings = {
            "roi_enabled": False,
            "segment_size": 128,
        }
        recipe = recipe_from_processing(settings)
        assert recipe.roi_enabled is False
        assert recipe.segment_size is None

    def test_roi_enabled_keeps_segment_size(self):
        settings = {
            "roi_enabled": True,
            "segment_size": 32,
        }
        recipe = recipe_from_processing(settings)
        assert recipe.roi_enabled is True
        assert recipe.segment_size == 32
