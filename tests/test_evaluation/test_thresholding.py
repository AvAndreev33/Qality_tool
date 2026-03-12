"""Tests for quality_tool.evaluation.thresholding."""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.models import MetricMapResult, ThresholdResult
from quality_tool.evaluation.thresholding import apply_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metric_map(
    scores: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> MetricMapResult:
    """Create a MetricMapResult for testing.

    Default: 2×3 map with scores [[0.1, 0.4, 0.7], [0.2, 0.5, 0.8]],
    all valid.
    """
    if scores is None:
        scores = np.array([[0.1, 0.4, 0.7],
                           [0.2, 0.5, 0.8]])
    if valid is None:
        valid = np.ones_like(scores, dtype=bool)
    return MetricMapResult(
        metric_name="test",
        score_map=scores,
        valid_map=valid,
    )


# ---------------------------------------------------------------------------
# Tests — keep above
# ---------------------------------------------------------------------------

class TestKeepAbove:
    def test_basic(self):
        mm = _make_metric_map()
        result = apply_threshold(mm, threshold=0.5, keep_rule="above")

        assert isinstance(result, ThresholdResult)
        expected = np.array([[False, False, True],
                             [False, True, True]])
        np.testing.assert_array_equal(result.mask, expected)

    def test_boundary_included(self):
        """score == threshold should be kept for 'above'."""
        mm = _make_metric_map()
        result = apply_threshold(mm, threshold=0.4, keep_rule="above")
        assert result.mask[0, 1] is np.bool_(True)  # score 0.4

    def test_rule_label(self):
        result = apply_threshold(_make_metric_map(), 0.5, "above")
        assert "score >= 0.5" in result.keep_rule


# ---------------------------------------------------------------------------
# Tests — keep below
# ---------------------------------------------------------------------------

class TestKeepBelow:
    def test_basic(self):
        mm = _make_metric_map()
        result = apply_threshold(mm, threshold=0.4, keep_rule="below")

        expected = np.array([[True, True, False],
                             [True, False, False]])
        np.testing.assert_array_equal(result.mask, expected)

    def test_boundary_included(self):
        """score == threshold should be kept for 'below'."""
        mm = _make_metric_map()
        result = apply_threshold(mm, threshold=0.7, keep_rule="below")
        assert result.mask[0, 2] is np.bool_(True)  # score 0.7

    def test_rule_label(self):
        result = apply_threshold(_make_metric_map(), 0.4, "below")
        assert "score <= 0.4" in result.keep_rule


# ---------------------------------------------------------------------------
# Tests — invalid pixels
# ---------------------------------------------------------------------------

class TestInvalidPixels:
    def test_invalid_always_rejected(self):
        scores = np.array([[10.0, 0.5],
                           [0.3,  10.0]])
        valid = np.array([[False, True],
                          [True, False]])
        mm = _make_metric_map(scores, valid)

        result = apply_threshold(mm, threshold=0.4, keep_rule="above")
        # Invalid pixels must be rejected even though score 10.0 > 0.4.
        assert result.mask[0, 0] is np.bool_(False)
        assert result.mask[1, 1] is np.bool_(False)
        # Valid pixels should follow the rule.
        assert result.mask[0, 1] is np.bool_(True)   # 0.5 >= 0.4
        assert result.mask[1, 0] is np.bool_(False)   # 0.3 < 0.4

    def test_invalid_rejected_below(self):
        scores = np.array([[0.0, 0.5]])
        valid = np.array([[False, True]])
        mm = _make_metric_map(scores, valid)

        result = apply_threshold(mm, threshold=0.4, keep_rule="below")
        assert result.mask[0, 0] is np.bool_(False)   # invalid → rejected
        assert result.mask[0, 1] is np.bool_(False)   # 0.5 > 0.4

    def test_nan_scores_invalid(self):
        """Invalid pixels with NaN scores should also be rejected."""
        scores = np.array([[np.nan, 0.6]])
        valid = np.array([[False, True]])
        mm = _make_metric_map(scores, valid)

        result = apply_threshold(mm, threshold=0.5, keep_rule="above")
        assert result.mask[0, 0] is np.bool_(False)
        assert result.mask[0, 1] is np.bool_(True)


# ---------------------------------------------------------------------------
# Tests — summary statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_all_valid(self):
        mm = _make_metric_map()
        result = apply_threshold(mm, threshold=0.5, keep_rule="above")
        s = result.stats
        assert s["total_pixels"] == 6
        assert s["valid_pixels"] == 6
        assert s["kept_pixels"] == 3
        assert s["rejected_pixels"] == 3
        assert s["kept_fraction"] == pytest.approx(0.5)

    def test_with_invalid_pixels(self):
        scores = np.array([[0.6, 0.8],
                           [0.9, 0.3]])
        valid = np.array([[True, True],
                          [True, False]])
        mm = _make_metric_map(scores, valid)

        result = apply_threshold(mm, threshold=0.5, keep_rule="above")
        s = result.stats
        assert s["total_pixels"] == 4
        assert s["valid_pixels"] == 3
        assert s["kept_pixels"] == 3       # 0.6, 0.8, 0.9
        assert s["rejected_pixels"] == 1   # the invalid pixel
        assert s["kept_fraction"] == pytest.approx(1.0)  # 3/3 valid kept

    def test_no_valid_pixels(self):
        scores = np.array([[1.0, 2.0]])
        valid = np.array([[False, False]])
        mm = _make_metric_map(scores, valid)

        result = apply_threshold(mm, threshold=0.5, keep_rule="above")
        assert result.stats["kept_fraction"] == 0.0
        assert result.stats["kept_pixels"] == 0


# ---------------------------------------------------------------------------
# Tests — unsupported rule
# ---------------------------------------------------------------------------

class TestUnsupportedRule:
    def test_raises_on_unknown_rule(self):
        mm = _make_metric_map()
        with pytest.raises(ValueError, match="unsupported keep_rule"):
            apply_threshold(mm, threshold=0.5, keep_rule="between")
