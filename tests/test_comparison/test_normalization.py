"""Tests for the normalization / comparison layer."""

import numpy as np
import pytest

from quality_tool.comparison.normalization import (
    normalize_score_map,
    normalize_single,
    reference_range_from_map,
)


# ------------------------------------------------------------------
# bounded_01 scale
# ------------------------------------------------------------------

class TestBounded01:
    """Metrics with score_scale='bounded_01'."""

    def test_higher_better_passthrough(self):
        scores = np.array([[0.2, 0.8], [0.5, 0.0]])
        valid = np.ones_like(scores, dtype=bool)
        result = normalize_score_map(scores, valid, "higher_better", "bounded_01")
        np.testing.assert_allclose(result, scores)

    def test_lower_better_flipped(self):
        scores = np.array([[0.0, 1.0], [0.3, 0.7]])
        valid = np.ones_like(scores, dtype=bool)
        result = normalize_score_map(scores, valid, "lower_better", "bounded_01")
        expected = 1.0 - scores
        np.testing.assert_allclose(result, expected)

    def test_values_clipped_to_01(self):
        scores = np.array([[1.2, -0.1]])
        valid = np.ones_like(scores, dtype=bool)
        result = normalize_score_map(scores, valid, "higher_better", "bounded_01")
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ------------------------------------------------------------------
# positive_unbounded scale
# ------------------------------------------------------------------

class TestPositiveUnbounded:
    """Metrics with score_scale='positive_unbounded'."""

    def test_higher_better_rescales(self):
        scores = np.array([[10.0, 50.0], [30.0, 20.0]])
        valid = np.ones_like(scores, dtype=bool)
        result = normalize_score_map(scores, valid, "higher_better", "positive_unbounded")
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        # Higher native score should give higher normalized score.
        assert result[0, 1] > result[0, 0]

    def test_lower_better_flips(self):
        scores = np.array([[10.0, 50.0], [30.0, 20.0]])
        valid = np.ones_like(scores, dtype=bool)
        result = normalize_score_map(scores, valid, "lower_better", "positive_unbounded")
        # Lower native score should give higher normalized score.
        assert result[0, 0] > result[0, 1]


# ------------------------------------------------------------------
# db_like scale
# ------------------------------------------------------------------

class TestDbLike:
    """Metrics with score_scale='db_like'."""

    def test_handles_negative_values(self):
        scores = np.array([[-5.0, 0.0, 10.0, 20.0]])
        valid = np.ones_like(scores, dtype=bool)
        result = normalize_score_map(scores, valid, "higher_better", "db_like")
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_higher_better_ordering(self):
        scores = np.array([[-5.0, 20.0]])
        valid = np.ones_like(scores, dtype=bool)
        result = normalize_score_map(scores, valid, "higher_better", "db_like")
        assert result[0, 1] > result[0, 0]


# ------------------------------------------------------------------
# Invalid pixel handling
# ------------------------------------------------------------------

class TestInvalidHandling:
    """Invalid pixels remain NaN after normalization."""

    def test_invalid_stays_nan(self):
        scores = np.array([[0.5, 0.8], [0.3, 0.9]])
        valid = np.array([[True, True], [False, True]])
        result = normalize_score_map(scores, valid, "higher_better", "bounded_01")
        assert np.isnan(result[1, 0])
        assert not np.isnan(result[0, 0])

    def test_all_invalid_returns_nan(self):
        scores = np.array([[1.0, 2.0]])
        valid = np.zeros_like(scores, dtype=bool)
        result = normalize_score_map(scores, valid, "higher_better", "bounded_01")
        assert np.all(np.isnan(result))


# ------------------------------------------------------------------
# normalize_single
# ------------------------------------------------------------------

class TestNormalizeSingle:
    """Per-pixel normalization using precomputed reference range."""

    def test_bounded_01(self):
        val = normalize_single(0.7, "higher_better", "bounded_01", 0.0, 1.0)
        assert pytest.approx(val) == 0.7

    def test_lower_better_flips(self):
        val = normalize_single(0.3, "lower_better", "bounded_01", 0.0, 1.0)
        assert pytest.approx(val) == 0.7

    def test_unbounded_rescales(self):
        val = normalize_single(50.0, "higher_better", "positive_unbounded", 10.0, 90.0)
        assert pytest.approx(val) == 0.5


# ------------------------------------------------------------------
# reference_range_from_map
# ------------------------------------------------------------------

class TestReferenceRange:
    """Reference range computation."""

    def test_bounded_always_01(self):
        scores = np.array([[0.3, 0.7]])
        valid = np.ones_like(scores, dtype=bool)
        lo, hi = reference_range_from_map(scores, valid, "bounded_01")
        assert lo == 0.0
        assert hi == 1.0

    def test_unbounded_uses_percentiles(self):
        rng = np.random.RandomState(42)
        scores = rng.uniform(0, 100, size=(10, 10))
        valid = np.ones_like(scores, dtype=bool)
        lo, hi = reference_range_from_map(scores, valid, "positive_unbounded")
        assert lo >= 0.0
        assert hi > lo

    def test_empty_valid_fallback(self):
        scores = np.array([[1.0]])
        valid = np.zeros_like(scores, dtype=bool)
        lo, hi = reference_range_from_map(scores, valid, "positive_unbounded")
        assert lo == 0.0
        assert hi == 1.0
