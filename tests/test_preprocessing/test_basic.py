"""Tests for quality_tool.preprocessing.basic."""

import numpy as np
import pytest

from quality_tool.preprocessing.basic import (
    normalize_amplitude,
    smooth,
    subtract_baseline,
)


# ── subtract_baseline ────────────────────────────────────────────────

class TestSubtractBaseline:

    def test_result_has_zero_mean(self):
        sig = np.array([1.0, 3.0, 5.0, 7.0])
        result = subtract_baseline(sig)
        assert result.mean() == pytest.approx(0.0)

    def test_does_not_mutate_input(self):
        sig = np.array([2.0, 4.0, 6.0])
        original = sig.copy()
        subtract_baseline(sig)
        np.testing.assert_array_equal(sig, original)

    def test_output_length_matches_input(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(subtract_baseline(sig)) == len(sig)

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError, match="1-D"):
            subtract_baseline(np.ones((3, 4)))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            subtract_baseline(np.array([]))


# ── normalize_amplitude ──────────────────────────────────────────────

class TestNormalizeAmplitude:

    def test_output_range_is_0_to_1(self):
        sig = np.array([2.0, 5.0, 10.0, 3.0])
        result = normalize_amplitude(sig)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_flat_signal_returns_zeros(self):
        sig = np.array([7.0, 7.0, 7.0])
        result = normalize_amplitude(sig)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_does_not_mutate_input(self):
        sig = np.array([1.0, 4.0, 2.0])
        original = sig.copy()
        normalize_amplitude(sig)
        np.testing.assert_array_equal(sig, original)

    def test_output_length_matches_input(self):
        sig = np.array([3.0, 6.0, 9.0])
        assert len(normalize_amplitude(sig)) == len(sig)

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError, match="1-D"):
            normalize_amplitude(np.ones((2, 3)))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_amplitude(np.array([]))


# ── smooth ───────────────────────────────────────────────────────────

class TestSmooth:

    def test_output_length_matches_input(self):
        sig = np.array([1.0, 3.0, 5.0, 3.0, 1.0])
        result = smooth(sig, window_size=3)
        assert len(result) == len(sig)

    def test_constant_signal_stays_constant_in_interior(self):
        sig = np.full(10, 4.0)
        result = smooth(sig, window_size=3)
        # Interior samples (away from edges) should be exact.
        np.testing.assert_allclose(result[1:-1], 4.0)

    def test_does_not_mutate_input(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        original = sig.copy()
        smooth(sig, window_size=3)
        np.testing.assert_array_equal(sig, original)

    def test_rejects_even_window(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="odd"):
            smooth(sig, window_size=4)

    def test_rejects_zero_window(self):
        sig = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match=">= 1"):
            smooth(sig, window_size=0)

    def test_rejects_window_larger_than_signal(self):
        sig = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="exceed"):
            smooth(sig, window_size=5)

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError, match="1-D"):
            smooth(np.ones((3, 4)), window_size=3)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            smooth(np.array([]), window_size=3)
