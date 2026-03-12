"""Tests for quality_tool.preprocessing.roi."""

import numpy as np
import pytest

from quality_tool.preprocessing.roi import extract_roi


class TestExtractRoi:

    # ── happy-path ───────────────────────────────────────────────────

    def test_output_has_correct_length(self):
        sig = np.array([0.0, 1.0, 4.0, 2.0, 0.0])
        result = extract_roi(sig, segment_size=3)
        assert len(result) == 3

    def test_centered_around_max(self):
        sig = np.array([0.0, 1.0, 10.0, 1.0, 0.0])
        result = extract_roi(sig, segment_size=3, mode="raw_max")
        np.testing.assert_array_equal(result, [1.0, 10.0, 1.0])

    def test_full_signal_extraction(self):
        sig = np.array([3.0, 1.0, 2.0])
        result = extract_roi(sig, segment_size=3)
        np.testing.assert_array_equal(result, sig)

    def test_returns_copy(self):
        sig = np.array([0.0, 5.0, 0.0, 0.0, 0.0])
        result = extract_roi(sig, segment_size=3)
        result[0] = 999.0
        assert sig[0] == 0.0  # original unchanged

    # ── edge clamping ────────────────────────────────────────────────

    def test_clamps_at_left_edge(self):
        sig = np.array([10.0, 1.0, 0.0, 0.0, 0.0])
        result = extract_roi(sig, segment_size=3)
        # max at index 0 → window clamped to [0, 3)
        np.testing.assert_array_equal(result, [10.0, 1.0, 0.0])

    def test_clamps_at_right_edge(self):
        sig = np.array([0.0, 0.0, 0.0, 1.0, 10.0])
        result = extract_roi(sig, segment_size=3)
        # max at index 4 → window clamped to [2, 5)
        np.testing.assert_array_equal(result, [0.0, 1.0, 10.0])

    # ── validation errors ────────────────────────────────────────────

    def test_rejects_segment_larger_than_signal(self):
        sig = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="exceeds"):
            extract_roi(sig, segment_size=5)

    def test_rejects_zero_segment_size(self):
        sig = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match=">= 1"):
            extract_roi(sig, segment_size=0)

    def test_rejects_unsupported_mode(self):
        sig = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="unsupported"):
            extract_roi(sig, segment_size=2, mode="unknown")

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError, match="1-D"):
            extract_roi(np.ones((3, 3)), segment_size=2)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            extract_roi(np.array([]), segment_size=1)
