"""Tests for quality_tool.io.z_axis_loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quality_tool.io.z_axis_loader import load_z_axis


def test_load_explicit_z_axis(tmp_path: Path) -> None:
    """File exists and length matches → load from file."""
    z = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    p = tmp_path / "z_axis.txt"
    np.savetxt(str(p), z)

    result, resolved = load_z_axis(p, signal_length=5)
    np.testing.assert_allclose(result, z)
    assert resolved is not None


def test_index_fallback_when_no_path() -> None:
    """No path given → index-based fallback."""
    result, resolved = load_z_axis(None, signal_length=4)
    np.testing.assert_array_equal(result, [0.0, 1.0, 2.0, 3.0])
    assert resolved is None


def test_index_fallback_when_file_missing(tmp_path: Path) -> None:
    """Path given but file does not exist → index-based fallback."""
    result, resolved = load_z_axis(tmp_path / "missing.txt", signal_length=3)
    np.testing.assert_array_equal(result, [0.0, 1.0, 2.0])
    assert resolved is None


def test_length_mismatch_raises(tmp_path: Path) -> None:
    """File exists but length differs from signal_length → ValueError."""
    z = np.array([1.0, 2.0, 3.0])
    p = tmp_path / "z_axis.txt"
    np.savetxt(str(p), z)

    with pytest.raises(ValueError, match="expected 5"):
        load_z_axis(p, signal_length=5)
