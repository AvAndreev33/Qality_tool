"""Tests for quality_tool.io.txt_matrix_loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quality_tool.io.txt_matrix_loader import load_txt_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_matrix(path: Path, data: np.ndarray) -> Path:
    np.savetxt(str(path), data)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_load_and_reshape(tmp_path: Path) -> None:
    """Load (N, M) matrix and validate canonical (H, W, M) output."""
    width, height, m = 3, 2, 5
    n = width * height
    data = np.arange(n * m, dtype=float).reshape(n, m)
    p = _write_matrix(tmp_path / "signals.txt", data)

    ss = load_txt_matrix(p, width=width, height=height)

    assert ss.signals.shape == (height, width, m)
    assert ss.width == width
    assert ss.height == height
    assert ss.z_axis.shape == (m,)
    np.testing.assert_array_equal(ss.z_axis, np.arange(m, dtype=float))
    assert ss.source_type == "txt_matrix"
    assert ss.metadata is None  # no sidecar


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    """N != width * height must raise ValueError."""
    data = np.zeros((6, 10))
    p = _write_matrix(tmp_path / "bad.txt", data)

    with pytest.raises(ValueError, match="does not match"):
        load_txt_matrix(p, width=2, height=2)  # expects 4 rows, got 6


def test_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_txt_matrix(tmp_path / "no_such_file.txt", width=1, height=1)


def test_z_axis_attached(tmp_path: Path) -> None:
    """When z_axis.txt exists next to data, it should be loaded."""
    width, height, m = 2, 2, 4
    data = np.ones((width * height, m))
    p = _write_matrix(tmp_path / "signals.txt", data)

    z = np.array([10.0, 20.0, 30.0, 40.0])
    np.savetxt(str(tmp_path / "z_axis.txt"), z)

    ss = load_txt_matrix(p, width=width, height=height)

    np.testing.assert_allclose(ss.z_axis, z)
    assert ss.z_axis_path is not None


def test_metadata_attached(tmp_path: Path) -> None:
    """When an info file exists next to data, metadata should be parsed."""
    width, height, m = 1, 1, 3
    data = np.ones((1, m))
    p = _write_matrix(tmp_path / "signals.txt", data)

    info = tmp_path / "image_stack_info.txt"
    info.write_text(
        "objective specific wavelength [nm]: 536\n", encoding="utf-8"
    )

    ss = load_txt_matrix(p, width=width, height=height)

    assert ss.metadata is not None
    assert ss.metadata["wavelength_nm"] == 536.0
