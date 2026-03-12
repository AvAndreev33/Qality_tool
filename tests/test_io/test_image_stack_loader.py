"""Tests for quality_tool.io.image_stack_loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from quality_tool.io.image_stack_loader import load_image_stack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiff_stack(path: Path, data: np.ndarray) -> Path:
    """Write a 3-D numpy array as a multi-page TIFF (M, H, W)."""
    tifffile.imwrite(str(path), data)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_stack_load(tmp_path: Path) -> None:
    """Load a simple (M, H, W) TIFF stack and check canonical output."""
    m, h, w = 5, 3, 4
    stack = np.random.default_rng(42).random((m, h, w)).astype(np.float32)
    p = _write_tiff_stack(tmp_path / "stack.tif", stack)

    ss = load_image_stack(p)

    assert ss.signals.shape == (h, w, m)
    assert ss.width == w
    assert ss.height == h
    assert ss.z_axis.shape == (m,)
    np.testing.assert_array_equal(ss.z_axis, np.arange(m, dtype=float))
    assert ss.source_type == "image_stack"


def test_signal_values_correct(tmp_path: Path) -> None:
    """Pixel signals should match the original stack data."""
    m, h, w = 3, 2, 2
    stack = np.arange(m * h * w, dtype=np.float32).reshape(m, h, w)
    p = _write_tiff_stack(tmp_path / "stack.tif", stack)

    ss = load_image_stack(p)

    # Signal at pixel (row=0, col=1) should be stack[:, 0, 1].
    expected = stack[:, 0, 1].astype(float)
    np.testing.assert_allclose(ss.signals[0, 1, :], expected)


def test_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image_stack(tmp_path / "missing.tif")


def test_z_axis_attached(tmp_path: Path) -> None:
    """When z_axis.txt exists next to the TIFF, it should be used."""
    m, h, w = 4, 2, 2
    stack = np.ones((m, h, w), dtype=np.float32)
    p = _write_tiff_stack(tmp_path / "stack.tif", stack)

    z = np.array([100.0, 200.0, 300.0, 400.0])
    np.savetxt(str(tmp_path / "z_axis.txt"), z)

    ss = load_image_stack(p)

    np.testing.assert_allclose(ss.z_axis, z)
    assert ss.z_axis_path is not None


def test_metadata_attached(tmp_path: Path) -> None:
    """When image_stack_info.txt exists next to TIFF, metadata is parsed."""
    m, h, w = 3, 2, 2
    stack = np.ones((m, h, w), dtype=np.float32)
    p = _write_tiff_stack(tmp_path / "stack.tif", stack)

    info = tmp_path / "image_stack_info.txt"
    info.write_text(
        "objective specific wavelength [nm]: 536\n", encoding="utf-8"
    )

    ss = load_image_stack(p)

    assert ss.metadata is not None
    assert ss.metadata["wavelength_nm"] == 536.0
