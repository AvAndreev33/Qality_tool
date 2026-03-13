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


def _write_tiff_frames(directory: Path, data: np.ndarray) -> None:
    """Write a (M, H, W) array as individual frame files.

    Creates ``Image_00001.tif``, ``Image_00002.tif``, ... inside *directory*.
    """
    for i in range(data.shape[0]):
        tifffile.imwrite(str(directory / f"Image_{i + 1:05d}.tif"), data[i])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_stack_load(tmp_path: Path) -> None:
    """Load a directory of sequential TIF frames and check canonical output."""
    m, h, w = 5, 3, 4
    stack = np.random.default_rng(42).random((m, h, w)).astype(np.float32)
    _write_tiff_frames(tmp_path, stack)

    ss = load_image_stack(tmp_path)

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
    _write_tiff_frames(tmp_path, stack)

    ss = load_image_stack(tmp_path)

    # Signal at pixel (row=0, col=1) should be stack[:, 0, 1].
    expected = stack[:, 0, 1].astype(float)
    np.testing.assert_allclose(ss.signals[0, 1, :], expected)


def test_directory_not_found(tmp_path: Path) -> None:
    """Error when the directory does not exist."""
    with pytest.raises(FileNotFoundError):
        load_image_stack(tmp_path / "missing_dir")


def test_empty_directory(tmp_path: Path) -> None:
    """Error when the directory contains no TIF files."""
    with pytest.raises(FileNotFoundError, match="No .tif/.tiff frame files"):
        load_image_stack(tmp_path)


def test_sort_order(tmp_path: Path) -> None:
    """Frames must be loaded in numeric order, not alphabetical."""
    h, w = 2, 2
    frame_2 = np.full((h, w), 2.0, dtype=np.float32)
    frame_10 = np.full((h, w), 10.0, dtype=np.float32)
    frame_1 = np.full((h, w), 1.0, dtype=np.float32)

    tifffile.imwrite(str(tmp_path / "Image_00002.tif"), frame_2)
    tifffile.imwrite(str(tmp_path / "Image_00010.tif"), frame_10)
    tifffile.imwrite(str(tmp_path / "Image_00001.tif"), frame_1)

    ss = load_image_stack(tmp_path)

    assert ss.signals.shape == (h, w, 3)
    # Frame order must be 1, 2, 10 (numeric).
    np.testing.assert_allclose(ss.signals[0, 0, :], [1.0, 2.0, 10.0])


def test_mixed_extensions(tmp_path: Path) -> None:
    """Both .tif and .tiff files should be discovered."""
    h, w = 2, 2
    tifffile.imwrite(
        str(tmp_path / "frame_01.tif"),
        np.ones((h, w), dtype=np.float32),
    )
    tifffile.imwrite(
        str(tmp_path / "frame_02.tiff"),
        np.full((h, w), 2.0, dtype=np.float32),
    )

    ss = load_image_stack(tmp_path)

    assert ss.signals.shape == (h, w, 2)
    np.testing.assert_allclose(ss.signals[0, 0, :], [1.0, 2.0])


def test_inconsistent_frame_shape(tmp_path: Path) -> None:
    """Error when frames have different dimensions."""
    tifffile.imwrite(
        str(tmp_path / "Image_00001.tif"),
        np.zeros((3, 4), dtype=np.float32),
    )
    tifffile.imwrite(
        str(tmp_path / "Image_00002.tif"),
        np.zeros((5, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="expected"):
        load_image_stack(tmp_path)


def test_z_axis_attached(tmp_path: Path) -> None:
    """When z_axis.txt exists in the directory, it should be used."""
    m, h, w = 4, 2, 2
    stack = np.ones((m, h, w), dtype=np.float32)
    _write_tiff_frames(tmp_path, stack)

    z = np.array([100.0, 200.0, 300.0, 400.0])
    np.savetxt(str(tmp_path / "z_axis.txt"), z)

    ss = load_image_stack(tmp_path)

    np.testing.assert_allclose(ss.z_axis, z)
    assert ss.z_axis_path is not None


def test_metadata_attached(tmp_path: Path) -> None:
    """When image_stack_info.txt exists in the directory, metadata is parsed."""
    m, h, w = 3, 2, 2
    stack = np.ones((m, h, w), dtype=np.float32)
    _write_tiff_frames(tmp_path, stack)

    info = tmp_path / "image_stack_info.txt"
    info.write_text(
        "objective specific wavelength [nm]: 536\n", encoding="utf-8"
    )

    ss = load_image_stack(tmp_path)

    assert ss.metadata is not None
    assert ss.metadata["wavelength_nm"] == 536.0
