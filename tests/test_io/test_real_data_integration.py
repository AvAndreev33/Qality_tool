"""Integration tests against real WLI testing data.

These tests are skipped automatically when the ``testing_data/`` directory
is not present (e.g. in CI environments without real data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Locate testing_data relative to project root.
# The project root contains pyproject.toml and testing_data/.
# Walk up from this file to find it.
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent  # tests/test_io -> tests -> project root

# testing_data may live in the main repo rather than a worktree, so also
# check the main repo location.
_TESTING_DATA = _PROJECT_ROOT / "testing_data"
if not _TESTING_DATA.is_dir():
    # Worktree layout: .claude/worktrees/<name>/ — main repo is 3 levels up.
    _MAIN_REPO = _PROJECT_ROOT.parent.parent.parent
    _TESTING_DATA = _MAIN_REPO / "testing_data"

_HAS_REAL_DATA = _TESTING_DATA.is_dir()

_REAL_DATA_TXT = _TESTING_DATA / "real_data_txt"
_REAL_DATA_STACK = _TESTING_DATA / "real_data_stack"

skip_no_data = pytest.mark.skipif(
    not _HAS_REAL_DATA,
    reason="testing_data/ directory not found",
)


# ---------------------------------------------------------------------------
# Metadata integration
# ---------------------------------------------------------------------------


@skip_no_data
def test_real_metadata_fields() -> None:
    """Parse the real sidecar info file and verify key fields."""
    from quality_tool.io.metadata_parser import parse_info_file

    info_path = _REAL_DATA_TXT / "image_stack_info.txt"
    if not info_path.is_file():
        pytest.skip("image_stack_info.txt not found")

    meta = parse_info_file(info_path)

    assert meta is not None
    assert meta["wavelength_nm"] == pytest.approx(566.0)
    assert meta["coherence_length_nm"] == pytest.approx(1556.0)
    assert meta["z_step_nm"] == pytest.approx(70.75)
    assert meta["z_start_mm"] == pytest.approx(30.3404)
    assert meta["z_end_mm"] == pytest.approx(30.37)
    assert meta["exposure_time_us"] == pytest.approx(1998.0)
    assert meta["illumination_intensity"] == pytest.approx(59.0)
    assert "pixel_size_x_mm" in meta
    assert "pixel_size_y_mm" in meta


# ---------------------------------------------------------------------------
# TXT matrix integration
# ---------------------------------------------------------------------------


@skip_no_data
def test_load_real_txt_matrix() -> None:
    """Load the real TXT signal matrix and verify shape and metadata."""
    from quality_tool.io.txt_matrix_loader import load_txt_matrix

    txt_path = _REAL_DATA_TXT / "correlogram_segments_surface_0_0.txt"
    if not txt_path.is_file():
        pytest.skip("correlogram_segments_surface_0_0.txt not found")

    ss = load_txt_matrix(txt_path, width=1920, height=1200)

    assert ss.signals.shape == (1200, 1920, 128)
    assert ss.width == 1920
    assert ss.height == 1200
    assert ss.z_axis.shape == (128,)
    # No z_axis.txt present — should fall back to index axis.
    assert ss.z_axis_path is None
    np.testing.assert_array_equal(ss.z_axis, np.arange(128, dtype=float))
    # Metadata should have been parsed from the sidecar info file.
    assert ss.metadata is not None
    assert ss.source_type == "txt_matrix"


@skip_no_data
def test_real_txt_signal_sanity() -> None:
    """Spot-check that loaded signal values are in a plausible range."""
    from quality_tool.io.txt_matrix_loader import load_txt_matrix

    txt_path = _REAL_DATA_TXT / "correlogram_segments_surface_0_0.txt"
    if not txt_path.is_file():
        pytest.skip("correlogram_segments_surface_0_0.txt not found")

    ss = load_txt_matrix(txt_path, width=1920, height=1200)

    # Values should be in plausible uint8-range (camera output).
    assert ss.signals.min() >= 0
    assert ss.signals.max() <= 255
    # Check a single pixel has 128 samples with non-zero variance.
    pixel_signal = ss.signals[0, 0, :]
    assert pixel_signal.shape == (128,)
    assert np.std(pixel_signal) > 0


# ---------------------------------------------------------------------------
# Image stack integration
# ---------------------------------------------------------------------------


@skip_no_data
def test_load_real_image_stack() -> None:
    """Load the real image stack directory and verify shape and metadata."""
    from quality_tool.io.image_stack_loader import load_image_stack

    stack_dir = _REAL_DATA_STACK
    if not stack_dir.is_dir():
        pytest.skip("real_data_stack/ directory not found")

    ss = load_image_stack(stack_dir)

    assert ss.signals.shape == (1200, 1920, 418)
    assert ss.width == 1920
    assert ss.height == 1200
    assert ss.z_axis.shape == (418,)
    # No z_axis.txt present — should fall back to index axis.
    assert ss.z_axis_path is None
    np.testing.assert_array_equal(ss.z_axis, np.arange(418, dtype=float))
    # Metadata should have been parsed from the sidecar info file.
    assert ss.metadata is not None
    assert ss.source_type == "image_stack"
