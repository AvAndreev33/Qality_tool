"""Tests for quality_tool.io.metadata_parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from quality_tool.io.metadata_parser import parse_info_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_info_file(tmp_path: Path) -> Path:
    """Create a representative sidecar info file."""
    content = textwrap.dedent("""\
        Objective info:

        objective magnification factor: 10x
        objective image scale x - y [mm/Pixel]: 0.000950727 - 0.000950727
        objective specific wavelength [nm]: 536
        objective specific coherence length [nm]: 2680


        Camera info:

        shutter time [us]: 1998


        Scanning (positioning) info:

        scanning device step size [nm]: 67
        oversampling factor: None
        scanning device start position [mm]: 0.700001
        scanning device end position [mm]: 0.74

        trigger rate (nominal / camera setting) [fps]: 168.921
        scan velocity [um/s]: 11.34
        scan distance during exposure time [nm]: 22.6573
        periods during exposure time: 0.0845419 (corresponding phase differenz [°]: 30.4351)


        Light control info:

        illumination intensity: 58


        Axes info:

        X axis offset [mm]: -0.922031
        Y axis offset [mm]: -0.553954
    """)
    p = tmp_path / "image_stack_info.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def minimal_info_file(tmp_path: Path) -> Path:
    """Info file with only a few fields."""
    content = textwrap.dedent("""\
        Objective info:

        objective specific wavelength [nm]: 536
    """)
    p = tmp_path / "info.txt"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_full_info(full_info_file: Path) -> None:
    meta = parse_info_file(full_info_file)
    assert meta is not None

    # Optics
    assert meta["wavelength_nm"] == 536.0
    assert meta["coherence_length_nm"] == 2680.0
    assert meta["objective_magnification"] == "10x"

    # Pixel size (split field)
    assert meta["pixel_size_x_mm"] == pytest.approx(0.000950727)
    assert meta["pixel_size_y_mm"] == pytest.approx(0.000950727)

    # Scanning
    assert meta["z_step_nm"] == 67.0
    assert meta["z_start_mm"] == pytest.approx(0.700001)
    assert meta["z_end_mm"] == pytest.approx(0.74)
    assert meta["oversampling_factor"] == "None"
    assert meta["trigger_rate_fps"] == pytest.approx(168.921)
    assert meta["scan_velocity_um_s"] == pytest.approx(11.34)
    assert meta["exposure_time_us"] == pytest.approx(1998.0)
    assert meta["scan_distance_during_exposure_nm"] == pytest.approx(22.6573)
    assert meta["periods_during_exposure"] == pytest.approx(0.0845419)

    # Light
    assert meta["illumination_intensity"] == 58.0

    # Axes
    assert meta["x_axis_offset_mm"] == pytest.approx(-0.922031)
    assert meta["y_axis_offset_mm"] == pytest.approx(-0.553954)


def test_parse_minimal_info(minimal_info_file: Path) -> None:
    meta = parse_info_file(minimal_info_file)
    assert meta is not None
    assert meta["wavelength_nm"] == 536.0
    # Fields not present should be absent, not None.
    assert "z_step_nm" not in meta


def test_parse_nonexistent_file(tmp_path: Path) -> None:
    result = parse_info_file(tmp_path / "does_not_exist.txt")
    assert result is None


def test_parse_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    result = parse_info_file(p)
    assert result is None
