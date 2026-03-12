"""Metadata parser for WLI acquisition sidecar info files.

Reads key-value info text files produced by the WLI acquisition system and
extracts useful, normalized metadata fields.  Irrelevant fields are discarded.
Missing optional fields are tolerated — the returned dict contains only
whatever was successfully parsed.
"""

from __future__ import annotations

import re
from pathlib import Path

# Mapping: (raw key in info file) → (normalised metadata key, conversion fn).
# Conversion functions accept a string and return the normalised value.
_FIELD_MAP: dict[str, tuple[str, type]] = {
    "objective image scale x - y [mm/pixel]": (
        "_pixel_size_xy",  # special: split into two fields
        str,
    ),
    "objective specific wavelength [nm]": ("wavelength_nm", float),
    "objective specific coherence length [nm]": ("coherence_length_nm", float),
    "objective magnification factor": ("objective_magnification", str),
    "scanning device step size [nm]": ("z_step_nm", float),
    "scanning device start position [mm]": ("z_start_mm", float),
    "scanning device end position [mm]": ("z_end_mm", float),
    "oversampling factor": ("oversampling_factor", str),
    "trigger rate (nominal / camera setting) [fps]": ("trigger_rate_fps", float),
    "scan velocity [um/s]": ("scan_velocity_um_s", float),
    "shutter time [us]": ("exposure_time_us", float),
    "scan distance during exposure time [nm]": (
        "scan_distance_during_exposure_nm",
        float,
    ),
    "periods during exposure time": ("_periods_during_exposure_raw", str),
    "illumination intensity": ("illumination_intensity", float),
    "x axis offset [mm]": ("x_axis_offset_mm", float),
    "y axis offset [mm]": ("y_axis_offset_mm", float),
}


def _try_float(value: str) -> float | None:
    """Attempt to parse a float from a string, return None on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_pixel_size(value: str) -> dict[str, float]:
    """Parse ``'0.000950727 - 0.000950727'`` into two float fields."""
    parts = [p.strip() for p in value.split("-")]
    result: dict[str, float] = {}
    if len(parts) >= 1:
        v = _try_float(parts[0])
        if v is not None:
            result["pixel_size_x_mm"] = v
    if len(parts) >= 2:
        v = _try_float(parts[1])
        if v is not None:
            result["pixel_size_y_mm"] = v
    return result


def _parse_periods(raw: str) -> float | None:
    """Extract the leading numeric value from the periods field.

    Example input: ``'0.0845419 (corresponding phase differenz [°]: 30.4351)'``
    """
    m = re.match(r"([0-9.]+)", raw.strip())
    if m:
        return _try_float(m.group(1))
    return None


def parse_info_file(path: str | Path) -> dict | None:
    """Parse a WLI acquisition sidecar info file.

    Parameters
    ----------
    path : str or Path
        Path to the info text file.

    Returns
    -------
    dict or None
        Dictionary of normalised metadata fields, or ``None`` if the file
        cannot be read.
    """
    path = Path(path)
    if not path.is_file():
        return None

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    result: dict = {}

    for line in text.splitlines():
        # Lines are expected as  "key: value" or "key:  value".
        if ":" not in line:
            continue
        raw_key, _, raw_value = line.partition(":")
        raw_key = raw_key.strip().lower()
        raw_value = raw_value.strip()

        if not raw_key or not raw_value:
            continue

        if raw_key not in _FIELD_MAP:
            continue

        norm_key, conv = _FIELD_MAP[raw_key]

        # Special-case: pixel size (two values in one line).
        if norm_key == "_pixel_size_xy":
            result.update(_parse_pixel_size(raw_value))
            continue

        # Special-case: periods during exposure (has trailing parenthetical).
        if norm_key == "_periods_during_exposure_raw":
            v = _parse_periods(raw_value)
            if v is not None:
                result["periods_during_exposure"] = v
            continue

        # General numeric / string conversion.
        if conv is float:
            v = _try_float(raw_value)
            if v is not None:
                result[norm_key] = v
        else:
            result[norm_key] = raw_value

    return result if result else None
