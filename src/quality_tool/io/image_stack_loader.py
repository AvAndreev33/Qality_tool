"""Image-stack loader for Quality_tool.

Loads a directory of sequential ``.tif`` / ``.tiff`` frames and returns a
validated ``SignalSet`` with signals in canonical ``(H, W, M)`` format.

Expected directory layout::

    stack_dir/
        Image_00001.tif
        Image_00002.tif
        ...
        image_stack_info.txt   (optional sidecar)
        z_axis.txt             (optional)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile

from quality_tool.core.models import SignalSet
from quality_tool.io.metadata_parser import parse_info_file
from quality_tool.io.z_axis_loader import load_z_axis

# Common names for sidecar info files, tried in order.
_INFO_FILE_CANDIDATES: Sequence[str] = (
    "image_stack_info.txt",
    "info.txt",
)

# Regex to extract the trailing integer from a filename stem.
_FRAME_NUMBER_RE = re.compile(r"(\d+)$")


def _find_info_file(directory: Path) -> Path | None:
    """Try to locate a sidecar info file inside *directory*."""
    for name in _INFO_FILE_CANDIDATES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _find_z_axis_file(directory: Path) -> Path | None:
    """Try to locate ``z_axis.txt`` inside *directory*."""
    candidate = directory / "z_axis.txt"
    return candidate if candidate.is_file() else None


def _sort_key(path: Path) -> int:
    """Extract the trailing integer from a filename for numeric sorting."""
    m = _FRAME_NUMBER_RE.search(path.stem)
    if m is None:
        raise ValueError(
            f"Cannot extract frame number from filename: {path.name}"
        )
    return int(m.group(1))


def _discover_frames(directory: Path) -> list[Path]:
    """Find and sort TIFF frame files in *directory*.

    Returns the files sorted by their trailing numeric index.

    Raises
    ------
    FileNotFoundError
        If no TIFF files are found in *directory*.
    """
    frames = sorted(
        [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in (".tif", ".tiff")
        ],
        key=_sort_key,
    )
    if not frames:
        raise FileNotFoundError(
            f"No .tif/.tiff frame files found in directory: {directory}"
        )
    return frames


def load_image_stack(
    path: str | Path,
    *,
    info_path: str | Path | None = None,
    z_axis_path: str | Path | None = None,
) -> SignalSet:
    """Load a directory of sequential TIFF frames and return a ``SignalSet``.

    Parameters
    ----------
    path : str or Path
        Path to the directory containing sequential ``.tif`` / ``.tiff``
        frame files.
    info_path : str, Path, or None
        Explicit path to a sidecar info file.  If ``None`` the loader
        tries to auto-discover one inside the directory.
    z_axis_path : str, Path, or None
        Explicit path to ``z_axis.txt``.  If ``None`` the loader tries
        to auto-discover it inside the directory.

    Returns
    -------
    SignalSet

    Raises
    ------
    FileNotFoundError
        If *path* is not a directory or contains no TIFF files.
    ValueError
        If frames have inconsistent shapes.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Image stack directory not found: {directory}"
        )

    frames = _discover_frames(directory)

    # Read the first frame to determine (H, W).
    first = tifffile.imread(str(frames[0]))
    if first.ndim != 2:
        raise ValueError(
            f"Expected 2-D frames (H, W), got shape {first.shape} "
            f"from {frames[0].name}"
        )
    expected_shape = first.shape

    # Pre-allocate the signal array in canonical format (H, W, M).
    h, w = expected_shape
    m = len(frames)
    signals = np.empty((h, w, m), dtype=float)
    signals[:, :, 0] = first.astype(float)

    for i, fp in enumerate(frames[1:], start=1):
        frame = tifffile.imread(str(fp))
        if frame.shape != expected_shape:
            raise ValueError(
                f"Frame {fp.name} has shape {frame.shape}, "
                f"expected {expected_shape}"
            )
        signals[:, :, i] = frame.astype(float)

    # --- sidecar info ---
    if info_path is None:
        info_file = _find_info_file(directory)
    else:
        info_file = Path(info_path)

    metadata = parse_info_file(info_file) if info_file is not None else None
    resolved_info = str(info_file.resolve()) if info_file is not None else None

    # --- z-axis ---
    if z_axis_path is None:
        z_axis_path = _find_z_axis_file(directory)

    z_axis, resolved_z = load_z_axis(z_axis_path, m)

    return SignalSet(
        signals=signals,
        width=w,
        height=h,
        z_axis=z_axis,
        metadata=metadata,
        source_type="image_stack",
        source_path=str(directory.resolve()),
        info_path=resolved_info,
        z_axis_path=resolved_z,
    )
