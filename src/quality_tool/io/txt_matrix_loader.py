"""TXT-matrix loader for Quality_tool.

Loads a ``.txt`` file containing a signal matrix of shape ``(N, M)`` and
returns a validated ``SignalSet`` with signals in canonical ``(H, W, M)``
format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from quality_tool.core.models import SignalSet
from quality_tool.io.metadata_parser import parse_info_file
from quality_tool.io.z_axis_loader import load_z_axis

# Common names for sidecar info files, tried in order.
_INFO_FILE_CANDIDATES: Sequence[str] = (
    "image_stack_info.txt",
    "info.txt",
)


def _find_info_file(data_path: Path) -> Path | None:
    """Try to locate a sidecar info file next to *data_path*."""
    parent = data_path.parent
    for name in _INFO_FILE_CANDIDATES:
        candidate = parent / name
        if candidate.is_file():
            return candidate
    return None


def _find_z_axis_file(data_path: Path) -> Path | None:
    """Try to locate ``z_axis.txt`` next to *data_path*."""
    candidate = data_path.parent / "z_axis.txt"
    return candidate if candidate.is_file() else None


def load_txt_matrix(
    path: str | Path,
    width: int,
    height: int,
    *,
    info_path: str | Path | None = None,
    z_axis_path: str | Path | None = None,
) -> SignalSet:
    """Load a TXT signal matrix and return a ``SignalSet``.

    Parameters
    ----------
    path : str or Path
        Path to the ``.txt`` data file.  Expected shape ``(N, M)``.
    width : int
        Image width.
    height : int
        Image height.
    info_path : str, Path, or None
        Explicit path to a sidecar info file.  If ``None`` the loader
        tries to auto-discover one next to *path*.
    z_axis_path : str, Path, or None
        Explicit path to ``z_axis.txt``.  If ``None`` the loader tries
        to auto-discover it next to *path*.

    Returns
    -------
    SignalSet

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If ``N != width * height``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"TXT data file not found: {p}")

    matrix = np.loadtxt(str(p))

    if matrix.ndim == 1:
        # Single signal — treat as (1, M).
        matrix = matrix.reshape(1, -1)

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2-D matrix (N, M), got shape {matrix.shape}"
        )

    n, m = matrix.shape

    if n != width * height:
        raise ValueError(
            f"Row count N={n} does not match width*height={width}*{height}={width * height}"
        )

    signals = matrix.reshape(height, width, m)

    # --- sidecar info ---
    if info_path is None:
        info_file = _find_info_file(p)
    else:
        info_file = Path(info_path)

    metadata = parse_info_file(info_file) if info_file is not None else None
    resolved_info = str(info_file.resolve()) if info_file is not None else None

    # --- z-axis ---
    if z_axis_path is None:
        z_axis_path = _find_z_axis_file(p)

    z_axis, resolved_z = load_z_axis(z_axis_path, m)

    return SignalSet(
        signals=signals,
        width=width,
        height=height,
        z_axis=z_axis,
        metadata=metadata,
        source_type="txt_matrix",
        source_path=str(p.resolve()),
        info_path=resolved_info,
        z_axis_path=resolved_z,
    )
