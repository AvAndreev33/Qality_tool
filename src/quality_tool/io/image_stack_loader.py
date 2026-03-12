"""Image-stack loader for Quality_tool.

Loads ``.tif`` / ``.tiff`` image stacks and returns a validated ``SignalSet``
with signals in canonical ``(H, W, M)`` format.
"""

from __future__ import annotations

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


def load_image_stack(
    path: str | Path,
    *,
    info_path: str | Path | None = None,
    z_axis_path: str | Path | None = None,
) -> SignalSet:
    """Load a TIFF image stack and return a ``SignalSet``.

    Parameters
    ----------
    path : str or Path
        Path to the ``.tif`` / ``.tiff`` stack file.
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
        If the stack cannot be interpreted as ``(M, H, W)`` frames.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image stack not found: {p}")

    stack = tifffile.imread(str(p))  # typically (M, H, W) or (M, H, W, C)

    if stack.ndim == 4:
        # Multi-channel image — use first channel.
        stack = stack[:, :, :, 0]

    if stack.ndim != 3:
        raise ValueError(
            f"Expected a 3-D stack (M, H, W), got shape {stack.shape}"
        )

    # Stack convention from tifffile: (M, H, W).
    # Canonical format: (H, W, M).
    signals = np.moveaxis(stack, 0, -1).astype(float)

    height, width, signal_length = signals.shape

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

    z_axis, resolved_z = load_z_axis(z_axis_path, signal_length)

    return SignalSet(
        signals=signals,
        width=width,
        height=height,
        z_axis=z_axis,
        metadata=metadata,
        source_type="image_stack",
        source_path=str(p.resolve()),
        info_path=resolved_info,
        z_axis_path=resolved_z,
    )
