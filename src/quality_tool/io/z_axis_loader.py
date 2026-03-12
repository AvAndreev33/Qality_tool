"""Z-axis loader for Quality_tool.

Supports two modes:
- Mode A: load an explicit z-axis from a text file.
- Mode B: generate an index-based z-axis ``[0, 1, ..., M-1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_z_axis(
    path: str | Path | None,
    signal_length: int,
) -> tuple[np.ndarray, str | None]:
    """Load or generate a z-axis array.

    Parameters
    ----------
    path : str, Path, or None
        Path to ``z_axis.txt``.  If ``None`` or the file does not exist,
        an index-based axis is generated instead.
    signal_length : int
        Expected number of samples (M).

    Returns
    -------
    z_axis : np.ndarray
        1-D array of length *signal_length*.
    resolved_path : str or None
        Absolute path to the file that was loaded, or ``None`` if an
        index-based fallback was used.

    Raises
    ------
    ValueError
        If the file exists but its length does not match *signal_length*.
    """
    if path is not None:
        p = Path(path)
        if p.is_file():
            z = np.loadtxt(p).ravel()
            if len(z) != signal_length:
                raise ValueError(
                    f"z_axis file has {len(z)} values, expected {signal_length}"
                )
            return z, str(p.resolve())

    # Mode B — index-based fallback.
    return np.arange(signal_length, dtype=float), None
