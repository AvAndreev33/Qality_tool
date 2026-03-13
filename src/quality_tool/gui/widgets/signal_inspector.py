"""Signal inspector widget for Quality_tool.

Displays the 1-D signal of a selected pixel.  Designed to be extended
later with envelope, ROI, or spectrum overlays.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class SignalInspector(QWidget):
    """Lower plot showing the signal of the currently selected pixel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._ax.set_xlabel("z")
        self._ax.set_ylabel("intensity")
        self._ax.set_title("Signal")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_signal(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray,
        *,
        label: str = "raw",
        title: str | None = None,
    ) -> None:
        """Plot a 1-D signal.

        Parameters
        ----------
        signal : np.ndarray
            1-D array of length M.
        z_axis : np.ndarray
            1-D array of length M — physical or index-based.
        label : str
            Legend label for the line.
        title : str or None
            Plot title.  Defaults to ``"Signal"``.
        """
        self._ax.clear()
        self._ax.plot(z_axis, signal, linewidth=0.8, label=label)
        self._ax.set_xlabel("z")
        self._ax.set_ylabel("intensity")
        self._ax.set_title(title or "Signal")
        self._ax.legend(loc="upper right", fontsize="small")
        self._canvas.draw_idle()

    def clear(self) -> None:
        """Clear the signal plot."""
        self._ax.clear()
        self._ax.set_xlabel("z")
        self._ax.set_ylabel("intensity")
        self._ax.set_title("Signal")
        self._canvas.draw_idle()
