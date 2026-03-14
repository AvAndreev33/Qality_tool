"""Signal inspector widget for Quality_tool.

Displays the 1-D signal of a selected pixel.  Supports three display
modes (raw, processed, spectrum) and an optional envelope overlay.
The widget only renders data — all processing is done by the caller
before passing data in.
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
        envelope: np.ndarray | None = None,
    ) -> None:
        """Plot a 1-D signal with optional envelope overlay.

        Parameters
        ----------
        signal : np.ndarray
            1-D array of length M.
        z_axis : np.ndarray
            1-D array of length M — physical or index-based.
        label : str
            Legend label for the signal line.
        title : str or None
            Plot title.  Defaults to ``"Signal"``.
        envelope : np.ndarray or None
            If provided, overlay this 1-D envelope on the plot.
        """
        self._ax.clear()
        self._ax.plot(z_axis, signal, linewidth=0.8, label=label)
        if envelope is not None:
            self._ax.plot(
                z_axis, envelope, linewidth=1.0, linestyle="--",
                color="tab:orange", label="envelope",
            )
        self._ax.set_xlabel("z")
        self._ax.set_ylabel("intensity")
        self._ax.set_title(title or "Signal")
        self._ax.legend(loc="upper right", fontsize="small")
        self._canvas.draw_idle()

    def update_spectrum(
        self,
        frequencies: np.ndarray,
        amplitude: np.ndarray,
        *,
        title: str | None = None,
    ) -> None:
        """Plot the amplitude spectrum with logarithmic y-scale.

        Parameters
        ----------
        frequencies : np.ndarray
            1-D frequency array.
        amplitude : np.ndarray
            1-D amplitude array.
        title : str or None
            Plot title.
        """
        self._ax.clear()
        self._ax.plot(frequencies, amplitude, linewidth=0.8, label="spectrum")
        self._ax.set_yscale("log")
        self._ax.set_xlabel("frequency")
        self._ax.set_ylabel("amplitude")
        self._ax.set_title(title or "Spectrum")
        self._ax.legend(loc="upper right", fontsize="small")
        self._canvas.draw_idle()

    def clear(self) -> None:
        """Clear the signal plot."""
        self._ax.clear()
        self._ax.set_xlabel("z")
        self._ax.set_ylabel("intensity")
        self._ax.set_title("Signal")
        self._canvas.draw_idle()
