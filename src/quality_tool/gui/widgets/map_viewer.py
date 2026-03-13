"""Generic 2D map viewer widget for Quality_tool.

Displays a single (H, W) array with a colorbar and emits pixel coordinates
when the user clicks on the map.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


class MapViewer(QWidget):
    """A generic 2D map display with click-to-select pixel interaction.

    Signals
    -------
    pixel_selected(int, int)
        Emitted when the user clicks a pixel.  Arguments are (row, col).
    """

    pixel_selected = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        self._colorbar = None
        self._image = None

        # Current map data for value look-up.
        self._data: np.ndarray | None = None

        # Selected pixel marker.
        self._marker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_map(
        self,
        data: np.ndarray,
        title: str = "",
        cmap: str = "viridis",
    ) -> None:
        """Display a 2D array in the viewer.

        Parameters
        ----------
        data : np.ndarray
            2-D array of shape (H, W).
        title : str
            Title shown above the map.
        cmap : str
            Matplotlib colormap name.
        """
        self._data = data
        self._ax.clear()

        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

        self._image = self._ax.imshow(
            data, origin="upper", aspect="equal", cmap=cmap,
        )
        self._colorbar = self._figure.colorbar(self._image, ax=self._ax)
        self._ax.set_title(title)
        self._marker = None
        self._canvas.draw_idle()

    def set_binary_mask(self, mask: np.ndarray, title: str = "") -> None:
        """Display a boolean mask (green = kept, red = rejected)."""
        rgb = np.zeros((*mask.shape, 3), dtype=float)
        rgb[mask] = [0.2, 0.7, 0.3]
        rgb[~mask] = [0.8, 0.2, 0.2]

        self._data = mask.astype(float)
        self._ax.clear()

        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

        self._image = self._ax.imshow(rgb, origin="upper", aspect="equal")
        self._ax.set_title(title)
        self._marker = None
        self._canvas.draw_idle()

    def clear(self) -> None:
        """Clear the viewer."""
        self._data = None
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except (AttributeError, ValueError):
                pass
            self._colorbar = None
        self._ax.clear()
        self._image = None
        self._marker = None
        self._canvas.draw_idle()

    def get_snapshot(self) -> tuple[np.ndarray | None, str]:
        """Return current data and title for the comparison window."""
        title = self._ax.get_title()
        if self._data is not None:
            return self._data.copy(), title
        return None, title

    def value_at(self, row: int, col: int) -> float | None:
        """Return the map value at (row, col), or None if no data."""
        if self._data is None:
            return None
        if 0 <= row < self._data.shape[0] and 0 <= col < self._data.shape[1]:
            return float(self._data[row, col])
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_click(self, event) -> None:
        """Handle matplotlib click and emit pixel_selected."""
        if event.inaxes != self._ax or self._data is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))

        h, w = self._data.shape[:2]
        if not (0 <= row < h and 0 <= col < w):
            return

        self._draw_marker(row, col)
        self.pixel_selected.emit(row, col)

    def _draw_marker(self, row: int, col: int) -> None:
        """Draw a crosshair on the selected pixel."""
        if self._marker is not None:
            self._marker.remove()
        (self._marker,) = self._ax.plot(
            col, row, marker="+", color="red",
            markersize=12, markeredgewidth=1.5,
        )
        self._canvas.draw_idle()
