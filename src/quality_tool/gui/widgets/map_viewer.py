"""Generic 2D map viewer widget for Quality_tool.

Displays a single (H, W) array with a colorbar and emits pixel coordinates
when the user clicks on the map.  Supports mouse-wheel zoom and reset-view.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from quality_tool.gui.style import apply_mpl_dark_style, create_dark_figure

# Zoom factor per scroll step (< 1 means zoom in).
_ZOOM_IN_FACTOR = 0.8
_ZOOM_OUT_FACTOR = 1.25


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

        self._figure = create_dark_figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        apply_mpl_dark_style(self._figure)
        self._colorbar = None
        self._image = None

        # Current map data for value look-up.
        self._data: np.ndarray | None = None

        # Selected pixel marker and its (row, col) position.
        self._marker = None
        self._selected_rc: tuple[int, int] | None = None

        # Home view limits stored when a new map is set.
        self._home_xlim: tuple[float, float] | None = None
        self._home_ylim: tuple[float, float] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_map(
        self,
        data: np.ndarray,
        title: str = "",
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
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
        vmin, vmax : float or None
            Optional explicit color range.  When ``None``, matplotlib
            auto-scales to the data range.
        """
        prev = self._snapshot_view(data)
        self._data = data
        self._remove_colorbar()
        self._ax.clear()

        self._image = self._ax.imshow(
            data, origin="upper", aspect="equal", cmap=cmap,
            vmin=vmin, vmax=vmax,
        )
        self._colorbar = self._figure.colorbar(self._image, ax=self._ax)
        self._ax.set_title(title)
        apply_mpl_dark_style(self._figure)
        self._marker = None
        self._store_home_limits()
        self._restore_view(prev)
        self._canvas.draw_idle()

    def set_masked_map(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        title: str = "",
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """Display a score map with rejected pixels shown in neutral gray.

        Parameters
        ----------
        data : np.ndarray
            2-D score array of shape (H, W).
        mask : np.ndarray
            Boolean mask — True = kept, False = rejected.
        title : str
            Title shown above the map.
        cmap : str
            Matplotlib colormap name.
        vmin, vmax : float | None
            Color range anchored to the full original score map.
        """
        prev = self._snapshot_view(data)
        self._data = data  # keep original scores for value_at()
        self._remove_colorbar()
        self._ax.clear()

        # Gray background first (lowest layer) for rejected pixels.
        gray_bg = np.full((*data.shape, 3), 0.85)
        self._ax.imshow(gray_bg, origin="upper", aspect="equal")

        # Masked score map on top — masked (rejected) pixels are
        # transparent, revealing the gray layer beneath.
        masked = np.ma.array(data, mask=~mask)
        self._image = self._ax.imshow(
            masked,
            origin="upper",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        self._colorbar = self._figure.colorbar(self._image, ax=self._ax)
        self._ax.set_title(title)
        apply_mpl_dark_style(self._figure)
        self._marker = None
        self._store_home_limits()
        self._restore_view(prev)
        self._canvas.draw_idle()

    def set_binary_mask(self, mask: np.ndarray, title: str = "") -> None:
        """Display a boolean mask (green = kept, red = rejected)."""
        bool_mask = mask.astype(bool)
        rgb = np.zeros((*bool_mask.shape, 3), dtype=float)
        rgb[bool_mask] = [0.2, 0.7, 0.3]
        rgb[~bool_mask] = [0.8, 0.2, 0.2]

        prev = self._snapshot_view(bool_mask)
        self._data = bool_mask
        self._remove_colorbar()
        self._ax.clear()

        self._image = self._ax.imshow(rgb, origin="upper", aspect="equal")
        self._ax.set_title(title)
        apply_mpl_dark_style(self._figure)
        self._marker = None
        self._store_home_limits()
        self._restore_view(prev)
        self._canvas.draw_idle()

    def reset_view(self) -> None:
        """Restore the map viewer to its default full-map extent."""
        if self._home_xlim is not None and self._home_ylim is not None:
            self._ax.set_xlim(self._home_xlim)
            self._ax.set_ylim(self._home_ylim)
            self._canvas.draw_idle()

    def clear(self) -> None:
        """Clear the viewer."""
        self._data = None
        self._remove_colorbar()
        self._ax.clear()
        apply_mpl_dark_style(self._figure)
        self._image = None
        self._marker = None
        self._selected_rc = None
        self._home_xlim = None
        self._home_ylim = None
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

    def _store_home_limits(self) -> None:
        """Capture the current axes limits as the home (full-map) view."""
        self._home_xlim = self._ax.get_xlim()
        self._home_ylim = self._ax.get_ylim()

    def _snapshot_view(
        self, new_data: np.ndarray,
    ) -> dict | None:
        """Capture current zoom and marker state before a redraw.

        Returns ``None`` when there is nothing to restore (first
        display or map shape changed).
        """
        if self._data is None:
            return None
        # Only restore when the new data has the same spatial shape —
        # a different shape means a completely new dataset.
        if new_data.shape[:2] != self._data.shape[:2]:
            return None
        return {
            "xlim": self._ax.get_xlim(),
            "ylim": self._ax.get_ylim(),
            "pixel": self._selected_rc,
        }

    def _restore_view(self, prev: dict | None) -> None:
        """Restore zoom and marker from a previous snapshot."""
        if prev is None:
            return
        self._ax.set_xlim(prev["xlim"])
        self._ax.set_ylim(prev["ylim"])
        if prev["pixel"] is not None:
            row, col = prev["pixel"]
            self._draw_marker(row, col)

    def _remove_colorbar(self) -> None:
        """Safely remove the current colorbar, if any.

        Must be called **before** ``self._ax.clear()`` because
        ``ax.clear()`` invalidates the subplot spec that the colorbar's
        axes depend on.
        """
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except (AttributeError, ValueError):
                pass
            self._colorbar = None

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

    def _on_scroll(self, event) -> None:
        """Zoom the map view on mouse-wheel scroll."""
        if event.inaxes != self._ax or self._data is None:
            return

        # Determine zoom direction.
        if event.button == "up":
            factor = _ZOOM_IN_FACTOR
        elif event.button == "down":
            factor = _ZOOM_OUT_FACTOR
        else:
            return

        # Current limits.
        xl, xr = self._ax.get_xlim()
        yb, yt = self._ax.get_ylim()

        # Cursor position in data coordinates.
        cx = event.xdata
        cy = event.ydata

        # Compute new limits centred on cursor.
        new_xl = cx - (cx - xl) * factor
        new_xr = cx + (xr - cx) * factor
        new_yb = cy - (cy - yb) * factor
        new_yt = cy + (yt - cy) * factor

        self._ax.set_xlim(new_xl, new_xr)
        self._ax.set_ylim(new_yb, new_yt)
        self._canvas.draw_idle()

    def _draw_marker(self, row: int, col: int) -> None:
        """Draw a crosshair on the selected pixel."""
        self._selected_rc = (row, col)
        if self._marker is not None:
            self._marker.remove()
        (self._marker,) = self._ax.plot(
            col, row, marker="+", color="red",
            markersize=12, markeredgewidth=1.5,
        )
        self._canvas.draw_idle()
