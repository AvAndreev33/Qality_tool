"""3D surface view of a 2D map for Quality_tool.

Displays a static snapshot of a metric map as a 3D surface plot.
Follows the same snapshot-window pattern as CompareWindow and
HistogramWindow — the window captures data at the moment it is
opened and does not update when the main window changes.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from quality_tool.gui.style import apply_mpl_dark_style, create_dark_figure


class Map3DWindow(QWidget):
    """A standalone window showing a 3D surface of a 2D map."""

    def __init__(
        self,
        data: np.ndarray,
        title: str = "",
        cmap: str = "viridis",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"3D — {title}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(600, 500)

        figure = create_dark_figure(tight_layout=True)
        canvas = FigureCanvasQTAgg(figure)
        ax = figure.add_subplot(111, projection="3d")
        apply_mpl_dark_style(figure)

        h, w = data.shape
        x = np.arange(w)
        y = np.arange(h)
        x_grid, y_grid = np.meshgrid(x, y)

        # Replace NaN with np.nan so plot_surface leaves gaps.
        z = data.astype(float)

        ax.plot_surface(
            x_grid, y_grid, z,
            cmap=cmap,
            edgecolor="none",
            rstride=1,
            cstride=1,
        )

        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_zlabel("value")
        ax.set_title(title)

        # Style 3D pane backgrounds for dark theme.
        pane_color = (0.18, 0.18, 0.19, 1.0)
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(canvas)
