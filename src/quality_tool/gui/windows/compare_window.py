"""Lightweight comparison window for Quality_tool.

Displays a static snapshot of a 2D map for side-by-side comparison.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class CompareWindow(QWidget):
    """A standalone window showing a fixed snapshot of a 2D map."""

    def __init__(
        self,
        data: np.ndarray,
        title: str = "",
        cmap: str = "viridis",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Compare — {title}")
        self.setAttribute(
            __import__("PySide6").QtCore.Qt.WidgetAttribute.WA_DeleteOnClose,
        )
        self.resize(500, 450)

        figure = Figure(tight_layout=True)
        canvas = FigureCanvasQTAgg(figure)
        ax = figure.add_subplot(111)

        is_bool = data.dtype == bool
        if is_bool:
            rgb = np.zeros((*data.shape, 3), dtype=float)
            rgb[data] = [0.2, 0.7, 0.3]
            rgb[~data] = [0.8, 0.2, 0.2]
            ax.imshow(rgb, origin="upper", aspect="equal")
        else:
            im = ax.imshow(data, origin="upper", aspect="equal", cmap=cmap)
            figure.colorbar(im, ax=ax)

        ax.set_title(title)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
