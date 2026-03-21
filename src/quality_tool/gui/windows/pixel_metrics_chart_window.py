"""Per-pixel normalized bar chart window for Quality_tool.

Displays a horizontal bar chart of normalized ``[0, 1]`` scores for
all computed metrics at a single pixel, enabling fast visual comparison.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PixelMetricsChartWindow(QWidget):
    """Standalone snapshot window with a horizontal normalized bar chart.

    Parameters
    ----------
    pixel : tuple[int, int]
        ``(row, col)`` of the inspected pixel.
    metrics_data : list[dict]
        One dict per metric with keys: ``name``, ``normalized_score``,
        ``valid``.
    """

    def __init__(
        self,
        pixel: tuple[int, int],
        metrics_data: list[dict],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(
            f"Pixel ({pixel[0]}, {pixel[1]}) — Normalized Scores",
        )
        self.resize(480, max(200, 30 * len(metrics_data) + 80))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        fig = Figure(tight_layout=True)
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)

        # Sort by normalized score (invalid last).
        sorted_data = sorted(
            metrics_data,
            key=lambda d: d["normalized_score"] if d["valid"] else -1.0,
        )

        names = [d["name"] for d in sorted_data]
        scores = [
            d["normalized_score"] if d["valid"] else 0.0
            for d in sorted_data
        ]
        valid_flags = [d["valid"] for d in sorted_data]

        y_pos = np.arange(len(names))
        colors = [
            "tab:blue" if v else "lightgray" for v in valid_flags
        ]

        ax.barh(y_pos, scores, color=colors, edgecolor="none", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0.0, 1.05)
        ax.set_xlabel("normalized score")
        ax.set_title(f"Pixel ({pixel[0]}, {pixel[1]})")

        # Mark invalid metrics.
        for i, v in enumerate(valid_flags):
            if not v:
                ax.text(
                    0.02, i, "invalid", va="center", fontsize=7,
                    color="gray", style="italic",
                )

        ax.invert_yaxis()
        canvas.draw_idle()
