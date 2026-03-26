"""Combined per-pixel metric window: table + bar chart.

Shows all computed metric values for a single pixel with both a table
(native + normalized scores) and a horizontal bar chart of normalized
scores in one unified window.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtWidgets import (
    QHeaderView,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from quality_tool.gui.style import apply_mpl_dark_style, create_dark_figure


class PixelMetricsWindow(QWidget):
    """Reusable window showing table + bar chart for one pixel.

    Call :meth:`update_data` to refresh contents when the selected
    pixel changes.  The window hides on close (preserving state)
    rather than destroying itself.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pixel Metrics")
        self.resize(560, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        # ── table ──────────────────────────────────────────────────
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Metric", "Category", "Native", "Normalized", "Valid"],
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 5):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        splitter.addWidget(self._table)

        # ── chart ──────────────────────────────────────────────────
        self._fig = create_dark_figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        apply_mpl_dark_style(self._fig)
        splitter.addWidget(self._canvas)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

    # ── public API ─────────────────────────────────────────────────

    def update_data(
        self,
        pixel: tuple[int, int],
        metrics_data: list[dict],
    ) -> None:
        """Replace table and chart contents for *pixel*."""
        row, col = pixel
        self.setWindowTitle(f"Pixel ({row}, {col}) — Metrics")
        self._fill_table(metrics_data)
        self._fill_chart(pixel, metrics_data)

    def closeEvent(self, ev) -> None:  # noqa: N802
        ev.ignore()
        self.hide()

    # ── internals ──────────────────────────────────────────────────

    def _fill_table(self, data: list[dict]) -> None:
        self._table.setRowCount(len(data))
        for i, entry in enumerate(data):
            self._table.setItem(i, 0, QTableWidgetItem(entry["name"]))
            self._table.setItem(i, 1, QTableWidgetItem(entry.get("category", "—")))

            if entry["valid"]:
                native_str = f"{entry['native_score']:.4g}"
                norm_str = f"{entry['normalized_score']:.3f}"
            else:
                native_str = "—"
                norm_str = "—"

            self._table.setItem(i, 2, QTableWidgetItem(native_str))
            self._table.setItem(i, 3, QTableWidgetItem(norm_str))
            self._table.setItem(i, 4, QTableWidgetItem(
                "yes" if entry["valid"] else "no",
            ))

    def _fill_chart(self, pixel: tuple[int, int], data: list[dict]) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        apply_mpl_dark_style(self._fig)

        sorted_data = sorted(
            data,
            key=lambda d: d["normalized_score"] if d["valid"] else -1.0,
        )

        names = [d["name"] for d in sorted_data]
        scores = [
            d["normalized_score"] if d["valid"] else 0.0
            for d in sorted_data
        ]
        valid_flags = [d["valid"] for d in sorted_data]

        y_pos = np.arange(len(names))
        colors = ["#4a86c8" if v else "#555555" for v in valid_flags]

        ax.barh(y_pos, scores, color=colors, edgecolor="none", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0.0, 1.05)
        ax.set_xlabel("normalized score")
        ax.set_title(f"Pixel ({pixel[0]}, {pixel[1]})")

        for i, v in enumerate(valid_flags):
            if not v:
                ax.text(
                    0.02, i, "invalid", va="center", fontsize=7,
                    color="gray", style="italic",
                )

        ax.invert_yaxis()
        self._canvas.draw_idle()
