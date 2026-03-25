"""Per-pixel metric table window for Quality_tool.

Displays a snapshot of all computed metric values for a single pixel
in tabular form with native and normalized scores.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class PixelMetricsTableWindow(QWidget):
    """Standalone snapshot window showing metric values for one pixel.

    Parameters
    ----------
    pixel : tuple[int, int]
        ``(row, col)`` of the inspected pixel.
    metrics_data : list[dict]
        One dict per metric with keys: ``name``, ``category``,
        ``native_score``, ``normalized_score``, ``valid``.
    """

    def __init__(
        self,
        pixel: tuple[int, int],
        metrics_data: list[dict],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.WidgetAttribute.WA_DeleteOnClose,
        )
        self.setWindowTitle(f"Pixel ({pixel[0]}, {pixel[1]}) — Metric Values")
        self.resize(520, 300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        columns = ["Metric", "Category", "Native", "Normalized", "Valid"]
        table = QTableWidget(len(metrics_data), len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)

        for row_idx, entry in enumerate(metrics_data):
            table.setItem(row_idx, 0, QTableWidgetItem(entry["name"]))
            table.setItem(row_idx, 1, QTableWidgetItem(entry.get("category", "—")))

            if entry["valid"]:
                native_str = f"{entry['native_score']:.4g}"
                norm_str = f"{entry['normalized_score']:.3f}"
            else:
                native_str = "—"
                norm_str = "—"

            table.setItem(row_idx, 2, QTableWidgetItem(native_str))
            table.setItem(row_idx, 3, QTableWidgetItem(norm_str))
            table.setItem(row_idx, 4, QTableWidgetItem("yes" if entry["valid"] else "no"))

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(columns)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(table)
