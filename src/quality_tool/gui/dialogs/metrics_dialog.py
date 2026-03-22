"""Metric selection dialog for Quality_tool.

Presents checkboxes for all registered metrics, grouped by category,
and returns the list of selected metric names.

Groups are arranged in a two-column layout to keep the dialog compact:
left column holds Baseline / Noise / Regularity metrics, right column
holds Envelope metrics.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from quality_tool.metrics.registry import MetricRegistry

# Category labels shown in the dialog header for each group.
_CATEGORY_LABELS: dict[str, str] = {
    "baseline": "Baseline metrics",
    "noise": "Noise metrics",
    "regularity": "Regularity metrics",
    "envelope": "Envelope metrics",
}

# Categories placed in the right column.
_RIGHT_COLUMN_CATEGORIES: set[str] = {"envelope"}


class MetricsDialog(QDialog):
    """Checkbox-based dialog for selecting multiple metrics, grouped
    by category in a two-column layout.

    Parameters
    ----------
    registry : MetricRegistry
        Registry containing all available metrics.
    selected : list[str]
        Metric names that should be pre-checked.
    parent : QWidget | None
        Parent widget.
    """

    def __init__(
        self,
        registry: MetricRegistry,
        selected: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select metrics")
        self.resize(560, 380)

        selected = selected or []
        self._checkboxes: dict[str, QCheckBox] = {}

        outer = QVBoxLayout(self)

        # --- two-column area ---
        columns = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        for category, items in registry.list_grouped():
            target = (
                right_col
                if category in _RIGHT_COLUMN_CATEGORIES
                else left_col
            )

            label_text = _CATEGORY_LABELS.get(category, category.title())
            header = QLabel(f"<b>{label_text}</b>")
            header.setContentsMargins(0, 6, 0, 2)
            target.addWidget(header)

            for name, display_name in items:
                cb = QCheckBox(display_name)
                cb.setChecked(name in selected)
                self._checkboxes[name] = cb
                target.addWidget(cb)

        left_col.addStretch()
        right_col.addStretch()

        columns.addLayout(left_col)
        columns.addSpacing(24)
        columns.addLayout(right_col)

        outer.addLayout(columns)

        # --- button row ---
        btn_row = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        outer.addLayout(btn_row)

    def selected_metrics(self) -> list[str]:
        """Return the list of checked metric names."""
        return [
            name for name, cb in self._checkboxes.items() if cb.isChecked()
        ]
