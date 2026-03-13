"""Metric selection dialog for Quality_tool.

Presents checkboxes for all registered metrics and returns the list
of selected metric names.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from quality_tool.metrics.registry import MetricRegistry


class MetricsDialog(QDialog):
    """Checkbox-based dialog for selecting multiple metrics.

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
        self.resize(300, 200)

        selected = selected or []
        self._checkboxes: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)

        for name in registry.list_metrics():
            cb = QCheckBox(name)
            cb.setChecked(name in selected)
            self._checkboxes[name] = cb
            layout.addWidget(cb)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def selected_metrics(self) -> list[str]:
        """Return the list of checked metric names."""
        return [
            name for name, cb in self._checkboxes.items() if cb.isChecked()
        ]
