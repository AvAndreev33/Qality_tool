"""Read-only information dialog for Quality_tool.

Shows dataset properties and current run state.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class InfoDialog(QDialog):
    """Modal dialog that displays dataset and run information."""

    def __init__(
        self,
        info: dict[str, str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset / Run Info")
        self.resize(460, 380)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setSpacing(4)

        for key, value in info.items():
            label = QLabel(
                f"<span style='color: #808080; font-size: 11px;'>{key}:</span>"
                f"  <span style='color: #d4d4d4;'>{value}</span>"
            )
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            content_layout.addWidget(label)

        content_layout.addStretch()
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(scroll)
