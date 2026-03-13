"""Read-only information dialog for Quality_tool.

Shows dataset properties and current run state.
"""

from __future__ import annotations

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
        self.resize(420, 340)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)

        for key, value in info.items():
            label = QLabel(f"<b>{key}:</b>  {value}")
            label.setWordWrap(True)
            content_layout.addWidget(label)

        content_layout.addStretch()
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
