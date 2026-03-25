"""Application entry point for the Quality_tool GUI.

Launch with::

    python -m quality_tool.gui.app
"""

from __future__ import annotations

import sys

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

from quality_tool.gui.main_window import MainWindow
from quality_tool.gui.style import (
    ACCENT,
    BG_BASE,
    BG_PANEL,
    BG_SELECTED,
    BG_WIDGET,
    DARK_STYLESHEET,
    TEXT_DISABLED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def _build_dark_palette() -> QPalette:
    """Build a QPalette matching the dark stylesheet colors."""
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(BG_BASE))
    p.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Base, QColor(BG_WIDGET))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_PANEL))
    p.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Button, QColor(BG_WIDGET))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_WIDGET))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_SECONDARY))
    # Disabled state.
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor(TEXT_DISABLED),
    )
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor(TEXT_DISABLED),
    )
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(TEXT_DISABLED),
    )
    return p


def main() -> None:
    """Create the QApplication and show the main window."""
    app = QApplication(sys.argv)
    app.setApplicationName("Quality_tool")

    # Apply dark theme.
    app.setPalette(_build_dark_palette())
    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
