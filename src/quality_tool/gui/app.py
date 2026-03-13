"""Application entry point for the Quality_tool GUI.

Launch with::

    python -m quality_tool.gui.app
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from quality_tool.gui.main_window import MainWindow


def main() -> None:
    """Create the QApplication and show the main window."""
    app = QApplication(sys.argv)
    app.setApplicationName("Quality_tool")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
