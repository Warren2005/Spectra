import sys

from PySide6.QtWidgets import QApplication

from spectra.ui.window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("SPECTRA")
    app.setOrganizationName("spectra")

    window = MainWindow()
    window.show()

    # Open file passed as CLI arg, or show open dialog
    if len(sys.argv) > 1:
        try:
            window._engine.open(sys.argv[1])
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(window, "Cannot open file", str(e))
    else:
        window.open_file()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
