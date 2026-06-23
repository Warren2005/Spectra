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
        window._engine.open(sys.argv[1])
    else:
        window.open_file()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
