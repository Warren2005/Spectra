from pathlib import Path

import cv2
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QStatusBar,
    QWidget,
)

from spectra.capture.webcam import WebcamCapture
from spectra.document.engine import DocumentEngine


class PdfWidget(QLabel):
    """Displays the current PDF page, scaled to fit the widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: #2b2b2b;")
        self._pixmap: QPixmap | None = None

    def set_page_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._update_display()

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)

    def _update_display(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPECTRA")
        self.resize(1200, 900)

        self._engine = DocumentEngine(self)
        self._webcam = WebcamCapture()

        self._build_ui()
        self._connect_signals()
        self._webcam.start()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._pdf_widget = PdfWidget()
        layout.addWidget(self._pdf_widget, stretch=1)

        # Webcam overlay (top-right corner, fixed size)
        self._cam_label = QLabel()
        self._cam_label.setFixedSize(240, 135)
        self._cam_label.setStyleSheet("background: black; border: 1px solid #555;")
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.move(self.width() - 250, 10)

        status = QStatusBar()
        self.setStatusBar(status)
        self._status_label = QLabel("No document open")
        status.addWidget(self._status_label)

        self._page_label = QLabel("")
        status.addPermanentWidget(self._page_label)

        self._fps_label = QLabel("cam: --fps")
        status.addPermanentWidget(self._fps_label)

    def _connect_signals(self) -> None:
        self._engine.page_rendered.connect(self._on_page_rendered)
        self._engine.page_count_changed.connect(self._on_page_count_changed)
        self._webcam.frame_ready.connect(self._on_frame)
        self._webcam.fps_update.connect(self._on_fps_update)

    # ── Menu actions ──────────────────────────────────────────────────────────

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", str(Path.home()), "PDF Files (*.pdf)"
        )
        if path:
            self._engine.open(path)
            self._status_label.setText(Path(path).name)

    # ── Slots ─────────────────────────────────────────────────────────────────

    @Slot(int, QPixmap)
    def _on_page_rendered(self, page_num: int, pixmap: QPixmap) -> None:
        self._pdf_widget.set_page_pixmap(pixmap)
        self._page_label.setText(
            f"Page {page_num + 1} / {self._engine.page_count}"
        )

    @Slot(int)
    def _on_page_count_changed(self, count: int) -> None:
        self._page_label.setText(f"Page 1 / {count}")

    @Slot(object)
    def _on_frame(self, frame) -> None:
        small = cv2.resize(frame, (240, 135))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._cam_label.setPixmap(QPixmap.fromImage(img))

    @Slot(float)
    def _on_fps_update(self, fps: float) -> None:
        self._fps_label.setText(f"cam: {fps:.0f}fps")

    # ── Keyboard shortcuts (temporary, until gestures work) ───────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Right or key == Qt.Key.Key_Space:
            self._engine.next_page()
        elif key == Qt.Key.Key_Left:
            self._engine.prev_page()
        elif key == Qt.Key.Key_O and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.open_file()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._webcam.stop()
        self._engine.close()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        # Keep webcam overlay pinned to top-right
        self._cam_label.setParent(self)
        self._cam_label.move(self.width() - 250, 10)
        self._cam_label.raise_()
        super().resizeEvent(event)
