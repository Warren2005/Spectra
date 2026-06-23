"""
PDF page display widget with overlay support for gesture cursor,
highlight anchor indicator, and live selection preview.
"""
from typing import Optional

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class PdfWidget(QLabel):
    """Displays a PDF page pixmap and draws gesture overlays on top."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: #2b2b2b;")

        self._source_pixmap: Optional[QPixmap] = None
        # Overlay rects in normalized page coords (0-1)
        self._cursor_norm: Optional[tuple[float, float]] = None
        self._anchor_norm: Optional[tuple[float, float, float, float]] = None  # x0,y0,x1,y1
        self._selection_norms: list[tuple[float, float, float, float]] = []

    # ── Pixmap ────────────────────────────────────────────────────────────────

    def set_page_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self.update()

    def resizeEvent(self, event) -> None:
        self.update()
        super().resizeEvent(event)

    # ── Overlay setters (call from main thread) ───────────────────────────────

    def set_cursor(self, norm_x: float, norm_y: float) -> None:
        self._cursor_norm = (norm_x, norm_y)
        self.update()

    def clear_cursor(self) -> None:
        self._cursor_norm = None
        self.update()

    def set_anchor(self, nx0: float, ny0: float, nx1: float, ny1: float) -> None:
        self._anchor_norm = (nx0, ny0, nx1, ny1)
        self.update()

    def set_selection(self, rects_norm: list[tuple[float, float, float, float]]) -> None:
        self._selection_norms = rects_norm
        self.update()

    def clear_overlays(self) -> None:
        self._cursor_norm = None
        self._anchor_norm = None
        self._selection_norms = []
        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#2b2b2b"))

        if self._source_pixmap is None:
            painter.end()
            return

        # Scale pixmap to fit, maintaining aspect ratio
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x_off = (self.width() - scaled.width()) // 2
        y_off = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x_off, y_off, scaled)

        # Draw overlays in widget space
        W, H = float(scaled.width()), float(scaled.height())

        def to_widget(nx, ny):
            return x_off + nx * W, y_off + ny * H

        def norm_rect_to_qrect(nx0, ny0, nx1, ny1) -> QRectF:
            wx0, wy0 = to_widget(nx0, ny0)
            wx1, wy1 = to_widget(nx1, ny1)
            return QRectF(wx0, wy0, wx1 - wx0, wy1 - wy0)

        # Selection preview (semi-transparent yellow)
        if self._selection_norms:
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 220, 0, 90))
            for r in self._selection_norms:
                painter.drawRect(norm_rect_to_qrect(*r))
            painter.restore()

        # Anchor indicator (orange border)
        if self._anchor_norm:
            painter.save()
            pen = QPen(QColor(255, 140, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 140, 0, 60))
            painter.drawRect(norm_rect_to_qrect(*self._anchor_norm))
            painter.restore()

        # Cursor crosshair (cyan)
        if self._cursor_norm:
            cx, cy = to_widget(*self._cursor_norm)
            painter.save()
            pen = QPen(QColor(0, 220, 220))
            pen.setWidth(2)
            painter.setPen(pen)
            size = 12
            painter.drawLine(int(cx - size), int(cy), int(cx + size), int(cy))
            painter.drawLine(int(cx), int(cy - size), int(cx), int(cy + size))
            painter.drawEllipse(int(cx - 4), int(cy - 4), 8, 8)
            painter.restore()

        painter.end()

    # ── Coordinate helper ─────────────────────────────────────────────────────

    def page_display_rect(self) -> tuple[int, int, int, int]:
        """Return (x_off, y_off, drawn_width, drawn_height) of the displayed page."""
        if self._source_pixmap is None:
            return 0, 0, self.width(), self.height()
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x_off = (self.width() - scaled.width()) // 2
        y_off = (self.height() - scaled.height()) // 2
        return x_off, y_off, scaled.width(), scaled.height()
