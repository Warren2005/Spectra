"""
HUD overlay widget: shows gesture state, active modality, and fps.
Rendered as a semi-transparent panel in the top-left corner of the window.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

_STATE_COLORS = {
    "IDLE":       "#888888",
    "COMMAND":    "#44dd88",
    "CURSOR":     "#44aaff",
    "ANCHOR_SET": "#ffaa44",
}

_CHEAT_SHEET = [
    ("Left hand open", "Enter command mode"),
    ("Right swipe",    "Next page"),
    ("Left swipe",     "Previous page"),
    ("Pinch × 2",      "Highlight word range"),
    ("Two-finger",     "Move cursor"),
]


class HudWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self._state = "IDLE"
        self._fps = 0.0
        self._show_cheat = False

    def set_state(self, state_name: str) -> None:
        self._state = state_name
        self.update()

    def set_fps(self, fps: float) -> None:
        self._fps = fps
        self.update()

    def toggle_cheat_sheet(self) -> None:
        self._show_cheat = not self._show_cheat
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background panel
        bg = QColor(20, 20, 20, 180)
        p.setBrush(bg)
        p.setPen(Qt.PenStyle.NoPen)
        panel_h = 130 if self._show_cheat else 56
        p.drawRoundedRect(8, 8, 220, panel_h, 8, 8)

        # State dot
        dot_color = QColor(_STATE_COLORS.get(self._state, "#888888"))
        p.setBrush(dot_color)
        p.drawEllipse(18, 18, 12, 12)

        # State label
        font = QFont("Menlo", 11, QFont.Weight.Bold)
        p.setFont(font)
        p.setPen(dot_color)
        p.drawText(36, 30, self._state)

        # FPS
        font2 = QFont("Menlo", 9)
        p.setFont(font2)
        p.setPen(QColor(150, 150, 150))
        p.drawText(18, 50, f"cam {self._fps:.0f} fps")

        if self._show_cheat:
            y = 70
            for gesture, action in _CHEAT_SHEET:
                p.setPen(QColor(200, 200, 200))
                p.drawText(18, y, gesture)
                p.setPen(QColor(120, 120, 120))
                p.drawText(120, y, action)
                y += 16

        p.end()
