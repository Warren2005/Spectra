"""Main application window. Wires gesture pipeline → document engine."""
from pathlib import Path
from typing import Optional

import fitz

import cv2
from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QWidget,
)

from spectra.capture.webcam import WebcamCapture
from spectra.document.annotation import AnnotationManager
from spectra.document.engine import DocumentEngine
from spectra.document.highlight import HighlightManager
from spectra.document.word_box import WordBox, snap_to_word
from spectra.inference.gesture.pipeline import GesturePipeline
from spectra.inference.gesture.state_machine import GestureState
from spectra.schema import IntentEvent
from spectra.ui.hud import HudWidget
from spectra.ui.pdf_widget import PdfWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPECTRA")
        self.resize(1280, 900)

        self._engine = DocumentEngine(self)
        self._highlight = HighlightManager()
        self._annotation = AnnotationManager(self._engine)
        self._cursor_word: Optional[WordBox] = None

        self._webcam = WebcamCapture()
        self._gesture_thread = QThread(self)
        self._gesture = GesturePipeline()
        self._gesture.moveToThread(self._gesture_thread)

        self._build_ui()
        self._connect_signals()

        self._gesture_thread.start()
        if not self._webcam.start():
            self._cam_label.setText("No camera")
            self._cam_label.setStyleSheet(
                "background: #1a1a1a; color: #666; border: 1px solid #444; font-size: 11px;"
            )

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._pdf_widget = PdfWidget()
        layout.addWidget(self._pdf_widget, stretch=1)

        # Webcam overlay — top-right corner, floated over the PDF widget
        self._cam_label = QLabel(self)
        self._cam_label.setFixedSize(240, 135)
        self._cam_label.setStyleSheet("background: black; border: 1px solid #555;")
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # HUD — top-left, floated
        self._hud = HudWidget(self)
        self._hud.setFixedSize(240, 140)

        status = QStatusBar()
        self.setStatusBar(status)
        self._status_lbl = QLabel("No document open")
        status.addWidget(self._status_lbl)
        self._page_lbl = QLabel("")
        status.addPermanentWidget(self._page_lbl)
        self._fps_lbl = QLabel("cam: --fps")
        status.addPermanentWidget(self._fps_lbl)
        self._state_lbl = QLabel("IDLE")
        status.addPermanentWidget(self._state_lbl)

    def _connect_signals(self) -> None:
        # Document engine → UI
        self._engine.page_rendered.connect(self._on_page_rendered)
        self._engine.page_count_changed.connect(self._on_page_count_changed)

        # Webcam → gesture pipeline (queued: inference runs on gesture thread)
        self._webcam.frame_ready.connect(
            self._gesture.process_frame, Qt.ConnectionType.QueuedConnection
        )
        self._webcam.frame_ready.connect(self._on_frame)
        self._webcam.fps_update.connect(self._on_fps_update)

        # Gesture pipeline → window (queued: slot runs on main thread)
        self._gesture.intent_event.connect(
            self._on_intent, Qt.ConnectionType.QueuedConnection
        )
        self._gesture.state_changed.connect(
            self._on_state_changed, Qt.ConnectionType.QueuedConnection
        )

    # ── Intent routing ────────────────────────────────────────────────────────

    @Slot(object)
    def _on_intent(self, event: IntentEvent) -> None:
        intent = event.intent
        if intent == "PAGE_NEXT":
            self._engine.next_page()
            self._pdf_widget.clear_overlays()
            self._highlight.cancel()
        elif intent == "PAGE_PREV":
            self._engine.prev_page()
            self._pdf_widget.clear_overlays()
            self._highlight.cancel()
        elif intent == "CURSOR_MOVE":
            self._handle_cursor_move(event.payload.get("cursor", (0.5, 0.5)))
        elif intent == "HIGHLIGHT_ANCHOR":
            self._handle_highlight_anchor()
        elif intent == "HIGHLIGHT_CONFIRM":
            self._handle_highlight_confirm()
        elif intent == "SCROLL_DOWN":
            self._engine.go_to_page(self._engine.current_page + 1)
        elif intent == "SCROLL_UP":
            self._engine.go_to_page(self._engine.current_page - 1)
        elif intent == "ZOOM_IN":
            pass  # TODO Phase 5
        elif intent == "ZOOM_OUT":
            pass
        elif intent == "UNDO_LAST":
            pass
        elif intent == "CODE_COPY":
            self._handle_code_copy()

    def _handle_cursor_move(self, cursor: tuple) -> None:
        norm_x, norm_y = cursor
        self._pdf_widget.set_cursor(norm_x, norm_y)

        page_num = self._engine.current_page
        if self._engine.is_scanned(page_num):
            return
        words = self._engine.get_words(page_num)
        if not words:
            return

        page_rect = self._engine.page_rect(page_num)
        word = snap_to_word(words, norm_x, norm_y, page_rect.width, page_rect.height)
        self._cursor_word = word

        # Show selection preview if anchor is set
        if word and self._highlight.has_anchor:
            a_idx = self._highlight.anchor_index
            b_idx = word.index
            if a_idx is not None:
                lo, hi = sorted([a_idx, b_idx])
                sel = words[lo : hi + 1]
                rects = HighlightManager.selection_to_rects(sel)
                page_w, page_h = page_rect.width, page_rect.height
                norms = [
                    (r.x0 / page_w, r.y0 / page_h, r.x1 / page_w, r.y1 / page_h)
                    for r in rects
                ]
                self._pdf_widget.set_selection(norms)

    def _handle_highlight_anchor(self) -> None:
        if self._cursor_word is None:
            return
        word = self._cursor_word
        self._highlight.set_anchor(word)
        page_rect = self._engine.page_rect(self._engine.current_page)
        pw, ph = page_rect.width, page_rect.height
        self._pdf_widget.set_anchor(
            word.x0 / pw, word.y0 / ph, word.x1 / pw, word.y1 / ph
        )

    def _handle_highlight_confirm(self) -> None:
        if self._cursor_word is None or not self._highlight.has_anchor:
            self._highlight.cancel()
            self._pdf_widget.clear_overlays()
            return

        page_num = self._engine.current_page
        words = self._engine.get_words(page_num)
        selection = self._highlight.confirm(self._cursor_word, words)
        if not selection:
            return

        rects = HighlightManager.selection_to_rects(selection)
        self._annotation.add_highlight(page_num, selection, rects)
        self._pdf_widget.clear_overlays()
        self._engine.invalidate_render_cache(page_num)
        self._engine.go_to_page(page_num)  # re-render to show annotation

    def _handle_code_copy(self) -> None:
        from spectra.document.code_detect import detect_code_blocks, extract_code_text
        page_num = self._engine.current_page
        if self._engine._doc is None:
            return
        page = self._engine._doc[page_num]
        blocks = detect_code_blocks(page)
        if not blocks or self._cursor_word is None:
            return
        page_rect = self._engine.page_rect(page_num)
        cx = self._cursor_word.cx
        cy = self._cursor_word.cy
        for block in blocks:
            if block.contains(fitz.Point(cx, cy)):
                text = extract_code_text(page, block)
                QApplication.clipboard().setText(text)
                self._status_lbl.setText("Code copied to clipboard")
                return

    # ── Qt slots ──────────────────────────────────────────────────────────────

    @Slot(int, QPixmap)
    def _on_page_rendered(self, page_num: int, pixmap: QPixmap) -> None:
        self._pdf_widget.set_page_pixmap(pixmap)
        self._page_lbl.setText(f"Page {page_num + 1} / {self._engine.page_count}")
        if self._engine.is_scanned(page_num):
            self._status_lbl.setText("⚠ No text layer — highlighting unavailable")
        else:
            path = self._engine._path
            self._status_lbl.setText(path.name if path else "")

    @Slot(int)
    def _on_page_count_changed(self, count: int) -> None:
        self._page_lbl.setText(f"Page 1 / {count}")

    @Slot(object)
    def _on_frame(self, frame) -> None:
        small = cv2.resize(frame, (240, 135))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._cam_label.setPixmap(QPixmap.fromImage(img))

    @Slot(float)
    def _on_fps_update(self, fps: float) -> None:
        self._fps_lbl.setText(f"cam: {fps:.0f}fps")
        self._hud.set_fps(fps)

    @Slot(str)
    def _on_state_changed(self, state_name: str) -> None:
        self._state_lbl.setText(state_name)
        self._hud.set_state(state_name)
        if state_name == "IDLE":
            self._pdf_widget.clear_cursor()

    # ── File open ─────────────────────────────────────────────────────────────

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", str(Path.home()), "PDF Files (*.pdf)"
        )
        if path:
            self._engine.open(path)
            self._pdf_widget.clear_overlays()
            self._highlight.cancel()

    # ── Keyboard (dev fallback until gestures are tuned) ──────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Space):
            self._engine.next_page()
        elif key == Qt.Key.Key_Left:
            self._engine.prev_page()
        elif key == Qt.Key.Key_O and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.open_file()
        elif key == Qt.Key.Key_H:
            self._hud.toggle_cheat_sheet()
        else:
            super().keyPressEvent(event)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        self._cam_label.move(self.width() - 250, 10)
        self._cam_label.raise_()
        self._hud.move(10, 10)
        self._hud.raise_()
        super().resizeEvent(event)

    def closeEvent(self, event) -> None:
        self._webcam.stop()
        self._gesture.close()
        self._gesture_thread.quit()
        self._gesture_thread.wait(2000)
        self._engine.close()
        super().closeEvent(event)
