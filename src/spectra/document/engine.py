import threading
from pathlib import Path

import fitz
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage, QPixmap

from spectra.document.word_box import WordBox, extract_words

_RENDER_CACHE_SIZE = 6
_WORD_CACHE_SIZE = 20


class DocumentEngine(QObject):
    page_rendered = Signal(int, QPixmap)
    page_count_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._doc: fitz.Document | None = None
        self._path: Path | None = None
        self._current_page = 0
        self._lock = threading.Lock()
        self._render_cache: dict[int, QPixmap] = {}
        self._word_cache: dict[int, list[WordBox]] = {}

    # ── File ─────────────────────────────────────────────────────────────────

    def open(self, path: str | Path) -> None:
        path = Path(path)
        with self._lock:
            if self._doc is not None:
                self._doc.save(str(self._path), garbage=4, deflate=True)
                self._doc.close()
            self._doc = fitz.open(str(path))
            self._path = path
            self._render_cache.clear()
            self._word_cache.clear()
            self._current_page = 0
        assert self._doc is not None
        self.page_count_changed.emit(len(self._doc))
        self._render_and_emit(0)
        self._schedule_prefetch(0)

    def close(self) -> None:
        with self._lock:
            if self._doc is not None:
                try:
                    if self._doc.is_dirty:
                        # Compact-save only when the doc has been modified
                        self._doc.saveIncr()
                except Exception:
                    pass
                self._doc.close()
                self._doc = None

    # ── Navigation ────────────────────────────────────────────────────────────

    @property
    def current_page(self) -> int:
        return self._current_page

    @property
    def page_count(self) -> int:
        return len(self._doc) if self._doc else 0

    def go_to_page(self, page_num: int) -> None:
        if self._doc is None:
            return
        page_num = max(0, min(page_num, len(self._doc) - 1))
        self._current_page = page_num
        self._render_and_emit(page_num)
        self._schedule_prefetch(page_num)

    def next_page(self) -> None:
        self.go_to_page(self._current_page + 1)

    def prev_page(self) -> None:
        self.go_to_page(self._current_page - 1)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_page(self, page_num: int) -> QPixmap:
        if page_num in self._render_cache:
            return self._render_cache[page_num]
        with self._lock:
            assert self._doc is not None
            page = self._doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        if len(self._render_cache) >= _RENDER_CACHE_SIZE:
            del self._render_cache[next(iter(self._render_cache))]
        self._render_cache[page_num] = pixmap
        return pixmap

    def _render_and_emit(self, page_num: int) -> None:
        self.page_rendered.emit(page_num, self._render_page(page_num))

    def _schedule_prefetch(self, current: int) -> None:
        candidates = [current + 1, current - 1, current + 2, current - 2]
        pages = [p for p in candidates if 0 <= p < self.page_count and p not in self._render_cache]
        if not pages:
            return
        def _prefetch():
            for p in pages:
                self._render_page(p)
        threading.Thread(target=_prefetch, daemon=True).start()

    def invalidate_render_cache(self, page_num: int) -> None:
        self._render_cache.pop(page_num, None)

    # ── Words ─────────────────────────────────────────────────────────────────

    def get_words(self, page_num: int) -> list[WordBox]:
        """Return reading-order WordBox list (cached, LRU eviction)."""
        if page_num in self._word_cache:
            return self._word_cache[page_num]
        with self._lock:
            assert self._doc is not None
            page = self._doc[page_num]
            words = extract_words(page)
        if len(self._word_cache) >= _WORD_CACHE_SIZE:
            del self._word_cache[next(iter(self._word_cache))]
        self._word_cache[page_num] = words
        return words

    def page_rect(self, page_num: int) -> fitz.Rect:
        with self._lock:
            assert self._doc is not None
            return self._doc[page_num].rect

    def is_scanned(self, page_num: int) -> bool:
        if self._doc is None:
            return False
        words = self.get_words(page_num)
        if words:
            return False
        with self._lock:
            assert self._doc is not None
            images = self._doc[page_num].get_images()
        return len(images) > 0

    # ── Annotations ───────────────────────────────────────────────────────────

    def save_incremental(self) -> None:
        if self._doc is not None and self._path is not None:
            with self._lock:
                self._doc.saveIncr()
