"""
PDF annotation persistence and JSON sidecar index.

Highlights are written as standard PDF annotation objects via PyMuPDF.
doc.saveIncr() after every annotation — no deferred save, no data loss.
The .spectra JSON sidecar is a search index only; the PDF is the source of truth.
"""
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fitz

from spectra.document.word_box import WordBox

if TYPE_CHECKING:
    from spectra.document.engine import DocumentEngine


class AnnotationManager:
    def __init__(self, engine: "DocumentEngine"):
        self._engine = engine

    def add_highlight(
        self,
        page_num: int,
        selection: list[WordBox],
        rects: list[fitz.Rect],
    ) -> None:
        """Write highlight to PDF and update sidecar. Saves incrementally."""
        if not rects or not selection or self._engine._doc is None:
            return

        page = self._engine._doc[page_num]
        annot = page.add_highlight_annot(rects)  # single call — one annotation object
        annot.update()
        self._engine.save_incremental()
        self._append_sidecar(page_num, selection)

        # Invalidate word cache so re-render picks up annotations
        self._engine._word_cache.pop(page_num, None)

    def _append_sidecar(self, page_num: int, selection: list[WordBox]) -> None:
        path = self._engine._path
        if path is None:
            return
        sidecar = Path(str(path) + ".spectra")
        entries: list[dict] = []
        if sidecar.exists():
            try:
                entries = json.loads(sidecar.read_text())
            except (json.JSONDecodeError, OSError):
                entries = []
        entries.append({
            "page": page_num,
            "text": " ".join(w.text for w in selection),
            "type": "highlight",
            "timestamp": time.time(),
        })
        sidecar.write_text(json.dumps(entries, indent=2))
