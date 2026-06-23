"""
Two-tap highlight model with reading-order-aware multi-line selection.

First pinch gesture sets anchor A. Second pinch sets anchor B.
Selection is every word in reading-order range [A, B]. One Rect per line span.
"""
from typing import Optional

import fitz

from spectra.document.word_box import WordBox


class HighlightManager:
    def __init__(self):
        self._anchor_idx: Optional[int] = None

    @property
    def has_anchor(self) -> bool:
        return self._anchor_idx is not None

    @property
    def anchor_index(self) -> Optional[int]:
        return self._anchor_idx

    def set_anchor(self, word: WordBox) -> None:
        self._anchor_idx = word.index

    def confirm(self, word_b: WordBox, words: list[WordBox]) -> list[WordBox]:
        """Return selected words [A..B] and reset anchor."""
        if self._anchor_idx is None:
            return []
        a, b = sorted([self._anchor_idx, word_b.index])
        selection = words[a : b + 1]
        self._anchor_idx = None
        return selection

    def cancel(self) -> None:
        self._anchor_idx = None

    @staticmethod
    def selection_to_rects(selection: list[WordBox]) -> list[fitz.Rect]:
        """
        One Rect per line in the selection.
        Start/end lines are clipped to the anchor words;
        middle lines span the full word range naturally.
        """
        by_line: dict[tuple, list[WordBox]] = {}
        for w in selection:
            by_line.setdefault((w.block_no, w.line_no), []).append(w)

        rects = []
        for line_words in by_line.values():
            rects.append(fitz.Rect(
                line_words[0].x0,
                line_words[0].y0,
                line_words[-1].x1,
                line_words[-1].y1,
            ))
        return rects
