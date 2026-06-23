"""Word-level text extraction and two-stage snap-to-word utilities."""
from dataclasses import dataclass
from typing import Optional

import fitz


@dataclass
class WordBox:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_no: int
    line_no: int
    word_no: int
    index: int  # position in reading-order flat list

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def rect(self) -> fitz.Rect:
        return fitz.Rect(self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_tuple(cls, t: tuple, index: int) -> "WordBox":
        return cls(
            x0=float(t[0]), y0=float(t[1]), x1=float(t[2]), y1=float(t[3]),
            text=str(t[4]),
            block_no=int(t[5]), line_no=int(t[6]), word_no=int(t[7]),
            index=index,
        )


def extract_words(page: fitz.Page) -> list[WordBox]:
    """
    Extract all words from a PyMuPDF page in reading order.
    Sort by (block_no, line_no, word_no) — block_no is not guaranteed
    contiguous, so sort is mandatory.
    """
    raw = page.get_text("words")
    raw.sort(key=lambda w: (w[5], w[6], w[7]))
    return [WordBox.from_tuple(t, i) for i, t in enumerate(raw)]


def snap_to_word(
    words: list[WordBox],
    norm_x: float,
    norm_y: float,
    page_width: float,
    page_height: float,
) -> Optional[WordBox]:
    """
    Two-stage snap-to-word for a cursor at normalized page coordinates.

    Stage 1: find the line whose vertical centre is nearest to cursor y.
             This prevents grabbing a word from an adjacent line when
             the cursor is between lines.
    Stage 2: within that line, find the word with the nearest x-centre.
    """
    if not words:
        return None

    px = norm_x * page_width
    py = norm_y * page_height

    # Group by line
    by_line: dict[tuple, list[WordBox]] = {}
    for w in words:
        by_line.setdefault((w.block_no, w.line_no), []).append(w)

    # Stage 1
    def _line_cy(ws: list[WordBox]) -> float:
        return (ws[0].y0 + ws[0].y1) / 2

    nearest_key = min(by_line, key=lambda k: abs(_line_cy(by_line[k]) - py))
    line_words = by_line[nearest_key]

    # Stage 2
    return min(line_words, key=lambda w: abs(w.cx - px))
