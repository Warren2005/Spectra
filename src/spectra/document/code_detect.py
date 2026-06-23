"""
Font-metadata monospace code block detector.

Scans span-level dict output from PyMuPDF and identifies spans whose font name
contains a known monospace keyword. Adjacent spans are merged into block regions.
Covers ~90% of born-digital technical books.
"""
import fitz

_MONO_KEYWORDS = ("mono", "courier", "code", "consolas", "inconsolata", "fira", "source code")
_MERGE_GAP = 5.0     # maximum vertical gap (PDF points) between adjacent spans to merge
_COLUMN_SLOP = 20.0  # maximum x0 drift to still consider same block column


def detect_code_blocks(page: fitz.Page) -> list[fitz.Rect]:
    """Return bounding boxes of detected monospace code block regions."""
    data = page.get_text("dict")
    mono_spans: list[fitz.Rect] = []

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "").lower()
                if any(kw in font for kw in _MONO_KEYWORDS):
                    mono_spans.append(fitz.Rect(span["bbox"]))

    return _merge_adjacent(mono_spans)


def _merge_adjacent(rects: list[fitz.Rect]) -> list[fitz.Rect]:
    """Merge vertically adjacent monospace spans into single block regions."""
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: r.y0)
    merged = [rects[0]]
    for r in rects[1:]:
        last = merged[-1]
        vertical_close = r.y0 - last.y1 <= _MERGE_GAP
        column_aligned = abs(r.x0 - last.x0) < _COLUMN_SLOP
        if vertical_close and column_aligned:
            merged[-1] = fitz.Rect(
                min(last.x0, r.x0), last.y0,
                max(last.x1, r.x1), r.y1,
            )
        else:
            merged.append(r)
    return merged


def extract_code_text(page: fitz.Page, rect: fitz.Rect) -> str:
    """Extract plain text from a detected code block region."""
    return page.get_text("text", clip=rect).strip()
