"""Phase 0/1c document engine tests — no display required for word extraction logic."""
from pathlib import Path

import fitz


def make_test_pdf(path: Path, text: str = "Hello world this is a test PDF document.") -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def make_scanned_pdf(path: Path) -> None:
    """PDF with an image but no text layer."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Insert a 1x1 white pixel as a placeholder image — simulates scanned page structure
    # Minimal valid PNG (1×1 white pixel)
    png_bytes = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02'
        b'\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff'
        b'\x3f\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    page.insert_image(fitz.Rect(0, 0, 595, 842), stream=png_bytes)
    doc.save(str(path))
    doc.close()


class TestWordExtraction:
    def test_words_sorted_reading_order(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        make_test_pdf(pdf_path, "Alpha Beta Gamma Delta")
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        words = page.get_text("words")
        words.sort(key=lambda w: (w[5], w[6], w[7]))
        texts = [w[4] for w in words]
        assert texts == ["Alpha", "Beta", "Gamma", "Delta"]
        doc.close()

    def test_word_tuple_structure(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        make_test_pdf(pdf_path)
        doc = fitz.open(str(pdf_path))
        words = doc[0].get_text("words")
        doc.close()
        assert len(words) > 0
        w = words[0]
        x0, y0, x1, y1, text, block_no, *_ = w
        assert isinstance(text, str) and len(text) > 0
        assert x0 < x1 and y0 < y1
        assert isinstance(block_no, int)

    def test_scanned_page_has_no_words(self, tmp_path):
        pdf_path = tmp_path / "scanned.pdf"
        make_scanned_pdf(pdf_path)
        doc = fitz.open(str(pdf_path))
        words = doc[0].get_text("words")
        doc.close()
        assert len(words) == 0

    def test_incremental_save(self, tmp_path):
        pdf_path = tmp_path / "annot.pdf"
        make_test_pdf(pdf_path)
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        words = page.get_text("words")
        words.sort(key=lambda w: (w[5], w[6], w[7]))
        # Highlight the first word
        word = words[0]
        rect = fitz.Rect(word[0], word[1], word[2], word[3])
        page.add_highlight_annot(rect)
        doc.saveIncr()
        doc.close()
        # Reopen and verify annotation is present
        doc2 = fitz.open(str(pdf_path))
        annots = list(doc2[0].annots())
        doc2.close()
        assert len(annots) >= 1


class TestSnapToWord:
    """Unit tests for snap-to-word logic without Qt dependencies."""

    def _make_word_list(self):
        return [
            # x0, y0, x1, y1, text, block, line, word
            (50, 100, 100, 120, "Hello", 0, 0, 0),
            (110, 100, 180, 120, "world", 0, 0, 1),
            (190, 100, 260, 120, "this", 0, 0, 2),
            (50, 130, 90, 150, "is", 0, 1, 0),
            (100, 130, 140, 150, "a", 0, 1, 1),
            (150, 130, 200, 150, "test", 0, 1, 2),
        ]

    def _snap(self, words, cx, cy):
        """Two-stage snap-to-word: find line first, then nearest word in line."""
        # Stage 1: find nearest line by y-centre
        by_line = {}
        for w in words:
            key = (w[5], w[6])  # block, line
            by_line.setdefault(key, []).append(w)
        # Find line whose vertical band is nearest to cy
        best_line_key = min(
            by_line,
            key=lambda k: abs(((by_line[k][0][1] + by_line[k][0][3]) / 2) - cy),
        )
        line_words = by_line[best_line_key]
        # Stage 2: nearest word in line by x-centre
        best = min(line_words, key=lambda w: abs(((w[0] + w[2]) / 2) - cx))
        return words.index(best)

    def test_snap_to_first_word(self):
        words = self._make_word_list()
        idx = self._snap(words, cx=75, cy=110)
        assert words[idx][4] == "Hello"

    def test_snap_to_second_line(self):
        words = self._make_word_list()
        idx = self._snap(words, cx=155, cy=140)
        assert words[idx][4] == "test"

    def test_snap_line_priority_over_horizontal_proximity(self):
        words = self._make_word_list()
        # Cursor between line 0 and line 1 vertically but closer to line 1's y
        idx = self._snap(words, cx=60, cy=128)
        assert words[idx][4] == "is"

    def test_multi_line_selection_spans_reading_order(self):
        words = self._make_word_list()
        anchor_a = 1   # "world"
        anchor_b = 4   # "a"
        selected = words[anchor_a:anchor_b + 1]
        texts = [w[4] for w in selected]
        assert texts == ["world", "this", "is", "a"]
