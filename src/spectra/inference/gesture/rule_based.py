"""
Rule-based gesture recognizers using landmark velocities and distances.
Used in Phase 1a. Replaced by the 1D-CNN classifier (Phase 1b) for swipe/pinch,
but clutch detection remains rule-based permanently (no training data needed).
"""
from collections import deque
from typing import Optional

import numpy as np

from spectra.inference.gesture.landmarks import (
    LandmarkArray,
    WRIST,
    THUMB_MCP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_TIP,
    MIDDLE_PIP, MIDDLE_TIP,
    RING_PIP, RING_TIP,
    PINKY_PIP, PINKY_TIP,
)

# PIP landmark for each non-thumb finger paired with its tip
_FINGER_PAIRS = [
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
]


def _finger_extended(lm: LandmarkArray, tip: int, pip: int) -> bool:
    """True when finger tip is above PIP joint (smaller y in normalized image space)."""
    return bool(lm[tip, 1] < lm[pip, 1])


def _thumb_extended(lm: LandmarkArray) -> bool:
    """True when thumb tip is farther from the index MCP than thumb MCP is."""
    d_tip = float(np.linalg.norm(lm[THUMB_TIP, :2] - lm[INDEX_MCP, :2]))
    d_mcp = float(np.linalg.norm(lm[THUMB_MCP, :2] - lm[INDEX_MCP, :2]))
    return d_tip > d_mcp


def all_fingers_extended(lm: LandmarkArray) -> bool:
    """All five fingers extended — used for clutch-on detection."""
    return all(_finger_extended(lm, t, p) for t, p in _FINGER_PAIRS) and _thumb_extended(lm)


def is_two_finger_point(lm: LandmarkArray) -> bool:
    """Index + middle extended, ring + pinky curled."""
    return (
        _finger_extended(lm, INDEX_TIP, INDEX_PIP)
        and _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
        and not _finger_extended(lm, RING_TIP, RING_PIP)
        and not _finger_extended(lm, PINKY_TIP, PINKY_PIP)
    )


def normalized_pinch_distance(lm: LandmarkArray) -> float:
    """
    Thumb-tip to index-tip distance normalized by hand size
    (wrist to middle-MCP distance). Scale-invariant across hand distances.
    """
    pinch = float(np.linalg.norm(lm[THUMB_TIP, :2] - lm[INDEX_TIP, :2]))
    hand_size = float(np.linalg.norm(lm[WRIST, :2] - lm[9, :2]))  # 9 = MIDDLE_MCP
    if hand_size < 1e-6:
        return 1.0
    return pinch / hand_size


class SwipeDetector:
    """
    Detects left/right swipe from the x-velocity of the index finger tip
    over a sliding window of recent frames.
    """
    WINDOW = 8
    THRESHOLD = 0.12   # minimum net x-displacement (in normalized coords)

    def __init__(self):
        self._history: deque[float] = deque(maxlen=self.WINDOW)

    def update(self, lm: LandmarkArray) -> Optional[str]:
        """Returns 'right', 'left', or None."""
        self._history.append(float(lm[INDEX_TIP, 0]))
        if len(self._history) < self.WINDOW:
            return None
        delta_x = list(self._history)[-1] - list(self._history)[0]
        if abs(delta_x) >= self.THRESHOLD:
            return "right" if delta_x > 0 else "left"
        return None

    def reset(self) -> None:
        self._history.clear()


class PinchDetector:
    """
    Detects pinch with hysteresis: fires once on the leading edge of a pinch,
    then requires the hand to open before firing again.
    """
    CLOSE_THRESHOLD = 0.18
    OPEN_THRESHOLD = 0.28

    def __init__(self):
        self._pinching = False

    def update(self, lm: LandmarkArray) -> bool:
        """Returns True only on the frame the pinch first closes."""
        dist = normalized_pinch_distance(lm)
        if not self._pinching and dist < self.CLOSE_THRESHOLD:
            self._pinching = True
            return True
        if self._pinching and dist > self.OPEN_THRESHOLD:
            self._pinching = False
        return False

    @property
    def is_pinching(self) -> bool:
        return self._pinching
