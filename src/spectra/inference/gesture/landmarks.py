"""MediaPipe Hands wrapper with One-Euro filter for landmark smoothing."""
import time
from enum import Enum
from typing import Optional

import mediapipe as mp
import numpy as np

# Landmark index constants (MediaPipe hand topology)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
N_LANDMARKS = 21

# (21, 3) numpy array of normalized [x, y, z] coords
LandmarkArray = np.ndarray


class Hand(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"


HandResult = dict[Hand, LandmarkArray]


class OneEuroFilter:
    """
    Adaptive low-pass filter that reduces smoothing during fast movement
    and increases it when the signal is near-stationary. Superior to a
    simple moving average for gesture interfaces.
    """

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.1, dcutoff: float = 1.0):
        self._freq = freq
        self._mincutoff = mincutoff
        self._beta = beta
        self._dcutoff = dcutoff
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None

    @staticmethod
    def _alpha(cutoff: np.ndarray, freq: float) -> np.ndarray:
        te = 1.0 / freq
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            return self._x.copy()

        dx = (x - self._x) * self._freq
        alpha_d = self._alpha(np.full_like(x, self._dcutoff), self._freq)
        assert self._dx is not None
        self._dx = alpha_d * dx + (1.0 - alpha_d) * self._dx

        cutoff = self._mincutoff + self._beta * np.abs(self._dx)
        alpha = self._alpha(cutoff, self._freq)
        self._x = alpha * x + (1.0 - alpha) * self._x
        return self._x.copy()

    def reset(self) -> None:
        self._x = None
        self._dx = None


class LandmarkDetector:
    """
    Wraps MediaPipe Hands. Applies One-Euro filter per hand.
    Runs on whatever thread process() is called from.
    """

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._filters: dict[Hand, OneEuroFilter] = {
            Hand.LEFT: OneEuroFilter(mincutoff=mincutoff, beta=beta),
            Hand.RIGHT: OneEuroFilter(mincutoff=mincutoff, beta=beta),
        }
        self._last_seen: dict[Hand, float] = {}

    def process(self, frame_bgr: np.ndarray) -> HandResult:
        """Return filtered landmarks keyed by Hand enum. Empty dict = no hands."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        detected: HandResult = {}
        now = time.time()

        if results.multi_hand_landmarks:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                if score < 0.7:
                    continue
                hand = Hand(label)
                raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                filtered = self._filters[hand](raw.flatten()).reshape(N_LANDMARKS, 3)
                detected[hand] = filtered
                self._last_seen[hand] = now

        # Reset filter for hands absent > 0.5 s to avoid stale state
        for hand in list(self._last_seen):
            if now - self._last_seen[hand] > 0.5:
                self._filters[hand].reset()
                del self._last_seen[hand]

        return detected

    def close(self) -> None:
        self._hands.close()
