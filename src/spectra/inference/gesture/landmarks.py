"""MediaPipe Hands wrapper (Tasks API, 0.10+) with One-Euro filter."""
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# Default model path
_MODEL_PATH = Path(__file__).parents[4] / "models" / "hand_landmarker.task"

# Landmark index constants
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
N_LANDMARKS = 21

LandmarkArray = np.ndarray  # shape (21, 3)


class Hand(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"


HandResult = dict[Hand, LandmarkArray]


class OneEuroFilter:
    """
    Adaptive low-pass filter: reduces smoothing during fast movement,
    increases it at rest. Superior to a moving average for gesture input.
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
        assert self._dx is not None
        dx = (x - self._x) * self._freq
        alpha_d = self._alpha(np.full_like(x, self._dcutoff), self._freq)
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
    Wraps MediaPipe HandLandmarker (Tasks API). Applies One-Euro filter
    per hand. Runs on whatever thread process() is called from.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        mincutoff: float = 1.0,
        beta: float = 0.1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        path = str(model_path or _MODEL_PATH)
        base_opts = mp_python.BaseOptions(model_asset_path=path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._filters: dict[Hand, OneEuroFilter] = {
            Hand.LEFT: OneEuroFilter(mincutoff=mincutoff, beta=beta),
            Hand.RIGHT: OneEuroFilter(mincutoff=mincutoff, beta=beta),
        }
        self._last_seen: dict[Hand, float] = {}
        self._start_time = time.time()

    def process(self, frame_bgr: np.ndarray) -> HandResult:
        """Return filtered landmarks keyed by Hand enum."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detected: HandResult = {}
        now = time.time()

        if result.hand_landmarks:
            for hand_landmarks, handedness_list in zip(
                result.hand_landmarks, result.handedness
            ):
                label = handedness_list[0].category_name  # "Left" or "Right"
                score = handedness_list[0].score
                if score < 0.7:
                    continue
                hand = Hand(label)
                raw = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                    dtype=np.float32,
                )
                filtered = self._filters[hand](raw.flatten()).reshape(N_LANDMARKS, 3)
                detected[hand] = filtered
                self._last_seen[hand] = now

        # Reset filter for hands absent > 0.5s
        for hand in list(self._last_seen):
            if now - self._last_seen[hand] > 0.5:
                self._filters[hand].reset()
                del self._last_seen[hand]

        return detected

    def close(self) -> None:
        self._landmarker.close()
