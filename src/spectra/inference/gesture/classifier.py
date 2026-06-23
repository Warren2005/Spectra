"""
1D-CNN temporal gesture classifier (Phase 1b).

Architecture: two Conv1D layers → GlobalMaxPool → Dense → Softmax.
Input is a 16-frame sliding window of flattened landmark coordinates.
Falls back gracefully when no trained weights are present.
"""
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from spectra.inference.gesture.landmarks import N_LANDMARKS

WINDOW_SIZE = 16
FEATURE_DIM = N_LANDMARKS * 3  # 63

CLASSES = ["left_swipe", "right_swipe", "pinch", "two_finger_point", "idle", "undefined"]
N_CLASSES = len(CLASSES)
IDX_TO_CLASS = dict(enumerate(CLASSES))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class GestureCNN(nn.Module):
    """
    Input:  (batch, WINDOW_SIZE, FEATURE_DIM)  — (B, 16, 63)
    Output: (batch, N_CLASSES)                 — class logits
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) → (B, F, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.max(dim=2).values  # GlobalMaxPool over time axis
        return self.head(x)


class GestureClassifier:
    """
    Inference wrapper: maintains a 16-frame sliding window of landmarks
    and emits a gesture class + confidence on every new frame once the
    window is full and weights are loaded.
    """

    CONFIDENCE_THRESHOLD = 0.65

    def __init__(self, model_path: Optional[Path] = None, device: str = "cpu"):
        self._device = torch.device(device)
        self._model = GestureCNN().to(self._device)
        self._model.eval()
        self._window: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self._ready = False

        if model_path is not None and Path(model_path).exists():
            state = torch.load(str(model_path), map_location=self._device, weights_only=True)
            self._model.load_state_dict(state)
            self._ready = True

    @property
    def ready(self) -> bool:
        """False until trained weights are loaded."""
        return self._ready

    def update(self, landmarks_flat: np.ndarray) -> tuple[Optional[str], float]:
        """
        Push one frame of flattened landmarks (shape: 63,).
        Returns (gesture_class, confidence) or (None, 0.0) if not ready.
        """
        self._window.append(landmarks_flat.astype(np.float32))
        if not self._ready or len(self._window) < WINDOW_SIZE:
            return None, 0.0

        seq = np.stack(self._window)  # (16, 63)
        x = torch.from_numpy(seq).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        if confidence < self.CONFIDENCE_THRESHOLD:
            return "idle", confidence
        return IDX_TO_CLASS[best_idx], confidence

    def reset_window(self) -> None:
        self._window.clear()
