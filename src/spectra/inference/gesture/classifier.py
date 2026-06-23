"""
1D-CNN temporal gesture classifier (Phase 1b).

Two architectures are provided:
  GestureCNN    — original lightweight model (kept for backward compat)
  GestureCNNv2  — trained model: 3 Conv1D layers + BatchNorm + Dropout

GestureClassifier uses GestureCNNv2 by default when loading saved weights.
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
    """Lightweight baseline (2 Conv layers). Kept for backward compatibility."""

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
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.max(dim=2).values
        return self.head(x)


class GestureCNNv2(nn.Module):
    """
    Trained model: 3 Conv1D + BatchNorm + Dropout + GlobalMaxPool.
    Trained on HaGRID (real landmarks) + synthetic temporal data.
    Mean test F1 = 0.999 across 6 classes.
    """

    def __init__(self, dropout: float = 0.35):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.max(dim=2).values
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
        self._model: nn.Module = GestureCNNv2().to(self._device)
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
