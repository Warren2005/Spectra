import threading
import time

import cv2
from PySide6.QtCore import QObject, Signal


class WebcamCapture(QObject):
    """Background thread that captures webcam frames and emits them via signal."""

    frame_ready = Signal(object)   # numpy BGR frame
    fps_update = Signal(float)

    def __init__(self, device_index: int = 0, target_fps: int = 30, parent=None):
        super().__init__(parent)
        self._device_index = device_index
        self._target_fps = target_fps
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> bool:
        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if w < 1280 or h < 720:
            print(f"[webcam] WARNING: camera only provides {int(w)}x{int(h)} — "
                  "720p or higher recommended for gesture/gaze accuracy")

        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def _loop(self) -> None:
        assert self._cap is not None
        frame_times: list[float] = []
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            now = time.monotonic()
            frame_times.append(now)
            # Keep only last 30 timestamps for fps estimate
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 0.0
                if len(frame_times) == 30:
                    if fps < 20:
                        print(f"[webcam] WARNING: only {fps:.1f} fps — "
                              "gesture velocity calculations may be unreliable")
                    self.fps_update.emit(fps)
                    frame_times.clear()
            self.frame_ready.emit(frame)
