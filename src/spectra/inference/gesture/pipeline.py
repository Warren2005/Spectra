"""
Gesture pipeline orchestrator (Phase 1a/1b).

Runs on a dedicated QThread. Receives webcam frames, runs landmark detection,
applies rule-based or ML classifier, drives the state machine, and emits
IntentEvents via Qt signal (queued to main thread automatically).
"""
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from spectra.inference.gesture.classifier import GestureClassifier
from spectra.inference.gesture.landmarks import (
    Hand, LandmarkDetector,
    INDEX_TIP,
)
from spectra.inference.gesture.rule_based import (
    SwipeDetector, PinchDetector,
    all_fingers_extended, is_two_finger_point,
)
from spectra.inference.gesture.state_machine import GestureState, GestureStateMachine
from spectra.schema import IntentEvent

# Default model path relative to models/ directory
_DEFAULT_MODEL = Path(__file__).parents[4] / "models" / "gesture_cnn.pt"


class GesturePipeline(QObject):
    """
    Processes one webcam frame at a time.
    Thread-safe: move this object to a QThread, connect webcam.frame_ready
    with QueuedConnection so inference runs off the main thread.
    """

    intent_event = Signal(object)  # IntentEvent
    state_changed = Signal(str)    # GestureState name — consumed by HUD

    def __init__(self, model_path: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self._detector = LandmarkDetector()
        self._swipe = SwipeDetector()
        self._pinch = PinchDetector()
        self._fsm = GestureStateMachine(self._emit_intent)

        mp = model_path or _DEFAULT_MODEL
        self._classifier = GestureClassifier(mp)
        self._use_classifier = self._classifier.ready
        self._prev_state = GestureState.IDLE

    def _emit_intent(self, event: IntentEvent) -> None:
        self.intent_event.emit(event)

    @Slot(object)
    def process_frame(self, frame: np.ndarray) -> None:
        """Called from webcam thread (via QueuedConnection → runs in this object's thread)."""
        hands = self._detector.process(frame)
        self._process_hands(hands)

        if self._fsm.state != self._prev_state:
            self._prev_state = self._fsm.state
            self.state_changed.emit(self._fsm.state.name)

    def _process_hands(self, hands: dict) -> None:
        left = hands.get(Hand.LEFT)
        right = hands.get(Hand.RIGHT)

        # Clutch logic — always rule-based
        if left is not None and all_fingers_extended(left):
            self._fsm.on_clutch_on()
        elif Hand.LEFT not in hands:
            self._fsm.on_clutch_off()

        if self._fsm.state == GestureState.IDLE or right is None:
            return

        # Two-finger point check (always rule-based — need index tip coords)
        if is_two_finger_point(right):
            self._fsm.on_two_finger_point(
                norm_x=float(right[INDEX_TIP, 0]),
                norm_y=float(right[INDEX_TIP, 1]),
            )
            return
        else:
            self._fsm.on_two_finger_released()

        if self._use_classifier:
            self._classify_gesture(right)
        else:
            self._rule_based_gesture(right)

    def _rule_based_gesture(self, right) -> None:
        if self._pinch.update(right):
            self._fsm.on_pinch()
            self._swipe.reset()
            return
        swipe = self._swipe.update(right)
        if swipe == "right":
            self._fsm.on_right_swipe()
            self._swipe.reset()
        elif swipe == "left":
            self._fsm.on_left_swipe()
            self._swipe.reset()

    def _classify_gesture(self, right) -> None:
        flat = right.flatten()
        gesture, _ = self._classifier.update(flat)
        if gesture == "right_swipe":
            self._fsm.on_right_swipe()
            self._classifier.reset_window()
        elif gesture == "left_swipe":
            self._fsm.on_left_swipe()
            self._classifier.reset_window()
        elif gesture == "pinch":
            if self._pinch.update(right):  # hysteresis guard
                self._fsm.on_pinch()
                self._classifier.reset_window()

    @property
    def fsm_state(self) -> GestureState:
        return self._fsm.state

    def close(self) -> None:
        self._detector.close()
