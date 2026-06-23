"""
Bimanual clutch state machine.

The state machine is the primary defence against the Midas Touch problem
(accidental activation from natural hand movement). All gesture recognition
is suppressed in IDLE state — the left-hand clutch must be raised first.
"""
import time
from enum import Enum, auto
from typing import Callable

from spectra.schema import IntentEvent, Modality


class GestureState(Enum):
    IDLE = auto()        # Clutch off — no gesture processing
    COMMAND = auto()     # Clutch on — listening for right-hand gestures
    CURSOR = auto()      # Two-finger point active — cursor move mode
    ANCHOR_SET = auto()  # First highlight anchor placed, awaiting second pinch


class GestureStateMachine:
    SWIPE_COOLDOWN = 0.5  # seconds between consecutive page-turn events

    def __init__(self, emit: Callable[[IntentEvent], None]):
        self._emit = emit
        self._state = GestureState.IDLE
        self._last_swipe_time = 0.0

    @property
    def state(self) -> GestureState:
        return self._state

    # ── Clutch ───────────────────────────────────────────────────────────────

    def on_clutch_on(self) -> None:
        if self._state == GestureState.IDLE:
            self._state = GestureState.COMMAND

    def on_clutch_off(self) -> None:
        self._state = GestureState.IDLE

    # ── Swipe (only valid in COMMAND or CURSOR) ───────────────────────────────

    def on_right_swipe(self) -> None:
        if self._state not in (GestureState.COMMAND, GestureState.CURSOR):
            return
        now = time.monotonic()
        if now - self._last_swipe_time < self.SWIPE_COOLDOWN:
            return
        self._last_swipe_time = now
        self._emit(IntentEvent(intent="PAGE_NEXT", confidence=0.9, modality=Modality.GESTURE))

    def on_left_swipe(self) -> None:
        if self._state not in (GestureState.COMMAND, GestureState.CURSOR):
            return
        now = time.monotonic()
        if now - self._last_swipe_time < self.SWIPE_COOLDOWN:
            return
        self._last_swipe_time = now
        self._emit(IntentEvent(intent="PAGE_PREV", confidence=0.9, modality=Modality.GESTURE))

    # ── Pinch ─────────────────────────────────────────────────────────────────

    def on_pinch(self) -> None:
        if self._state in (GestureState.COMMAND, GestureState.CURSOR):
            self._state = GestureState.ANCHOR_SET
            self._emit(IntentEvent(
                intent="HIGHLIGHT_ANCHOR", confidence=0.85, modality=Modality.GESTURE
            ))
        elif self._state == GestureState.ANCHOR_SET:
            self._state = GestureState.COMMAND
            self._emit(IntentEvent(
                intent="HIGHLIGHT_CONFIRM", confidence=0.85, modality=Modality.GESTURE
            ))

    # ── Two-finger point ──────────────────────────────────────────────────────

    def on_two_finger_point(self, norm_x: float, norm_y: float) -> None:
        if self._state == GestureState.COMMAND:
            self._state = GestureState.CURSOR
        if self._state == GestureState.CURSOR:
            self._emit(IntentEvent(
                intent="CURSOR_MOVE",
                confidence=0.8,
                modality=Modality.GESTURE,
                payload={"cursor": (norm_x, norm_y)},
            ))

    def on_two_finger_released(self) -> None:
        if self._state == GestureState.CURSOR:
            self._state = GestureState.COMMAND
