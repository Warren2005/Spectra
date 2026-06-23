"""
Phase 1a replay tests — rule-based gesture recognizers and state machine.
No camera required: uses synthetic landmark arrays.
"""
import numpy as np

from spectra.inference.gesture.landmarks import (
    LandmarkArray, N_LANDMARKS,
    INDEX_TIP, THUMB_TIP, INDEX_MCP,
    INDEX_PIP, MIDDLE_TIP, MIDDLE_PIP,
    RING_TIP, RING_PIP, PINKY_TIP, PINKY_PIP,
    THUMB_MCP, WRIST,
)
from spectra.inference.gesture.rule_based import (
    SwipeDetector, PinchDetector,
    all_fingers_extended, is_two_finger_point, normalized_pinch_distance,
)
from spectra.inference.gesture.state_machine import GestureState, GestureStateMachine
from spectra.schema import IntentEvent


# ── Synthetic landmark helpers ─────────────────────────────────────────────────

def make_landmarks() -> LandmarkArray:
    """Base landmark array: all landmarks at (0.5, 0.5, 0)."""
    return np.full((N_LANDMARKS, 3), 0.5, dtype=np.float32)


def open_hand(lm: LandmarkArray) -> LandmarkArray:
    """Set all finger tips above PIP joints (extended) and thumb out."""
    lm = lm.copy()
    # Fingers: tip.y < pip.y  →  tips are higher (smaller y in image space)
    for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                     (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]:
        lm[tip, 1] = 0.2   # tip high up
        lm[pip, 1] = 0.4   # pip lower
    # Thumb extended: tip farther from INDEX_MCP than THUMB_MCP
    lm[WRIST, :2] = [0.5, 0.9]
    lm[THUMB_MCP, :2] = [0.4, 0.7]
    lm[THUMB_TIP, :2] = [0.2, 0.5]  # far from INDEX_MCP
    lm[INDEX_MCP, :2] = [0.5, 0.6]
    return lm


def curled_hand(lm: LandmarkArray) -> LandmarkArray:
    """All finger tips below PIP (curled)."""
    lm = lm.copy()
    for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                     (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]:
        lm[tip, 1] = 0.7   # tip lower (curled)
        lm[pip, 1] = 0.4
    return lm


def two_finger_hand(lm: LandmarkArray) -> LandmarkArray:
    """Index + middle extended, ring + pinky curled."""
    lm = curled_hand(lm)
    lm[INDEX_TIP, 1] = 0.2
    lm[INDEX_PIP, 1] = 0.4
    lm[MIDDLE_TIP, 1] = 0.2
    lm[MIDDLE_PIP, 1] = 0.4
    return lm


def pinch_hand(lm: LandmarkArray) -> LandmarkArray:
    """Thumb tip very close to index tip (small normalized distance)."""
    lm = lm.copy()
    lm[THUMB_TIP, :2] = [0.45, 0.50]
    lm[INDEX_TIP, :2] = [0.47, 0.50]
    lm[WRIST, :2] = [0.50, 0.90]
    lm[9, :2] = [0.50, 0.60]   # MIDDLE_MCP — sets hand_size
    return lm


def open_hand_no_pinch(lm: LandmarkArray) -> LandmarkArray:
    """Thumb and index far apart."""
    lm = lm.copy()
    lm[THUMB_TIP, :2] = [0.20, 0.50]
    lm[INDEX_TIP, :2] = [0.80, 0.50]
    lm[WRIST, :2] = [0.50, 0.90]
    lm[9, :2] = [0.50, 0.60]
    return lm


# ── Finger state tests ─────────────────────────────────────────────────────────

class TestFingerState:
    def test_open_hand_all_extended(self):
        lm = open_hand(make_landmarks())
        assert all_fingers_extended(lm)

    def test_curled_hand_not_extended(self):
        lm = curled_hand(make_landmarks())
        assert not all_fingers_extended(lm)

    def test_two_finger_point_detected(self):
        lm = two_finger_hand(make_landmarks())
        assert is_two_finger_point(lm)

    def test_open_hand_not_two_finger_point(self):
        lm = open_hand(make_landmarks())
        assert not is_two_finger_point(lm)

    def test_pinch_distance_small_when_closed(self):
        lm = pinch_hand(make_landmarks())
        assert normalized_pinch_distance(lm) < 0.18

    def test_pinch_distance_large_when_open(self):
        lm = open_hand_no_pinch(make_landmarks())
        assert normalized_pinch_distance(lm) > 0.5


# ── SwipeDetector tests ───────────────────────────────────────────────────────

class TestSwipeDetector:
    def _feed_swipe(self, direction: str, n_frames: int = 8) -> list:
        det = SwipeDetector()
        results = []
        for i in range(n_frames):
            lm = make_landmarks()
            if direction == "right":
                lm[INDEX_TIP, 0] = 0.2 + (i / n_frames) * 0.5  # 0.2 → 0.7
            else:
                lm[INDEX_TIP, 0] = 0.7 - (i / n_frames) * 0.5  # 0.7 → 0.2
            results.append(det.update(lm))
        return results

    def _feed_natural(self, n_frames: int = 8) -> list:
        det = SwipeDetector()
        results = []
        for i in range(n_frames):
            lm = make_landmarks()
            lm[INDEX_TIP, 0] = 0.5 + 0.01 * np.sin(i)  # tiny oscillation
            results.append(det.update(lm))
        return results

    def test_right_swipe_detected(self):
        results = self._feed_swipe("right")
        assert "right" in results

    def test_left_swipe_detected(self):
        results = self._feed_swipe("left")
        assert "left" in results

    def test_natural_movement_no_swipe(self):
        results = self._feed_natural()
        assert all(r is None for r in results)

    def test_reset_clears_history(self):
        det = SwipeDetector()
        lm = make_landmarks()
        lm[INDEX_TIP, 0] = 0.8
        for _ in range(8):
            det.update(lm)
        det.reset()
        assert len(det._history) == 0


# ── PinchDetector tests ────────────────────────────────────────────────────────

class TestPinchDetector:
    def test_fires_on_close(self):
        det = PinchDetector()
        assert det.update(pinch_hand(make_landmarks()))

    def test_hysteresis_no_double_fire(self):
        det = PinchDetector()
        lm = pinch_hand(make_landmarks())
        assert det.update(lm)   # first frame: fires
        assert not det.update(lm)  # still pinching: does not re-fire

    def test_fires_again_after_release(self):
        det = PinchDetector()
        lm_closed = pinch_hand(make_landmarks())
        lm_open = open_hand_no_pinch(make_landmarks())
        det.update(lm_closed)      # fire
        det.update(lm_open)        # release
        assert det.update(lm_closed)  # fire again

    def test_no_fire_when_open(self):
        det = PinchDetector()
        assert not det.update(open_hand_no_pinch(make_landmarks()))


# ── State machine tests ────────────────────────────────────────────────────────

class TestGestureStateMachine:
    def setup_method(self):
        self.emitted: list[IntentEvent] = []
        self.fsm = GestureStateMachine(self.emitted.append)

    def test_starts_idle(self):
        assert self.fsm.state == GestureState.IDLE

    def test_clutch_on_enters_command(self):
        self.fsm.on_clutch_on()
        assert self.fsm.state == GestureState.COMMAND

    def test_clutch_off_returns_to_idle(self):
        self.fsm.on_clutch_on()
        self.fsm.on_clutch_off()
        assert self.fsm.state == GestureState.IDLE

    def test_swipe_suppressed_in_idle(self):
        self.fsm.on_right_swipe()
        assert len(self.emitted) == 0

    def test_right_swipe_emits_page_next(self):
        self.fsm.on_clutch_on()
        self.fsm.on_right_swipe()
        assert len(self.emitted) == 1
        assert self.emitted[0].intent == "PAGE_NEXT"

    def test_left_swipe_emits_page_prev(self):
        self.fsm.on_clutch_on()
        self.fsm.on_left_swipe()
        assert self.emitted[0].intent == "PAGE_PREV"

    def test_pinch_sets_anchor_then_confirm(self):
        self.fsm.on_clutch_on()
        self.fsm.on_pinch()
        assert self.fsm.state == GestureState.ANCHOR_SET
        assert self.emitted[-1].intent == "HIGHLIGHT_ANCHOR"
        self.fsm.on_pinch()
        assert self.fsm.state == GestureState.COMMAND
        assert self.emitted[-1].intent == "HIGHLIGHT_CONFIRM"

    def test_two_finger_point_enters_cursor(self):
        self.fsm.on_clutch_on()
        self.fsm.on_two_finger_point(0.5, 0.5)
        assert self.fsm.state == GestureState.CURSOR
        assert self.emitted[-1].intent == "CURSOR_MOVE"
        assert self.emitted[-1].payload["cursor"] == (0.5, 0.5)

    def test_two_finger_released_returns_to_command(self):
        self.fsm.on_clutch_on()
        self.fsm.on_two_finger_point(0.5, 0.5)
        self.fsm.on_two_finger_released()
        assert self.fsm.state == GestureState.COMMAND

    def test_swipe_cooldown_prevents_double_fire(self):
        self.fsm.on_clutch_on()
        self.fsm.on_right_swipe()
        self.fsm.on_right_swipe()  # immediately — should be suppressed
        assert len(self.emitted) == 1

    def test_confidence_is_valid(self):
        self.fsm.on_clutch_on()
        self.fsm.on_right_swipe()
        assert 0.0 <= self.emitted[0].confidence <= 1.0


# ── One-Euro filter tests ─────────────────────────────────────────────────────

class TestOneEuroFilter:
    def test_filters_jitter(self):
        from spectra.inference.gesture.landmarks import OneEuroFilter
        f = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.1)
        # Feed a stable signal with small noise; output should converge near true value
        true_val = np.array([0.5] * 63, dtype=np.float32)
        out = true_val.copy()
        for _ in range(30):
            noisy = true_val + np.random.normal(0, 0.005, 63).astype(np.float32)
            out = f(noisy)
        assert np.allclose(out, true_val, atol=0.02)

    def test_reset_reinitialises_filter(self):
        from spectra.inference.gesture.landmarks import OneEuroFilter
        f = OneEuroFilter()
        f(np.zeros(10, dtype=np.float32))
        assert f._x is not None
        f.reset()
        assert f._x is None
