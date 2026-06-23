"""
End-to-end gesture pipeline latency profiler (Phase 1d).

Instruments the pipeline from frame-arrival to intent-emission.
Requires a real webcam. Run for at least 500 gesture events.

Usage:
    python evaluation/latency_profile.py

Reports: mean latency, p95 latency, histogram.
Target: p95 < 80ms.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import cv2

from spectra.inference.gesture.landmarks import Hand, LandmarkDetector
from spectra.inference.gesture.rule_based import (
    SwipeDetector, PinchDetector,
    all_fingers_extended, is_two_finger_point,
)
from spectra.inference.gesture.state_machine import GestureState, GestureStateMachine
from spectra.schema import IntentEvent

TARGET_P95_MS = 80.0
N_SAMPLES = 500


def main():
    latencies_ms: list[float] = []
    events: list[IntentEvent] = []

    def on_intent(event: IntentEvent) -> None:
        events.append(event)

    detector = LandmarkDetector()
    swipe = SwipeDetector()
    pinch = PinchDetector()
    fsm = GestureStateMachine(on_intent)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"\nLatency profiler — perform gestures until {N_SAMPLES} events collected.")
    print("Q = quit early\n")

    n_events_prev = 0
    while len(latencies_ms) < N_SAMPLES:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            continue

        hands = detector.process(frame)
        left = hands.get(Hand.LEFT)
        right = hands.get(Hand.RIGHT)

        if left is not None and all_fingers_extended(left):
            fsm.on_clutch_on()
        elif Hand.LEFT not in hands:
            fsm.on_clutch_off()

        if fsm.state != GestureState.IDLE and right is not None:
            if is_two_finger_point(right):
                fsm.on_two_finger_point(float(right[8, 0]), float(right[8, 1]))
            else:
                fsm.on_two_finger_released()
                if pinch.update(right):
                    fsm.on_pinch()
                else:
                    d = swipe.update(right)
                    if d == "right":
                        fsm.on_right_swipe()
                        swipe.reset()
                    elif d == "left":
                        fsm.on_left_swipe()
                        swipe.reset()

        t1 = time.perf_counter()

        if len(events) > n_events_prev:
            latency_ms = (t1 - t0) * 1000
            latencies_ms.append(latency_ms)
            n_events_prev = len(events)

        frame_copy = frame.copy()
        cv2.putText(frame_copy,
                    f"Events: {len(latencies_ms)}/{N_SAMPLES}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Latency Profiler", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if not latencies_ms:
        print("No events recorded.")
        return

    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    p99_ms = float(np.percentile(arr, 99))

    print(f"\n{'Metric':<12}  {'ms':>8}")
    print("-" * 24)
    print(f"{'mean':<12}  {mean_ms:>8.1f}")
    print(f"{'p50':<12}  {p50_ms:>8.1f}")
    print(f"{'p95':<12}  {p95_ms:>8.1f}")
    print(f"{'p99':<12}  {p99_ms:>8.1f}")
    print()
    gate = "✓ PASS" if p95_ms < TARGET_P95_MS else "✗ FAIL"
    print(f"Phase 1d exit gate (p95 < {TARGET_P95_MS}ms): {gate}  ({p95_ms:.1f}ms)")

    # Simple ASCII histogram
    print("\nLatency distribution (ms):")
    bins = np.linspace(0, max(100, p99_ms), 11)
    counts, edges = np.histogram(arr, bins=bins)
    for i, c in enumerate(counts):
        bar = "█" * int(c / max(counts) * 30)
        print(f"  {edges[i]:5.0f}–{edges[i+1]:5.0f}ms  {bar} {c}")


if __name__ == "__main__":
    main()
