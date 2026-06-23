"""
False-activation evaluation (Phase 1d).

Replays a recorded natural-movement landmark file through the full gesture
pipeline (without camera) and counts how many IntentEvents fire.

Usage:
    python evaluation/false_activation.py data/landmarks/natural_movement/session1_*.json

Reports: false-activations/hour, with a threshold sweep.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path when run as a script
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from spectra.inference.gesture.landmarks import Hand
from spectra.inference.gesture.rule_based import (
    SwipeDetector, PinchDetector, all_fingers_extended, is_two_finger_point,
)
from spectra.inference.gesture.state_machine import GestureState, GestureStateMachine
from spectra.schema import IntentEvent


def load_fixture(path: Path) -> list[dict]:
    """Load a landmark JSON file and return list of frame dicts."""
    data = json.loads(path.read_text())
    # Support both tools/record_landmarks.py format (list of sequences)
    # and replay fixture format (flat list of frames)
    if isinstance(data, dict) and "sequences" in data:
        frames: list[dict] = []
        for seq in data["sequences"]:
            frames.extend(seq)
        return frames
    assert isinstance(data, list)
    return data


def replay_pipeline(frames: list[dict], swipe_threshold: float = 0.12) -> tuple[int, float]:
    """
    Replay frames through rule-based pipeline.
    Returns (n_false_activations, duration_seconds).
    """
    events: list[IntentEvent] = []
    fsm = GestureStateMachine(events.append)
    swipe = SwipeDetector()
    swipe.THRESHOLD = swipe_threshold
    pinch = PinchDetector()

    t_start = frames[0]["timestamp"] if frames else time.time()
    t_end = frames[-1]["timestamp"] if frames else time.time()

    for frame in frames:
        hand_data = frame.get("hands", {})
        left_raw = hand_data.get("Left") or hand_data.get("left")
        right_raw = hand_data.get("Right") or hand_data.get("right")

        left = np.array(left_raw, dtype=np.float32) if left_raw else None
        right = np.array(right_raw, dtype=np.float32) if right_raw else None

        # Clutch
        if left is not None and all_fingers_extended(left):
            fsm.on_clutch_on()
        elif left is None:
            fsm.on_clutch_off()

        if fsm.state == GestureState.IDLE or right is None:
            continue

        # Two-finger point
        if is_two_finger_point(right):
            fsm.on_two_finger_point(float(right[8, 0]), float(right[8, 1]))
            continue
        else:
            fsm.on_two_finger_released()

        if pinch.update(right):
            fsm.on_pinch()
            swipe.reset()
            continue
        direction = swipe.update(right)
        if direction == "right":
            fsm.on_right_swipe()
            swipe.reset()
        elif direction == "left":
            fsm.on_left_swipe()
            swipe.reset()

    duration = t_end - t_start
    return len(events), duration


def main():
    parser = argparse.ArgumentParser(description="False-activation evaluator")
    parser.add_argument("files", nargs="+", help="JSON landmark files to replay")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.08, 0.10, 0.12, 0.15, 0.18],
                        help="Swipe threshold values to sweep")
    args = parser.parse_args()

    all_frames: list[dict] = []
    for path in args.files:
        all_frames.extend(load_fixture(Path(path)))

    if not all_frames:
        print("No frames loaded.")
        return

    total_seconds = all_frames[-1]["timestamp"] - all_frames[0]["timestamp"]
    total_hours = total_seconds / 3600
    print(f"\nLoaded {len(all_frames)} frames  ({total_seconds:.0f}s = {total_hours:.2f}h)\n")
    print(f"{'Threshold':>12}  {'False Activations':>18}  {'FA/hour':>10}  {'Pass (<3/hr)':>12}")
    print("-" * 60)

    for thresh in args.thresholds:
        n, duration = replay_pipeline(all_frames, swipe_threshold=thresh)
        fa_per_hour = n / max(duration / 3600, 1e-6)
        passed = "✓" if fa_per_hour < 3.0 else "✗"
        print(f"{thresh:>12.3f}  {n:>18d}  {fa_per_hour:>10.1f}  {passed:>12}")

    print()
    print("Target: < 3 false activations per hour (Phase 1d exit gate)")


if __name__ == "__main__":
    main()
