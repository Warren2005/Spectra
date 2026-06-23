"""
Gesture training data collection tool.

Usage:
    python tools/record_landmarks.py --class right_swipe --session 1

Controls:
    SPACE  — start/stop recording a gesture sequence
    Q      — quit and save all recorded sequences
    D      — discard the last sequence

Each JSON file saved to data/landmarks/<class>/<session>_<timestamp>.json
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

SAVE_DIR = Path(__file__).parents[1] / "data" / "landmarks"

CLASSES = [
    "left_swipe", "right_swipe", "pinch",
    "two_finger_point", "idle", "natural_movement",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--class", dest="gesture_class", choices=CLASSES, required=True)
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = SAVE_DIR / args.gesture_class
    out_dir.mkdir(parents=True, exist_ok=True)

    hands_mp = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording = False
    sequences: list[list[dict]] = []
    current_seq: list[dict] = []

    print(f"\nRecording class: {args.gesture_class}  session: {args.session}")
    print("SPACE = start/stop  |  D = discard last  |  Q = quit & save\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_mp.process(rgb)

        frame_data: dict[str, list] = {}
        if results.multi_hand_landmarks:
            for lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                coords = [[lm.x, lm.y, lm.z] for lm in lms.landmark]
                frame_data[label] = coords
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, lms, mp.solutions.hands.HAND_CONNECTIONS
                )

        if recording and frame_data:
            current_seq.append({"timestamp": time.time(), "hands": frame_data})

        # Status overlay
        status = "● REC" if recording else "○ READY"
        color = (0, 0, 220) if recording else (0, 180, 0)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Seqs: {len(sequences)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f"Class: {args.gesture_class}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.imshow("SPECTRA — Gesture Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if not recording:
                recording = True
                current_seq = []
                print(f"  Recording... (SPACE to stop)")
            else:
                recording = False
                if len(current_seq) >= 8:
                    sequences.append(current_seq)
                    print(f"  Saved sequence ({len(current_seq)} frames). Total: {len(sequences)}")
                else:
                    print(f"  Sequence too short ({len(current_seq)} frames), discarded")
                current_seq = []
        elif key == ord('d') and sequences:
            sequences.pop()
            print(f"  Discarded last sequence. Remaining: {len(sequences)}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_mp.close()

    if sequences:
        ts = int(time.time())
        out_path = out_dir / f"session{args.session}_{ts}.json"
        payload = {
            "gesture_class": args.gesture_class,
            "session": args.session,
            "recorded_at": ts,
            "sequences": sequences,
        }
        out_path.write_text(json.dumps(payload))
        print(f"\nSaved {len(sequences)} sequences → {out_path}")
    else:
        print("\nNo sequences recorded.")


if __name__ == "__main__":
    main()
