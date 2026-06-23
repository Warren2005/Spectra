"""
Stream HaGRID images, run MediaPipe HandLandmarker on each, and save
landmark sequences to data/landmarks/<spectra_class>/hagrid.json.

Dataset: Jayabalambika/hagrid-classification-512p-dataset (HuggingFace)
18 gesture classes → mapped to 3 of our Spectra classes:
  peace / peace_inverted / two_up / two_up_inverted → two_finger_point
  palm / stop / stop_inverted                        → idle
  fist / ok / rock / like / dislike                  → undefined

For each detected hand we build a short pseudo-sequence by repeating
the static frame with per-frame noise + drift, giving the temporal CNN
examples of what "holding a static pose" looks like over 20 frames.

Usage:
    python scripts/extract_hagrid_landmarks.py
    python scripts/extract_hagrid_landmarks.py --per-class 500
"""
import argparse
import json
import time
from pathlib import Path

import mediapipe as mp
import numpy as np
from datasets import load_dataset
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path(__file__).parents[1] / "models" / "hand_landmarker.task"
OUT_DIR = Path(__file__).parents[1] / "data" / "landmarks"

DATASET_ID = "Jayabalambika/hagrid-classification-512p-dataset"

# HaGRID class name → Spectra class
LABEL_MAP: dict[str, str] = {
    "palm":            "idle",
    "stop":            "idle",
    "stop_inverted":   "idle",
    "peace":           "two_finger_point",
    "peace_inverted":  "two_finger_point",
    "two_up":          "two_finger_point",
    "two_up_inverted": "two_finger_point",
    "fist":            "undefined",
    "ok":              "undefined",
    "rock":            "undefined",
    "like":            "undefined",
    "dislike":         "undefined",
    # call, mute, one, four, three, three2 → skipped (no clear mapping)
}

N_LANDMARKS = 21
SKIP_LABELS = {"call", "mute", "one", "four", "three", "three2"}


def make_image_landmarker() -> mp_vision.HandLandmarker:
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def to_array(hand_landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)


def make_static_sequence(lm: np.ndarray, n_frames: int) -> list[dict]:
    """Build a pseudo-temporal sequence from a single detected landmark frame."""
    t0 = time.time()
    drift_x = np.random.uniform(-0.004, 0.004)
    drift_y = np.random.uniform(-0.003, 0.003)
    noise_std = np.random.uniform(0.002, 0.005)
    frames = []
    for i in range(n_frames):
        noisy = lm.copy()
        noisy[:, 0] += drift_x * i + np.random.normal(0, noise_std, N_LANDMARKS)
        noisy[:, 1] += drift_y * i + np.random.normal(0, noise_std, N_LANDMARKS)
        noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)
        frames.append({"timestamp": t0 + i / 30.0, "hands": {"Right": noisy.tolist()}})
    return frames


def augment_sequence(lm: np.ndarray, n_frames: int) -> list[dict]:
    """Augmented copy: flip + noise variation."""
    lm_aug = lm.copy()
    if np.random.random() < 0.5:
        lm_aug[:, 0] = 1.0 - lm_aug[:, 0]
    lm_aug += np.random.normal(0, 0.006, lm_aug.shape).astype(np.float32)
    lm_aug = np.clip(lm_aug, 0.0, 1.0)
    return make_static_sequence(lm_aug, n_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=400,
                        help="Images to extract per Spectra class")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--augments", type=int, default=3,
                        help="Augmented copies per extracted frame")
    args = parser.parse_args()

    print(f"Loading {DATASET_ID} (streaming)...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    label_feature = ds.features["label"]

    landmarker = make_image_landmarker()
    sequences: dict[str, list] = {cls: [] for cls in set(LABEL_MAP.values())}
    counts: dict[str, int] = {cls: 0 for cls in sequences}
    processed = skipped = 0

    target_per_class = args.per_class
    print(f"Target: {target_per_class} images per class\n")

    for sample in ds:
        if all(counts[c] >= target_per_class for c in counts):
            break

        # Decode integer label to string
        raw_label = sample["label"]
        if isinstance(raw_label, int):
            hagrid_name = label_feature.int2str(raw_label)
        else:
            hagrid_name = str(raw_label)

        if hagrid_name in SKIP_LABELS or hagrid_name not in LABEL_MAP:
            skipped += 1
            continue

        spectra_cls = LABEL_MAP[hagrid_name]
        if counts[spectra_cls] >= target_per_class:
            continue

        # Run MediaPipe
        try:
            img = sample["image"]
            rgb = np.array(img.convert("RGB"), dtype=np.uint8)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)
        except Exception:
            skipped += 1
            continue

        if not result.hand_landmarks:
            skipped += 1
            continue

        lm = to_array(result.hand_landmarks[0])

        # Base + augmented sequences
        seqs = [make_static_sequence(lm, args.seq_len)]
        for _ in range(args.augments):
            seqs.append(augment_sequence(lm, args.seq_len))

        sequences[spectra_cls].extend(seqs)
        counts[spectra_cls] += 1
        processed += 1

        if processed % 100 == 0:
            bar = "  ".join(f"{c}:{n}/{target_per_class}" for c, n in counts.items())
            print(f"  [{processed}] {bar}")

    landmarker.close()

    print(f"\nExtracted {processed} hands  (skipped {skipped})\n")

    for spectra_cls, seqs in sequences.items():
        if not seqs:
            print(f"  {spectra_cls}: 0 sequences — no data extracted")
            continue
        out_dir = OUT_DIR / spectra_cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "hagrid.json"
        out_path.write_text(json.dumps({
            "gesture_class": spectra_cls,
            "session": "hagrid",
            "recorded_at": int(time.time()),
            "sequences": seqs,
        }))
        total_seq = len(seqs)
        print(f"  {spectra_cls}: {counts[spectra_cls]} images → {total_seq} sequences → {out_path}")


if __name__ == "__main__":
    main()
