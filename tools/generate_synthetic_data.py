"""
Synthetic gesture training data generator.

Generates realistic MediaPipe-style hand landmark sequences for all gesture
classes without a webcam or recording session.

Usage:
    python tools/generate_synthetic_data.py              # default: 600 per class
    python tools/generate_synthetic_data.py --n 300      # fewer examples (faster)

Output: data/landmarks/<class>/synthetic.json  (same format as record_landmarks.py)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parents[1] / "data" / "landmarks"

# ── Hand template ─────────────────────────────────────────────────────────────
# Landmark offsets from wrist in "hand units" where 1.0 = hand height.
# y is negative = upward in image space (y=0 at top, y=1 at bottom).
# z is depth: positive = toward camera.

_EXTENDED = np.array([
    # 0  WRIST
    [0.000,  0.000,  0.000],
    # 1-4  THUMB
    [-0.150, -0.080,  0.010],
    [-0.220, -0.200,  0.015],
    [-0.240, -0.320,  0.010],
    [-0.200, -0.420,  0.005],   # 4 TIP
    # 5-8  INDEX
    [-0.080, -0.280,  0.000],   # 5 MCP
    [-0.078, -0.520,  0.000],   # 6 PIP
    [-0.074, -0.645,  0.000],   # 7 DIP
    [-0.070, -0.748,  0.000],   # 8 TIP  ← primary swipe signal
    # 9-12  MIDDLE
    [ 0.010, -0.295,  0.000],
    [ 0.010, -0.538,  0.000],
    [ 0.010, -0.670,  0.000],
    [ 0.010, -0.775,  0.000],   # 12 TIP
    # 13-16  RING
    [ 0.090, -0.278,  0.000],
    [ 0.098, -0.498,  0.000],
    [ 0.098, -0.618,  0.000],
    [ 0.100, -0.718,  0.000],   # 16 TIP
    # 17-20  PINKY
    [ 0.158, -0.238,  0.000],
    [ 0.168, -0.408,  0.000],
    [ 0.168, -0.498,  0.000],
    [ 0.178, -0.578,  0.000],   # 20 TIP
], dtype=np.float32)

_CURLED = np.array([
    [ 0.000,  0.000,  0.000],   # 0 WRIST
    [-0.150, -0.080,  0.010],
    [-0.180, -0.150,  0.020],
    [-0.160, -0.178,  0.030],
    [-0.120, -0.148,  0.025],   # 4 THUMB TIP (resting near palm)
    [-0.080, -0.278,  0.000],   # 5 INDEX MCP
    [-0.100, -0.368,  0.045],   # 6 PIP (curled inward)
    [-0.080, -0.398,  0.068],   # 7 DIP
    [-0.058, -0.368,  0.055],   # 8 INDEX TIP
    [ 0.010, -0.298,  0.000],   # 9 MIDDLE MCP
    [-0.008, -0.388,  0.045],   # 10 PIP
    [ 0.012, -0.418,  0.065],   # 11 DIP
    [ 0.032, -0.388,  0.052],   # 12 MIDDLE TIP
    [ 0.090, -0.278,  0.000],   # 13 RING MCP
    [ 0.098, -0.368,  0.042],   # 14 PIP
    [ 0.098, -0.398,  0.062],   # 15 DIP
    [ 0.088, -0.368,  0.050],   # 16 RING TIP
    [ 0.158, -0.238,  0.000],   # 17 PINKY MCP
    [ 0.163, -0.318,  0.040],   # 18 PIP
    [ 0.163, -0.343,  0.060],   # 19 DIP
    [ 0.158, -0.323,  0.048],   # 20 PINKY TIP
], dtype=np.float32)

# Two-finger point: index+middle extended, ring+pinky curled
_TWO_FINGER = _EXTENDED.copy()
_TWO_FINGER[13:17] = _CURLED[13:17]  # ring curled
_TWO_FINGER[17:21] = _CURLED[17:21]  # pinky curled

# Pre-pinch: all extended, thumb and index at normal distance
_PRE_PINCH = _EXTENDED.copy()

# Post-pinch: thumb tip moved close to index tip
_POST_PINCH = _EXTENDED.copy()
_POST_PINCH[4] = _EXTENDED[8] + np.array([0.01, -0.01, 0.01])  # thumb tip near index tip


# ── Core utilities ─────────────────────────────────────────────────────────────

def place_hand(
    offsets: np.ndarray,
    wrist_x: float,
    wrist_y: float,
    scale: float,
    noise_std: float = 0.004,
) -> np.ndarray:
    """
    Position a hand pose at (wrist_x, wrist_y) with the given scale,
    adding small Gaussian noise to every landmark.
    """
    lm = offsets.copy() * scale
    lm[:, 0] += wrist_x
    lm[:, 1] += wrist_y
    lm += np.random.normal(0, noise_std, lm.shape).astype(np.float32)
    return np.clip(lm, 0.0, 1.0).astype(np.float32)


def smooth_step(t: float) -> float:
    """S-curve easing for natural motion deceleration."""
    return t * t * (3.0 - 2.0 * t)


def to_frame(lm: np.ndarray, timestamp: float) -> dict:
    return {"timestamp": timestamp, "hands": {"Right": lm.tolist()}}


def vary(base: float, spread: float) -> float:
    return base + random.uniform(-spread, spread)


# ── Per-class generators ───────────────────────────────────────────────────────

def gen_swipe(direction: str, n: int) -> list[list[dict]]:
    """Right swipe: INDEX_TIP x increases. Left swipe: decreases."""
    seqs = []
    for _ in range(n):
        scale = random.uniform(0.24, 0.42)
        cy = random.uniform(0.48, 0.74)
        noise = random.uniform(0.003, 0.008)
        n_frames = random.randint(14, 26)

        if direction == "right":
            x_start = random.uniform(0.12, 0.35)
            x_end = random.uniform(0.55, 0.82)
        else:
            x_start = random.uniform(0.55, 0.82)
            x_end = random.uniform(0.12, 0.35)

        # Small random vertical drift during the swipe
        y_drift = random.uniform(-0.04, 0.04)

        t0 = time.time()
        frames = []
        for i in range(n_frames):
            t = smooth_step(i / max(n_frames - 1, 1))
            cx = x_start + t * (x_end - x_start)
            cy_i = cy + t * y_drift
            frames.append(to_frame(place_hand(_EXTENDED, cx, cy_i, scale, noise), t0 + i / 30))
        seqs.append(frames)
    return seqs


def gen_pinch(n: int) -> list[list[dict]]:
    """Thumb tip converges to index tip over ~18 frames, then releases."""
    seqs = []
    for _ in range(n):
        scale = random.uniform(0.24, 0.42)
        cx = random.uniform(0.25, 0.75)
        cy = random.uniform(0.48, 0.74)
        noise = random.uniform(0.003, 0.007)
        n_close = random.randint(10, 18)
        n_hold = random.randint(2, 6)
        n_open = random.randint(6, 12)
        n_frames = n_close + n_hold + n_open

        t0 = time.time()
        frames = []
        for i in range(n_frames):
            if i < n_close:
                t = smooth_step(i / n_close)
                pose = _PRE_PINCH * (1 - t) + _POST_PINCH * t
            elif i < n_close + n_hold:
                pose = _POST_PINCH.copy()
                # Slight wiggle while pinching
                pose[4] += np.random.normal(0, 0.008, 3).astype(np.float32)
                pose[8] += np.random.normal(0, 0.008, 3).astype(np.float32)
            else:
                t = smooth_step((i - n_close - n_hold) / n_open)
                pose = _POST_PINCH * (1 - t) + _PRE_PINCH * t
            frames.append(to_frame(place_hand(pose, cx, cy, scale, noise), t0 + i / 30))
        seqs.append(frames)
    return seqs


def gen_two_finger_point(n: int) -> list[list[dict]]:
    """Mostly static two-finger pose with small position drift."""
    seqs = []
    for _ in range(n):
        scale = random.uniform(0.24, 0.42)
        cx = random.uniform(0.25, 0.75)
        cy = random.uniform(0.48, 0.74)
        noise = random.uniform(0.003, 0.007)
        n_frames = random.randint(12, 24)

        t0 = time.time()
        frames = []
        drift_x = random.uniform(-0.006, 0.006)  # very slow drift per frame
        drift_y = random.uniform(-0.004, 0.004)
        for i in range(n_frames):
            cx_i = cx + drift_x * i
            cy_i = cy + drift_y * i
            frames.append(to_frame(place_hand(_TWO_FINGER, cx_i, cy_i, scale, noise), t0 + i / 30))
        seqs.append(frames)
    return seqs


def gen_idle(n: int) -> list[list[dict]]:
    """
    Natural movement: random walk with no dominant directional velocity.
    Constrains net x-displacement per 8 frames to < swipe threshold (0.10).
    Mixes extended and curled poses.
    """
    seqs = []
    for _ in range(n):
        scale = random.uniform(0.24, 0.42)
        cx = random.uniform(0.25, 0.75)
        cy = random.uniform(0.48, 0.74)
        noise = random.uniform(0.004, 0.010)
        n_frames = random.randint(16, 32)
        pose = random.choice([_EXTENDED, _CURLED, _TWO_FINGER])

        # Oscillatory or drift motion, capped so no 8-frame window exceeds 0.09 x-displacement
        t0 = time.time()
        frames = []
        positions_x: list[float] = [cx]
        positions_y: list[float] = [cy]

        for i in range(n_frames):
            # Propose a small random step, reject if it would create a swipe-like window
            new_x = positions_x[-1]
            new_y = positions_y[-1]
            for _ in range(10):
                dx = random.gauss(0, 0.008)
                dy = random.gauss(0, 0.006)
                cand_x = float(np.clip(positions_x[-1] + dx, 0.1, 0.9))
                cand_y = float(np.clip(positions_y[-1] + dy, 0.3, 0.85))
                window_start_x = positions_x[-min(7, len(positions_x))]
                if abs(cand_x - window_start_x) < 0.09:
                    new_x, new_y = cand_x, cand_y
                    break
            positions_x.append(new_x)
            positions_y.append(new_y)
            frames.append(to_frame(place_hand(pose, new_x, new_y, scale, noise), t0 + i / 30))
        seqs.append(frames)
    return seqs


def gen_undefined(n: int) -> list[list[dict]]:
    """
    Ambiguous or transitional poses: half-extended, borderline pinch,
    very slow drift. Not clearly any gesture.
    """
    seqs = []
    for _ in range(n):
        scale = random.uniform(0.24, 0.42)
        cx = random.uniform(0.25, 0.75)
        cy = random.uniform(0.48, 0.74)
        noise = random.uniform(0.006, 0.012)  # more noise = more ambiguous
        n_frames = random.randint(12, 20)

        # Mix between extended and curled randomly per frame
        t0 = time.time()
        frames = []
        for i in range(n_frames):
            mix = random.uniform(0.3, 0.7)
            pose = _EXTENDED * mix + _CURLED * (1 - mix)
            cx_i = cx + random.gauss(0, 0.004) * i
            cy_i = cy + random.gauss(0, 0.004) * i
            frames.append(to_frame(place_hand(pose, cx_i, cy_i, scale, noise), t0 + i / 30))
        seqs.append(frames)
    return seqs


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_sequence(frames: list[dict], n_augments: int = 3) -> list[list[dict]]:
    """
    Apply random time-warp, noise, and horizontal flip to produce
    additional examples from one recorded/generated sequence.
    """
    augmented = []
    for _ in range(n_augments):
        aug_frames = []
        flip = random.random() < 0.5
        noise_scale = random.uniform(0.5, 1.5)
        # Resample at slightly different speed (time warp)
        speed = random.uniform(0.75, 1.30)
        src_len = len(frames)
        dst_len = max(8, int(src_len / speed))
        src_indices = np.linspace(0, src_len - 1, dst_len)
        t0 = time.time()
        for i, si in enumerate(src_indices):
            lo = int(si)
            hi = min(lo + 1, src_len - 1)
            alpha = si - lo
            lm_lo = np.array(frames[lo]["hands"]["Right"], np.float32)
            lm_hi = np.array(frames[hi]["hands"]["Right"], np.float32)
            lm = lm_lo * (1 - alpha) + lm_hi * alpha
            lm += np.random.normal(0, 0.004 * noise_scale, lm.shape).astype(np.float32)
            if flip:
                lm[:, 0] = 1.0 - lm[:, 0]  # horizontal mirror
            lm = np.clip(lm, 0.0, 1.0)
            aug_frames.append(to_frame(lm, t0 + i / 30))
        augmented.append(aug_frames)
    return augmented


# ── Main ───────────────────────────────────────────────────────────────────────

GENERATORS = {
    "right_swipe":      lambda n: gen_swipe("right", n),
    "left_swipe":       lambda n: gen_swipe("left", n),
    "pinch":            gen_pinch,
    "two_finger_point": gen_two_finger_point,
    "idle":             gen_idle,
    "undefined":        gen_undefined,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600,
                        help="Base examples per class before augmentation")
    parser.add_argument("--augments", type=int, default=2,
                        help="Number of augmented copies per base sequence")
    args = parser.parse_args()

    # For idle class, generate more to balance the negative class
    counts = {cls: args.n for cls in GENERATORS}
    counts["idle"] = args.n * 2

    total_seqs = 0
    for cls, gen_fn in GENERATORS.items():
        out_dir = OUT_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "synthetic.json"

        n_base = counts[cls]
        print(f"  Generating {cls}... ", end="", flush=True)

        base_seqs = gen_fn(n_base)

        # Augment
        all_seqs = list(base_seqs)
        for seq in base_seqs:
            all_seqs.extend(augment_sequence(seq, args.augments))

        payload = {
            "gesture_class": cls,
            "session": "synthetic",
            "recorded_at": int(time.time()),
            "sequences": all_seqs,
        }
        out_path.write_text(json.dumps(payload))
        print(f"{len(all_seqs)} sequences → {out_path}")
        total_seqs += len(all_seqs)

    print(f"\nDone. {total_seqs} total sequences across {len(GENERATORS)} classes.")
    print(f"Next: python tools/train_classifier.py")


if __name__ == "__main__":
    main()
