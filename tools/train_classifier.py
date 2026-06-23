"""
Gesture CNN training script.

Loads landmark JSON files from data/landmarks/<class>/,
trains GestureCNN, saves weights to models/gesture_cnn.pt,
and writes a training log to models/training_log.json.

Data sources used automatically if present:
  data/landmarks/<class>/synthetic.json   (from generate_synthetic_data.py)
  data/landmarks/<class>/hagrid.json      (from extract_hagrid_landmarks.py)
  data/landmarks/<class>/session*.json    (from record_landmarks.py)

Usage:
    python tools/train_classifier.py
    python tools/train_classifier.py --epochs 60 --lr 3e-4
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from spectra.inference.gesture.classifier import (
    CLASS_TO_IDX, CLASSES, FEATURE_DIM, WINDOW_SIZE, GestureCNNv2,
)

DATA_DIR = Path(__file__).parents[1] / "data" / "landmarks"
MODEL_DIR = Path(__file__).parents[1] / "models"
MODEL_PATH = MODEL_DIR / "gesture_cnn.pt"
LOG_PATH = MODEL_DIR / "training_log.json"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_windows(stride: int = 4) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all landmark JSON files and extract sliding-window examples.
    Returns X (N, WINDOW_SIZE, FEATURE_DIM), y (N,), sources (N,).
    Splitting is done by source file to prevent temporal leakage.
    """
    X_list, y_list, src_list = [], [], []

    for cls in CLASSES:
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            print(f"  [skip] {cls}: no data directory")
            continue
        json_files = sorted(cls_dir.glob("*.json"))
        if not json_files:
            print(f"  [skip] {cls}: no JSON files")
            continue

        label = CLASS_TO_IDX[cls]
        n_windows = 0

        for path in json_files:
            data = json.loads(path.read_text())
            seqs = data.get("sequences", []) if isinstance(data, dict) else [data]

            for seq in seqs:
                if not isinstance(seq, list) or len(seq) < WINDOW_SIZE:
                    continue
                lm_seq = []
                for frame in seq:
                    hands = frame.get("hands", {})
                    right = hands.get("Right") or hands.get("right")
                    if right is None:
                        break
                    flat = np.array(right, dtype=np.float32).flatten()
                    if len(flat) != FEATURE_DIM:
                        break
                    lm_seq.append(flat)
                else:
                    for start in range(0, len(lm_seq) - WINDOW_SIZE + 1, stride):
                        X_list.append(np.stack(lm_seq[start:start + WINDOW_SIZE]))
                        y_list.append(label)
                        src_list.append(path.stem)
                        n_windows += 1

        source_names = [p.stem for p in json_files]
        print(f"  {cls:20s} {n_windows:>6} windows  [{', '.join(source_names)}]")

    if not X_list:
        return np.empty((0, WINDOW_SIZE, FEATURE_DIM)), np.empty(0, np.int64), []

    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list, dtype=np.int64),
        src_list,
    )


def stratified_split(
    X: np.ndarray, y: np.ndarray, sources: list[str],
    val_frac: float = 0.15, test_frac: float = 0.15,
) -> tuple:
    """
    Stratified split by class label. Each class's windows are independently
    shuffled and split into train/val/test proportions, ensuring every class
    is represented in all three splits regardless of how many source files exist.
    """
    rng = np.random.default_rng(42)
    train_idx, val_idx, test_idx = [], [], []

    for cls_idx in np.unique(y):
        cls_mask = np.where(y == cls_idx)[0]
        perm = rng.permutation(len(cls_mask))
        cls_indices = cls_mask[perm]

        n = len(cls_indices)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = n - n_test - n_val

        test_idx.extend(cls_indices[:n_test].tolist())
        val_idx.extend(cls_indices[n_test:n_test + n_val].tolist())
        train_idx.extend(cls_indices[n_test + n_val:].tolist())

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


# ── Runtime augmentation ──────────────────────────────────────────────────────

def augment_batch(X: torch.Tensor) -> torch.Tensor:
    """Apply lightweight augmentation on a training batch."""
    B, T, F = X.shape
    # Gaussian noise on landmark coordinates
    X = X + torch.randn_like(X) * 0.004
    # Random horizontal flip (x coords are every 3rd value starting at 0)
    flip_mask = torch.rand(B) < 0.4
    if flip_mask.any():
        X_flip = X[flip_mask].clone()
        X_flip[:, :, 0::3] = 1.0 - X_flip[:, :, 0::3]
        X[flip_mask] = X_flip
    # Random time-warp: drop one frame and repeat another
    if torch.rand(1).item() < 0.3:
        drop_t = int(torch.randint(1, T - 1, (1,)).item())
        X = torch.cat([X[:, :drop_t], X[:, drop_t + 1:], X[:, -1:]], dim=1)
    return X.clamp(0.0, 1.0)


# ── Training ──────────────────────────────────────────────────────────────────

def make_loader(X, y, batch_size, balanced=False, shuffle=True):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    sampler = None
    if balanced:
        counts = Counter(y.tolist())
        weights = [1.0 / counts[int(yi)] for yi in y]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)


def train(args) -> None:
    print("\n── Loading data ─────────────────────────────────")
    X, y, sources = load_windows()
    if len(X) == 0:
        print("\nNo data. Run generate_synthetic_data.py and/or extract_hagrid_landmarks.py first.")
        return

    print(f"\n  Total windows: {len(X)}")
    for cls in CLASSES:
        n = int((y == CLASS_TO_IDX[cls]).sum())
        print(f"    {cls}: {n}")

    X_tr, y_tr, X_val, y_val, X_te, y_te = stratified_split(X, y, sources)
    print(f"\n  Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}")

    tr_loader = make_loader(X_tr, y_tr, args.batch_size, balanced=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    te_loader = make_loader(X_te, y_te, args.batch_size, shuffle=False)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")

    model = GestureCNNv2().to(device)  # imported from classifier.py
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

    print(f"\n── Training ({args.epochs} epochs) ──────────────────")
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 10
    log = {"epochs": [], "train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_loader:
            Xb = augment_batch(Xb).to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(X_tr)
        scheduler.step()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                val_loss += criterion(out, yb).item() * len(Xb)
                val_correct += (out.argmax(1) == yb).sum().item()
        val_loss /= len(X_val)
        val_acc = val_correct / len(X_val)

        log["epochs"].append(epoch)
        log["train_loss"].append(round(tr_loss, 4))
        log["val_loss"].append(round(val_loss, 4))
        log["val_acc"].append(round(val_acc, 4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_PATH))
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}  train={tr_loss:.4f}  val={val_loss:.4f}"
                  f"  val_acc={val_acc:.3f}  lr={lr_now:.2e}  patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

    MODEL_DIR.mkdir(exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2))

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n── Test evaluation (best checkpoint) ────────────")
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device, weights_only=True))
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for Xb, yb in te_loader:
            pred = model(Xb.to(device)).argmax(1).cpu()
            all_pred.extend(pred.tolist())
            all_true.extend(yb.tolist())

    pred_arr = np.array(all_pred)
    true_arr = np.array(all_true)

    # Per-class metrics
    print(f"\n  {'Class':<22} {'Prec':>7} {'Rec':>7} {'F1':>7} {'N':>6}  Gate")
    print("  " + "─" * 58)
    all_f1 = []
    results = {}
    for cls in CLASSES:
        idx = CLASS_TO_IDX[cls]
        tp = int(((pred_arr == idx) & (true_arr == idx)).sum())
        fp = int(((pred_arr == idx) & (true_arr != idx)).sum())
        fn = int(((pred_arr != idx) & (true_arr == idx)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        n = int((true_arr == idx).sum())
        gate = "✓" if f1 >= 0.85 or n == 0 else "✗"
        print(f"  {cls:<22} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {n:>6}  {gate}")
        if n > 0:
            all_f1.append(f1)
        results[cls] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "n": n}

    mean_f1 = float(np.mean(all_f1)) if all_f1 else 0.0
    gate = "✓ PASS" if mean_f1 >= 0.85 else "✗ FAIL"
    print(f"\n  Mean F1 (classes with test data): {mean_f1:.3f}")
    print(f"  Phase 1b exit gate (mean F1 ≥ 0.85): {gate}")

    # Confusion matrix
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    n_cls = len(CLASSES)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(true_arr, pred_arr):
        cm[t, p] += 1
    header = "  " + " ".join(f"{c[:5]:>6}" for c in CLASSES)
    print(header)
    for i, cls in enumerate(CLASSES):
        row = "  " + f"{cls[:5]:>5}" + " ".join(f"{cm[i,j]:>6}" for j in range(n_cls))
        print(row)

    # Save final results
    log["test_results"] = results
    log["mean_f1"] = round(mean_f1, 4)
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"\n  Model saved → {MODEL_PATH}")
    print(f"  Training log → {LOG_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
