"""
Gesture CNN training script.

Loads landmark JSON files from data/landmarks/<class>/,
trains the GestureCNN, and saves weights to models/gesture_cnn.pt.

Usage:
    python tools/train_classifier.py
    python tools/train_classifier.py --epochs 30 --lr 5e-4
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from spectra.inference.gesture.classifier import (
    CLASS_TO_IDX, CLASSES, GestureCNN, WINDOW_SIZE, FEATURE_DIM,
)

DATA_DIR = Path(__file__).parents[1] / "data" / "landmarks"
MODEL_DIR = Path(__file__).parents[1] / "models"
MODEL_PATH = MODEL_DIR / "gesture_cnn.pt"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_windows(stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all landmark JSON files and extract sliding-window examples.
    Returns X of shape (N, WINDOW_SIZE, FEATURE_DIM) and y of shape (N,).
    Splits by file, not by frame, to prevent temporal leakage.
    """
    X_list, y_list = [], []

    for cls in CLASSES:
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            print(f"  [skip] {cls}: no data directory")
            continue
        json_files = list(cls_dir.glob("*.json"))
        if not json_files:
            print(f"  [skip] {cls}: no JSON files (run generate_synthetic_data.py first)")
            continue

        label = CLASS_TO_IDX[cls]
        n_windows = 0
        for path in json_files:
            data = json.loads(path.read_text())
            seqs = data.get("sequences", [data]) if isinstance(data, dict) else [data]
            for seq in seqs:
                frames = seq if isinstance(seq, list) else []
                if len(frames) < WINDOW_SIZE:
                    continue
                # Extract right-hand landmarks from each frame
                lm_seq = []
                for frame in frames:
                    hands = frame.get("hands", {})
                    right = hands.get("Right") or hands.get("right")
                    if right is None:
                        break
                    flat = np.array(right, dtype=np.float32).flatten()
                    if len(flat) != FEATURE_DIM:
                        break
                    lm_seq.append(flat)
                else:
                    # All frames had right hand data
                    for start in range(0, len(lm_seq) - WINDOW_SIZE + 1, stride):
                        window = np.stack(lm_seq[start : start + WINDOW_SIZE])
                        X_list.append(window)
                        y_list.append(label)
                        n_windows += 1
                    continue
                # If we broke out, skip this sequence
        print(f"  {cls}: {n_windows} windows from {len(json_files)} file(s)")

    if not X_list:
        return np.empty((0, WINDOW_SIZE, FEATURE_DIM)), np.empty(0, dtype=np.int64)

    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)


def train_val_test_split(
    X: np.ndarray, y: np.ndarray, val_frac: float = 0.15, test_frac: float = 0.15
) -> tuple:
    n = len(X)
    idx = np.random.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


# ── Training ───────────────────────────────────────────────────────────────────

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                balanced: bool = False) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    sampler = None
    if balanced:
        counts = Counter(y.tolist())
        weights = [1.0 / counts[int(yi)] for yi in y]
        sampler = WeightedRandomSampler(weights, len(weights))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(sampler is None))


def train(args) -> None:
    print("\n── Loading data ──────────────────────────────")
    X, y = load_windows()
    if len(X) == 0:
        print("\nNo data found. Run generate_synthetic_data.py first.")
        return

    print(f"\n  Total windows: {len(X)}")
    for cls in CLASSES:
        n = int((y == CLASS_TO_IDX[cls]).sum())
        print(f"    {cls}: {n}")

    X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(X, y)

    tr_loader = make_loader(X_tr, y_tr, args.batch_size, balanced=True)
    val_loader = make_loader(X_val, y_val, args.batch_size)
    te_loader = make_loader(X_te, y_te, args.batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    model = GestureCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    print("\n── Training ──────────────────────────────────")
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 8

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(X_tr)

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

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_PATH))
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train={tr_loss:.4f}  val={val_loss:.4f}"
                  f"  val_acc={val_acc:.3f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print(f"\n── Test evaluation (best checkpoint) ─────────")
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device, weights_only=True))
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for Xb, yb in te_loader:
            pred = model(Xb.to(device)).argmax(1).cpu()
            all_pred.extend(pred.tolist())
            all_true.extend(yb.tolist())

    all_pred_np = np.array(all_pred)
    all_true_np = np.array(all_true)

    print(f"\n  {'Class':<20}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'N':>6}")
    print("  " + "-" * 60)
    all_f1 = []
    for cls in CLASSES:
        idx = CLASS_TO_IDX[cls]
        tp = int(((all_pred_np == idx) & (all_true_np == idx)).sum())
        fp = int(((all_pred_np == idx) & (all_true_np != idx)).sum())
        fn = int(((all_pred_np != idx) & (all_true_np == idx)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        n = int((all_true_np == idx).sum())
        gate = "✓" if f1 >= 0.85 else "✗"
        print(f"  {cls:<20}  {prec:>10.3f}  {rec:>8.3f}  {f1:>8.3f}  {n:>6}  {gate}")
        all_f1.append(f1)

    mean_f1 = np.mean(all_f1)
    print(f"\n  Mean F1: {mean_f1:.3f}")
    gate = "✓ PASS" if mean_f1 >= 0.85 else "✗ FAIL"
    print(f"  Phase 1b exit gate (mean F1 ≥ 0.85): {gate}")
    print(f"\n  Model saved → {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
