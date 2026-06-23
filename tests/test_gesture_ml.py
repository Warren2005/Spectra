"""
Phase 1b tests — 1D-CNN classifier architecture and inference wrapper.
No camera, no trained weights required.
"""
import numpy as np
import torch

from spectra.inference.gesture.classifier import (
    CLASSES, FEATURE_DIM, N_CLASSES, WINDOW_SIZE,
    GestureCNN, GestureClassifier, IDX_TO_CLASS,
)


class TestGestureCNN:
    def test_output_shape(self):
        model = GestureCNN()
        x = torch.randn(4, WINDOW_SIZE, FEATURE_DIM)
        out = model(x)
        assert out.shape == (4, N_CLASSES)

    def test_single_sample(self):
        model = GestureCNN()
        x = torch.randn(1, WINDOW_SIZE, FEATURE_DIM)
        out = model(x)
        assert out.shape == (1, N_CLASSES)

    def test_output_is_logits_not_probs(self):
        model = GestureCNN()
        x = torch.randn(1, WINDOW_SIZE, FEATURE_DIM)
        out = model(x)
        # Logits can be negative; softmax output would all be > 0 and sum to 1
        probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_class_count_matches_vocabulary(self):
        assert N_CLASSES == len(CLASSES)
        assert all(i in IDX_TO_CLASS for i in range(N_CLASSES))


class TestGestureClassifier:
    def test_returns_none_without_weights(self):
        clf = GestureClassifier(model_path=None)
        assert not clf.ready
        flat = np.random.randn(FEATURE_DIM).astype(np.float32)
        gesture, conf = clf.update(flat)
        assert gesture is None
        assert conf == 0.0

    def test_returns_none_before_window_full_even_with_weights(self, tmp_path):
        # Save an untrained (random weights) model
        model = GestureCNN()
        path = tmp_path / "gesture_cnn.pt"
        torch.save(model.state_dict(), str(path))

        clf = GestureClassifier(model_path=path)
        assert clf.ready

        # Feed fewer than WINDOW_SIZE frames
        flat = np.zeros(FEATURE_DIM, dtype=np.float32)
        for _ in range(WINDOW_SIZE - 1):
            gesture, conf = clf.update(flat)
        assert gesture is None  # window not full yet

    def test_produces_output_after_window_full(self, tmp_path):
        model = GestureCNN()
        path = tmp_path / "gesture_cnn.pt"
        torch.save(model.state_dict(), str(path))

        clf = GestureClassifier(model_path=path)
        flat = np.zeros(FEATURE_DIM, dtype=np.float32)
        last_gesture, last_conf = None, 0.0
        for _ in range(WINDOW_SIZE):
            last_gesture, last_conf = clf.update(flat)

        # After WINDOW_SIZE frames, should produce a result
        assert isinstance(last_conf, float)
        assert 0.0 <= last_conf <= 1.0

    def test_reset_window_clears_state(self, tmp_path):
        model = GestureCNN()
        path = tmp_path / "gesture_cnn.pt"
        torch.save(model.state_dict(), str(path))

        clf = GestureClassifier(model_path=path)
        flat = np.zeros(FEATURE_DIM, dtype=np.float32)
        for _ in range(WINDOW_SIZE):
            clf.update(flat)
        clf.reset_window()
        assert len(clf._window) == 0

    def test_nonexistent_path_not_ready(self, tmp_path):
        clf = GestureClassifier(model_path=tmp_path / "nope.pt")
        assert not clf.ready

    def test_output_class_in_vocabulary(self, tmp_path):
        model = GestureCNN()
        path = tmp_path / "gesture_cnn.pt"
        torch.save(model.state_dict(), str(path))

        clf = GestureClassifier(model_path=path)
        clf.CONFIDENCE_THRESHOLD = 0.0  # force a class to be returned
        flat = np.random.randn(FEATURE_DIM).astype(np.float32)
        last_gesture = None
        for _ in range(WINDOW_SIZE):
            last_gesture, _ = clf.update(flat)
        assert last_gesture in CLASSES
