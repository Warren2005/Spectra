"""Phase 0 — verify the shared IntentEvent schema."""
import pytest
from spectra.schema import IntentEvent, Modality


def test_valid_intent():
    e = IntentEvent(intent="PAGE_NEXT", confidence=0.9, modality=Modality.GESTURE)
    assert e.intent == "PAGE_NEXT"
    assert e.modality == Modality.GESTURE
    assert 0.0 <= e.confidence <= 1.0
    assert isinstance(e.timestamp, float)
    assert e.payload == {}


def test_invalid_intent_raises():
    with pytest.raises(ValueError, match="Unknown intent"):
        IntentEvent(intent="NONEXISTENT", confidence=0.5, modality=Modality.VOICE)


def test_invalid_confidence_raises():
    with pytest.raises(ValueError, match="Confidence"):
        IntentEvent(intent="PAGE_NEXT", confidence=1.5, modality=Modality.GAZE)


def test_payload_passed_through():
    e = IntentEvent(
        intent="CURSOR_MOVE",
        confidence=0.7,
        modality=Modality.GESTURE,
        payload={"cursor": (0.4, 0.6)},
    )
    assert e.payload["cursor"] == (0.4, 0.6)


def test_all_modalities():
    for mod in Modality:
        e = IntentEvent(intent="PAGE_PREV", confidence=0.5, modality=mod)
        assert e.modality == mod
