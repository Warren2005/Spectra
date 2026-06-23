from dataclasses import dataclass, field
from enum import Enum
import time


class Modality(Enum):
    GESTURE = "gesture"
    GAZE = "gaze"
    VOICE = "voice"


# All valid intent names. Document engine and fusion layer only see these strings.
INTENTS = frozenset({
    "PAGE_NEXT",
    "PAGE_PREV",
    "HIGHLIGHT_ANCHOR",
    "HIGHLIGHT_CONFIRM",
    "HIGHLIGHT_MODE_TOGGLE",
    "CURSOR_MOVE",
    "BOOKMARK_ADD",
    "CODE_COPY",
    "ZOOM_IN",
    "ZOOM_OUT",
    "UNDO_LAST",
    "SCROLL_DOWN",
    "SCROLL_UP",
})


@dataclass
class IntentEvent:
    intent: str
    confidence: float          # 0–1, pipeline self-estimate
    modality: Modality
    timestamp: float = field(default_factory=time.time)
    payload: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.intent not in INTENTS:
            raise ValueError(f"Unknown intent: {self.intent!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
