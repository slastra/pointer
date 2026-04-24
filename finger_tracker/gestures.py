"""Gesture classifier.

Standard mapping (per user plan):
  - PINCH_L: thumb tip ↔ index PIP close  → left click
  - PINCH_R: thumb tip ↔ middle PIP close → right click
  - FIST_SCROLL: all fingers curled, palm vertical motion → scroll
  - PAUSE_TOGGLE: open palm (all fingers extended) held ≥1s → toggles paused
  - IDLE: none of the above

Each gesture requires N stable frames to latch (via GestureDebouncer).
Classification is pure — no state — so it is trivially testable with
fabricated landmark fixtures.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol


class Gesture(Enum):
    IDLE = auto()
    PINCH_L = auto()
    PINCH_R = auto()
    FIST = auto()
    OPEN_PALM = auto()
    TWO_FINGER = auto()  # index + middle extended, ring + pinky curled → scroll
    POINT = auto()  # only index extended → cursor tracking


# MediaPipe hand landmark IDs (21 total).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

FINGER_TIP_IDS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_MCP_IDS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


class Landmarks(Protocol):
    """Minimal shape of MediaPipe hand_landmarks.landmark[i] objects (.x/.y/.z normalized)."""

    def __getitem__(self, idx: int) -> object: ...


def _dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _finger_extended(lm, tip_id: int, mcp_id: int, wrist) -> bool:
    """Finger extended iff MCP→PIP and PIP→TIP point in similar directions.

    Uses the cosine of the angle between the proximal (MCP→PIP) and distal
    (PIP→TIP) segments of the finger: ~0° when straight, ~180° when curled.
    Orientation-robust (works whether the hand points up, sideways, etc.),
    unlike a simple tip-vs-MCP Euclidean distance.

    Assumes MediaPipe's 4-landmarks-per-finger layout: mcp, pip=mcp+1, dip=mcp+2,
    tip=mcp+3.
    """
    del wrist  # retained in signature for back-compat, unused now
    pip_id = mcp_id + 1
    ax = lm[pip_id].x - lm[mcp_id].x
    ay = lm[pip_id].y - lm[mcp_id].y
    bx = lm[tip_id].x - lm[pip_id].x
    by = lm[tip_id].y - lm[pip_id].y
    mag_a = (ax * ax + ay * ay) ** 0.5
    mag_b = (bx * bx + by * by) ** 0.5
    if mag_a < 1e-6 or mag_b < 1e-6:
        return False
    cos_angle = (ax * bx + ay * by) / (mag_a * mag_b)
    # cos > 0.5 → angle < 60° → finger roughly straight.
    return cos_angle > 0.5


@dataclass
class GestureResult:
    gesture: Gesture
    confidence: float = 1.0
    metadata: dict[str, float] | None = None


def classify(landmark_list, hand_scale: float = 1.0, pinch_threshold: float = 0.06) -> GestureResult:
    """Classify a single hand landmark frame into a Gesture.

    `landmark_list` supports [i] indexing returning .x/.y normalized MediaPipe
    landmarks. `hand_scale` rescales the pinch threshold so near/far hands
    behave the same (threshold is in normalized-coordinate units).
    """
    lm = landmark_list
    wrist = lm[WRIST]
    thumb = lm[THUMB_TIP]

    thumb_index = _dist(thumb, lm[INDEX_PIP])
    thumb_middle = _dist(thumb, lm[MIDDLE_PIP])

    scaled_pinch = pinch_threshold * hand_scale

    extended = {
        tip: _finger_extended(lm, tip, mcp, wrist)
        for tip, mcp in zip(FINGER_TIP_IDS, FINGER_MCP_IDS, strict=True)
    }
    fingers_extended = sum(1 for v in extended.values() if v)

    # Pinches take precedence over pose-based classifications (OPEN_PALM/FIST)
    # since a pinch naturally keeps the non-pinching fingers extended.
    if thumb_middle < scaled_pinch and thumb_middle < thumb_index:
        return GestureResult(Gesture.PINCH_R, metadata={"distance": thumb_middle})

    if thumb_index < scaled_pinch:
        return GestureResult(Gesture.PINCH_L, metadata={"distance": thumb_index})

    # Two-finger peace sign: index + middle extended, ring + pinky curled → scroll.
    if (
        extended[INDEX_TIP]
        and extended[MIDDLE_TIP]
        and not extended[RING_TIP]
        and not extended[PINKY_TIP]
    ):
        return GestureResult(Gesture.TWO_FINGER)

    # Pointing: only index extended → cursor tracking.
    if (
        extended[INDEX_TIP]
        and not extended[MIDDLE_TIP]
        and not extended[RING_TIP]
        and not extended[PINKY_TIP]
    ):
        return GestureResult(Gesture.POINT)

    if fingers_extended == 0 and thumb_index > scaled_pinch and thumb_middle > scaled_pinch:
        return GestureResult(Gesture.FIST)

    if fingers_extended == 4:
        return GestureResult(Gesture.OPEN_PALM)

    return GestureResult(Gesture.IDLE)


class GestureDebouncer:
    """Latches a gesture only after N consecutive matching classifications."""

    def __init__(self, stable_frames: int = 3) -> None:
        self.stable_frames = stable_frames
        self._history: deque[Gesture] = deque(maxlen=stable_frames)
        self._latched = Gesture.IDLE

    def update(self, gesture: Gesture) -> Gesture:
        self._history.append(gesture)
        if len(self._history) == self.stable_frames and len(set(self._history)) == 1:
            self._latched = gesture
        return self._latched

    @property
    def latched(self) -> Gesture:
        return self._latched

    def reset(self) -> None:
        self._history.clear()
        self._latched = Gesture.IDLE
