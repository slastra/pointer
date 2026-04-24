"""Tests for gesture classification and debouncing."""

from __future__ import annotations

from dataclasses import dataclass

from finger_tracker.gestures import (
    FINGER_MCP_IDS,
    FINGER_TIP_IDS,
    INDEX_PIP,
    MIDDLE_PIP,
    THUMB_TIP,
    WRIST,
    Gesture,
    GestureDebouncer,
    classify,
)


@dataclass
class _LM:
    x: float
    y: float
    z: float = 0.0


class _FakeLandmarks:
    """Dict-backed stand-in for hand_landmarks.landmark (normalized coords)."""

    def __init__(self, points: dict[int, tuple[float, float]]):
        self.points = {i: _LM(*xy) for i, xy in points.items()}

    def __getitem__(self, idx: int) -> _LM:
        return self.points[idx]


def _base_hand(extended: bool = False) -> dict[int, tuple[float, float]]:
    """21 landmarks in a rough hand shape. `extended` makes fingers straight."""
    wrist = (0.5, 0.9)
    pts: dict[int, tuple[float, float]] = {WRIST: wrist, THUMB_TIP: (0.55, 0.75)}
    # MCP joints in a row across the palm, vertically above the wrist.
    mcp_y = 0.65
    mcp_xs = {5: 0.42, 9: 0.48, 13: 0.54, 17: 0.60}
    for mcp, x in mcp_xs.items():
        pts[mcp] = (x, mcp_y)
        # PIP just above MCP.
        pts[mcp + 1] = (x, mcp_y - 0.04)
        # DIP
        pts[mcp + 2] = (x, mcp_y - 0.08 if extended else mcp_y + 0.02)
        # TIP
        pts[mcp + 3] = (x, mcp_y - 0.14 if extended else mcp_y + 0.06)
    # Thumb MCP/IP/CMC filler (unused by classifier but must exist).
    pts[1] = (0.48, 0.85)
    pts[2] = (0.50, 0.82)
    pts[3] = (0.52, 0.78)
    return pts


class TestClassify:
    def test_idle_when_hand_open_but_thumb_far(self):
        lm = _FakeLandmarks(_base_hand(extended=True))
        # Open palm = 4 fingers extended → OPEN_PALM, not IDLE.
        assert classify(lm).gesture is Gesture.OPEN_PALM

    def test_pinch_l_when_thumb_near_index_pip(self):
        pts = _base_hand(extended=True)
        pts[THUMB_TIP] = (pts[INDEX_PIP][0] + 0.01, pts[INDEX_PIP][1])
        lm = _FakeLandmarks(pts)
        assert classify(lm).gesture is Gesture.PINCH_L

    def test_pinch_r_when_thumb_near_middle_pip(self):
        pts = _base_hand(extended=True)
        pts[THUMB_TIP] = (pts[MIDDLE_PIP][0] + 0.01, pts[MIDDLE_PIP][1])
        lm = _FakeLandmarks(pts)
        assert classify(lm).gesture is Gesture.PINCH_R

    def test_fist_when_all_fingers_curled(self):
        pts = _base_hand(extended=False)
        # Ensure thumb is not near PIPs (so not a pinch).
        pts[THUMB_TIP] = (0.2, 0.85)
        lm = _FakeLandmarks(pts)
        assert classify(lm).gesture is Gesture.FIST

    def test_two_finger_when_index_and_middle_extended(self):
        pts = _base_hand(extended=False)  # all curled
        # Straighten index + middle: tips far above their MCPs (above = lower y).
        for mcp in (5, 9):
            mcp_y = pts[mcp][1]
            x = pts[mcp][0]
            pts[mcp + 1] = (x, mcp_y - 0.04)  # PIP
            pts[mcp + 2] = (x, mcp_y - 0.08)  # DIP
            pts[mcp + 3] = (x, mcp_y - 0.14)  # TIP
        pts[THUMB_TIP] = (0.2, 0.85)  # thumb far from pips
        lm = _FakeLandmarks(pts)
        assert classify(lm).gesture is Gesture.TWO_FINGER

    def test_point_when_only_index_extended(self):
        pts = _base_hand(extended=False)
        mcp_y = pts[5][1]
        x = pts[5][0]
        pts[6] = (x, mcp_y - 0.04)
        pts[7] = (x, mcp_y - 0.08)
        pts[8] = (x, mcp_y - 0.14)
        pts[THUMB_TIP] = (0.2, 0.85)
        lm = _FakeLandmarks(pts)
        assert classify(lm).gesture is Gesture.POINT

    def test_hand_scale_relaxes_pinch_threshold(self):
        pts = _base_hand(extended=True)
        # Place thumb offset to the LEFT of index_pip so it's not closer to
        # middle_pip than to index_pip (else it classifies as PINCH_R).
        pts[THUMB_TIP] = (pts[INDEX_PIP][0] - 0.08, pts[INDEX_PIP][1])
        lm = _FakeLandmarks(pts)
        assert classify(lm, hand_scale=1.0).gesture is not Gesture.PINCH_L
        # Bigger hand → larger effective threshold → same pose reads as pinch.
        assert classify(lm, hand_scale=2.0).gesture is Gesture.PINCH_L


class TestDebouncer:
    def test_requires_n_matching_frames(self):
        d = GestureDebouncer(stable_frames=3)
        assert d.update(Gesture.PINCH_L) is Gesture.IDLE
        assert d.update(Gesture.PINCH_L) is Gesture.IDLE
        assert d.update(Gesture.PINCH_L) is Gesture.PINCH_L

    def test_mixed_history_keeps_latched(self):
        d = GestureDebouncer(stable_frames=3)
        for _ in range(3):
            d.update(Gesture.PINCH_L)
        assert d.latched is Gesture.PINCH_L
        # Single intruder shouldn't flip.
        d.update(Gesture.IDLE)
        assert d.latched is Gesture.PINCH_L

    def test_reset_clears(self):
        d = GestureDebouncer(stable_frames=2)
        d.update(Gesture.FIST)
        d.update(Gesture.FIST)
        assert d.latched is Gesture.FIST
        d.reset()
        assert d.latched is Gesture.IDLE


def test_constants_align():
    assert FINGER_TIP_IDS == (8, 12, 16, 20)
    assert FINGER_MCP_IDS == (5, 9, 13, 17)
