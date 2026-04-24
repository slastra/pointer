"""Per-frame state produced by FingerTracker.update(), consumed by render()."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrameState:
    has_hand: bool = False
    predicted: bool = False

    thumb_tip: tuple[int, int] | None = None
    index_tip: tuple[int, int] | None = None
    index_pip: tuple[int, int] | None = None
    smoothed_px: tuple[int, int] | None = None
    screen_xy: tuple[int, int] | None = None

    click_distance: float = 0.0
    is_touching: bool = False
    effective_threshold: float = 0.0

    mode: str = "hold"
    mouse_down: bool = False
    paused: bool = False

    calibration_mode: bool = False
    calibration_complete: bool = False
    calibration_bounds: dict[str, int] | None = None
    calibration_points: list[tuple[int, int]] = field(default_factory=list)

    smoothing_enabled: bool = True
    smoothing_factor: float = 0.7
    strategy: str = "one_euro"
    threshold: int = 10
    fps: float = 0.0
    gesture: str = "IDLE"

    hand_landmarks: object = None  # MediaPipe landmarks, passed through to renderer
