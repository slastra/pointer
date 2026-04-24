"""Immutable configuration for the finger tracker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Init-time configuration. Runtime-mutable values live on controllers."""

    # Hand detection
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    # Low value keeps tracking through partial occlusions — e.g. when the
    # user's wrist dips below the camera frame while pointing low.
    min_tracking_confidence: float = 0.1

    # Motion prediction (short-horizon only; used when hand is briefly lost)
    prediction_enabled: bool = True
    max_prediction_distance: int = 100
    max_frames_to_predict: int = 20
    velocity_decay: float = 0.9
    velocity_smoothing: float = 0.7

    # Calibration
    calibration_file: str = "calibration.json"
    # Shift the active calibration rectangle upward (in pixels) so the user
    # doesn't have to drop their hand off the bottom of the frame to reach
    # the rectangle's lower edge. Applied to both mapping and rendering.
    bounds_y_offset_px: int = 120

    # Capture resolution — frames we pull from the webcam. MediaPipe still
    # downscales to detection_{width,height} for detection regardless.
    capture_width: int = 1280
    capture_height: int = 720

    # Performance
    detection_width: int = 640  # downscale target for MediaPipe (0 = no downscale)
    detection_height: int = 360
