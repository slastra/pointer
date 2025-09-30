"""Configuration settings for the finger tracker."""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration settings for finger tracking."""

    # Hand detection settings
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.3

    # Click detection settings
    click_distance_threshold: int = 10
    click_cooldown: float = 0.3
    click_mode: str = 'hold'  # 'click' or 'hold'

    # Smoothing settings
    smoothing_enabled: bool = True
    smoothing_factor: float = 0.7  # 0-1, higher = less smoothing
    history_size: int = 3

    # Motion prediction settings
    prediction_enabled: bool = True
    max_prediction_distance: int = 100
    max_frames_to_predict: int = 20
    velocity_decay: float = 0.9
    velocity_smoothing: float = 0.7

    # Calibration settings
    calibration_file: str = "calibration.json"

    # PyAutoGUI safety
    failsafe_enabled: bool = False
    mouse_pause: float = 0