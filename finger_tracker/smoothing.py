"""Position smoothing and prediction for reducing jitter."""

import numpy as np
from collections import deque
from typing import Optional, Tuple

from .config import Config


class PositionSmoother:
    """Handles position smoothing and motion prediction."""

    def __init__(self, config: Config):
        """
        Initialize the position smoother.

        Args:
            config: Configuration object with smoothing settings
        """
        self.config = config

        # Smoothing state
        self.position_history = deque(maxlen=config.history_size)
        self.last_smooth_position: Optional[Tuple[float, float]] = None

        # Prediction state
        self.last_raw_position: Optional[Tuple[int, int]] = None
        self.last_velocity = (0.0, 0.0)
        self.frames_without_detection = 0

    def reset(self):
        """Reset all smoothing state."""
        self.position_history.clear()
        self.last_smooth_position = None
        self.last_raw_position = None
        self.last_velocity = (0.0, 0.0)
        self.frames_without_detection = 0

    def update_velocity(self, x: int, y: int):
        """
        Update velocity based on position change.

        Args:
            x: Current X coordinate
            y: Current Y coordinate
        """
        if self.last_raw_position is not None:
            velocity_x = x - self.last_raw_position[0]
            velocity_y = y - self.last_raw_position[1]
            # Smooth velocity to reduce noise
            self.last_velocity = (
                self.config.velocity_smoothing * self.last_velocity[0] + (1 - self.config.velocity_smoothing) * velocity_x,
                self.config.velocity_smoothing * self.last_velocity[1] + (1 - self.config.velocity_smoothing) * velocity_y
            )
        self.last_raw_position = (x, y)
        self.frames_without_detection = 0

    def smooth_position(self, x: int, y: int) -> Tuple[int, int]:
        """
        Apply smoothing to reduce jitter in cursor movement.

        Args:
            x: Current X coordinate
            y: Current Y coordinate

        Returns:
            Smoothed (x, y) coordinates
        """
        if not self.config.smoothing_enabled:
            return x, y

        # Add current position to history
        self.position_history.append((x, y))

        # Moving average smoothing
        if len(self.position_history) > 0:
            avg_x = sum(p[0] for p in self.position_history) / len(self.position_history)
            avg_y = sum(p[1] for p in self.position_history) / len(self.position_history)

            # Exponential smoothing on top of moving average
            if self.last_smooth_position is not None:
                # Give more weight to current position to reduce lag
                smooth_x = self.config.smoothing_factor * avg_x + (1 - self.config.smoothing_factor) * self.last_smooth_position[0]
                smooth_y = self.config.smoothing_factor * avg_y + (1 - self.config.smoothing_factor) * self.last_smooth_position[1]
            else:
                smooth_x, smooth_y = avg_x, avg_y

            self.last_smooth_position = (smooth_x, smooth_y)
            return int(smooth_x), int(smooth_y)

        return x, y

    def predict_position(self) -> Optional[Tuple[int, int]]:
        """
        Predict position when hand is temporarily lost.

        Returns:
            Predicted (x, y) coordinates or None if prediction unavailable
        """
        self.frames_without_detection += 1

        if self.last_raw_position is None or not self.config.prediction_enabled:
            return None

        if self.frames_without_detection > self.config.max_frames_to_predict:
            return None

        # Use velocity to predict next position
        predicted_x = self.last_raw_position[0] + self.last_velocity[0]
        predicted_y = self.last_raw_position[1] + self.last_velocity[1]

        # Limit prediction distance
        distance = self._calculate_distance(self.last_raw_position, (predicted_x, predicted_y))
        if distance > self.config.max_prediction_distance:
            # Scale down to max distance
            scale = self.config.max_prediction_distance / distance
            predicted_x = self.last_raw_position[0] + self.last_velocity[0] * scale
            predicted_y = self.last_raw_position[1] + self.last_velocity[1] * scale

        # Decay velocity over time (friction)
        self.last_velocity = (
            self.last_velocity[0] * self.config.velocity_decay,
            self.last_velocity[1] * self.config.velocity_decay
        )

        return int(predicted_x), int(predicted_y)

    @staticmethod
    def _calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)