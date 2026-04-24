"""Position smoothing — One-Euro filter + short-horizon velocity prediction."""

from __future__ import annotations

import time

import numpy as np

from .config import Config
from .filters import OneEuroStrategy


class PositionSmoother:
    """One-Euro smoothing with velocity-based prediction when the hand is lost."""

    def __init__(
        self,
        config: Config,
        min_cutoff: float = 0.3,
        beta: float = 0.05,
        smoothing_enabled: bool = True,
    ) -> None:
        self.config = config
        self._smoothing_enabled = smoothing_enabled
        self._strategy = OneEuroStrategy(min_cutoff=min_cutoff, beta=beta)

        self.last_raw_position: tuple[int, int] | None = None
        self.last_velocity: tuple[float, float] = (0.0, 0.0)
        self.frames_without_detection = 0

    # --- runtime-mutable state ---------------------------------------------

    @property
    def smoothing_enabled(self) -> bool:
        return self._smoothing_enabled

    @property
    def min_cutoff(self) -> float:
        return self._strategy.min_cutoff

    @property
    def beta(self) -> float:
        return self._strategy.beta

    def set_min_cutoff(self, value: float) -> float:
        return self._strategy.adjust_min_cutoff(value - self._strategy.min_cutoff)

    def set_beta(self, value: float) -> float:
        return self._strategy.set_beta(value)

    def toggle_enabled(self) -> bool:
        self._smoothing_enabled = not self._smoothing_enabled
        if not self._smoothing_enabled:
            self.reset()
        return self._smoothing_enabled

    # --- state --------------------------------------------------------------

    def reset(self) -> None:
        self._strategy.reset()
        self.last_raw_position = None
        self.last_velocity = (0.0, 0.0)
        self.frames_without_detection = 0

    def update_velocity(self, x: int, y: int) -> None:
        if self.last_raw_position is not None:
            vx = x - self.last_raw_position[0]
            vy = y - self.last_raw_position[1]
            vs = self.config.velocity_smoothing
            self.last_velocity = (
                vs * self.last_velocity[0] + (1 - vs) * vx,
                vs * self.last_velocity[1] + (1 - vs) * vy,
            )
        self.last_raw_position = (x, y)
        self.frames_without_detection = 0

    def smooth_position(self, x: float, y: float) -> tuple[float, float]:
        if not self._smoothing_enabled:
            return float(x), float(y)
        return self._strategy.filter(float(x), float(y), time.monotonic())

    def predict_position(self) -> tuple[int, int] | None:
        self.frames_without_detection += 1

        if self.last_raw_position is None or not self.config.prediction_enabled:
            return None
        if self.frames_without_detection > self.config.max_frames_to_predict:
            return None

        px = self.last_raw_position[0] + self.last_velocity[0]
        py = self.last_raw_position[1] + self.last_velocity[1]

        distance = float(np.hypot(px - self.last_raw_position[0], py - self.last_raw_position[1]))
        if distance > self.config.max_prediction_distance:
            scale = self.config.max_prediction_distance / distance
            px = self.last_raw_position[0] + self.last_velocity[0] * scale
            py = self.last_raw_position[1] + self.last_velocity[1] * scale

        self.last_velocity = (
            self.last_velocity[0] * self.config.velocity_decay,
            self.last_velocity[1] * self.config.velocity_decay,
        )

        return int(px), int(py)
