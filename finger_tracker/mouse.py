"""Mouse control — click pipeline (hysteresis + debounce) over a backend."""

from __future__ import annotations

import logging
import time
from collections import deque

from .backends import MouseBackend
from .config import Config

logger = logging.getLogger(__name__)


class MouseController:
    MIN_THRESHOLD = 10
    MAX_THRESHOLD = 150
    RELEASE_DELTA = 4  # release threshold = press + delta (hysteresis)
    DEBOUNCE_FRAMES = 3

    def __init__(
        self,
        backend: MouseBackend,
        config: Config,
        click_distance_threshold: int = 10,
        click_cooldown: float = 0.3,
        click_mode: str = "click",
    ) -> None:
        self.config = config
        self.backend = backend
        logger.info("Mouse backend: %s", self.backend.name)

        self._threshold = click_distance_threshold
        self._cooldown = click_cooldown
        self._mode = click_mode

        self.mouse_down = False
        self.last_click_time = 0.0

        self._touch_history: deque[bool] = deque(maxlen=self.DEBOUNCE_FRAMES)
        self._debounced_touching = False

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def mode(self) -> str:
        return self._mode

    def set_threshold(self, value: int) -> int:
        self._threshold = max(self.MIN_THRESHOLD, min(self.MAX_THRESHOLD, value))
        return self._threshold

    def adjust_threshold(self, delta: int) -> int:
        return self.set_threshold(self._threshold + delta)

    def toggle_mode(self) -> str:
        self.reset()
        self._mode = "click" if self._mode == "hold" else "hold"
        logger.info("Click mode: %s", self._mode.upper())
        return self._mode

    def effective_threshold(self, hand_scale: float = 1.0) -> float:
        return self._threshold * hand_scale

    def update_touch(self, click_distance: float, hand_scale: float = 1.0) -> tuple[bool, float]:
        eff = self.effective_threshold(hand_scale)
        press = eff
        release = eff + self.RELEASE_DELTA

        instant = click_distance < release if self._debounced_touching else click_distance < press
        self._touch_history.append(instant)

        if len(self._touch_history) == self.DEBOUNCE_FRAMES:
            if all(self._touch_history):
                self._debounced_touching = True
            elif not any(self._touch_history):
                self._debounced_touching = False

        return self._debounced_touching, eff

    def move_to(self, x: int, y: int) -> None:
        try:
            self.backend.move_to(x, y)
        except OSError as e:
            logger.warning("Failed to move mouse: %s", e)

    def scroll(self, amount: int) -> None:
        try:
            self.backend.scroll(amount)
        except OSError as e:
            logger.warning("Failed to scroll: %s", e)

    def right_click(self) -> None:
        try:
            self.backend.right_click()
            logger.info("Right click")
        except OSError as e:
            logger.warning("Failed to right-click: %s", e)

    def handle_click(self, is_touching: bool, x: int, y: int) -> bool:
        current_time = time.time()

        if is_touching:
            if not self.mouse_down:
                try:
                    if self._mode == "hold":
                        self.backend.mouse_down()
                        logger.info("Mouse DOWN at (%d, %d)", x, y)
                    elif (current_time - self.last_click_time) > self._cooldown:
                        self.backend.click()
                        self.last_click_time = current_time
                        logger.info("Click at (%d, %d)", x, y)
                    self.mouse_down = True
                except OSError as e:
                    logger.warning("Failed to click: %s", e)
        else:
            if self.mouse_down and self._mode == "hold":
                try:
                    self.backend.mouse_up()
                    logger.info("Mouse UP at (%d, %d)", x, y)
                except OSError as e:
                    logger.warning("Failed to release mouse: %s", e)
            self.mouse_down = False

        return self.mouse_down

    def reset(self) -> None:
        if self.mouse_down:
            try:
                self.backend.mouse_up()
            except OSError as e:
                logger.warning("Failed to release mouse: %s", e)
        self.mouse_down = False
        self.last_click_time = 0.0
        self._touch_history.clear()
        self._debounced_touching = False

    def close(self) -> None:
        self.backend.close()
