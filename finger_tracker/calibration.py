"""Calibration — maps hand coordinates to screen coordinates.

Bounds are stored NORMALIZED (0..1 in frame space) so the rectangle is
resolution-independent. Legacy pixel-format files are migrated on load.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# Assumed capture resolution of legacy calibration files (pre-normalization).
_LEGACY_FRAME_W = 640
_LEGACY_FRAME_H = 480


class CalibrationManager:
    """Manages calibration for mapping hand coordinates to screen coordinates."""

    def __init__(
        self,
        calibration_file: str,
        screen_width: int,
        screen_height: int,
        y_offset: int = 0,
    ) -> None:
        self.calibration_file = calibration_file
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Shift active rectangle upward (in pixels) so the user's wrist stays
        # in frame when pointing at the bottom edge.
        self.y_offset = y_offset

        self.calibration_mode = False
        self.calibration_points: list[tuple[int, int]] = []
        self.calibration_complete = False
        # Normalized bounds: keys = min_x, max_x, min_y, max_y in [0..1].
        self.bounds_norm: dict[str, float] | None = None
        # Current frame size — set by the tracker each frame so we can
        # convert normalized bounds to pixel-space on demand.
        self._frame_w: int = _LEGACY_FRAME_W
        self._frame_h: int = _LEGACY_FRAME_H

        self.load()

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_w = width
        self._frame_h = height

    # --- bounds access ------------------------------------------------------

    @property
    def calibration_bounds(self) -> dict[str, int] | None:
        """Pixel-space bounds for the CURRENT frame size (no y-offset)."""
        if self.bounds_norm is None:
            return None
        w, h = self._frame_w, self._frame_h
        return {
            "min_x": int(self.bounds_norm["min_x"] * w),
            "max_x": int(self.bounds_norm["max_x"] * w),
            "min_y": int(self.bounds_norm["min_y"] * h),
            "max_y": int(self.bounds_norm["max_y"] * h),
        }

    @property
    def effective_bounds(self) -> dict[str, int] | None:
        """Pixel bounds with y-offset applied — what's drawn and mapped."""
        b = self.calibration_bounds
        if b is None:
            return None
        return {
            "min_x": b["min_x"],
            "max_x": b["max_x"],
            "min_y": b["min_y"] - self.y_offset,
            "max_y": b["max_y"] - self.y_offset,
        }

    # --- persistence --------------------------------------------------------

    def save(self) -> bool:
        if self.bounds_norm:
            try:
                with open(self.calibration_file, "w") as f:
                    json.dump(self.bounds_norm, f)
                logger.info("Calibration saved (normalized)")
                return True
            except OSError as e:
                logger.warning("Failed to save calibration: %s", e)
                return False
        return False

    def load(self) -> bool:
        if not os.path.exists(self.calibration_file):
            return False
        try:
            with open(self.calibration_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load calibration file: %s", e)
            return False

        # Heuristic: normalized values are all in [0, 1]. Legacy pixel values
        # will be > 1 (e.g. min_x=111). Migrate legacy on load.
        try:
            if all(0.0 <= float(v) <= 1.0 for v in data.values()):
                self.bounds_norm = {k: float(v) for k, v in data.items()}
                logger.info("Calibration loaded (normalized) from %s", self.calibration_file)
            else:
                self.bounds_norm = {
                    "min_x": data["min_x"] / _LEGACY_FRAME_W,
                    "max_x": data["max_x"] / _LEGACY_FRAME_W,
                    "min_y": data["min_y"] / _LEGACY_FRAME_H,
                    "max_y": data["max_y"] / _LEGACY_FRAME_H,
                }
                logger.info(
                    "Calibration migrated from legacy pixel format (%dx%d base)",
                    _LEGACY_FRAME_W,
                    _LEGACY_FRAME_H,
                )
                self.save()
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Calibration file unrecognized, ignoring: %s", e)
            return False

        self.calibration_complete = True
        return True

    # --- workflow -----------------------------------------------------------

    def start(self) -> None:
        self.calibration_mode = True
        self.calibration_points = []
        self.calibration_complete = False
        logger.info("Calibration started — move index to 4 corners, SPACE to record, ESC to cancel")

    def add_point(self, x: int, y: int) -> None:
        self.calibration_points.append((x, y))
        logger.info("Calibration point %d: (%d, %d)", len(self.calibration_points), x, y)

        if len(self.calibration_points) >= 4:
            w, h = max(1, self._frame_w), max(1, self._frame_h)
            xs = [p[0] / w for p in self.calibration_points]
            ys = [p[1] / h for p in self.calibration_points]

            self.bounds_norm = {
                "min_x": min(xs),
                "max_x": max(xs),
                "min_y": min(ys),
                "max_y": max(ys),
            }

            self.calibration_complete = True
            self.calibration_mode = False
            self.save()
            logger.info(
                "Calibration complete: X[%.2f,%.2f] Y[%.2f,%.2f]",
                self.bounds_norm["min_x"],
                self.bounds_norm["max_x"],
                self.bounds_norm["min_y"],
                self.bounds_norm["max_y"],
            )

    def cancel(self) -> None:
        self.calibration_mode = False
        self.calibration_points = []
        logger.info("Calibration cancelled")

    def delete(self) -> None:
        self.bounds_norm = None
        self.calibration_complete = False
        if os.path.exists(self.calibration_file):
            os.remove(self.calibration_file)
            logger.info("Calibration deleted")

    # --- mapping ------------------------------------------------------------

    def map_to_screen(
        self, x: float, y: float, frame_width: int, frame_height: int
    ) -> tuple[int, int]:
        # Make sure bounds reflect the current frame dimensions.
        self.set_frame_size(frame_width, frame_height)

        b = self.effective_bounds
        if self.calibration_complete and b is not None:
            cal_w = b["max_x"] - b["min_x"]
            cal_h = b["max_y"] - b["min_y"]
            if cal_w > 0 and cal_h > 0:
                norm_x = (x - b["min_x"]) / cal_w
                norm_y = (y - b["min_y"]) / cal_h
                screen_x = int(norm_x * self.screen_width)
                screen_y = int(norm_y * self.screen_height)
            else:
                screen_x = int((x / frame_width) * self.screen_width)
                screen_y = int((y / frame_height) * self.screen_height)
        else:
            screen_x = int((x / frame_width) * self.screen_width)
            screen_y = int((y / frame_height) * self.screen_height)

        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        return screen_x, screen_y
