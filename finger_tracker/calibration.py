"""Calibration management for coordinate mapping."""

import json
import os
from typing import Optional, Tuple, Dict


class CalibrationManager:
    """Manages calibration for mapping hand coordinates to screen coordinates."""

    def __init__(self, calibration_file: str, screen_width: int, screen_height: int):
        """
        Initialize the calibration manager.

        Args:
            calibration_file: Path to the calibration JSON file
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels
        """
        self.calibration_file = calibration_file
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_complete = False
        self.calibration_bounds: Optional[Dict[str, int]] = None

        # Load saved calibration if exists
        self.load()

    def save(self) -> bool:
        """Save calibration bounds to file."""
        if self.calibration_bounds:
            try:
                with open(self.calibration_file, 'w') as f:
                    json.dump(self.calibration_bounds, f)
                print("Calibration saved!")
                return True
            except Exception as e:
                print(f"Failed to save calibration: {e}")
                return False
        return False

    def load(self) -> bool:
        """Load calibration bounds from file."""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    self.calibration_bounds = json.load(f)
                self.calibration_complete = True
                print("Calibration loaded from file!")
                return True
            except Exception as e:
                print(f"Failed to load calibration file: {e}")
        return False

    def start(self):
        """Start calibration mode."""
        self.calibration_mode = True
        self.calibration_points = []
        self.calibration_complete = False
        print("\n=== CALIBRATION MODE ===")
        print("Move your index finger to the corners of your movement area")
        print("Press SPACE to record each corner (need 4 corners)")
        print("Press ESC to cancel calibration")

    def add_point(self, x: int, y: int):
        """
        Add a calibration point.

        Args:
            x: X coordinate in frame
            y: Y coordinate in frame
        """
        self.calibration_points.append((x, y))
        print(f"Calibration point {len(self.calibration_points)} recorded: ({x}, {y})")

        if len(self.calibration_points) >= 4:
            # Calculate bounds from recorded points
            xs = [p[0] for p in self.calibration_points]
            ys = [p[1] for p in self.calibration_points]

            self.calibration_bounds = {
                'min_x': min(xs),
                'max_x': max(xs),
                'min_y': min(ys),
                'max_y': max(ys)
            }

            self.calibration_complete = True
            self.calibration_mode = False
            self.save()
            print("Calibration complete!")
            print(f"Movement area: X[{self.calibration_bounds['min_x']}, {self.calibration_bounds['max_x']}], "
                  f"Y[{self.calibration_bounds['min_y']}, {self.calibration_bounds['max_y']}]")

    def cancel(self):
        """Cancel calibration mode."""
        self.calibration_mode = False
        self.calibration_points = []
        print("Calibration cancelled")

    def delete(self):
        """Delete saved calibration."""
        self.calibration_bounds = None
        self.calibration_complete = False
        if os.path.exists(self.calibration_file):
            os.remove(self.calibration_file)
            print("Calibration deleted!")

    def map_to_screen(self, x: int, y: int, frame_width: int, frame_height: int) -> Tuple[int, int]:
        """
        Map hand coordinates to screen coordinates.

        Args:
            x: X coordinate in frame
            y: Y coordinate in frame
            frame_width: Width of the frame
            frame_height: Height of the frame

        Returns:
            Tuple of (screen_x, screen_y)
        """
        if self.calibration_complete and self.calibration_bounds:
            # Map from calibration bounds to screen
            cal_width = self.calibration_bounds['max_x'] - self.calibration_bounds['min_x']
            cal_height = self.calibration_bounds['max_y'] - self.calibration_bounds['min_y']

            if cal_width > 0 and cal_height > 0:
                # Normalize position within calibration bounds
                norm_x = (x - self.calibration_bounds['min_x']) / cal_width
                norm_y = (y - self.calibration_bounds['min_y']) / cal_height

                # Map to screen coordinates
                screen_x = int(norm_x * self.screen_width)
                screen_y = int(norm_y * self.screen_height)
            else:
                # Fallback to full frame mapping
                screen_x = int((x / frame_width) * self.screen_width)
                screen_y = int((y / frame_height) * self.screen_height)
        else:
            # No calibration, use full frame
            screen_x = int((x / frame_width) * self.screen_width)
            screen_y = int((y / frame_height) * self.screen_height)

        # Clamp to screen bounds
        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))

        return screen_x, screen_y