"""Mouse control using PyAutoGUI."""

import pyautogui
import time

from .config import Config


class MouseController:
    """Handles mouse movement and clicking."""

    def __init__(self, config: Config):
        """
        Initialize the mouse controller.

        Args:
            config: Configuration object with mouse settings
        """
        self.config = config

        # Configure PyAutoGUI
        pyautogui.FAILSAFE = config.failsafe_enabled
        pyautogui.PAUSE = config.mouse_pause

        # Mouse state
        self.mouse_down = False
        self.last_click_time = 0

    def move_to(self, x: int, y: int):
        """
        Move mouse to position.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        try:
            pyautogui.moveTo(x, y, duration=0)
        except Exception as e:
            print(f"Failed to move mouse: {e}")

    def handle_click(self, is_touching: bool, x: int, y: int) -> bool:
        """
        Handle click based on finger touch state.

        Args:
            is_touching: Whether fingers are touching
            x: Current X coordinate
            y: Current Y coordinate

        Returns:
            True if mouse is currently down, False otherwise
        """
        current_time = time.time()

        if is_touching:
            # Fingers are touching - mouse should be down
            if not self.mouse_down:
                try:
                    if self.config.click_mode == 'hold':
                        pyautogui.mouseDown()
                        print(f"Mouse DOWN at ({x}, {y})")
                    else:  # click mode
                        if (current_time - self.last_click_time) > self.config.click_cooldown:
                            pyautogui.click()
                            self.last_click_time = current_time
                            print(f"Click at ({x}, {y})")
                    self.mouse_down = True
                except Exception as e:
                    print(f"Failed to click: {e}")
        else:
            # Fingers are apart - mouse should be up
            if self.mouse_down and self.config.click_mode == 'hold':
                try:
                    pyautogui.mouseUp()
                    print(f"Mouse UP at ({x}, {y})")
                except Exception as e:
                    print(f"Failed to release mouse: {e}")
            self.mouse_down = False

        return self.mouse_down

    def reset(self):
        """Reset mouse state (release if pressed)."""
        if self.mouse_down:
            try:
                pyautogui.mouseUp()
            except Exception as e:
                print(f"Failed to release mouse: {e}")
        self.mouse_down = False
        self.last_click_time = 0

    def toggle_mode(self):
        """Toggle between click and hold modes."""
        if self.config.click_mode == 'hold':
            self.config.click_mode = 'click'
            # Make sure to release mouse if it was down
            self.reset()
        else:
            self.config.click_mode = 'hold'
        print(f"Click mode changed to: {self.config.click_mode.upper()}")

    def adjust_threshold(self, adjustment: int):
        """
        Adjust click distance threshold.

        Args:
            adjustment: Amount to adjust (positive or negative)
        """
        self.config.click_distance_threshold = max(10, self.config.click_distance_threshold + adjustment)
        print(f"Click threshold adjusted to {self.config.click_distance_threshold}")