"""Main finger tracker application."""

import cv2
import pyautogui

from .config import Config
from .calibration import CalibrationManager
from .smoothing import PositionSmoother
from .detection import HandDetector
from .mouse import MouseController


class FingerTracker:
    """Main application that orchestrates hand tracking and mouse control."""

    def __init__(self, config: Config = None):
        """
        Initialize the finger tracker.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()

        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()

        # Initialize components
        self.detector = HandDetector(self.config)
        self.smoother = PositionSmoother(self.config)
        self.calibration = CalibrationManager(
            self.config.calibration_file,
            screen_width,
            screen_height
        )
        self.mouse = MouseController(self.config)

        # State for calibration UI
        self.current_index_position = None

    def process_frame(self, frame):
        """
        Process a single frame.

        Args:
            frame: BGR image frame from OpenCV

        Returns:
            Processed frame with visualizations
        """
        frame = cv2.flip(frame, 1)

        # Draw calibration area if calibrated
        if self.calibration.calibration_complete and self.calibration.calibration_bounds:
            bounds = self.calibration.calibration_bounds
            cv2.rectangle(frame,
                         (bounds['min_x'], bounds['min_y']),
                         (bounds['max_x'], bounds['max_y']),
                         (0, 255, 0), 2)

        # Draw calibration points during calibration
        if self.calibration.calibration_mode:
            for i, point in enumerate(self.calibration.calibration_points):
                cv2.circle(frame, point, 8, (0, 0, 255), -1)
                cv2.putText(frame, f"{i+1}", (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.putText(frame, f"Calibrating: {len(self.calibration.calibration_points)}/4 points (SPACE to mark)",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        # Detect hands
        results = self.detector.detect(frame)

        if results.multi_hand_landmarks:
            self._process_hand_detected(frame, results)
        else:
            self._process_hand_lost(frame)

        # Draw controls at bottom
        self._draw_controls(frame)

        return frame

    def _process_hand_detected(self, frame, results):
        """Process frame when hand is detected."""
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        self.detector.draw_landmarks(frame, hand_landmarks)

        # Get key points
        thumb_tip = self.detector.get_landmark_position(hand_landmarks, 4, frame.shape)
        index_tip = self.detector.get_landmark_position(hand_landmarks, 8, frame.shape)
        index_pip = self.detector.get_landmark_position(hand_landmarks, 6, frame.shape)

        # Update velocity for prediction
        self.smoother.update_velocity(index_tip[0], index_tip[1])

        # Draw markers
        cv2.circle(frame, thumb_tip, 10, (255, 0, 0), -1)  # Blue for thumb
        cv2.circle(frame, index_tip, 10, (0, 255, 0), -1)  # Green for index tip
        cv2.circle(frame, index_pip, 8, (255, 165, 0), -1)  # Orange for click target

        # Store current index position for calibration
        if self.calibration.calibration_mode:
            self.current_index_position = index_tip

        # Calculate distance between thumb tip and index first knuckle (PIP joint)
        click_distance = self.detector.calculate_distance(thumb_tip, index_pip)

        # Draw line between thumb and click target
        cv2.line(frame, thumb_tip, index_pip, (0, 255, 255), 2)

        # Apply smoothing to index finger position
        smooth_x, smooth_y = self.smoother.smooth_position(index_tip[0], index_tip[1])

        # Map smoothed position to screen
        screen_x, screen_y = self.calibration.map_to_screen(
            smooth_x, smooth_y,
            frame.shape[1], frame.shape[0]
        )

        # Draw smoothed position indicator
        if self.config.smoothing_enabled and len(self.smoother.position_history) > 1:
            cv2.circle(frame, (smooth_x, smooth_y), 5, (255, 0, 255), -1)  # Magenta for smoothed position

        # Only move cursor and handle clicks if not in calibration mode
        if not self.calibration.calibration_mode:
            self.mouse.move_to(screen_x, screen_y)

            # Check if fingers are touching
            is_touching = click_distance < self.config.click_distance_threshold
            self.mouse.handle_click(is_touching, screen_x, screen_y)

            # Visual feedback for mouse down
            if is_touching:
                cv2.circle(frame, index_pip, 20, (0, 0, 255), 3)
                cv2.circle(frame, thumb_tip, 15, (0, 0, 255), 3)

            # Draw status bar
            self._draw_status(frame)

    def _process_hand_lost(self, frame):
        """Process frame when hand is not detected."""
        self.current_index_position = None

        # Try prediction
        predicted_pos = self.smoother.predict_position()

        if predicted_pos is not None and self.smoother.frames_without_detection <= self.config.max_frames_to_predict:
            # Draw predicted position
            cv2.circle(frame, predicted_pos, 12, (255, 0, 0), 2)
            cv2.putText(frame, "?", (predicted_pos[0] + 15, predicted_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Apply smoothing to predicted position
            smooth_x, smooth_y = self.smoother.smooth_position(predicted_pos[0], predicted_pos[1])

            # Map to screen and move cursor
            if not self.calibration.calibration_mode:
                screen_x, screen_y = self.calibration.map_to_screen(
                    smooth_x, smooth_y,
                    frame.shape[1], frame.shape[0]
                )
                self.mouse.move_to(screen_x, screen_y)
        else:
            # Too many frames without detection
            cv2.putText(frame, "No hand", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            # Release mouse if it was down
            if self.mouse.mouse_down and self.config.click_mode == 'hold':
                self.mouse.reset()

    def _draw_status(self, frame):
        """Draw status bar at top."""
        mode_text = "HOLD" if self.config.click_mode == 'hold' else "CLICK"
        status_parts = [f"Mode: {mode_text}"]

        # Only show click state when actually clicking
        if self.mouse.mouse_down:
            if self.config.click_mode == 'hold':
                status_parts.append("DRAGGING")
            else:
                status_parts.append("CLICKED")

        # Combine into single status line
        status_text = " | ".join(status_parts)
        cv2.putText(frame, status_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _draw_controls(self, frame):
        """Draw controls at bottom."""
        controls_text = "q:quit | m:mode | c:cal | s:smooth"
        cv2.putText(frame, controls_text,
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    def _handle_key(self, key: int):
        """
        Handle keyboard input.

        Args:
            key: Key code from cv2.waitKey()

        Returns:
            True to continue, False to quit
        """
        if key == ord('q'):
            return False
        elif key == ord('c'):
            self.calibration.start()
        elif key == ord('d'):
            self.calibration.delete()
        elif key == ord(' ') and self.calibration.calibration_mode:
            if self.current_index_position is not None:
                self.calibration.add_point(
                    self.current_index_position[0],
                    self.current_index_position[1]
                )
        elif key == 27 and self.calibration.calibration_mode:  # ESC key
            self.calibration.cancel()
        elif key == ord('m'):
            self.mouse.toggle_mode()
        elif key == ord('r'):
            self.mouse.reset()
            print("Reset!")
        elif key == ord('+'):
            self.mouse.adjust_threshold(5)
        elif key == ord('-'):
            self.mouse.adjust_threshold(-5)
        elif key == ord('s'):
            self.config.smoothing_enabled = not self.config.smoothing_enabled
            if not self.config.smoothing_enabled:
                self.smoother.reset()
            print(f"Smoothing: {'ON' if self.config.smoothing_enabled else 'OFF'}")
        elif key == ord('['):
            self.config.smoothing_factor = max(0.1, self.config.smoothing_factor - 0.1)
            print(f"Smoothing factor decreased to {self.config.smoothing_factor:.2f} (more smoothing)")
        elif key == ord(']'):
            self.config.smoothing_factor = min(0.9, self.config.smoothing_factor + 0.1)
            print(f"Smoothing factor increased to {self.config.smoothing_factor:.2f} (less smoothing)")

        return True

    def run(self):
        """Run the main tracking loop."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        self._print_welcome()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame")
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow('Finger Tracker', processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

    def _print_welcome(self):
        """Print welcome message with controls."""
        print("\n=== Finger Tracker Started! ===")
        print("\nControls:")
        print("- Move your index finger to control the cursor")
        print("- Touch your thumb and index finger together to click/drag")
        print("- Press 'm' to toggle between HOLD (drag) and CLICK modes")
        print("- Press 's' to toggle smoothing on/off")
        print("- Press '[/]' to adjust smoothing amount")
        print("- Press 'c' to start calibration")
        print("- Press 'd' to delete calibration")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset")
        print("- Press '+/-' to adjust click sensitivity")
        print(f"\nCurrent settings:")
        print(f"- Click threshold: {self.config.click_distance_threshold} pixels")
        print(f"- Click mode: {self.config.click_mode.upper()} (hold for drag, click for single clicks)")
        print(f"- Smoothing: {'ON' if self.config.smoothing_enabled else 'OFF'} (factor: {self.config.smoothing_factor:.2f})")
        if self.calibration.calibration_complete:
            print("- Calibration: LOADED")
        else:
            print("- Calibration: NOT SET (using full frame)")