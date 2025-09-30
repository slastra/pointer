"""Hand detection using MediaPipe."""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

from .config import Config


class HandDetector:
    """Wrapper for MediaPipe hand detection."""

    def __init__(self, config: Config):
        """
        Initialize the hand detector.

        Args:
            config: Configuration object with detection settings
        """
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_num_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame):
        """
        Detect hands in the frame.

        Args:
            frame: BGR image frame from OpenCV

        Returns:
            MediaPipe detection results
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    def draw_landmarks(self, frame, hand_landmarks):
        """
        Draw hand landmarks on the frame.

        Args:
            frame: BGR image frame
            hand_landmarks: MediaPipe hand landmarks
        """
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
        )

    @staticmethod
    def get_landmark_position(hand_landmarks, landmark_id: int, frame_shape: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        Get the pixel position of a landmark.

        Args:
            hand_landmarks: MediaPipe hand landmarks
            landmark_id: ID of the landmark to get
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            Tuple of (x, y) pixel coordinates
        """
        landmark = hand_landmarks.landmark[landmark_id]
        height, width = frame_shape[:2]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y

    @staticmethod
    def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            point1: First point (x, y)
            point2: Second point (x, y)

        Returns:
            Distance between points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def close(self):
        """Release resources."""
        self.hands.close()