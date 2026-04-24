"""Hand detection using MediaPipe."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec

from . import palette
from .config import Config

# Adwaita-themed landmark + connection specs for MediaPipe's drawing util.
_LANDMARK_SPEC = DrawingSpec(color=palette.BLUE, thickness=2, circle_radius=3)
_CONNECTION_SPEC = DrawingSpec(color=palette.GREEN, thickness=2)


class HandDetector:
    """Wrapper for MediaPipe hand detection."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_num_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame):
        # Optionally downscale before MediaPipe — landmark coords are normalized
        # so the caller still gets them in original frame-space via
        # get_landmark_position(frame_shape).
        dw, dh = self.config.detection_width, self.config.detection_height
        h, w = frame.shape[:2]
        if dw and dh and (w > dw or h > dh):
            scale = min(dw / w, dh / h)
            target = (max(1, int(w * scale)), max(1, int(h * scale)))
            frame = cv2.resize(frame, target, interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    def draw_landmarks(self, frame, hand_landmarks) -> None:
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            _LANDMARK_SPEC,
            _CONNECTION_SPEC,
        )

    @staticmethod
    def get_landmark_position(hand_landmarks, landmark_id: int, frame_shape: tuple[int, int, int]) -> tuple[int, int]:
        landmark = hand_landmarks.landmark[landmark_id]
        height, width = frame_shape[:2]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y

    @staticmethod
    def calculate_distance(point1: tuple[int, int], point2: tuple[int, int]) -> float:
        return float(np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))

    # Reference palm size (pixels) at which hand_scale == 1.0. Typical wrist →
    # middle-MCP at arm's length ~120 px; tweak if the default feels off.
    REFERENCE_PALM_PX = 120.0

    @classmethod
    def hand_scale(cls, wrist: tuple[int, int], middle_mcp: tuple[int, int]) -> float:
        """Palm size as a scalar around 1.0. Close hand → >1, far hand → <1."""
        palm_px = cls.calculate_distance(wrist, middle_mcp)
        if palm_px <= 0:
            return 1.0
        return palm_px / cls.REFERENCE_PALM_PX

    def close(self) -> None:
        self.hands.close()
