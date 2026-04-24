"""Orchestrator: update() runs the detect→smooth→mouse pipeline, render() draws overlays."""

from __future__ import annotations

import logging
import time

import cv2
import pyautogui

from .backends import build_backend
from .calibration import CalibrationManager
from .config import Config
from .detection import HandDetector
from .gestures import Gesture, GestureDebouncer, classify
from .mouse import MouseController
from .smoothing import PositionSmoother
from .state import FrameState

SCROLL_PX_PER_STEP = 20.0  # px of vertical motion = one wheel click

from . import palette as C  # noqa: E402 (grouped with other local imports visually)

logger = logging.getLogger(__name__)


class FingerTracker:
    """Top-level app. update() mutates state + cursor. render() draws overlays."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

        screen_width, screen_height = pyautogui.size()

        self.detector = HandDetector(self.config)
        self.smoother = PositionSmoother(self.config)
        self.calibration = CalibrationManager(
            self.config.calibration_file,
            screen_width,
            screen_height,
            y_offset=self.config.bounds_y_offset_px,
        )
        backend = build_backend(screen_width, screen_height)
        self.mouse = MouseController(backend, self.config)

        self.current_index_position: tuple[int, int] | None = None
        self._last_frame_time = time.time()
        self._fps = 0.0

        self.gestures = GestureDebouncer(stable_frames=3)
        self.paused = False
        self._pinch_r_fired = False
        self._scroll_last_y: int | None = None
        self._scroll_accum = 0.0

        # Previous gesture used to detect POINT-latching transitions — we reset
        # the smoothing filter when POINT activates so stale state from earlier
        # non-tracking frames doesn't drag the first tracked cursor position.
        self._prev_gesture: Gesture = Gesture.IDLE

    # ------------------------------------------------------------------ update

    def update(self, frame) -> FrameState:
        """Run detection/smoothing/mouse dispatch. Returns FrameState for rendering."""
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt

        if now - getattr(self, "_last_heartbeat", 0.0) > 2.0:
            self._last_heartbeat = now
            logger.info(
                "heartbeat fps=%.1f paused=%s mode=%s gesture=%s thresh=%d smooth=%s",
                self._fps,
                self.paused,
                self.mouse.mode,
                self.gestures.latched.name,
                self.mouse.threshold,
                "on" if self.smoother.smoothing_enabled else "off",
            )

        state = FrameState(
            mode=self.mouse.mode,
            mouse_down=self.mouse.mouse_down,
            strategy="one_euro",
            paused=self.paused,
            calibration_mode=self.calibration.calibration_mode,
            calibration_complete=self.calibration.calibration_complete,
            calibration_bounds=self.calibration.effective_bounds,
            calibration_points=list(self.calibration.calibration_points),
            smoothing_enabled=self.smoother.smoothing_enabled,
            smoothing_factor=self.smoother.min_cutoff,
            threshold=self.mouse.threshold,
            effective_threshold=float(self.mouse.threshold),
            fps=self._fps,
        )

        results = self.detector.detect(frame)

        if results.multi_hand_landmarks:
            self._update_hand_detected(frame, results, state)
        else:
            self._update_hand_lost(frame, state)

        state.mouse_down = self.mouse.mouse_down
        return state

    def _update_hand_detected(self, frame, results, state: FrameState) -> None:
        hand_landmarks = results.multi_hand_landmarks[0]
        state.hand_landmarks = hand_landmarks
        state.has_hand = True

        thumb_tip = self.detector.get_landmark_position(hand_landmarks, 4, frame.shape)
        index_tip = self.detector.get_landmark_position(hand_landmarks, 8, frame.shape)
        index_pip = self.detector.get_landmark_position(hand_landmarks, 6, frame.shape)
        wrist = self.detector.get_landmark_position(hand_landmarks, 0, frame.shape)
        middle_mcp = self.detector.get_landmark_position(hand_landmarks, 9, frame.shape)

        state.thumb_tip = thumb_tip
        state.index_tip = index_tip
        state.index_pip = index_pip

        self.smoother.update_velocity(index_tip[0], index_tip[1])

        if self.calibration.calibration_mode:
            self.current_index_position = index_tip

        click_distance = self.detector.calculate_distance(thumb_tip, index_pip)
        state.click_distance = click_distance
        hand_scale = self.detector.hand_scale(wrist, middle_mcp)

        result = classify(hand_landmarks.landmark, hand_scale=hand_scale)
        gesture = self.gestures.update(result.gesture)
        state.gesture = gesture.name

        # Reset smoother state on latching transition into POINT so the filter
        # starts fresh at the current finger position instead of outputting a
        # lagged combination of whatever it accumulated during non-tracking.
        tracking_gestures = (Gesture.POINT, Gesture.PINCH_L)
        if gesture in tracking_gestures and self._prev_gesture not in tracking_gestures:
            self.smoother.reset()
            self.smoother.update_velocity(index_tip[0], index_tip[1])
        self._prev_gesture = gesture

        smooth_x, smooth_y = self.smoother.smooth_position(index_tip[0], index_tip[1])
        state.smoothed_px = (int(smooth_x), int(smooth_y))

        screen_x, screen_y = self.calibration.map_to_screen(
            smooth_x, smooth_y, frame.shape[1], frame.shape[0]
        )
        state.screen_xy = (screen_x, screen_y)

        if not self.calibration.calibration_mode:
            self._handle_gesture(
                gesture, index_tip, click_distance, hand_scale, screen_x, screen_y, state
            )

    def _handle_gesture(
        self,
        gesture: Gesture,
        index_tip: tuple[int, int],
        click_distance: float,
        hand_scale: float,
        screen_x: int,
        screen_y: int,
        state: FrameState,
    ) -> None:
        if self.paused:
            state.effective_threshold = float(self.mouse.threshold)
            return

        # Cursor tracks on POINT or PINCH_L (so drag works).
        tracking = gesture in (Gesture.POINT, Gesture.PINCH_L)

        if gesture is not Gesture.TWO_FINGER:
            self._scroll_last_y = None
            self._scroll_accum = 0.0
            if tracking:
                logger.debug(
                    "move_to screen=(%d,%d) smoothed=%s raw=%s gesture=%s",
                    screen_x, screen_y, state.smoothed_px, index_tip, gesture.name,
                )
                self.mouse.move_to(screen_x, screen_y)

        if gesture is Gesture.TWO_FINGER:
            if self._scroll_last_y is None:
                self._scroll_last_y = index_tip[1]
            else:
                dy = self._scroll_last_y - index_tip[1]
                self._scroll_accum += dy / SCROLL_PX_PER_STEP
                steps = int(self._scroll_accum)
                if steps != 0:
                    self.mouse.scroll(steps)
                    self._scroll_accum -= steps
                self._scroll_last_y = index_tip[1]
            return

        if gesture is Gesture.PINCH_R:
            if not self._pinch_r_fired:
                self.mouse.right_click()
                self._pinch_r_fired = True
            state.is_touching = False
            state.effective_threshold = float(self.mouse.threshold)
            return
        self._pinch_r_fired = False

        # Click: gesture-latched PINCH_L drives mouse down/up.
        is_touching = gesture is Gesture.PINCH_L
        state.is_touching = is_touching
        state.effective_threshold = float(self.mouse.threshold)
        self.mouse.handle_click(is_touching, screen_x, screen_y)

    def _update_hand_lost(self, frame, state: FrameState) -> None:
        self.current_index_position = None
        # Reset prev-gesture so that when hand reappears, whatever gesture
        # latches first is treated as a new transition (smoother resets if POINT).
        self._prev_gesture = Gesture.IDLE

        predicted_pos = self.smoother.predict_position()
        if (
            predicted_pos is not None
            and self.smoother.frames_without_detection <= self.config.max_frames_to_predict
        ):
            state.predicted = True
            state.index_tip = predicted_pos
        elif self.mouse.mouse_down and self.mouse.mode == "hold":
            self.mouse.reset()

    # ------------------------------------------------------------------ render

    def render(self, frame, state: FrameState):
        if state.calibration_complete and state.calibration_bounds:
            b = state.calibration_bounds
            cv2.rectangle(frame, (b["min_x"], b["min_y"]), (b["max_x"], b["max_y"]), C.GREEN, 2)

        if state.calibration_mode:
            for point in state.calibration_points:
                cv2.circle(frame, point, 8, C.RED, -1)

        if state.has_hand and state.hand_landmarks is not None:
            self.detector.draw_landmarks(frame, state.hand_landmarks)
            if state.thumb_tip:
                cv2.circle(frame, state.thumb_tip, 10, C.BLUE, -1)
            if state.index_tip:
                cv2.circle(frame, state.index_tip, 10, C.GREEN, -1)
            if state.index_pip:
                cv2.circle(frame, state.index_pip, 8, C.ORANGE, -1)
            if state.thumb_tip and state.index_pip:
                cv2.line(frame, state.thumb_tip, state.index_pip, C.YELLOW, 2)
            if state.smoothed_px and state.smoothing_enabled:
                cv2.circle(frame, state.smoothed_px, 5, C.PURPLE, -1)
            if state.is_touching and state.index_pip and state.thumb_tip:
                cv2.circle(frame, state.index_pip, 20, C.RED, 3)
                cv2.circle(frame, state.thumb_tip, 15, C.RED, 3)

        if state.predicted and state.index_tip:
            cv2.circle(frame, state.index_tip, 12, C.BLUE, 2)

        if state.calibration_mode:
            self._draw_calibration_preview(frame, state)

        return frame

    def _draw_calibration_preview(self, frame, state: FrameState) -> None:
        h, w = frame.shape[:2]
        inset_w, inset_h = 160, 90
        x0, y0 = w - inset_w - 10, 10
        cv2.rectangle(frame, (x0, y0), (x0 + inset_w, y0 + inset_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + inset_w, y0 + inset_h), (200, 200, 200), 1)
        pts = list(state.calibration_points)
        if self.current_index_position is not None:
            pts.append(self.current_index_position)
        if not pts:
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 8
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(1, max_x - min_x)
        span_y = max(1, max_y - min_y)

        for i, p in enumerate(pts):
            px = x0 + pad + int((p[0] - min_x) / span_x * (inset_w - 2 * pad))
            py = y0 + pad + int((p[1] - min_y) / span_y * (inset_h - 2 * pad))
            if i < len(state.calibration_points):
                cv2.circle(frame, (px, py), 3, C.RED, -1)
            else:
                cv2.drawMarker(frame, (px, py), C.YELLOW, cv2.MARKER_CROSS, 10, 1)
