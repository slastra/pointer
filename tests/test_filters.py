"""Tests for smoothing + MouseController hysteresis/debounce."""

from __future__ import annotations

import pytest

from finger_tracker.config import Config
from finger_tracker.detection import HandDetector
from finger_tracker.filters import OneEuroStrategy
from finger_tracker.mouse import MouseController


class _NullBackend:
    """Stub backend — never touches the real mouse."""

    name = "null"

    def move_to(self, x: int, y: int) -> None: ...
    def mouse_down(self) -> None: ...
    def mouse_up(self) -> None: ...
    def click(self) -> None: ...
    def right_click(self) -> None: ...
    def scroll(self, amount: int) -> None: ...
    def close(self) -> None: ...


class TestOneEuro:
    def test_static_input_converges(self):
        f = OneEuroStrategy(min_cutoff=1.0, beta=0.0)
        out = (0.0, 0.0)
        t = 0.0
        for _ in range(200):
            out = f.filter(100.0, 200.0, t)
            t += 1 / 60
        assert abs(out[0] - 100) <= 1
        assert abs(out[1] - 200) <= 1

    def test_filter_lags_large_jump_with_low_cutoff(self):
        f = OneEuroStrategy(min_cutoff=0.5, beta=0.0)
        f.filter(0.0, 0.0, 0.0)
        x_out, _ = f.filter(100.0, 0.0, 1 / 60)
        assert 0 < x_out < 100

    def test_adjust_bounds(self):
        f = OneEuroStrategy()
        f.min_cutoff = 0.5
        f.adjust_min_cutoff(-10)
        assert f.min_cutoff == pytest.approx(OneEuroStrategy.MIN_CUTOFF_BOUNDS[0])
        f.adjust_min_cutoff(100)
        assert f.min_cutoff == pytest.approx(OneEuroStrategy.MIN_CUTOFF_BOUNDS[1])

    def test_reset_clears_state(self):
        f = OneEuroStrategy()
        f.filter(10.0, 20.0, 0.0)
        f.reset()
        assert f._fx._last_ts is None
        assert f._fx._x.y is None


class TestMouseHysteresisAndDebounce:
    def _mc(self, **kwargs):
        return MouseController(
            _NullBackend(), Config(), click_distance_threshold=20, **kwargs
        )

    def test_requires_debounce_frames_to_engage(self):
        mc = self._mc()
        assert mc.update_touch(5, 1.0)[0] is False
        assert mc.update_touch(5, 1.0)[0] is False
        assert mc.update_touch(5, 1.0)[0] is True

    def test_hysteresis_uses_release_threshold_once_engaged(self):
        mc = self._mc()
        for _ in range(3):
            mc.update_touch(5, 1.0)
        assert mc._debounced_touching is True

        t, eff = mc.update_touch(22, 1.0)
        assert eff == pytest.approx(20.0)
        assert t is True

        for _ in range(3):
            t, _ = mc.update_touch(30, 1.0)
        assert t is False

    def test_hand_scale_scales_effective_threshold(self):
        mc = self._mc()
        _, eff = mc.update_touch(15, 0.5)
        assert eff == pytest.approx(10.0)

    def test_threshold_clamped_upper(self):
        mc = MouseController(_NullBackend(), Config(), click_distance_threshold=10)
        mc.set_threshold(9999)
        assert mc.threshold == MouseController.MAX_THRESHOLD

    def test_toggle_mode_resets_and_switches(self):
        mc = MouseController(_NullBackend(), Config(), click_mode="hold")
        mc.last_click_time = 123.0
        mc.toggle_mode()
        assert mc.mode == "click"
        assert mc.last_click_time == 0.0
        mc.toggle_mode()
        assert mc.mode == "hold"


class TestHandScale:
    def test_unit_scale_when_palm_equals_reference(self):
        assert HandDetector.hand_scale(
            (0, 0), (0, int(HandDetector.REFERENCE_PALM_PX))
        ) == pytest.approx(1.0)

    def test_zero_distance_returns_one(self):
        assert HandDetector.hand_scale((5, 5), (5, 5)) == 1.0

    def test_bigger_hand_scales_above_one(self):
        s = HandDetector.hand_scale((0, 0), (0, int(HandDetector.REFERENCE_PALM_PX * 2)))
        assert s == pytest.approx(2.0)


class TestSmootherReset:
    def test_reset_clears_filter_state(self):
        from finger_tracker.smoothing import PositionSmoother

        s = PositionSmoother(Config())
        s.smooth_position(100, 200)
        s.update_velocity(100, 200)
        s.reset()
        assert s.last_raw_position is None
        assert s.last_velocity == (0.0, 0.0)
