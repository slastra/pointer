"""Tests for CalibrationManager.map_to_screen (normalized bounds)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finger_tracker.calibration import CalibrationManager


@pytest.fixture
def cal_file(tmp_path: Path) -> str:
    return str(tmp_path / "cal.json")


def _set_pixel_bounds(m: CalibrationManager, bounds: dict, frame_w: int, frame_h: int) -> None:
    m.bounds_norm = {
        "min_x": bounds["min_x"] / frame_w,
        "max_x": bounds["max_x"] / frame_w,
        "min_y": bounds["min_y"] / frame_h,
        "max_y": bounds["max_y"] / frame_h,
    }
    m.calibration_complete = True
    m.set_frame_size(frame_w, frame_h)


class TestMapToScreenUncalibrated:
    def test_maps_full_frame_to_screen(self, cal_file):
        m = CalibrationManager(cal_file, screen_width=1920, screen_height=1080)
        assert m.map_to_screen(0, 0, frame_width=640, frame_height=480) == (0, 0)

    def test_center_of_frame_maps_to_center(self, cal_file):
        m = CalibrationManager(cal_file, 1920, 1080)
        assert m.map_to_screen(320, 240, 640, 480) == (960, 540)

    def test_clamped_to_screen_minus_one(self, cal_file):
        m = CalibrationManager(cal_file, 1920, 1080)
        assert m.map_to_screen(640, 480, 640, 480) == (1919, 1079)

    def test_negative_input_clamped_to_zero(self, cal_file):
        m = CalibrationManager(cal_file, 1920, 1080)
        assert m.map_to_screen(-100, -50, 640, 480) == (0, 0)


class TestMapToScreenCalibrated:
    def test_maps_within_calibration_bounds(self, cal_file):
        m = CalibrationManager(cal_file, 1920, 1080)
        _set_pixel_bounds(m, {"min_x": 100, "max_x": 500, "min_y": 100, "max_y": 400}, 640, 480)

        assert m.map_to_screen(100, 100, 640, 480) == (0, 0)
        assert m.map_to_screen(300, 250, 640, 480) == (960, 540)

    def test_zero_width_bounds_fall_back_to_full_frame(self, cal_file):
        m = CalibrationManager(cal_file, 1920, 1080)
        _set_pixel_bounds(m, {"min_x": 200, "max_x": 200, "min_y": 100, "max_y": 400}, 640, 480)
        assert m.map_to_screen(320, 240, 640, 480) == (960, 540)

    def test_zero_height_bounds_fall_back_to_full_frame(self, cal_file):
        m = CalibrationManager(cal_file, 1920, 1080)
        _set_pixel_bounds(m, {"min_x": 100, "max_x": 500, "min_y": 250, "max_y": 250}, 640, 480)
        assert m.map_to_screen(320, 240, 640, 480) == (960, 540)

    def test_out_of_bounds_clamped(self, cal_file):
        m = CalibrationManager(cal_file, 1920, 1080)
        _set_pixel_bounds(m, {"min_x": 100, "max_x": 500, "min_y": 100, "max_y": 400}, 640, 480)
        x, y = m.map_to_screen(9999, 9999, 640, 480)
        assert x == 1919 and y == 1079


class TestPersistence:
    def test_save_and_load_roundtrip(self, cal_file):
        m1 = CalibrationManager(cal_file, 1920, 1080)
        m1.bounds_norm = {"min_x": 0.1, "max_x": 0.9, "min_y": 0.2, "max_y": 0.8}
        assert m1.save() is True

        m2 = CalibrationManager(cal_file, 1920, 1080)
        assert m2.calibration_complete is True
        assert m2.bounds_norm == {"min_x": 0.1, "max_x": 0.9, "min_y": 0.2, "max_y": 0.8}

    def test_legacy_pixel_format_is_migrated(self, cal_file):
        Path(cal_file).write_text(json.dumps({"min_x": 64, "max_x": 576, "min_y": 48, "max_y": 432}))
        m = CalibrationManager(cal_file, 1920, 1080)
        assert m.calibration_complete is True
        # Migrated using legacy 640x480 base → [0.1, 0.9] × [0.1, 0.9].
        assert m.bounds_norm is not None
        assert m.bounds_norm["min_x"] == pytest.approx(0.1)
        assert m.bounds_norm["max_x"] == pytest.approx(0.9)

    def test_load_corrupt_file_does_not_raise(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        m = CalibrationManager(str(bad), 1920, 1080)
        assert m.calibration_complete is False
        assert m.bounds_norm is None

    def test_add_point_completes_on_four(self, cal_file):
        m = CalibrationManager(cal_file, 1920, 1080)
        m.set_frame_size(640, 480)
        m.start()
        for pt in [(64, 48), (576, 48), (576, 432), (64, 432)]:
            m.add_point(*pt)
        assert m.calibration_complete is True
        assert m.calibration_mode is False
        assert m.bounds_norm is not None
        assert m.bounds_norm["min_x"] == pytest.approx(0.1)
        assert m.bounds_norm["max_x"] == pytest.approx(0.9)
        saved = json.loads(Path(cal_file).read_text())
        assert saved == m.bounds_norm

    def test_cancel_clears_points(self, cal_file):
        m = CalibrationManager(cal_file, 1920, 1080)
        m.start()
        m.add_point(10, 10)
        m.cancel()
        assert m.calibration_mode is False
        assert m.calibration_points == []

    def test_delete_removes_file_and_state(self, cal_file):
        m = CalibrationManager(cal_file, 1920, 1080)
        m.bounds_norm = {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0}
        m.save()
        m.calibration_complete = True
        m.delete()
        assert m.bounds_norm is None
        assert m.calibration_complete is False
        assert not Path(cal_file).exists()
