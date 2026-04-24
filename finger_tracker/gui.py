"""GTK4 + libadwaita UI for the finger tracker.

Replaces the OpenCV window with a modern Adwaita app:
  - AdwApplicationWindow with an AdwHeaderBar
  - Live webcam preview (GtkDrawingArea, cairo blit of rendered frame)
  - Side panel with live-tunable smoothing / threshold sliders,
    strategy dropdown, mode switch, gesture + backend status
  - Header-bar actions for pause, calibrate, reset, quit

Threading model:
  - Main thread: GTK main loop + UI event handlers
  - Worker thread: capture + detect + smooth + mouse dispatch
  - Lock-protected "latest rendered frame + FrameState" slot

Entry point: `pointer-gui` script (see pyproject.toml).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

import cairo
import cv2
import gi
import numpy as np

from .config import Config
from .state import FrameState
from .tracker import FingerTracker

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GLib, Gtk  # noqa: E402

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Worker thread
# ----------------------------------------------------------------------------


class TrackerWorker(threading.Thread):
    """Runs the full capture→detect→mouse pipeline off the GTK main thread."""

    def __init__(self, tracker: FingerTracker) -> None:
        super().__init__(daemon=True, name="tracker-worker")
        self.tracker = tracker
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._state: FrameState | None = None
        self._cap: cv2.VideoCapture | None = None

    def run(self) -> None:
        self._cap = _open_camera(self.tracker.config)
        if self._cap is None or not self._cap.isOpened():
            logger.error("Could not open camera")
            return

        try:
            while not self._stop_evt.is_set():
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                state = self.tracker.update(frame)
                rendered = self.tracker.render(frame, state)

                with self._lock:
                    self._frame = rendered
                    self._state = state
        finally:
            if self._cap is not None:
                self._cap.release()
            self.tracker.detector.close()
            self.tracker.mouse.close()

    def latest(self):
        with self._lock:
            return self._frame, self._state

    def stop(self) -> None:
        self._stop_evt.set()


def _open_camera(config) -> cv2.VideoCapture | None:
    """Open /dev/video0 and force a high-fps config.

    Many webcams (incl. this machine's UVC one) ship with
    `exposure_dynamic_framerate=1`, which drops capture FPS to extend
    exposure in low light — we lose 30→15 fps just by being indoors. Turn it
    off before opening so the whole session runs at 30 fps. Also force MJPG:
    YUYV at any resolution above 640x480 caps the camera at ≤20 fps.
    """
    if shutil.which("v4l2-ctl"):
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", "/dev/video0", "-c", "exposure_dynamic_framerate=0"],
                check=False,
                capture_output=True,
                timeout=1.0,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.info("v4l2-ctl exposure_dynamic_framerate tweak skipped: %s", e)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    logger.info(
        "Camera: %dx%d @ %.0f fps",
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cap.get(cv2.CAP_PROP_FPS),
    )
    return cap


# ----------------------------------------------------------------------------
# Live webcam preview widget
# ----------------------------------------------------------------------------


class CameraPreview(Gtk.DrawingArea):
    """Draws the most recent processed frame; scales to widget size."""

    def __init__(self) -> None:
        super().__init__()
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_content_width(640)
        self.set_content_height(360)
        self.set_draw_func(self._on_draw)

        self._frame_bgra: np.ndarray | None = None
        self._frame_w = 0
        self._frame_h = 0

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        # cairo ARGB32 on little-endian == BGRA memory layout.
        bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        self._frame_bgra = np.ascontiguousarray(bgra)
        self._frame_h, self._frame_w = bgra.shape[:2]
        self.queue_draw()

    def _on_draw(self, _area, ctx: cairo.Context, width: int, height: int) -> None:
        if self._frame_bgra is None:
            ctx.set_source_rgb(0.1, 0.1, 0.12)
            ctx.paint()
            return

        stride = cairo.ImageSurface.format_stride_for_width(
            cairo.FORMAT_ARGB32, self._frame_w
        )
        surface = cairo.ImageSurface.create_for_data(
            memoryview(self._frame_bgra).cast("B"),  # type: ignore[arg-type]
            cairo.FORMAT_ARGB32,
            self._frame_w,
            self._frame_h,
            stride,
        )

        # Preserve aspect; letterbox with dark fill.
        ctx.set_source_rgb(0.08, 0.08, 0.1)
        ctx.paint()

        fw, fh = self._frame_w, self._frame_h
        scale = min(width / fw, height / fh)
        draw_w, draw_h = fw * scale, fh * scale
        x0 = (width - draw_w) / 2
        y0 = (height - draw_h) / 2

        ctx.save()
        ctx.translate(x0, y0)
        ctx.scale(scale, scale)
        ctx.set_source_surface(surface, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_BILINEAR)
        ctx.paint()
        ctx.restore()


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application, worker: TrackerWorker) -> None:
        super().__init__(application=app, title="Pointer")
        self.set_default_size(1100, 680)

        self.worker = worker
        self.tracker = worker.tracker

        # --- content ------------------------------------------------------
        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)

        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)

        # Header buttons
        self._pause_btn = Gtk.ToggleButton(icon_name="media-playback-pause-symbolic")
        self._pause_btn.set_tooltip_text("Pause tracking (open-palm 1s or here)")
        self._pause_btn.connect("toggled", self._on_pause_toggled)
        header.pack_start(self._pause_btn)

        cal_btn = Gtk.Button(icon_name="view-grid-symbolic")
        cal_btn.set_tooltip_text("Start 4-point calibration")
        cal_btn.connect("clicked", lambda *_: self.tracker.calibration.start())
        header.pack_start(cal_btn)

        reset_btn = Gtk.Button(icon_name="edit-undo-symbolic")
        reset_btn.set_tooltip_text("Reset mouse state")
        reset_btn.connect("clicked", lambda *_: self.tracker.mouse.reset())
        header.pack_start(reset_btn)

        menu = Gtk.MenuButton(icon_name="open-menu-symbolic")
        menu_model = _build_menu()
        popover = Gtk.PopoverMenu.new_from_model(menu_model)
        menu.set_popover(popover)
        header.pack_end(menu)

        # --- main body (split) -------------------------------------------
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_position(740)
        paned.set_shrink_start_child(False)
        paned.set_shrink_end_child(False)
        toolbar.set_content(paned)

        self.preview = CameraPreview()
        paned.set_start_child(self.preview)

        paned.set_end_child(self._build_side_panel())

        # Status bar
        self.status = Gtk.Label(xalign=0.0)
        self.status.add_css_class("dim-label")
        self.status.set_margin_start(12)
        self.status.set_margin_end(12)
        self.status.set_margin_top(6)
        self.status.set_margin_bottom(6)
        toolbar.add_bottom_bar(self.status)

        # --- app actions -------------------------------------------------
        self._add_action("calibrate", lambda *_: self.tracker.calibration.start())
        self._add_action("delete-calibration", lambda *_: self.tracker.calibration.delete())
        self._add_action("reset-mouse", lambda *_: self.tracker.mouse.reset())
        self._add_action("quit", lambda *_: app.quit())

        # --- refresh loop (GTK main thread) ------------------------------
        GLib.timeout_add(33, self._tick)  # ~30Hz

    # ---- side panel --------------------------------------------------------

    def _build_side_panel(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_size_request(320, -1)

        ctrl_group = Adw.PreferencesGroup(title="Controls")
        box.append(ctrl_group)

        self._smooth_row = Adw.SpinRow.new_with_range(0.1, 5.0, 0.05)
        self._smooth_row.set_title("Smoothing min-cutoff")
        self._smooth_row.set_subtitle("Lower = smoother at rest")
        self._smooth_row.set_value(self.tracker.smoother.min_cutoff)
        self._smooth_row.connect("notify::value", self._on_smooth_changed)
        ctrl_group.add(self._smooth_row)

        self._beta_row = Adw.SpinRow.new_with_range(0.0, 1.0, 0.01)
        self._beta_row.set_title("Smoothing beta")
        self._beta_row.set_subtitle("Higher = more responsive to motion")
        self._beta_row.set_value(self.tracker.smoother.beta)
        self._beta_row.connect("notify::value", self._on_beta_changed)
        ctrl_group.add(self._beta_row)

        self._thresh_row = Adw.SpinRow.new_with_range(10, 150, 1)
        self._thresh_row.set_title("Click threshold (px)")
        self._thresh_row.set_subtitle("Pinch distance below which a click fires")
        self._thresh_row.set_value(self.tracker.mouse.threshold)
        self._thresh_row.connect("notify::value", self._on_thresh_changed)
        ctrl_group.add(self._thresh_row)

        self._mode_row = Adw.SwitchRow()
        self._mode_row.set_title("Hold-to-drag mode")
        self._mode_row.set_subtitle("Off = discrete click, on = hold")
        self._mode_row.set_active(self.tracker.mouse.mode == "hold")
        self._mode_row.connect("notify::active", self._on_mode_changed)
        ctrl_group.add(self._mode_row)

        self._smooth_enable_row = Adw.SwitchRow()
        self._smooth_enable_row.set_title("Smoothing enabled")
        self._smooth_enable_row.set_active(self.tracker.smoother.smoothing_enabled)
        self._smooth_enable_row.connect("notify::active", self._on_smooth_enable_changed)
        ctrl_group.add(self._smooth_enable_row)

        # Calibration rectangle — live-tunable normalized bounds.
        rect_group = Adw.PreferencesGroup(title="Tracking rectangle")
        rect_group.set_description("Portion of the camera frame mapped to the screen")
        box.append(rect_group)

        self._rect_rows: dict[str, Adw.SpinRow] = {}
        for key, title in (
            ("min_x", "Left"),
            ("max_x", "Right"),
            ("min_y", "Top"),
            ("max_y", "Bottom"),
        ):
            row = Adw.SpinRow.new_with_range(0.0, 1.0, 0.01)
            row.set_title(title)
            row.set_digits(2)
            row.set_value(self._get_bound(key))
            row.connect("notify::value", self._on_rect_changed, key)
            rect_group.add(row)
            self._rect_rows[key] = row

        # Y-offset — shifts the rectangle upward so the user's wrist stays
        # in frame when pointing at the bottom edge.
        self._yoff_row = Adw.SpinRow.new_with_range(0, 300, 5)
        self._yoff_row.set_title("Bottom headroom (px)")
        self._yoff_row.set_subtitle("Pixels of space below the rectangle for your wrist")
        self._yoff_row.set_value(self.tracker.calibration.y_offset)
        self._yoff_row.connect("notify::value", self._on_yoff_changed)
        rect_group.add(self._yoff_row)

        reset_rect = Gtk.Button(label="Fit to full frame")
        reset_rect.add_css_class("flat")
        reset_rect.connect("clicked", self._on_reset_rect)
        rect_group.add(reset_rect)

        return box

    def _on_yoff_changed(self, row, _pspec) -> None:
        self.tracker.calibration.y_offset = int(row.get_value())

    def _get_bound(self, key: str) -> float:
        bn = self.tracker.calibration.bounds_norm
        if bn is None:
            return {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0}[key]
        return float(bn[key])

    def _on_rect_changed(self, row, _pspec, key: str) -> None:
        cal = self.tracker.calibration
        if cal.bounds_norm is None:
            cal.bounds_norm = {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0}
        cal.bounds_norm[key] = float(row.get_value())
        cal.calibration_complete = True
        cal.save()

    def _on_reset_rect(self, _btn) -> None:
        cal = self.tracker.calibration
        cal.bounds_norm = {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0}
        cal.calibration_complete = True
        cal.save()
        for key, row in self._rect_rows.items():
            row.handler_block_by_func(self._on_rect_changed)
            row.set_value(cal.bounds_norm[key])
            row.handler_unblock_by_func(self._on_rect_changed)

    def _add_action(self, name: str, callback: Callable) -> None:
        from gi.repository import Gio

        action = Gio.SimpleAction.new(name, None)
        action.connect("activate", callback)
        self.get_application().add_action(action)

    # ---- control change handlers ------------------------------------------

    def _on_smooth_changed(self, row, _pspec) -> None:
        self.tracker.smoother.set_min_cutoff(row.get_value())

    def _on_thresh_changed(self, row, _pspec) -> None:
        self.tracker.mouse.set_threshold(int(row.get_value()))

    def _on_beta_changed(self, row, _pspec) -> None:
        self.tracker.smoother.set_beta(row.get_value())

    def _on_mode_changed(self, row, _pspec) -> None:
        want_hold = row.get_active()
        have_hold = self.tracker.mouse.mode == "hold"
        if want_hold != have_hold:
            self.tracker.mouse.toggle_mode()

    def _on_smooth_enable_changed(self, row, _pspec) -> None:
        want = row.get_active()
        if want != self.tracker.smoother.smoothing_enabled:
            self.tracker.smoother.toggle_enabled()

    def _on_pause_toggled(self, btn) -> None:
        self.tracker.paused = btn.get_active()

    # ---- tick: pull latest frame+state from worker ------------------------

    def _tick(self) -> bool:
        frame, state = self.worker.latest()
        if frame is not None:
            self.preview.set_frame(frame)

        if state is not None:
            self.status.set_text(
                f"fps {state.fps:5.1f}   gesture {state.gesture:10s}   "
                f"mode {state.mode}   threshold {state.threshold}px   "
                f"{'paused' if state.paused else 'live'}"
            )
            if self._pause_btn.get_active() != state.paused:
                self._pause_btn.handler_block_by_func(self._on_pause_toggled)
                self._pause_btn.set_active(state.paused)
                self._pause_btn.handler_unblock_by_func(self._on_pause_toggled)

        return True


def _build_menu():
    from gi.repository import Gio

    menu = Gio.Menu()
    menu.append("Start calibration", "app.calibrate")
    menu.append("Delete calibration", "app.delete-calibration")
    menu.append("Reset mouse", "app.reset-mouse")
    menu.append("Quit", "app.quit")
    return menu


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------


class PointerApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id="dev.shaun.Pointer")
        self.tracker: FingerTracker | None = None
        self.worker: TrackerWorker | None = None
        self.connect("activate", self._on_activate)
        self.connect("shutdown", self._on_shutdown)

    def _on_activate(self, _app) -> None:
        self.tracker = FingerTracker(Config())
        self.worker = TrackerWorker(self.tracker)
        self.worker.start()

        win = MainWindow(self, self.worker)
        win.present()

        # Global Ctrl+Q quits.
        self.set_accels_for_action("app.quit", ["<Ctrl>q"])

    def _on_shutdown(self, _app) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=2.0)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = PointerApp()
    return app.run(None)


if __name__ == "__main__":
    import sys

    sys.exit(main())


# Silence unused-import warning for Gdk (kept for future input-event handling).
_ = Gdk
