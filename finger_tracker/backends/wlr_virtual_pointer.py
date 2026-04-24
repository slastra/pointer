"""Wayland-native mouse backend via wlr-virtual-pointer-unstable-v1.

This is the proper path on wlroots-based compositors (Hyprland, Sway): we
create a compositor-recognized virtual pointer device through the
zwlr_virtual_pointer_manager_v1 global and send motion_absolute + button
events. Unlike the uinput-tablet backend, this keeps the native cursor
visible and routes button events correctly as clicks.

Generated protocol bindings live under finger_tracker._wayland_generated/.
Regenerate with:
    uv run python -m pywayland.scanner \\
        -i /usr/share/wayland/wayland.xml \\
           finger_tracker/_wayland_protocols/wlr-virtual-pointer-unstable-v1.xml \\
        -o finger_tracker/_wayland_generated/
"""

from __future__ import annotations

import logging
import time

from pywayland.client import Display

# Linux input-event-codes constants (from <linux/input-event-codes.h>).
BTN_LEFT = 0x110
BTN_RIGHT = 0x111
BTN_MIDDLE = 0x112

# wl_pointer.button state enum.
STATE_RELEASED = 0
STATE_PRESSED = 1

# wl_pointer.axis enum (vertical scroll).
AXIS_VERTICAL_SCROLL = 0

# wl_pointer.axis_source enum.
AXIS_SOURCE_WHEEL = 0

logger = logging.getLogger(__name__)


class WlrVirtualPointerBackend:
    name = "wlr-virtual-pointer"

    def __init__(self, screen_width: int, screen_height: int) -> None:
        from finger_tracker._wayland_generated.wayland.wl_seat import WlSeat
        from finger_tracker._wayland_generated.wlr_virtual_pointer_unstable_v1.zwlr_virtual_pointer_manager_v1 import (
            ZwlrVirtualPointerManagerV1,
        )

        self._sw = screen_width
        self._sh = screen_height

        self._display = Display()
        self._display.connect()

        registry = self._display.get_registry()

        self._seat = None
        self._manager = None

        def on_global(_registry, id_num: int, iface: str, version: int) -> None:
            if iface == WlSeat.name:
                self._seat = _registry.bind(id_num, WlSeat, min(version, 7))
            elif iface == ZwlrVirtualPointerManagerV1.name:
                self._manager = _registry.bind(
                    id_num, ZwlrVirtualPointerManagerV1, min(version, 2)
                )

        registry.dispatcher["global"] = on_global
        self._display.roundtrip()

        if self._manager is None:
            raise RuntimeError(
                "compositor does not advertise zwlr_virtual_pointer_manager_v1"
            )

        self._pointer = self._manager.create_virtual_pointer(self._seat)
        self._display.roundtrip()

        self._start_ms = int(time.monotonic() * 1000)
        logger.info("wlr-virtual-pointer bound")

    def _ts(self) -> int:
        return int(time.monotonic() * 1000) - self._start_ms

    def _flush(self) -> None:
        self._display.flush()

    def move_to(self, x: int, y: int) -> None:
        x = max(0, min(self._sw - 1, x))
        y = max(0, min(self._sh - 1, y))
        # motion_absolute(time, x, y, x_extent, y_extent) — coords within the
        # specified extent. Using screen dims maps 1:1 to pixels.
        self._pointer.motion_absolute(self._ts(), x, y, self._sw, self._sh)
        self._pointer.frame()
        self._flush()

    def _button(self, code: int, state: int) -> None:
        self._pointer.button(self._ts(), code, state)
        self._pointer.frame()
        self._flush()

    def mouse_down(self) -> None:
        self._button(BTN_LEFT, STATE_PRESSED)

    def mouse_up(self) -> None:
        self._button(BTN_LEFT, STATE_RELEASED)

    def click(self) -> None:
        self._button(BTN_LEFT, STATE_PRESSED)
        self._button(BTN_LEFT, STATE_RELEASED)

    def right_click(self) -> None:
        self._button(BTN_RIGHT, STATE_PRESSED)
        self._button(BTN_RIGHT, STATE_RELEASED)

    def scroll(self, amount: int) -> None:
        if amount == 0:
            return
        ts = self._ts()
        self._pointer.axis_source(AXIS_SOURCE_WHEEL)
        # Negate: pyautogui-style "positive = up" vs wl_pointer's screen Y.
        value = -amount * 10.0
        discrete = -amount
        self._pointer.axis_discrete(ts, AXIS_VERTICAL_SCROLL, value, discrete)
        self._pointer.frame()
        self._flush()

    def close(self) -> None:
        try:
            self._pointer.destroy()
            self._display.flush()
            self._display.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed closing wlr-virtual-pointer: %s", e)
