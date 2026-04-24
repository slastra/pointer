"""Mouse control backend — wlr-virtual-pointer for Hyprland/Sway/wlroots."""

from __future__ import annotations

from .base import MouseBackend
from .wlr_virtual_pointer import WlrVirtualPointerBackend

__all__ = ["MouseBackend", "WlrVirtualPointerBackend", "build_backend"]


def build_backend(screen_width: int, screen_height: int) -> MouseBackend:
    """Return the one backend we support: wlr-virtual-pointer.

    Raises RuntimeError if the compositor doesn't advertise
    zwlr_virtual_pointer_manager_v1 (i.e. not running on wlroots).
    """
    return WlrVirtualPointerBackend(screen_width, screen_height)
