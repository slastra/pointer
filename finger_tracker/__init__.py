"""Finger tracking mouse controller using MediaPipe and OpenCV."""

from __future__ import annotations

from .config import Config
from .gui import main
from .tracker import FingerTracker

__version__ = "1.0.0"
__all__ = ["Config", "FingerTracker", "main"]
