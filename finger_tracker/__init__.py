"""Finger tracking mouse controller using MediaPipe and OpenCV."""

from .config import Config
from .tracker import FingerTracker

__version__ = "1.0.0"
__all__ = ["Config", "FingerTracker"]