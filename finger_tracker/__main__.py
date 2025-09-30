"""Entry point for running the finger tracker as a module."""

from .tracker import FingerTracker

if __name__ == "__main__":
    tracker = FingerTracker()
    tracker.run()