#!/usr/bin/env python3
"""
Simple launcher for the finger tracker application.

This file maintains backwards compatibility by allowing users to run:
    python finger_tracker.py

The actual implementation is in the finger_tracker package.
"""

from finger_tracker import FingerTracker

if __name__ == "__main__":
    tracker = FingerTracker()
    tracker.run()