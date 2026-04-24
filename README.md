# Finger Tracker Mouse Controller

A computer vision-based mouse controller that uses hand tracking for cursor movement and gesture-based clicking. Built with MediaPipe and OpenCV.

## Features

- Real-time hand detection and tracking using MediaPipe
- Cursor control via index finger movement
- Click detection through thumb-finger proximity
- Configurable motion smoothing to reduce jitter
- Motion prediction for handling brief occlusions
- Coordinate calibration system for improved accuracy
- Multiple interaction modes (single-click and drag-hold)

## Requirements

- Python 3.7+
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

```bash
git clone <repository-url>
cd pointer
uv sync
```

Requires [uv](https://github.com/astral-sh/uv). Pinned to Python 3.11 / 3.12.

## Usage

The application can be launched with:

```bash
uv run pointer
# or
uv run python -m finger_tracker
```

For programmatic usage:

```python
from finger_tracker import FingerTracker
tracker = FingerTracker()
tracker.run()
```

## Controls

### Gestures
- Move index finger to control cursor
- Thumb ↔ index finger PIP (first knuckle) — left click / drag
- Thumb ↔ middle finger PIP — right click
- Closed fist, move vertically — scroll
- Open palm held for ~1 second — toggle pause (cursor + clicks frozen)

### Keyboard Commands
- `q` - Exit application
- `m` - Toggle between HOLD (drag) and CLICK modes
- `c` - Enter calibration mode
- `d` - Delete saved calibration
- `s` - Toggle smoothing on/off
- `f` - Cycle smoothing filter (one_euro → legacy → kalman)
- `[` / `]` - Adjust smoothing parameter (filter-dependent)
- `+` / `-` - Adjust click sensitivity threshold
- `r` - Reset mouse state

### Calibration
1. Press `c` to enter calibration mode
2. Position index finger at each corner of desired tracking area
3. Press `SPACE` at each corner (4 points required)
4. Calibration data is saved automatically to `calibration.json`
5. Press `ESC` to cancel

## Configuration

Custom configurations can be passed to the tracker:

```python
from finger_tracker import Config, FingerTracker

config = Config(
    click_distance_threshold=10,
    smoothing_factor=0.7,
    click_mode='hold',
    prediction_enabled=True
)

tracker = FingerTracker(config)
tracker.run()
```

## Architecture

```
pointer/
├── finger_tracker/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # Module entry point
│   ├── config.py            # Immutable Config dataclass
│   ├── state.py             # Per-frame FrameState dataclass
│   ├── capture.py           # Sync + threaded camera capture
│   ├── calibration.py       # Coordinate mapping + calibration
│   ├── detection.py         # MediaPipe hand detection + downscale
│   ├── gestures.py          # Gesture classifier + debouncer
│   ├── mouse.py             # PyAutoGUI mouse control + hysteresis/debounce
│   ├── smoothing.py         # Smoother facade with pluggable strategies
│   ├── filters/             # One-Euro, Kalman, legacy MA+EMA strategies
│   └── tracker.py           # Orchestrator (update/render split)
├── tests/                   # pytest suite
├── pyproject.toml
├── uv.lock
└── README.md
```

## Technical Overview

The application processes webcam frames through the following pipeline:

1. Hand landmark detection using MediaPipe
2. Extraction of index finger tip and thumb positions
3. Motion smoothing via moving average and exponential smoothing
4. Velocity-based position prediction during occlusions
5. Coordinate mapping from camera space to screen space
6. Click detection via Euclidean distance measurement
7. Mouse control through PyAutoGUI

## Troubleshooting

### Jittery Cursor
- Increase smoothing by pressing `[`
- Ensure adequate lighting conditions
- Keep hand within camera field of view

### Click Detection Issues
- Increase threshold with `+`
- Touch thumb to index finger knuckle (not fingertip)
- Monitor yellow distance indicator in video feed

### Performance Issues
- Decrease smoothing with `]`
- Close resource-intensive applications
- Reduce camera resolution (modify source)

### Stuck Mouse Button
- Press `r` to reset
- Switch to CLICK mode with `m`

## License

MIT

## Dependencies

- [MediaPipe](https://google.github.io/mediapipe/) - Hand tracking
- [OpenCV](https://opencv.org/) - Computer vision
- [PyAutoGUI](https://pyautogui.readthedocs.io/) - Mouse control