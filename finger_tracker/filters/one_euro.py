"""One-Euro filter (Casiez et al. 2012) — adaptive low-pass tuned by velocity.

High-frequency jitter at rest gets heavier smoothing; fast motion gets lighter
smoothing to minimize lag. Two knobs:
  - min_cutoff: low-speed cutoff (Hz). Lower = more smoothing at rest.
  - beta: speed coefficient. Higher = more aggressive tracking during motion.
"""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _LowPass:
    def __init__(self) -> None:
        self.y: float | None = None

    def filter(self, x: float, alpha: float) -> float:
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y

    def reset(self) -> None:
        self.y = None


class _OneEuro1D:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_ts: float | None = None
        self._last_x: float | None = None

    def filter(self, x: float, timestamp: float) -> float:
        dt = (
            1.0 / 60.0
            if self._last_ts is None or timestamp <= self._last_ts
            else timestamp - self._last_ts
        )
        self._last_ts = timestamp

        dx = 0.0 if self._last_x is None else (x - self._last_x) / dt
        self._last_x = x

        dx_hat = self._dx.filter(dx, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        return self._x.filter(x, _alpha(cutoff, dt))

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last_ts = None
        self._last_x = None


class OneEuroStrategy:
    """2D One-Euro filter: independent filters per axis."""

    name = "one_euro"

    MIN_CUTOFF_BOUNDS = (0.1, 5.0)
    BETA_BOUNDS = (0.0, 1.0)

    def __init__(self, min_cutoff: float = 0.3, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._fx = _OneEuro1D(min_cutoff, beta, d_cutoff)
        self._fy = _OneEuro1D(min_cutoff, beta, d_cutoff)

    def filter(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        return self._fx.filter(x, timestamp), self._fy.filter(y, timestamp)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def adjust_min_cutoff(self, delta: float) -> float:
        lo, hi = self.MIN_CUTOFF_BOUNDS
        self.min_cutoff = max(lo, min(hi, self.min_cutoff + delta))
        self._fx.min_cutoff = self.min_cutoff
        self._fy.min_cutoff = self.min_cutoff
        return self.min_cutoff

    def adjust_beta(self, delta: float) -> float:
        lo, hi = self.BETA_BOUNDS
        self.beta = max(lo, min(hi, self.beta + delta))
        self._fx.beta = self.beta
        self._fy.beta = self.beta
        return self.beta

    def set_beta(self, value: float) -> float:
        lo, hi = self.BETA_BOUNDS
        self.beta = max(lo, min(hi, value))
        self._fx.beta = self.beta
        self._fy.beta = self.beta
        return self.beta
