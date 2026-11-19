"""Simulated discrete clock with fixed dt = 0.1 s."""

from __future__ import annotations


class SimClock:
    """Deterministic simulated time. No wall-clock calls."""

    def __init__(self, dt: float = 0.1) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self._t = 0.0
        self._step = 0

    @property
    def t(self) -> float:
        return self._t

    @property
    def step(self) -> int:
        return self._step

    def tick(self) -> float:
        """Advance one step and return new time."""
        self._step += 1
        self._t = self._step * self.dt
        return self._t

    def reset(self) -> None:
        self._t = 0.0
        self._step = 0

    def __repr__(self) -> str:
        return f"SimClock(t={self._t:.1f}, step={self._step}, dt={self.dt})"
