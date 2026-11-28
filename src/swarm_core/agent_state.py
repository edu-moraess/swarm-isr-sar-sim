"""Per-agent state estimation from that agent's reported telemetry.

This adapter deliberately accepts measurements, not Drone or World objects.
Each estimator instance belongs to exactly one agent.
"""
from __future__ import annotations

from math import isfinite

from swarm_core.contracts import AgentLocalState


class LocalStateEstimator:
    """Build immutable local snapshots from timestamped reported positions.

    Velocity is a finite-difference estimate from successive reported
    positions. The first sample has zero estimated velocity because no
    temporal baseline exists yet. No physical Drone state is read here.
    """

    def __init__(self, source_id: int) -> None:
        if type(source_id) is not int or source_id < 0:
            raise ValueError("source_id must be a non-negative integer")
        self.source_id = source_id
        self._previous_timestamp_s: float | None = None
        self._previous_position: tuple[float, float, float] | None = None

    def update(
        self,
        *,
        timestamp_s: float,
        reported_position_xyz_m: tuple[float, float, float],
        battery_pct: float,
    ) -> AgentLocalState:
        """Return a snapshot using only this agent's reported measurements."""
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)):
            raise ValueError("timestamp_s must be a finite number")
        if not isfinite(timestamp_s):
            raise ValueError("timestamp_s must be a finite number")
        if type(reported_position_xyz_m) is not tuple or len(reported_position_xyz_m) != 3:
            raise ValueError("reported_position_xyz_m must be a 3-value tuple")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in reported_position_xyz_m
        ):
            raise ValueError("reported_position_xyz_m must contain finite numbers")
        if self._previous_timestamp_s is not None and timestamp_s <= self._previous_timestamp_s:
            raise ValueError("timestamp_s must increase for each agent")

        if self._previous_timestamp_s is None or self._previous_position is None:
            estimated_velocity = (0.0, 0.0, 0.0)
        else:
            dt = timestamp_s - self._previous_timestamp_s
            estimated_velocity = tuple(
                (current - previous) / dt
                for current, previous in zip(
                    reported_position_xyz_m, self._previous_position, strict=True
                )
            )

        snapshot = AgentLocalState(
            source_id=self.source_id,
            timestamp_s=float(timestamp_s),
            estimated_position_xyz_m=reported_position_xyz_m,
            estimated_velocity_xyz_mps=estimated_velocity,
            battery_pct=battery_pct,
        )
        self._previous_timestamp_s = float(timestamp_s)
        self._previous_position = reported_position_xyz_m
        return snapshot
