"""Explicit adapter from reported telemetry messages to local agent state."""

from __future__ import annotations

from swarm_core.agent_state import LocalStateEstimator
from swarm_core.contracts import AgentLocalState
from swarm_core.messages import Point


class AgentTelemetryAdapter:
    """Convert a reported Point and battery reading into an immutable local state.

    This boundary accepts telemetry values only; it does not depend on the
    physical Drone or World implementation.
    """

    def __init__(self, source_id: int) -> None:
        self._estimator = LocalStateEstimator(source_id=source_id)

    def update(
        self,
        *,
        timestamp_s: float,
        reported_position: Point,
        battery_pct: float,
    ) -> AgentLocalState:
        """Project one reported telemetry sample into the agent-local domain."""
        if not isinstance(reported_position, Point):
            raise TypeError("reported_position must be a Point message")
        return self._estimator.update(
            timestamp_s=timestamp_s,
            reported_position_xyz_m=(
                reported_position.x,
                reported_position.y,
                reported_position.z,
            ),
            battery_pct=battery_pct,
        )
