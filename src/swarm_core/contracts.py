"""Immutable state contracts for the Dual-World architecture.

PhysicalTruthState belongs to the simulator/evaluation boundary. AgentLocalState
and BeliefSnapshot are the only state views intended for agent-side consumers.
All vector and collection fields use tuples and reject mutable containers.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


Vector3 = tuple[float, float, float]


def _validate_vector(name: str, value: Vector3) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be an immutable 3-value tuple")
    if not all(isfinite(component) for component in value):
        raise ValueError(f"{name} must contain three finite values")


@dataclass(frozen=True, slots=True)
class PhysicalTruthState:
    """Authoritative simulator snapshot; never expose this to agent planners."""

    entity_id: int
    timestamp_s: float
    position_xyz_m: Vector3
    velocity_xyz_mps: Vector3
    terrain_elevation_m: float
    cell_states: tuple[tuple[int, int], ...] = ()
    target_states: tuple[tuple[int, int, bool], ...] = ()

    def __post_init__(self) -> None:
        _validate_vector("position_xyz_m", self.position_xyz_m)
        _validate_vector("velocity_xyz_mps", self.velocity_xyz_mps)
        if not isfinite(self.timestamp_s) or not isfinite(self.terrain_elevation_m):
            raise ValueError("timestamp and terrain elevation must be finite")
        if not isinstance(self.cell_states, tuple) or not isinstance(self.target_states, tuple):
            raise ValueError("physical state collections must be tuples")
        if any(not isinstance(item, tuple) for item in self.cell_states):
            raise ValueError("cell state entries must be tuples")
        if any(not isinstance(item, tuple) for item in self.target_states):
            raise ValueError("target state entries must be tuples")


@dataclass(frozen=True, slots=True)
class AgentLocalState:
    """Noisy/estimated state available to one agent, not simulator ground truth."""

    source_id: int
    timestamp_s: float
    estimated_position_xyz_m: Vector3
    estimated_velocity_xyz_mps: Vector3
    battery_pct: float

    def __post_init__(self) -> None:
        _validate_vector("estimated_position_xyz_m", self.estimated_position_xyz_m)
        _validate_vector("estimated_velocity_xyz_mps", self.estimated_velocity_xyz_mps)
        if not isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not isfinite(self.battery_pct) or not 0.0 <= self.battery_pct <= 100.0:
            raise ValueError("battery_pct must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class BeliefSnapshot:
    """Immutable local probabilistic map and contiguous Gossip watermarks."""

    source_id: int
    timestamp_s: float
    probabilities: tuple[float, ...]
    applied_watermarks: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not isinstance(self.probabilities, tuple) or not isinstance(self.applied_watermarks, tuple):
            raise ValueError("belief collections must be tuples")
        if any(not isfinite(p) or not 0.0 <= p <= 1.0 for p in self.probabilities):
            raise ValueError("belief probabilities must be finite and in [0, 1]")
        if any(not isinstance(item, tuple) for item in self.applied_watermarks):
            raise ValueError("watermarks must be immutable tuples")
        if any(origin < 0 or sequence < 0 for origin, sequence in self.applied_watermarks):
            raise ValueError("watermarks must be non-negative")
