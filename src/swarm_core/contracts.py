"""Immutable state contracts for the Dual-World architecture.

PhysicalTruthState belongs to the simulator/evaluation boundary. AgentLocalState
and BeliefSnapshot are agent-facing immutable snapshots. Nested collections are
validated recursively so frozen dataclasses cannot retain mutable payloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

Vector3: TypeAlias = tuple[float, float, float]


def _finite_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _non_negative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_vector(name: str, value: Vector3) -> None:
    if type(value) is not tuple or len(value) != 3:
        raise ValueError(f"{name} must be an immutable 3-value tuple")
    for component in value:
        _finite_number(name, component)


def _validate_cell_states(value: tuple[tuple[int, int], ...]) -> None:
    if type(value) is not tuple:
        raise ValueError("cell_states must be a tuple")
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("each cell state must be an immutable (cell_id, state) tuple")
        _non_negative_int("cell_states.cell_id", item[0])
        _non_negative_int("cell_states.state", item[1])


def _validate_target_states(value: tuple[tuple[int, int, bool], ...]) -> None:
    if type(value) is not tuple:
        raise ValueError("target_states must be a tuple")
    for item in value:
        if type(item) is not tuple or len(item) != 3:
            raise ValueError("each target state must be an immutable (target_id, cell_id, confirmed) tuple")
        _non_negative_int("target_states.target_id", item[0])
        _non_negative_int("target_states.cell_id", item[1])
        if type(item[2]) is not bool:
            raise ValueError("target_states.confirmed must be bool")


def _validate_watermarks(value: tuple[tuple[int, int], ...]) -> None:
    if type(value) is not tuple:
        raise ValueError("applied_watermarks must be a tuple")
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("each watermark must be an immutable (origin_id, sequence) tuple")
        _non_negative_int("applied_watermarks.origin_id", item[0])
        _non_negative_int("applied_watermarks.sequence", item[1])


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
        _non_negative_int("entity_id", self.entity_id)
        _finite_number("timestamp_s", self.timestamp_s)
        _validate_vector("position_xyz_m", self.position_xyz_m)
        _validate_vector("velocity_xyz_mps", self.velocity_xyz_mps)
        _finite_number("terrain_elevation_m", self.terrain_elevation_m)
        _validate_cell_states(self.cell_states)
        _validate_target_states(self.target_states)


@dataclass(frozen=True, slots=True)
class AgentLocalState:
    """Noisy/estimated state available to one agent, not simulator ground truth."""

    source_id: int
    timestamp_s: float
    estimated_position_xyz_m: Vector3
    estimated_velocity_xyz_mps: Vector3
    battery_pct: float

    def __post_init__(self) -> None:
        _non_negative_int("source_id", self.source_id)
        _finite_number("timestamp_s", self.timestamp_s)
        _validate_vector("estimated_position_xyz_m", self.estimated_position_xyz_m)
        _validate_vector("estimated_velocity_xyz_mps", self.estimated_velocity_xyz_mps)
        _finite_number("battery_pct", self.battery_pct)
        if not 0.0 <= self.battery_pct <= 100.0:
            raise ValueError("battery_pct must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class BeliefSnapshot:
    """Immutable local probabilistic map and contiguous Gossip watermarks."""

    source_id: int
    timestamp_s: float
    probabilities: tuple[float, ...]
    applied_watermarks: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        _non_negative_int("source_id", self.source_id)
        _finite_number("timestamp_s", self.timestamp_s)
        if type(self.probabilities) is not tuple:
            raise ValueError("probabilities must be an immutable tuple")
        for probability in self.probabilities:
            _finite_number("probabilities", probability)
            if not 0.0 <= probability <= 1.0:
                raise ValueError("belief probabilities must be in [0, 1]")
        _validate_watermarks(self.applied_watermarks)
