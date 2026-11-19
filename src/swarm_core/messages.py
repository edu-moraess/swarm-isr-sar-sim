"""Dataclasses mirroring the ROS 2 IDL messages (Tier A side)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MissionPhase(IntEnum):
    INIT = 0
    TAKEOFF = 1
    SEARCH = 2
    VERIFY = 3
    RETURN = 4
    LAND = 5
    COMPLETE = 6
    DEGRADED = 7
    RTL = 8
    ABORT = 9


class DroneRole(IntEnum):
    WORKER = 0
    LEADER = 1
    RELAY = 2
    VERIFIER = 3


class AlertType(IntEnum):
    TARGET = 0
    NEW_OBJECT = 1
    MOTION = 2
    NODE_LOST = 3
    LOW_BATTERY = 4


class AlertSeverity(IntEnum):
    INFO = 0
    WARN = 1
    CRIT = 2


@dataclass(frozen=True)
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class SwarmState:
    drone_id: int
    position: Point
    velocity: Vector3
    battery_pct: float
    phase: MissionPhase
    role: DroneRole
    leader_term: int = 0
    timestamp: float = 0.0


@dataclass
class TaskAssignment:
    drone_id: int
    task_id: int
    waypoints: list[Point] = field(default_factory=list)
    score: float = 0.0
    epoch: int = 0
    timestamp: float = 0.0


@dataclass
class ProbabilityGrid:
    resolution: float
    width: int
    height: int
    origin: Point
    p_target: list[float]
    timestamp: float = 0.0


@dataclass
class Alert:
    type: AlertType
    severity: AlertSeverity
    position: Point
    text: str = ""
    timestamp: float = 0.0


@dataclass
class LeaderElection:
    leader_id: int
    term: int
    kind: int
    sender_id: int
    timestamp: float = 0.0


@dataclass
class NetworkStatus:
    latency_ms: float
    loss_pct: float
    partitioned: bool
    profile: str
    timestamp: float = 0.0


@dataclass
class Observation:
    """Single cell observation. seq is per-drone monotonic counter."""

    drone_id: int
    seq: int
    cell_idx: int
    detected: bool
    sensitivity: float
    false_alarm: float
    timestamp: float = 0.0


@dataclass
class ObservationBatch:
    """All observations from one drone at one sense step.

    seq is a single integer per batch (monotonic per drone).
    Gossip retransmits from the receiver's watermark for that drone.
    """

    drone_id: int
    seq: int
    timestamp: float
    observations: list[Observation] = field(default_factory=list)


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, IntEnum):
        return int(obj)
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, tuple):
        return list(obj)
    return obj
