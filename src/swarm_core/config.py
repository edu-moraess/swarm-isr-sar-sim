"""Pydantic models and YAML loaders for mission, swarm, perception and network."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class AreaConfig(BaseModel):
    type: Literal["rect", "polygon"] = "rect"
    size_m: list[float] = Field(default_factory=lambda: [200.0, 200.0])
    cell_m: float = 5.0
    origin: list[float] = Field(default_factory=lambda: [0.0, 0.0])

    @field_validator("size_m")
    @classmethod
    def size_positive(cls, v: list[float]) -> list[float]:
        if len(v) != 2 or any(x <= 0 for x in v):
            raise ValueError("size_m must be two positive numbers")
        return v


class StopConfig(BaseModel):
    coverage_pct: float = 95.0
    all_targets_confirmed: bool = True


class TargetsConfig(BaseModel):
    count: int = 8
    clutter: int = 12


class PriorConfig(BaseModel):
    type: Literal["uniform", "hotspots"] = "uniform"
    hotspots: list[dict[str, Any]] = Field(default_factory=list)


class SectorsConfig(BaseModel):
    size_m: float = 20.0
    priorities: dict[str, float] = Field(default_factory=dict)


class OperatorConfig(BaseModel):
    mode: Literal["active", "shadow"] = "active"
    script: str | None = "config/operator_script.yaml"


class MissionConfig(BaseModel):
    mission_id: str = "sar_open_field_001"
    seed: int = 0
    area: AreaConfig = Field(default_factory=AreaConfig)
    base: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    time_limit_s: float = 900.0
    stop: StopConfig = Field(default_factory=StopConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    prior: PriorConfig = Field(default_factory=PriorConfig)
    sectors: SectorsConfig = Field(default_factory=SectorsConfig)
    operator: OperatorConfig = Field(default_factory=OperatorConfig)


class FormationConfig(BaseModel):
    type: Literal["line", "column", "grid"] = "line"
    spacing_m: float = 15.0


class BatteryConfig(BaseModel):
    drain_hover_pct_s: float = 0.040
    drain_per_mps_pct_s: float = 0.004
    rtl_reserve_pct: float = 15.0


class SwarmConfig(BaseModel):
    n_drones: int = 10
    formation: FormationConfig = Field(default_factory=FormationConfig)
    allocation: Literal["cbba", "auction", "greedy", "b1_fixed", "b2_central"] = "cbba"
    consensus: Literal["raft_like", "bully"] = "raft_like"
    cruise_mps: float = 5.0
    search_alt_m: float = 30.0
    verify_alt_m: float = 15.0
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    comm_range_m: float = 150.0

    @field_validator("n_drones")
    @classmethod
    def n_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_drones must be >= 1")
        return v


class DetectorConfig(BaseModel):
    kind: Literal["aruco", "yolo"] = "aruco"
    sensitivity: float = 0.92
    false_alarm: float = 0.02


class PerceptionConfig(BaseModel):
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    gps_sigma_m: float = 1.5
    footprint_radius_m: dict[str, float] = Field(default_factory=lambda: {"30": 15.0, "15": 7.5})


class NetworkProfile(BaseModel):
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    loss_pct: float = 0.0
    burst: bool = False
    partition: dict[str, Any] | None = None


class JammingConfig(BaseModel):
    region: dict[str, Any] = Field(
        default_factory=lambda: {"center": [100.0, 100.0], "radius_m": 40.0}
    )
    start_s: float = 120.0
    duration_s: float = 60.0
    loss_pct: float = 60.0


class QoSConfig(BaseModel):
    reliable_retries: int = 3
    ack_timeout_ms: float = 200.0


class NetworkConfig(BaseModel):
    profiles: dict[str, NetworkProfile] = Field(
        default_factory=lambda: {
            "clean": NetworkProfile(),
            "mild": NetworkProfile(latency_ms=50, jitter_ms=20, loss_pct=5),
            "harsh": NetworkProfile(latency_ms=200, jitter_ms=80, loss_pct=20, burst=True),
            "partition": NetworkProfile(
                latency_ms=50,
                jitter_ms=20,
                loss_pct=5,
                partition={
                    "start_s": 300,
                    "duration_s": 60,
                    "groups": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                },
            ),
        }
    )
    jamming: JammingConfig = Field(default_factory=JammingConfig)
    qos: QoSConfig = Field(default_factory=QoSConfig)


class FullConfig(BaseModel):
    mission: MissionConfig = Field(default_factory=MissionConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)

    @model_validator(mode="after")
    def check_consistency(self) -> FullConfig:
        if self.swarm.n_drones < 1:
            raise ValueError("n_drones must be positive")
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {p}")
    return data


def load_mission(path: str | Path = "config/mission.yaml") -> MissionConfig:
    return MissionConfig.model_validate(load_yaml(path))


def load_swarm(path: str | Path = "config/swarm.yaml") -> SwarmConfig:
    return SwarmConfig.model_validate(load_yaml(path))


def load_perception(path: str | Path = "config/perception.yaml") -> PerceptionConfig:
    return PerceptionConfig.model_validate(load_yaml(path))


def load_network(path: str | Path = "config/network.yaml") -> NetworkConfig:
    return NetworkConfig.model_validate(load_yaml(path))


def load_full(
    mission_path: str | Path = "config/mission.yaml",
    swarm_path: str | Path = "config/swarm.yaml",
    perception_path: str | Path = "config/perception.yaml",
    network_path: str | Path = "config/network.yaml",
) -> FullConfig:
    """Load and validate all config files into a single FullConfig."""
    return FullConfig(
        mission=load_mission(mission_path),
        swarm=load_swarm(swarm_path),
        perception=load_perception(perception_path),
        network=load_network(network_path),
    )
