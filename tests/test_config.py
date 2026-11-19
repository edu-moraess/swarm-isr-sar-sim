"""Config loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from swarm_core.config import (
    load_mission,
    load_network,
    load_perception,
    load_swarm,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def test_load_mission() -> None:
    m = load_mission(CONFIG / "mission.yaml")
    assert m.mission_id == "sar_open_field_001"
    assert m.area.size_m == [200, 200]
    assert m.targets.count == 8


def test_load_swarm() -> None:
    s = load_swarm(CONFIG / "swarm.yaml")
    assert s.n_drones == 10
    assert s.allocation == "cbba"


def test_load_perception() -> None:
    p = load_perception(CONFIG / "perception.yaml")
    assert p.detector.sensitivity == 0.92
    assert p.gps_sigma_m == 1.5


def test_load_network() -> None:
    n = load_network(CONFIG / "network.yaml")
    assert "clean" in n.profiles
    assert n.profiles["harsh"].loss_pct == 20


def test_invalid_n_drones() -> None:
    with pytest.raises(ValidationError):
        from swarm_core.config import SwarmConfig

        SwarmConfig(n_drones=0)
