from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from swarm_core.config import PerceptionConfig
from swarm_core.contracts import AgentLocalState
from swarm_core.messages import Point
from swarm_core.rng import RNGManager
from swarm_core.sensors import cells_in_footprint, sense
from swarm_core.world import CellState, World


def _world() -> World:
    return World(
        width=20,
        height=20,
        cell_m=5.0,
        origin=Point(0.0, 0.0, 0.0),
        grid=np.full(400, CellState.FREE, dtype=np.int8),
    )


def _state(
    position: tuple[float, float, float],
    source_id: int = 4,
    timestamp_s: float = 1.0,
) -> AgentLocalState:
    return AgentLocalState(
        source_id=source_id,
        timestamp_s=timestamp_s,
        estimated_position_xyz_m=position,
        estimated_velocity_xyz_mps=(0.0, 0.0, 0.0),
        battery_pct=90.0,
    )


def test_footprint_uses_agent_estimated_position() -> None:
    world = _world()
    inside = cells_in_footprint(world, (50.0, 50.0, 30.0), radius=15.0)
    outside = cells_in_footprint(world, (150.0, 150.0, 30.0), radius=15.0)

    assert inside
    assert outside == []


def test_sense_uses_agent_local_pose_and_source_identity() -> None:
    world = _world()
    cfg = PerceptionConfig()
    rng = RNGManager(23).spawn("agent-local-sensor")[0]

    batch = sense(_state((50.0, 50.0, 30.0), source_id=9), world, cfg, 1.0, rng)

    assert batch.drone_id == 9
    assert batch.seq == 1
    assert batch.observations
    assert all(observation.drone_id == 9 for observation in batch.observations)
    assert world.sensor_sequences == {9: 1}


def test_sense_outside_world_does_not_fall_back_to_physical_pose() -> None:
    world = _world()
    cfg = PerceptionConfig()
    rng = RNGManager(24).spawn("agent-local-sensor")[0]

    batch = sense(_state((150.0, 150.0, 30.0)), world, cfg, 1.0, rng)

    assert batch.observations == []
    assert batch.drone_id == 4


def test_sensor_module_has_no_drone_dependency() -> None:
    source = Path("src/swarm_core/sensors.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "swarm_core.drone" not in imported_modules
    assert "swarm_core.contracts" in imported_modules


def test_sense_rejects_mutable_or_invalid_agent_state_contract() -> None:
    # AgentLocalState validates the immutable pose contract before perception.
    try:
        AgentLocalState(
            source_id=1,
            timestamp_s=0.0,
            estimated_position_xyz_m=[1.0, 2.0, 3.0],  # type: ignore[arg-type]
            estimated_velocity_xyz_mps=(0.0, 0.0, 0.0),
            battery_pct=100.0,
        )
    except ValueError as exc:
        assert "immutable 3-value tuple" in str(exc)
    else:
        raise AssertionError("mutable pose must not cross the perception boundary")



def test_mission_routes_latest_local_snapshot_to_sensor(monkeypatch) -> None:
    from swarm_core.config import (
        AreaConfig,
        MissionConfig,
        SwarmConfig,
        TargetsConfig,
    )
    from swarm_core.mission import MissionLoop
    from swarm_core.messages import ObservationBatch

    mission = MissionConfig(
        mission_id="local_perception_test",
        seed=5,
        area=AreaConfig(size_m=[40.0, 40.0], cell_m=5.0),
        base=[2.5, 2.5],
        targets=TargetsConfig(count=0, clutter=0),
    )
    loop = MissionLoop(
        mission=mission,
        swarm=SwarmConfig(n_drones=1),
        perception=PerceptionConfig(),
        obstacle_fraction=0.0,
    )
    loop.start()
    captured = []

    def fake_sense(agent_state, world, cfg, timestamp_s, rng):
        captured.append(agent_state)
        return ObservationBatch(
            drone_id=agent_state.source_id,
            seq=1,
            timestamp=timestamp_s,
            observations=[],
        )

    monkeypatch.setattr("swarm_core.mission.sense", fake_sense)
    loop._sense_and_share()

    assert len(captured) == 1
    assert captured[0] is loop.agent_local_states[loop.drones[0].id]
    assert isinstance(captured[0], AgentLocalState)
