"""Sensor sequence ownership and isolation across independent worlds."""

from __future__ import annotations

from swarm_core.config import MissionConfig, PerceptionConfig, SwarmConfig
from swarm_core.drone import create_drones
from swarm_core.rng import RNGManager
from swarm_core.sensors import sense
from swarm_core.telemetry import AgentTelemetryAdapter
from swarm_core.world import build_world


def _make_run(seed: int = 7):
    rng_manager = RNGManager(seed)
    world_rng = rng_manager.spawn("world")[0]
    drone_rng = rng_manager.spawn("drones")[0]
    sensor_rng = rng_manager.spawn("sensors")[0]
    mission = MissionConfig(seed=seed)
    world = build_world(mission, world_rng)
    drones = create_drones(SwarmConfig(n_drones=1), mission.base, drone_rng)
    drone = drones[0]
    telemetry = AgentTelemetryAdapter(source_id=drone.id)
    return world, drone, sensor_rng, telemetry


def test_sensor_sequences_are_scoped_to_world() -> None:
    cfg = PerceptionConfig()
    world_a, drone_a, rng_a, telemetry_a = _make_run()
    world_b, drone_b, rng_b, telemetry_b = _make_run()

    state_a_1 = telemetry_a.update(
        timestamp_s=0.1,
        reported_position=drone_a.reported_position(),
        battery_pct=drone_a.battery_pct,
    )
    state_a_2 = telemetry_a.update(
        timestamp_s=0.2,
        reported_position=drone_a.reported_position(),
        battery_pct=drone_a.battery_pct,
    )
    state_b_1 = telemetry_b.update(
        timestamp_s=0.1,
        reported_position=drone_b.reported_position(),
        battery_pct=drone_b.battery_pct,
    )

    first = sense(state_a_1, world_a, cfg, 0.1, rng_a)
    second = sense(state_a_2, world_a, cfg, 0.2, rng_a)
    independent = sense(state_b_1, world_b, cfg, 0.1, rng_b)

    assert (first.seq, second.seq) == (1, 2)
    assert independent.seq == 1
    assert world_a.sensor_sequences == {drone_a.id: 2}
    assert world_b.sensor_sequences == {drone_b.id: 1}
