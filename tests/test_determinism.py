"""Two runs with same seed produce identical event-log SHA-256."""

from __future__ import annotations

from swarm_core.clock import SimClock
from swarm_core.config import MissionConfig, PerceptionConfig, SwarmConfig
from swarm_core.drone import create_drones, step_drone
from swarm_core.events import EventLog
from swarm_core.rng import RNGManager
from swarm_core.sensors import sense
from swarm_core.telemetry import AgentTelemetryAdapter
from swarm_core.world import build_world


def _run_short(seed: int) -> str:
    rng_mgr = RNGManager(seed)
    world_rng = rng_mgr.spawn("world")[0]
    sensor_rng = rng_mgr.spawn("sensors")[0]
    drone_rng = rng_mgr.spawn("drones")[0]

    mission = MissionConfig(seed=seed)
    swarm = SwarmConfig(n_drones=3)
    perc = PerceptionConfig()
    world = build_world(mission, world_rng)
    drones = create_drones(swarm, mission.base, drone_rng)
    telemetry = {
        drone.id: AgentTelemetryAdapter(source_id=drone.id)
        for drone in drones
    }
    clock = SimClock(dt=0.1)
    log = EventLog()

    for step in range(20):
        t = clock.tick()
        world.update_wind(clock.dt)
        for d in drones:
            step_drone(d, clock.dt, swarm, swarm.battery, world, perc.gps_sigma_m, drone_rng)
            if step % 5 == 0:
                local_state = telemetry[d.id].update(
                    timestamp_s=t,
                    reported_position=d.reported_position(),
                    battery_pct=d.battery_pct,
                )
                batch = sense(local_state, world, perc, t, sensor_rng)
                log.emit(t, "sense", drone_id=d.id, n_obs=len(batch.observations), seq=batch.seq)
        log.emit(t, "tick", step=clock.step)

    return log.sha256()


def test_same_seed_same_hash() -> None:
    h1 = _run_short(42)
    h2 = _run_short(42)
    assert h1 == h2


def test_different_seed_different_hash() -> None:
    h1 = _run_short(1)
    h2 = _run_short(2)
    assert h1 != h2
