"""Two runs with same seed produce identical event-log SHA-256."""

from __future__ import annotations

from swarm_core.clock import SimClock
from swarm_core.config import MissionConfig, PerceptionConfig, SwarmConfig
from swarm_core.drone import create_drones, step_drone
from swarm_core.events import EventLog
from swarm_core.rng import RNGManager
from swarm_core.sensors import sense
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
    clock = SimClock(dt=0.1)
    log = EventLog()

    for _ in range(20):
        t = clock.tick()
        world.update_wind(clock.dt)
        for d in drones:
            step_drone(d, clock.dt, swarm, swarm.battery, world, perc.gps_sigma_m, drone_rng)
            obs = sense(d, world, perc, t, sensor_rng)
            log.emit(t, "sense", drone_id=d.id, n_obs=len(obs))
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
