"""Acceptance: 1 drone covers ≥ 90 %; belief diverges from prior on observed cells."""

from __future__ import annotations

import time

import numpy as np

from swarm_core.belief import BeliefGrid
from swarm_core.clock import SimClock
from swarm_core.config import MissionConfig, PerceptionConfig, SwarmConfig
from swarm_core.drone import create_drones, step_drone
from swarm_core.planning.boustrophedon import boustrophedon_waypoints
from swarm_core.rng import RNGManager
from swarm_core.sensors import reset_sense_seq, sense
from swarm_core.world import build_world


def test_one_drone_coverage_ge_90() -> None:
    t0 = time.perf_counter()
    seed = 0
    reset_sense_seq()
    rng_mgr = RNGManager(seed)
    world_rng = rng_mgr.spawn("world")[0]
    sensor_rng = rng_mgr.spawn("sensors")[0]
    drone_rng = rng_mgr.spawn("drones")[0]

    mission = MissionConfig(seed=seed)
    swarm = SwarmConfig(n_drones=1)
    perc = PerceptionConfig()
    world = build_world(mission, world_rng, obstacle_fraction=0.02)
    drones = create_drones(swarm, mission.base, drone_rng)
    drone = drones[0]

    wps = boustrophedon_waypoints(world, altitude=30.0)
    drone.waypoints = [np.array([p.x, p.y, p.z]) for p in wps]
    drone.position[2] = 30.0

    prior = 0.1
    belief = BeliefGrid(world.n_cells, prior=prior)
    clock = SimClock(dt=0.1)
    observed: set[int] = set()
    free = set(world.free_cells())

    sense_period = 5
    max_steps = int(900 / clock.dt)
    for step in range(max_steps):
        t = clock.tick()
        step_drone(drone, clock.dt, swarm, swarm.battery, world, perc.gps_sigma_m, drone_rng)
        if step % sense_period == 0:
            batch = sense(drone, world, perc, t, sensor_rng)
            belief.update_batch(batch)
            for o in batch.observations:
                observed.add(o.cell_idx)
        if drone.wp_idx >= len(drone.waypoints):
            break

    coverage = len(observed & free) / max(len(free), 1)
    assert coverage >= 0.90, f"coverage={coverage:.3f} < 0.90"

    changed = sum(1 for i in observed if abs(belief.p(i) - prior) > 1e-6)
    assert changed >= 0.5 * len(observed), (
        f"only {changed}/{len(observed)} observed cells changed from prior"
    )

    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"test took {elapsed:.2f}s > 5s"
