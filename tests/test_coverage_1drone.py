"""Acceptance: 1 drone covers ≥ 90 % of free cells without intervention."""

from __future__ import annotations

import numpy as np

from swarm_core.belief import BeliefGrid
from swarm_core.clock import SimClock
from swarm_core.config import MissionConfig, PerceptionConfig, SwarmConfig
from swarm_core.drone import create_drones, step_drone
from swarm_core.planning.boustrophedon import boustrophedon_waypoints
from swarm_core.rng import RNGManager
from swarm_core.sensors import sense
from swarm_core.world import build_world


def test_one_drone_coverage_ge_90() -> None:
    seed = 0
    rng_mgr = RNGManager(seed)
    world_rng = rng_mgr.spawn("world")[0]
    sensor_rng = rng_mgr.spawn("sensors")[0]
    drone_rng = rng_mgr.spawn("drones")[0]

    mission = MissionConfig(seed=seed)
    swarm = SwarmConfig(n_drones=1, cruise_mps=8.0)
    perc = PerceptionConfig()
    world = build_world(mission, world_rng, obstacle_fraction=0.02)
    drones = create_drones(swarm, mission.base, drone_rng)
    drone = drones[0]

    wps = boustrophedon_waypoints(world, altitude=30.0)
    drone.waypoints = [np.array([p.x, p.y, p.z]) for p in wps]
    drone.position[2] = 30.0

    belief = BeliefGrid(world.n_cells, prior=0.1)
    clock = SimClock(dt=0.1)
    observed: set[int] = set()
    free = set(world.free_cells())

    max_steps = int(900 / clock.dt)
    for _ in range(max_steps):
        t = clock.tick()
        step_drone(drone, clock.dt, swarm, swarm.battery, world, perc.gps_sigma_m, drone_rng)
        obs = sense(drone, world, perc, t, sensor_rng)
        belief.fuse(obs)
        for o in obs:
            observed.add(o.cell_idx)
        if drone.wp_idx >= len(drone.waypoints):
            break

    coverage = len(observed & free) / max(len(free), 1)
    assert coverage >= 0.90, f"coverage={coverage:.3f} < 0.90"
