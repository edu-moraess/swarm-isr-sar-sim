"""Regression tests for separated pads and stable landing completion."""

import numpy as np

from swarm_core.config import BatteryConfig, MissionConfig, PerceptionConfig, SwarmConfig
from swarm_core.drone import Drone, step_drone
from swarm_core.messages import Point
from swarm_core.mission import MissionLoop
from swarm_core.world import World


def make_loop(n_drones: int = 10) -> MissionLoop:
    return MissionLoop(
        mission=MissionConfig(seed=42),
        swarm=SwarmConfig(n_drones=n_drones),
        perception=PerceptionConfig(),
    )


def test_landing_pads_are_deterministic_and_separated() -> None:
    loop = make_loop()
    first = loop._make_landing_positions()
    second = loop._make_landing_positions()

    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)

    positions = list(first.values())
    distances = [
        float(np.linalg.norm(positions[i][:2] - positions[j][:2]))
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    ]
    assert min(distances) >= 3.0
    assert all(
        0.0 <= float(position[0]) <= loop.mission_config.area.size_m[0]
        and 0.0 <= float(position[1]) <= loop.mission_config.area.size_m[1]
        for position in positions
    )


def test_landing_completion_requires_position_and_velocity_tolerances() -> None:
    loop = make_loop(n_drones=4)
    loop.landing_positions = loop._make_landing_positions()

    for drone in loop.drones:
        pad = loop.landing_positions[drone.id]
        drone.waypoints = [pad.copy()]
        drone.wp_idx = 1
        drone.position = pad.copy()
        drone.velocity = np.zeros(3)

    assert loop._all_drones_landed()

    loop.drones[0].position[2] = -0.30
    assert not loop._all_drones_landed()
    loop.drones[0].position = loop.landing_positions[1].copy()
    loop.drones[0].velocity[2] = -0.50
    assert not loop._all_drones_landed()


def test_point_mass_brakes_and_settles_before_consuming_landing_waypoint() -> None:
    swarm = SwarmConfig(n_drones=1)
    battery = BatteryConfig()
    world = World(
        width=1,
        height=1,
        cell_m=5.0,
        origin=Point(x=0.0, y=0.0, z=0.0),
        grid=np.zeros(1, dtype=np.int8),
    )
    drone = Drone(
        id=1,
        position=np.array([0.0, 0.0, 30.0]),
        velocity=np.zeros(3),
    )
    drone.waypoints = [np.array([0.0, 0.0, 0.0])]
    rng = np.random.default_rng(42)

    for _ in range(2_000):
        step_drone(
            drone,
            0.1,
            swarm,
            battery,
            world,
            gps_sigma=0.0,
            rng=rng,
            tol=0.20,
            arrival_speed_mps=0.15,
            braking=True,
        )
        if drone.wp_idx == 1:
            break

    assert drone.wp_idx == 1
    assert abs(float(drone.position[2])) <= 0.20
    assert float(np.linalg.norm(drone.velocity)) <= 0.25
