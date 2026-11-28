"""Point-mass drone dynamics, battery and noisy GPS."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator

from swarm_core.config import BatteryConfig, SwarmConfig
from swarm_core.messages import DroneRole, MissionPhase, Point
from swarm_core.world import World


@dataclass
class Drone:
    id: int
    position: np.ndarray  # true 3D [x,y,z]
    velocity: np.ndarray
    battery_pct: float = 100.0
    phase: MissionPhase = MissionPhase.INIT
    role: DroneRole = DroneRole.WORKER
    waypoints: list[np.ndarray] = field(default_factory=list)
    wp_idx: int = 0
    leader_term: int = 0
    alive: bool = True
    gps_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def reported_position(self) -> Point:
        return Point(
            x=float(self.gps_pos[0]), y=float(self.gps_pos[1]), z=float(self.gps_pos[2])
        )

    def true_position(self) -> Point:
        return Point(
            x=float(self.position[0]), y=float(self.position[1]), z=float(self.position[2])
        )


def create_drones(
    swarm: SwarmConfig,
    base: list[float],
    rng: Generator,
) -> list[Drone]:
    """Spawn n drones near base with formation offset."""
    drones: list[Drone] = []
    spacing = swarm.formation.spacing_m
    for i in range(swarm.n_drones):
        if swarm.formation.type == "line":
            offset = np.array([i * spacing, 0.0, 0.0])
        elif swarm.formation.type == "column":
            offset = np.array([0.0, i * spacing, 0.0])
        else:  # grid
            cols = int(np.ceil(np.sqrt(swarm.n_drones)))
            row, col = divmod(i, cols)
            offset = np.array([col * spacing, row * spacing, 0.0])
        pos = np.array([base[0], base[1], 0.0]) + offset
        d = Drone(
            id=i + 1,
            position=pos.copy(),
            velocity=np.zeros(3),
            battery_pct=100.0,
            gps_pos=pos.copy(),
        )
        drones.append(d)
    return drones


def step_drone(
    drone: Drone,
    dt: float,
    swarm: SwarmConfig,
    battery: BatteryConfig,
    world: World,
    gps_sigma: float,
    rng: Generator,
    a_max: float = 2.0,
    tol: float = 1.5,
    arrival_speed_mps: float = 0.25,
) -> None:
    """Integrate a bounded-acceleration point mass with braking near waypoints.

    The desired speed is limited by both cruise speed and a stopping-distance
    envelope. A waypoint is consumed only after position and speed tolerances
    are simultaneously satisfied; this avoids carrying high residual velocity
    through a waypoint.
    """
    if not drone.alive:
        return

    speed = float(np.linalg.norm(drone.velocity[:2]))
    drain = battery.drain_hover_pct_s + battery.drain_per_mps_pct_s * speed
    drone.battery_pct = max(0.0, drone.battery_pct - drain * dt)

    if drone.waypoints and drone.wp_idx < len(drone.waypoints):
        target = drone.waypoints[drone.wp_idx]
        err = target - drone.position
        dist = float(np.linalg.norm(err))
        if dist > 1e-9:
            direction = err / dist
            stopping_speed = float(np.sqrt(2.0 * a_max * dist))
            proportional_speed = 1.5 * dist
            desired_speed = min(swarm.cruise_mps, stopping_speed, proportional_speed)
            desired_v = direction * desired_speed
        else:
            desired_v = np.zeros(3)

        dv = desired_v - drone.velocity
        max_dv = a_max * dt
        dv_norm = float(np.linalg.norm(dv))
        if dv_norm > max_dv:
            dv = dv / dv_norm * max_dv
        drone.velocity = drone.velocity + dv

        if dist <= tol and float(np.linalg.norm(drone.velocity)) <= arrival_speed_mps:
            drone.wp_idx += 1
    else:
        drone.velocity *= 0.9

    wind = world.wind_velocity()
    drone.velocity[:2] += wind * dt * 0.1
    drone.position = drone.position + drone.velocity * dt

    noise = rng.normal(0, gps_sigma, size=3)
    drone.gps_pos = drone.position + noise


def min_separation(drones: list[Drone]) -> float:
    """Return minimum pairwise horizontal distance among alive drones."""
    alive = [d for d in drones if d.alive]
    if len(alive) < 2:
        return float("inf")
    mins = []
    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            d = np.linalg.norm(alive[i].position[:2] - alive[j].position[:2])
            mins.append(d)
    return float(min(mins)) if mins else float("inf")
