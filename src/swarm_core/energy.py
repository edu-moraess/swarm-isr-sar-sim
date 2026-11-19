"""Battery model and RTL reserve computation."""

from __future__ import annotations

import numpy as np

from swarm_core.config import BatteryConfig
from swarm_core.drone import Drone
from swarm_core.messages import Point


def estimated_return_cost_pct(
    drone: Drone,
    base: Point | list[float],
    battery: BatteryConfig,
    cruise_mps: float = 5.0,
) -> float:
    if isinstance(base, list):
        bx, by = base[0], base[1]
    else:
        bx, by = base.x, base.y
    dist = float(np.hypot(drone.position[0] - bx, drone.position[1] - by))
    time_s = dist / max(cruise_mps, 0.1)
    drain_rate = battery.drain_hover_pct_s + battery.drain_per_mps_pct_s * cruise_mps
    return drain_rate * time_s


def rtl_reserve_pct(
    drone: Drone,
    base: Point | list[float],
    battery: BatteryConfig,
    cruise_mps: float = 5.0,
) -> float:
    return battery.rtl_reserve_pct + estimated_return_cost_pct(
        drone, base, battery, cruise_mps
    )


def should_rtl(
    drone: Drone,
    base: Point | list[float],
    battery: BatteryConfig,
    cruise_mps: float = 5.0,
    low_threshold_pct: float = 20.0,
) -> bool:
    reserve = rtl_reserve_pct(drone, base, battery, cruise_mps)
    return drone.battery_pct <= max(reserve, low_threshold_pct)
