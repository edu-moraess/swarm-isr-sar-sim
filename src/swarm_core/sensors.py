"""Footprint, line-of-sight (Bresenham) and detection model."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.random import Generator

from swarm_core.config import PerceptionConfig
from swarm_core.drone import Drone
from swarm_core.messages import Observation
from swarm_core.world import CellState, World


def footprint_radius(alt_m: float, cfg: PerceptionConfig) -> float:
    """Lookup or interpolate footprint radius by altitude."""
    table = {float(k): v for k, v in cfg.footprint_radius_m.items()}
    if alt_m in table:
        return table[alt_m]
    keys = sorted(table.keys())
    if alt_m <= keys[0]:
        return table[keys[0]]
    if alt_m >= keys[-1]:
        return table[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= alt_m <= keys[i + 1]:
            t = (alt_m - keys[i]) / (keys[i + 1] - keys[i])
            return table[keys[i]] * (1 - t) + table[keys[i + 1]] * t
    return 15.0


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Grid cells on the line from (x0,y0) to (x1,y1)."""
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return cells


def line_of_sight(world: World, from_idx: int, to_idx: int) -> bool:
    """True if no OCCUPIED cell blocks the Bresenham path (excluding endpoints)."""
    r0, c0 = divmod(from_idx, world.width)
    r1, c1 = divmod(to_idx, world.width)
    path = bresenham_line(c0, r0, c1, r1)
    for c, r in path[1:-1]:
        idx = r * world.width + c
        if 0 <= idx < world.n_cells and world.grid[idx] == CellState.OCCUPIED:
            return False
    return True


def cells_in_footprint(world: World, drone: Drone, radius: float) -> list[int]:
    """Return free cell indices inside circular footprint of the drone."""
    cx, cy = drone.position[0], drone.position[1]
    result: list[int] = []
    for idx in range(world.n_cells):
        if not world.is_free(idx):
            continue
        x, y = world.idx_to_xy(idx)
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
            d_idx = world.xy_to_idx(cx, cy)
            if d_idx is None or line_of_sight(world, d_idx, idx):
                result.append(idx)
    return result


def sensitivity(dist: float, radius: float, s0: float = 0.92) -> float:
    """s(d) = s0 for d <= 0.5r, linear decay to 0.60 at d = r."""
    if dist <= 0.5 * radius:
        return s0
    if dist >= radius:
        return 0.60
    t = (dist - 0.5 * radius) / (0.5 * radius)
    return s0 + t * (0.60 - s0)


def sense(
    drone: Drone,
    world: World,
    cfg: PerceptionConfig,
    t: float,
    rng: Generator,
) -> list[Observation]:
    """Produce observations for all cells in the current footprint."""
    alt = float(drone.position[2])
    radius = footprint_radius(alt, cfg)
    cells = cells_in_footprint(world, drone, radius)
    s0 = cfg.detector.sensitivity
    f = cfg.detector.false_alarm
    obs: list[Observation] = []

    target_cells = {tg.cell_idx for tg in world.targets}
    clutter_cells = {c.cell_idx for c in world.clutter}

    for cell in cells:
        x, y = world.idx_to_xy(cell)
        dist = float(np.hypot(x - drone.position[0], y - drone.position[1]))
        s = sensitivity(dist, radius, s0)

        if cell in target_cells:
            detected = bool(rng.random() < s)
        elif cell in clutter_cells:
            detected = bool(rng.random() < f)
        else:
            detected = False

        obs_id = (drone.id, t, cell)
        obs.append(
            Observation(
                obs_id=obs_id,
                cell_idx=cell,
                detected=detected,
                sensitivity=s,
                false_alarm=f,
                timestamp=t,
            )
        )
    return obs
