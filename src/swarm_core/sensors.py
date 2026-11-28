"""Footprint (rectangle-bounded), Bresenham LoS and detection model."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from swarm_core.config import PerceptionConfig
from swarm_core.contracts import AgentLocalState
from swarm_core.messages import Observation, ObservationBatch
from swarm_core.world import CellState, World


def footprint_radius(alt_m: float, cfg: PerceptionConfig) -> float:
    table = {float(k): v for k, v in cfg.footprint_radius_m.items()}
    if alt_m in table:
        return table[alt_m]
    keys = sorted(table.keys())
    if not keys:
        return 15.0
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
    r0, c0 = divmod(from_idx, world.width)
    r1, c1 = divmod(to_idx, world.width)
    path = bresenham_line(c0, r0, c1, r1)
    for c, r in path[1:-1]:
        idx = r * world.width + c
        if 0 <= idx < world.n_cells and world.grid[idx] == CellState.OCCUPIED:
            return False
    return True


def cells_in_footprint(
    world: World, position_xyz_m: tuple[float, float, float], radius: float
) -> list[int]:
    """Iterate the footprint around an agent-reported position."""
    cx, cy = float(position_xyz_m[0]), float(position_xyz_m[1])
    cell = world.cell_m
    r_cells = int(np.ceil(radius / cell)) + 1
    col0 = max(0, int((cx - world.origin.x) / cell) - r_cells)
    col1 = min(world.width - 1, int((cx - world.origin.x) / cell) + r_cells)
    row0 = max(0, int((cy - world.origin.y) / cell) - r_cells)
    row1 = min(world.height - 1, int((cy - world.origin.y) / cell) + r_cells)

    d_idx = world.xy_to_idx(cx, cy)
    result: list[int] = []
    r2 = radius * radius
    for row in range(row0, row1 + 1):
        for col in range(col0, col1 + 1):
            idx = row * world.width + col
            if not world.is_free(idx):
                continue
            x = world.origin.x + (col + 0.5) * cell
            y = world.origin.y + (row + 0.5) * cell
            if (x - cx) ** 2 + (y - cy) ** 2 > r2:
                continue
            if d_idx is None or line_of_sight(world, d_idx, idx):
                result.append(idx)
    return result


def sensitivity(dist: float, radius: float, s0: float = 0.92) -> float:
    if dist <= 0.5 * radius:
        return s0
    if dist >= radius:
        return 0.60
    t = (dist - 0.5 * radius) / (0.5 * radius)
    return s0 + t * (0.60 - s0)




def sense(
    agent_state: AgentLocalState,
    world: World,
    cfg: PerceptionConfig,
    t: float,
    rng: Generator,
) -> ObservationBatch:
    """Produce observations using only the agent-local estimated pose.

    World remains the environment/evaluation model (free cells, occlusion,
    target and clutter labels); the sensor geometry never reads true drone pose.
    """
    position = agent_state.estimated_position_xyz_m
    alt = float(position[2])
    radius = footprint_radius(alt, cfg)
    cells = cells_in_footprint(world, position, radius)
    s0 = cfg.detector.sensitivity
    f = cfg.detector.false_alarm

    seq = world.sensor_sequences.get(agent_state.source_id, 0) + 1
    world.sensor_sequences[agent_state.source_id] = seq

    target_cells = {tg.cell_idx for tg in world.targets}
    clutter_cells = {c.cell_idx for c in world.clutter}

    obs_list: list[Observation] = []
    for cell in cells:
        x, y = world.idx_to_xy(cell)
        dist = float(np.hypot(x - position[0], y - position[1]))
        s = sensitivity(dist, radius, s0)

        if cell in target_cells:
            detected = bool(rng.random() < s)
        elif cell in clutter_cells:
            detected = bool(rng.random() < f)
        else:
            detected = False

        obs_list.append(
            Observation(
                drone_id=agent_state.source_id,
                seq=seq,
                cell_idx=cell,
                detected=detected,
                sensitivity=s,
                false_alarm=f,
                timestamp=t,
            )
        )

    return ObservationBatch(
        drone_id=agent_state.source_id,
        seq=seq,
        timestamp=t,
        observations=obs_list,
    )

