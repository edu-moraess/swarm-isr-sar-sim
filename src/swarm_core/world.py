"""Grid world: cells, obstacles, synthetic targets, clutter, wind."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.random import Generator

from swarm_core.config import MissionConfig
from swarm_core.messages import Point


class CellState(IntEnum):
    FREE = 0
    OCCUPIED = 1
    UNKNOWN = 2


@dataclass
class Target:
    id: int
    cell_idx: int
    position: Point
    confirmed: bool = False
    detections: int = 0


@dataclass
class Clutter:
    id: int
    cell_idx: int
    position: Point


@dataclass
class World:
    width: int
    height: int
    cell_m: float
    origin: Point
    grid: np.ndarray  # CellState
    targets: list[Target] = field(default_factory=list)
    clutter: list[Clutter] = field(default_factory=list)
    wind: np.ndarray = field(default_factory=lambda: np.zeros(2))
    gust_ar1: float = 0.0
    rng: Generator | None = None

    @property
    def n_cells(self) -> int:
        return self.width * self.height

    def idx_to_xy(self, idx: int) -> tuple[float, float]:
        row, col = divmod(idx, self.width)
        x = self.origin.x + (col + 0.5) * self.cell_m
        y = self.origin.y + (row + 0.5) * self.cell_m
        return x, y

    def xy_to_idx(self, x: float, y: float) -> int | None:
        col = int((x - self.origin.x) / self.cell_m)
        row = int((y - self.origin.y) / self.cell_m)
        if 0 <= col < self.width and 0 <= row < self.height:
            return row * self.width + col
        return None

    def is_free(self, idx: int) -> bool:
        return bool(self.grid[idx] == CellState.FREE)

    def free_cells(self) -> list[int]:
        return [i for i in range(self.n_cells) if self.is_free(i)]

    def update_wind(self, dt: float, sigma: float = 0.5) -> None:
        """Constant wind + AR(1) gust."""
        if self.rng is None:
            return
        self.gust_ar1 = 0.9 * self.gust_ar1 + self.rng.normal(0, sigma * np.sqrt(dt))

    def wind_velocity(self) -> Any:
        return self.wind + np.array([self.gust_ar1, 0.0])


def build_world(
    mission: MissionConfig,
    rng: Generator,
    obstacle_fraction: float = 0.05,
) -> World:
    """Create a deterministic world from mission config and RNG stream."""
    area = mission.area
    cell_m = area.cell_m
    w = int(area.size_m[0] / cell_m)
    h = int(area.size_m[1] / cell_m)
    n = w * h
    grid = np.full(n, CellState.FREE, dtype=np.int8)

    n_obs = int(n * obstacle_fraction)
    if n_obs > 0:
        obs_idx = rng.choice(n, size=n_obs, replace=False)
        grid[obs_idx] = CellState.OCCUPIED

    free = [i for i in range(n) if grid[i] == CellState.FREE]
    origin = Point(x=area.origin[0], y=area.origin[1], z=0.0)

    tgt_cfg = mission.targets
    n_tgt = min(tgt_cfg.count, len(free))
    tgt_cells = rng.choice(free, size=n_tgt, replace=False)
    targets: list[Target] = []
    used = set(tgt_cells.tolist())
    for i, c in enumerate(tgt_cells):
        x, y = _idx_to_xy(c, w, cell_m, origin)
        targets.append(Target(id=i, cell_idx=int(c), position=Point(x=x, y=y, z=0.0)))

    remaining = [i for i in free if i not in used]
    n_clut = min(tgt_cfg.clutter, len(remaining))
    clut_cells: list[int] = (
        list(rng.choice(remaining, size=n_clut, replace=False)) if n_clut else []
    )
    clutter: list[Clutter] = []
    for i, c in enumerate(clut_cells):
        x, y = _idx_to_xy(int(c), w, cell_m, origin)
        clutter.append(Clutter(id=i, cell_idx=int(c), position=Point(x=x, y=y, z=0.0))

    wind = rng.normal(0, 0.3, size=2)

    return World(
        width=w,
        height=h,
        cell_m=cell_m,
        origin=origin,
        grid=grid,
        targets=targets,
        clutter=clutter,
        wind=wind,
        rng=rng,
    )


def _idx_to_xy(idx: int, width: int, cell_m: float, origin: Point) -> tuple[float, float]:
    row, col = divmod(idx, width)
    x = origin.x + (col + 0.5) * cell_m
    y = origin.y + (row + 0.5) * cell_m
    return x, y
