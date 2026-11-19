"""Sector decomposition (20×20 m default)."""

from __future__ import annotations

from swarm_core.world import World


def sector_id(cell_idx: int, world: World, sector_m: float = 20.0) -> int:
    cells_per_sector = max(1, int(sector_m / world.cell_m))
    r, c = divmod(cell_idx, world.width)
    sr = r // cells_per_sector
    sc = c // cells_per_sector
    n_sc = max(1, (world.width + cells_per_sector - 1) // cells_per_sector)
    return sr * n_sc + sc


def sector_cells(world: World, sector_m: float = 20.0) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for idx in world.free_cells():
        sid = sector_id(idx, world, sector_m)
        result.setdefault(sid, []).append(idx)
    return result


def n_sectors(world: World, sector_m: float = 20.0) -> int:
    cells_per = max(1, int(sector_m / world.cell_m))
    nx = max(1, (world.width + cells_per - 1) // cells_per)
    ny = max(1, (world.height + cells_per - 1) // cells_per)
    return nx * ny
