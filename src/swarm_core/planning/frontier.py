"""Frontier-based re-planning: UNKNOWN cells adjacent to SEARCHED."""

from __future__ import annotations

from swarm_core.belief import BeliefGrid
from swarm_core.world import World


def frontier_cells(
    world: World,
    belief: BeliefGrid,
    searched: set[int],
    p_unknown_threshold: float = 0.4,
) -> list[int]:
    frontiers: list[int] = []
    for idx in world.free_cells():
        if idx in searched:
            continue
        if belief.p(idx) < p_unknown_threshold or belief.p(idx) > (1.0 - p_unknown_threshold):
            continue
        r, c = divmod(idx, world.width)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < world.height and 0 <= nc < world.width:
                nidx = nr * world.width + nc
                if nidx in searched:
                    frontiers.append(idx)
                    break
    return frontiers
