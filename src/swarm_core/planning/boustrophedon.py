"""Boustrophedon (lawnmower) coverage path generation."""

from __future__ import annotations

import numpy as np

from swarm_core.messages import Point
from swarm_core.world import World


def boustrophedon_waypoints(
    world: World,
    altitude: float = 30.0,
    stripe_width_m: float | None = None,
    overlap: float = 0.2,
    phase_offset: float = 0.0,
) -> list[Point]:
    if stripe_width_m is None:
        stripe_width_m = 2 * 15.0 * (1.0 - overlap)

    cell = world.cell_m
    n_stripes = max(1, int(np.ceil(world.height * cell / stripe_width_m)))
    waypoints: list[Point] = []

    for s in range(n_stripes):
        y = world.origin.y + (s + 0.5) * stripe_width_m + phase_offset
        y = min(max(y, world.origin.y + cell / 2), world.origin.y + world.height * cell - cell / 2)
        row = int((y - world.origin.y) / cell)
        if row < 0 or row >= world.height:
            continue

        cols = []
        for c in range(world.width):
            idx = row * world.width + c
            if world.is_free(idx):
                cols.append(c)
        if not cols:
            continue

        xs = [world.origin.x + (c + 0.5) * cell for c in cols]
        if s % 2 == 1:
            xs = list(reversed(xs))

        for x in xs:
            waypoints.append(Point(x=x, y=y, z=altitude))

    return waypoints


def path_length(waypoints: list[Point]) -> float:
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i].x - waypoints[i - 1].x
        dy = waypoints[i].y - waypoints[i - 1].y
        total += float(np.hypot(dx, dy))
    return total
