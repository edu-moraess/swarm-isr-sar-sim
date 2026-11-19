"""Formation offset generators (line, column, grid)."""

from __future__ import annotations

import numpy as np


def formation_offsets(
    n: int,
    kind: str = "line",
    spacing_m: float = 15.0,
) -> list[np.ndarray]:
    offsets: list[np.ndarray] = []
    for i in range(n):
        if kind == "line":
            o = np.array([i * spacing_m, 0.0, 0.0])
        elif kind == "column":
            o = np.array([0.0, i * spacing_m, 0.0])
        else:
            cols = int(np.ceil(np.sqrt(n)))
            row, col = divmod(i, cols)
            o = np.array([col * spacing_m, row * spacing_m, 0.0])
        offsets.append(o)
    return offsets
