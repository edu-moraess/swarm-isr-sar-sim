"""Deterministic RNG based on numpy SeedSequence.spawn."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator, SeedSequence


class RNGManager:
    """Root SeedSequence that spawns independent streams per component."""

    def __init__(self, seed: int = 0) -> None:
        self.root = SeedSequence(seed)
        self._streams: dict[str, Generator] = {}

    def spawn(self, name: str, n: int = 1) -> list[Generator]:
        """Spawn n independent Generators for a named component."""
        children = self.root.spawn(n)
        gens = [np.random.default_rng(child) for child in children]
        if n == 1:
            self._streams[name] = gens[0]
        else:
            for i, g in enumerate(gens):
                self._streams[f"{name}_{i}"] = g
        return gens

    def get(self, name: str) -> Generator:
        """Return a previously spawned stream or spawn a new one."""
        if name not in self._streams:
            self.spawn(name, 1)
        return self._streams[name]

    def integers(self, name: str, low: int, high: int | None = None, size: Any = None) -> Any:
        return self.get(name).integers(low, high, size=size)

    def random(self, name: str, size: Any = None) -> Any:
        return self.get(name).random(size=size)

    def normal(self, name: str, loc: float = 0.0, scale: float = 1.0, size: Any = None) -> Any:
        return self.get(name).normal(loc, scale, size=size)

    def choice(
        self, name: str, a: Any, size: Any = None, replace: bool = True, p: Any = None
    ) -> Any:
        return self.get(name).choice(a, size=size, replace=replace, p=p)
