"""Bayesian log-odds occupancy / target belief grid with idempotent fusion."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from swarm_core.messages import Observation


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def sigmoid(log_odds: float) -> float:
    if log_odds >= 0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    e = math.exp(log_odds)
    return e / (1.0 + e)


class BeliefGrid:
    """Per-cell log-odds. Prior uniform → log-odds = 0 (p=0.5) or low prior."""

    def __init__(self, n_cells: int, prior: float = 0.1) -> None:
        self.n_cells = n_cells
        self.log_odds = np.full(n_cells, logit(prior), dtype=np.float64)
        self._seen_obs: set[tuple[int, float, int]] = set()

    def p(self, idx: int) -> float:
        return sigmoid(float(self.log_odds[idx]))

    def probabilities(self) -> np.ndarray:
        return np.array([sigmoid(float(lo)) for lo in self.log_odds])

    def update(self, obs: Observation) -> None:
        """Idempotent Bayesian update. Ignores duplicate obs_id."""
        if obs.obs_id in self._seen_obs:
            return
        self._seen_obs.add(obs.obs_id)

        s = obs.sensitivity
        f = obs.false_alarm
        p = self.p(obs.cell_idx)

        if obs.detected:
            num = s * p
            den = num + f * (1.0 - p)
            p_new = num / den if den > 0 else p
        else:
            num = (1.0 - s) * p
            den = num + (1.0 - f) * (1.0 - p)
            p_new = num / den if den > 0 else p

        self.log_odds[obs.cell_idx] = logit(p_new)

    def fuse(self, observations: Iterable[Observation]) -> None:
        for o in observations:
            self.update(o)

    def entropy(self, idx: int) -> float:
        p = self.p(idx)
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    def mean_entropy(self, cells: list[int] | None = None) -> float:
        idxs = cells if cells is not None else list(range(self.n_cells))
        if not idxs:
            return 0.0
        return float(np.mean([self.entropy(i) for i in idxs]))

    def copy(self) -> BeliefGrid:
        b = BeliefGrid(self.n_cells)
        b.log_odds = self.log_odds.copy()
        b._seen_obs = set(self._seen_obs)
        return b

    def max_divergence(self, other: BeliefGrid) -> float:
        """Max absolute difference in probability (for convergence tests)."""
        return float(np.max(np.abs(self.probabilities() - other.probabilities())))


def sector_priority(
    belief: BeliefGrid,
    cells: list[int],
    lambda_h: float = 0.5,
) -> float:
    """priority = mean(p) * (1 + λ * mean(H))."""
    if not cells:
        return 0.0
    ps = [belief.p(i) for i in cells]
    hs = [belief.entropy(i) for i in cells]
    return float(np.mean(ps) * (1.0 + lambda_h * np.mean(hs)))
