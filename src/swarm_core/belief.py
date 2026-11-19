"""Bayesian log-odds target belief grid with watermark dedup and vectorized updates."""

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
    """Per-cell log-odds belief.

    Deduplication uses a per-drone watermark: only observations with
    sequence number strictly greater than the last accepted for that
    drone are applied.
    """

    def __init__(self, n_cells: int, prior: float = 0.1) -> None:
        self.n_cells = n_cells
        self.log_odds = np.full(n_cells, logit(prior), dtype=np.float64)
        self._watermark: dict[int, float] = {}

    def p(self, idx: int) -> float:
        return float(sigmoid(float(self.log_odds[idx])))

    def probabilities(self) -> np.ndarray:
        lo = self.log_odds
        pos = lo >= 0
        out = np.empty_like(lo)
        out[pos] = 1.0 / (1.0 + np.exp(-lo[pos]))
        exp_lo = np.exp(lo[~pos])
        out[~pos] = exp_lo / (1.0 + exp_lo)
        return out

    def update(self, obs: Observation) -> None:
        drone_id = obs.obs_id[0]
        seq = obs.obs_id[1]
        last = self._watermark.get(drone_id, -1.0)
        if seq <= last:
            return
        self._watermark[drone_id] = seq

        s = obs.sensitivity
        f = obs.false_alarm
        if obs.detected:
            if f <= 0:
                delta = 10.0
            else:
                delta = math.log(max(s, 1e-9) / max(f, 1e-9))
        else:
            delta = math.log(max(1.0 - s, 1e-9) / max(1.0 - f, 1e-9))

        self.log_odds[obs.cell_idx] += delta

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
        b._watermark = dict(self._watermark)
        return b

    def max_divergence(self, other: BeliefGrid) -> float:
        return float(np.max(np.abs(self.probabilities() - other.probabilities())))


def sector_priority(
    belief: BeliefGrid,
    cells: list[int],
    lambda_h: float = 0.5,
) -> float:
    if not cells:
        return 0.0
    ps = [belief.p(i) for i in cells]
    hs = [belief.entropy(i) for i in cells]
    return float(np.mean(ps) * (1.0 + lambda_h * np.mean(hs)))
