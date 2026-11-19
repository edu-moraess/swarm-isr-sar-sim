"""Bayesian log-odds belief with per-drone watermark and batch updates."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from swarm_core.messages import Observation, ObservationBatch


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def sigmoid(log_odds: float) -> float:
    if log_odds >= 0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    e = math.exp(log_odds)
    return e / (1.0 + e)


class BeliefGrid:
    """Per-cell log-odds.

    Watermark is per-drone batch sequence: a batch is applied only if
    batch.seq > watermark[drone_id]. All cells in the batch are then
    updated atomically. Gossip can retransmit batches with seq >
    receiver.watermark[sender].
    """

    def __init__(self, n_cells: int, prior: float = 0.1) -> None:
        self.n_cells = n_cells
        self.prior = prior
        self.log_odds = np.full(n_cells, logit(prior), dtype=np.float64)
        self._watermark: dict[int, int] = {}

    def watermark(self, drone_id: int) -> int:
        return self._watermark.get(drone_id, -1)

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

    def _apply_obs(self, obs: Observation) -> None:
        s = obs.sensitivity
        f = obs.false_alarm
        if obs.detected:
            delta = 10.0 if f <= 0 else math.log(max(s, 1e-9) / max(f, 1e-9))
        else:
            delta = math.log(max(1.0 - s, 1e-9) / max(1.0 - f, 1e-9))
        self.log_odds[obs.cell_idx] += delta

    def update(self, obs: Observation) -> None:
        last = self._watermark.get(obs.drone_id, -1)
        if obs.seq <= last:
            return
        self._watermark[obs.drone_id] = obs.seq
        self._apply_obs(obs)

    def update_batch(self, batch: ObservationBatch) -> int:
        """Apply entire batch if seq is newer. Returns number of cells updated."""
        last = self._watermark.get(batch.drone_id, -1)
        if batch.seq <= last:
            return 0
        self._watermark[batch.drone_id] = batch.seq
        for obs in batch.observations:
            self._apply_obs(obs)
        return len(batch.observations)

    def fuse(self, observations: Iterable[Observation]) -> None:
        for o in observations:
            self.update(o)

    def fuse_batches(self, batches: Iterable[ObservationBatch]) -> int:
        n = 0
        for b in batches:
            n += self.update_batch(b)
        return n

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
        b = BeliefGrid(self.n_cells, prior=self.prior)
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
