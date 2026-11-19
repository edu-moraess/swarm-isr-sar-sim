"""Belief: batch updates, watermark, Hypothesis properties."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swarm_core.belief import BeliefGrid, logit, sigmoid
from swarm_core.messages import Observation, ObservationBatch


def test_p_in_unit_interval() -> None:
    b = BeliefGrid(10, prior=0.1)
    assert all(0.0 <= b.p(i) <= 1.0 for i in range(10))


def test_batch_updates_n_cells() -> None:
    """A batch with N cells must alter N cells (not just 1)."""
    b = BeliefGrid(20, prior=0.2)
    prior_p = b.p(0)
    n = 7
    obs = [
        Observation(
            drone_id=1,
            seq=1,
            cell_idx=i,
            detected=True,
            sensitivity=0.92,
            false_alarm=0.02,
        )
        for i in range(n)
    ]
    batch = ObservationBatch(drone_id=1, seq=1, timestamp=0.0, observations=obs)
    updated = b.update_batch(batch)
    assert updated == n
    for i in range(n):
        assert b.p(i) != pytest.approx(prior_p)
    assert b.update_batch(batch) == 0


def test_detection_increases_p() -> None:
    b = BeliefGrid(5, prior=0.2)
    p0 = b.p(0)
    batch = ObservationBatch(
        drone_id=1,
        seq=1,
        timestamp=0.0,
        observations=[Observation(1, 1, 0, True, 0.92, 0.02, 0.0)],
    )
    b.update_batch(batch)
    assert b.p(0) > p0


def test_nondetection_decreases_p() -> None:
    b = BeliefGrid(5, prior=0.5)
    p0 = b.p(0)
    batch = ObservationBatch(
        drone_id=1,
        seq=1,
        timestamp=0.0,
        observations=[Observation(1, 1, 0, False, 0.92, 0.02, 0.0)],
    )
    b.update_batch(batch)
    assert b.p(0) < p0


def test_watermark_idempotence() -> None:
    b = BeliefGrid(5, prior=0.3)
    batch = ObservationBatch(
        drone_id=1,
        seq=5,
        timestamp=1.0,
        observations=[Observation(1, 5, 2, True, 0.9, 0.01, 1.0)],
    )
    b.update_batch(batch)
    p1 = b.p(2)
    b.update_batch(batch)
    assert b.p(2) == pytest.approx(p1)
    assert b.watermark(1) == 5


def test_logit_sigmoid_roundtrip() -> None:
    for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-6)


@given(st.floats(0.01, 0.99), st.integers(1, 50))
@settings(max_examples=40)
def test_hypothesis_p_in_unit(prior: float, n: int) -> None:
    b = BeliefGrid(n, prior=prior)
    probs = b.probabilities()
    assert bool(np.all((probs >= 0.0) & (probs <= 1.0)))


@given(st.integers(5, 20), st.floats(0.05, 0.4))
@settings(max_examples=20)
def test_hypothesis_cross_fusion(n_cells: int, prior: float) -> None:
    b_a = BeliefGrid(n_cells, prior=prior)
    b_b = BeliefGrid(n_cells, prior=prior)
    obs = [
        Observation(1, 1, i % n_cells, i % 2 == 0, 0.9, 0.02, 0.0)
        for i in range(n_cells)
    ]
    batch = ObservationBatch(1, 1, 0.0, obs)
    b_a.update_batch(batch)
    b_b.update_batch(batch)
    assert b_a.max_divergence(b_b) <= 1e-6
