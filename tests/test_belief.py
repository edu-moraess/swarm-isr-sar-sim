"""Belief grid: bounds, detection increases P, non-detection decreases, idempotence."""

from __future__ import annotations

import pytest

from swarm_core.belief import BeliefGrid, logit, sigmoid
from swarm_core.messages import Observation


def test_p_in_unit_interval() -> None:
    b = BeliefGrid(10, prior=0.1)
    for i in range(10):
        p = b.p(i)
        assert 0.0 <= p <= 1.0


def test_detection_increases_p() -> None:
    b = BeliefGrid(5, prior=0.2)
    p0 = b.p(0)
    obs = Observation(
        obs_id=(1, 0.0, 0),
        cell_idx=0,
        detected=True,
        sensitivity=0.92,
        false_alarm=0.02,
    )
    b.update(obs)
    assert b.p(0) > p0


def test_nondetection_decreases_p() -> None:
    b = BeliefGrid(5, prior=0.5)
    p0 = b.p(0)
    obs = Observation(
        obs_id=(1, 0.0, 0),
        cell_idx=0,
        detected=False,
        sensitivity=0.92,
        false_alarm=0.02,
    )
    b.update(obs)
    assert b.p(0) < p0


def test_idempotence_by_obs_id() -> None:
    b = BeliefGrid(5, prior=0.3)
    obs = Observation(
        obs_id=(1, 1.0, 2),
        cell_idx=2,
        detected=True,
        sensitivity=0.9,
        false_alarm=0.01,
    )
    b.update(obs)
    p1 = b.p(2)
    b.update(obs)  # same obs_id
    assert b.p(2) == pytest.approx(p1)


def test_logit_sigmoid_roundtrip() -> None:
    for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-6)
