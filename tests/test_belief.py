"""Belief grid: bounds, detection, non-detection, idempotence, Hypothesis properties."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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


def test_idempotence_by_watermark() -> None:
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
    b.update(obs)
    assert b.p(2) == pytest.approx(p1)


def test_logit_sigmoid_roundtrip() -> None:
    for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-6)


def test_vectorized_probabilities_match_scalar() -> None:
    b = BeliefGrid(20, prior=0.15)
    for i in range(5):
        b.update(
            Observation(
                obs_id=(1, float(i), i),
                cell_idx=i,
                detected=True,
                sensitivity=0.9,
                false_alarm=0.02,
            )
        )
    vec = b.probabilities()
    for i in range(20):
        assert vec[i] == pytest.approx(b.p(i), abs=1e-9)


@given(st.floats(0.01, 0.99), st.integers(1, 50))
@settings(max_examples=50)
def test_hypothesis_p_always_in_unit_interval(prior: float, n: int) -> None:
    b = BeliefGrid(n, prior=prior)
    probs = b.probabilities()
    assert bool(np.all((probs >= 0.0) & (probs <= 1.0)))


@given(
    st.lists(
        st.tuples(
            st.integers(0, 9),
            st.booleans(),
            st.floats(0.5, 0.99),
            st.floats(0.001, 0.1),
        ),
        min_size=1,
        max_size=12,
    )
)
@settings(max_examples=30)
def test_hypothesis_order_independence(
    obs_specs: list[tuple[int, bool, float, float]],
) -> None:
    n = 10
    b1 = BeliefGrid(n, prior=0.2)
    b2 = BeliefGrid(n, prior=0.2)
    for i, (cell, det, s, f) in enumerate(obs_specs):
        o = Observation(
            obs_id=(1, float(i), cell),
            cell_idx=cell,
            detected=det,
            sensitivity=s,
            false_alarm=f,
        )
        b1.update(o)
    for i, (cell, det, s, f) in enumerate(reversed(obs_specs)):
        o = Observation(
            obs_id=(2, float(i), cell),
            cell_idx=cell,
            detected=det,
            sensitivity=s,
            false_alarm=f,
        )
        b2.update(o)
    assert b1.max_divergence(b2) <= 1e-6


@given(st.integers(5, 25), st.floats(0.05, 0.4))
@settings(max_examples=25)
def test_hypothesis_cross_fusion_convergence(n_cells: int, prior: float) -> None:
    b_a = BeliefGrid(n_cells, prior=prior)
    b_b = BeliefGrid(n_cells, prior=prior)
    obs_list = [
        Observation(
            obs_id=(1, float(i), i % n_cells),
            cell_idx=i % n_cells,
            detected=(i % 2 == 0),
            sensitivity=0.9,
            false_alarm=0.02,
        )
        for i in range(n_cells)
    ]
    mid = len(obs_list) // 2
    for o in obs_list[:mid]:
        b_a.update(o)
    for o in obs_list[mid:]:
        b_b.update(o)
    for i, o in enumerate(obs_list[mid:]):
        o2 = Observation(
            obs_id=(1, float(100 + i), o.cell_idx),
            cell_idx=o.cell_idx,
            detected=o.detected,
            sensitivity=o.sensitivity,
            false_alarm=o.false_alarm,
        )
        b_a.update(o2)
    for i, o in enumerate(obs_list[:mid]):
        o2 = Observation(
            obs_id=(2, float(100 + i), o.cell_idx),
            cell_idx=o.cell_idx,
            detected=o.detected,
            sensitivity=o.sensitivity,
            false_alarm=o.false_alarm,
        )
        b_b.update(o2)
    assert b_a.max_divergence(b_b) <= 1e-6
