from __future__ import annotations

import ast
from pathlib import Path

import pytest

from swarm_core.agent_state import LocalStateEstimator


def test_first_reported_sample_has_no_assumed_motion() -> None:
    estimator = LocalStateEstimator(source_id=3)

    state = estimator.update(
        timestamp_s=10.0,
        reported_position_xyz_m=(1.0, 2.0, 3.0),
        battery_pct=95.0,
    )

    assert state.source_id == 3
    assert state.estimated_position_xyz_m == (1.0, 2.0, 3.0)
    assert state.estimated_velocity_xyz_mps == (0.0, 0.0, 0.0)


def test_velocity_is_estimated_from_successive_reported_positions() -> None:
    estimator = LocalStateEstimator(source_id=7)
    estimator.update(
        timestamp_s=2.0,
        reported_position_xyz_m=(0.0, 0.0, 1.0),
        battery_pct=90.0,
    )

    state = estimator.update(
        timestamp_s=4.0,
        reported_position_xyz_m=(4.0, -2.0, 3.0),
        battery_pct=88.0,
    )

    assert state.estimated_velocity_xyz_mps == (2.0, -1.0, 1.0)


def test_estimators_keep_independent_per_agent_history() -> None:
    first = LocalStateEstimator(source_id=1)
    second = LocalStateEstimator(source_id=2)
    first.update(
        timestamp_s=1.0,
        reported_position_xyz_m=(0.0, 0.0, 0.0),
        battery_pct=100.0,
    )

    state = second.update(
        timestamp_s=5.0,
        reported_position_xyz_m=(9.0, 9.0, 9.0),
        battery_pct=80.0,
    )

    assert state.estimated_velocity_xyz_mps == (0.0, 0.0, 0.0)
    assert state.source_id == 2


def test_non_monotonic_timestamp_is_rejected_without_advancing_history() -> None:
    estimator = LocalStateEstimator(source_id=1)
    estimator.update(
        timestamp_s=1.0,
        reported_position_xyz_m=(0.0, 0.0, 0.0),
        battery_pct=100.0,
    )

    with pytest.raises(ValueError, match="timestamp_s must increase"):
        estimator.update(
            timestamp_s=1.0,
            reported_position_xyz_m=(100.0, 0.0, 0.0),
            battery_pct=99.0,
        )

    state = estimator.update(
        timestamp_s=2.0,
        reported_position_xyz_m=(2.0, 0.0, 0.0),
        battery_pct=99.0,
    )
    assert state.estimated_velocity_xyz_mps == (2.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "position",
    [
        [0.0, 0.0, 0.0],
        (0.0, float("nan"), 0.0),
        (0.0, 0.0),
    ],
)
def test_invalid_reported_position_is_rejected(position: object) -> None:
    estimator = LocalStateEstimator(source_id=1)
    with pytest.raises(ValueError):
        estimator.update(
            timestamp_s=1.0,
            reported_position_xyz_m=position,  # type: ignore[arg-type]
            battery_pct=100.0,
        )


def test_agent_estimator_has_no_physical_world_or_drone_dependency() -> None:
    source = Path("src/swarm_core/agent_state.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "swarm_core.drone" not in imported_modules
    assert "swarm_core.world" not in imported_modules
    assert "swarm_core.contracts" in imported_modules
