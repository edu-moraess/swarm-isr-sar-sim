"""Tests for agent-local route-lane assignment."""
from __future__ import annotations

import pytest

from swarm_core.contracts import AgentLocalState
from swarm_core.messages import Point
from swarm_core.planning.agent_assignment import assign_route_lanes


def _state(source_id: int, x: float, y: float, z: float = 0.0) -> AgentLocalState:
    return AgentLocalState(
        source_id=source_id,
        timestamp_s=1.0,
        estimated_position_xyz_m=(x, y, z),
        estimated_velocity_xyz_mps=(0.0, 0.0, 0.0),
        battery_pct=100.0,
    )


def _route() -> list[Point]:
    return [
        Point(0.0, 0.0, 10.0),
        Point(10.0, 0.0, 10.0),
        Point(0.0, 10.0, 10.0),
        Point(10.0, 10.0, 10.0),
    ]


def test_assigns_nearest_estimated_agent_to_each_interleaved_lane() -> None:
    route = _route()
    states = {
        1: _state(1, 10.0, 0.0, 10.0),
        2: _state(2, 0.0, 0.0, 10.0),
    }

    assigned = assign_route_lanes(route, states)

    assert assigned[1] == [route[1], route[3]]
    assert assigned[2] == [route[0], route[2]]


def test_assignment_is_deterministic_when_distances_tie() -> None:
    states = {1: _state(1, 5.0, 0.0, 10.0), 2: _state(2, 5.0, 0.0, 10.0)}

    first = assign_route_lanes(_route(), states)
    second = assign_route_lanes(_route(), states)

    assert first == second
    assert first[1] == [_route()[0], _route()[2]]
    assert first[2] == [_route()[1], _route()[3]]


def test_all_route_waypoints_are_assigned_once_when_lanes_are_nonempty() -> None:
    route = _route()
    assigned = assign_route_lanes(
        route,
        {1: _state(1, 0.0, 0.0), 2: _state(2, 10.0, 0.0)},
    )

    flattened = [point for lane in assigned.values() for point in lane]
    assert len(flattened) == len(route)
    assert sorted((p.x, p.y, p.z) for p in flattened) == sorted(
        (p.x, p.y, p.z) for p in route
    )


def test_preserves_legacy_fallback_when_agents_outnumber_waypoints() -> None:
    route = [Point(1.0, 2.0, 3.0)]
    assigned = assign_route_lanes(
        route,
        {1: _state(1, 1.0, 2.0, 3.0), 2: _state(2, 1.0, 2.0, 3.0)},
    )

    assert assigned == {1: [route[0]], 2: [route[0]]}


def test_rejects_state_mapping_with_mismatched_identity() -> None:
    with pytest.raises(ValueError, match="source_id"):
        assign_route_lanes(_route(), {1: _state(2, 0.0, 0.0)})


def test_rejects_empty_route_or_agent_set() -> None:
    with pytest.raises(ValueError, match="route"):
        assign_route_lanes([], {1: _state(1, 0.0, 0.0)})
    with pytest.raises(ValueError, match="agent_states"):
        assign_route_lanes(_route(), {})
