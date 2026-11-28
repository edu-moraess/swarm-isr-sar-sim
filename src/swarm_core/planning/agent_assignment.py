"""Deterministic route-lane assignment using agent-local estimated poses.

The geometric coverage route remains generated from the environment model.
This module only assigns its existing interleaved lanes to agents.
"""
from __future__ import annotations

from math import dist
from typing import Mapping, Sequence

from swarm_core.contracts import AgentLocalState
from swarm_core.messages import Point


def assign_route_lanes(
    route: Sequence[Point],
    agent_states: Mapping[int, AgentLocalState],
) -> dict[int, list[Point]]:
    """Assign interleaved route lanes to agents by estimated start proximity.

    Candidate agent/lane pairs are greedily matched in ascending 3D distance.
    Ties are resolved by agent ID and lane index, making the heuristic stable.
    This is a deterministic heuristic, not a globally optimal assignment.
    """
    if not route:
        raise ValueError("route must contain at least one waypoint")
    if not agent_states:
        raise ValueError("agent_states must contain at least one agent")

    agent_ids = sorted(agent_states)
    if any(type(agent_id) is not int or agent_id < 0 for agent_id in agent_ids):
        raise ValueError("agent IDs must be non-negative integers")
    if any(
        not isinstance(state, AgentLocalState) or state.source_id != agent_id
        for agent_id, state in agent_states.items()
    ):
        raise ValueError("each mapping key must match its AgentLocalState.source_id")

    lane_count = len(agent_ids)
    lanes = [list(route[index::lane_count]) for index in range(lane_count)]
    for index, lane in enumerate(lanes):
        if not lane:
            # Preserve the legacy fallback for more agents than route points.
            lanes[index] = [route[index % len(route)]]

    candidates: list[tuple[float, int, int]] = []
    for agent_id in agent_ids:
        position = agent_states[agent_id].estimated_position_xyz_m
        for lane_index, lane in enumerate(lanes):
            start = lane[0]
            distance = dist(position, (start.x, start.y, start.z))
            candidates.append((distance, agent_id, lane_index))

    assigned_agents: set[int] = set()
    assigned_lanes: set[int] = set()
    result: dict[int, list[Point]] = {}
    for _, agent_id, lane_index in sorted(candidates):
        if agent_id in assigned_agents or lane_index in assigned_lanes:
            continue
        result[agent_id] = lanes[lane_index]
        assigned_agents.add(agent_id)
        assigned_lanes.add(lane_index)
        if len(assigned_agents) == lane_count:
            break

    return {agent_id: result[agent_id] for agent_id in agent_ids}
