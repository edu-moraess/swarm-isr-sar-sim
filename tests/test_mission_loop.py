"""Deterministic integration tests for the Tier A mission orchestrator."""

from __future__ import annotations

from swarm_core.config import (
    AreaConfig,
    MissionConfig,
    PerceptionConfig,
    SwarmConfig,
    TargetsConfig,
)
from swarm_core.messages import MissionPhase
from swarm_core.mission import MissionLoop


def _make_loop(seed: int = 7) -> MissionLoop:
    mission = MissionConfig(
        mission_id="mission_loop_test",
        seed=seed,
        area=AreaConfig(size_m=[40.0, 40.0], cell_m=5.0),
        base=[2.5, 2.5],
        time_limit_s=180.0,
        targets=TargetsConfig(count=2, clutter=0),
    )
    swarm = SwarmConfig(n_drones=1, search_alt_m=10.0, cruise_mps=5.0)
    return MissionLoop(
        mission=mission,
        swarm=swarm,
        perception=PerceptionConfig(),
        dt=0.1,
        sense_period_steps=5,
        obstacle_fraction=0.0,
    )


def test_mission_loop_same_seed_same_event_hash() -> None:
    first = _make_loop(seed=17).run(max_steps=25)
    second = _make_loop(seed=17).run(max_steps=25)

    assert first.event_sha256 == second.event_sha256
    assert first.steps == second.steps == 25
    assert first.completion_reason == "step_limit"
    assert first.event_chain_valid
    assert second.event_chain_valid


def test_mission_loop_completes_full_lifecycle() -> None:
    loop = _make_loop()
    result = loop.run()

    assert result.completed
    assert result.phase == MissionPhase.COMPLETE
    assert result.completion_reason == "coverage_routes_completed"
    assert result.steps > 0
    assert result.simulated_time_s == result.steps * loop.clock.dt
    assert result.event_chain_valid
    assert result.event_count > result.steps
    assert result.total_free_cells > 0
    assert 0.0 <= result.coverage_pct <= 100.0
    assert [transition.to for transition in loop.fsm.history] == [
        MissionPhase.TAKEOFF,
        MissionPhase.SEARCH,
        MissionPhase.RETURN,
        MissionPhase.LAND,
        MissionPhase.COMPLETE,
    ]


def test_mission_loop_requires_start_before_manual_step() -> None:
    loop = _make_loop()
    try:
        loop.step()
    except RuntimeError as exc:
        assert "start()" in str(exc)
    else:
        raise AssertionError("step() must reject an unstarted mission")


def test_mission_loop_rejects_invalid_sense_period() -> None:
    mission = MissionConfig()
    swarm = SwarmConfig(n_drones=1)
    try:
        MissionLoop(
            mission,
            swarm,
            PerceptionConfig(),
            sense_period_steps=0,
        )
    except ValueError as exc:
        assert "sense_period_steps" in str(exc)
    else:
        raise AssertionError("zero sense period must be rejected")


def test_mission_loop_rejects_unknown_network_profile() -> None:
    try:
        MissionLoop(
            MissionConfig(),
            SwarmConfig(n_drones=1),
            PerceptionConfig(),
            network_profile="missing-profile",
        )
    except ValueError as exc:
        assert "Unknown network profile" in str(exc)
    else:
        raise AssertionError("unknown network profile must be rejected")
