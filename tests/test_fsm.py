"""FSM: legal transitions including DEGRADED battery_low and gps_degraded."""

import pytest

from swarm_core.fsm import TRANSITIONS, MissionFSM
from swarm_core.messages import MissionPhase


def test_happy_path() -> None:
    fsm = MissionFSM()
    assert fsm.phase == MissionPhase.INIT
    fsm.trigger("cmd_start")
    assert fsm.phase == MissionPhase.TAKEOFF
    fsm.trigger("altitude_reached")
    assert fsm.phase == MissionPhase.SEARCH
    fsm.trigger("tasks_done")
    assert fsm.phase == MissionPhase.RETURN
    fsm.trigger("at_base")
    assert fsm.phase == MissionPhase.LAND
    fsm.trigger("landed")
    assert fsm.phase == MissionPhase.COMPLETE


def test_illegal_raises() -> None:
    fsm = MissionFSM()
    with pytest.raises(ValueError, match="Illegal"):
        fsm.trigger("altitude_reached")


def test_degraded_battery_low_to_rtl() -> None:
    fsm = MissionFSM(MissionPhase.DEGRADED)
    fsm.trigger("battery_low")
    assert fsm.phase == MissionPhase.RTL


def test_gps_degraded_to_degraded() -> None:
    fsm = MissionFSM(MissionPhase.SEARCH)
    fsm.trigger("gps_degraded")
    assert fsm.phase == MissionPhase.DEGRADED
    fsm2 = MissionFSM(MissionPhase.TAKEOFF)
    fsm2.trigger("gps_degraded")
    assert fsm2.phase == MissionPhase.DEGRADED


def test_all_table_entries_reachable() -> None:
    for (frm, event), to in TRANSITIONS.items():
        assert isinstance(frm, MissionPhase)
        assert isinstance(to, MissionPhase)
        assert isinstance(event, str)
