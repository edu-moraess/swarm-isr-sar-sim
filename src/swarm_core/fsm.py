"""Mission finite-state machine with strict transition table."""

from __future__ import annotations

from dataclasses import dataclass

from swarm_core.messages import MissionPhase


TRANSITIONS: dict[tuple[MissionPhase, str], MissionPhase] = {
    (MissionPhase.INIT, "cmd_start"): MissionPhase.TAKEOFF,
    (MissionPhase.TAKEOFF, "altitude_reached"): MissionPhase.SEARCH,
    (MissionPhase.SEARCH, "candidate"): MissionPhase.VERIFY,
    (MissionPhase.VERIFY, "confirm_done"): MissionPhase.SEARCH,
    (MissionPhase.SEARCH, "tasks_done"): MissionPhase.RETURN,
    (MissionPhase.VERIFY, "tasks_done"): MissionPhase.RETURN,
    (MissionPhase.SEARCH, "battery_low"): MissionPhase.RTL,
    (MissionPhase.VERIFY, "battery_low"): MissionPhase.RTL,
    (MissionPhase.SEARCH, "lost_leader"): MissionPhase.DEGRADED,
    (MissionPhase.VERIFY, "lost_leader"): MissionPhase.DEGRADED,
    (MissionPhase.DEGRADED, "contact_restored"): MissionPhase.SEARCH,
    (MissionPhase.DEGRADED, "timeout_rtl"): MissionPhase.RTL,
    (MissionPhase.RETURN, "at_base"): MissionPhase.LAND,
    (MissionPhase.RTL, "at_base"): MissionPhase.LAND,
    (MissionPhase.LAND, "landed"): MissionPhase.COMPLETE,
    (MissionPhase.INIT, "cmd_abort"): MissionPhase.ABORT,
    (MissionPhase.TAKEOFF, "cmd_abort"): MissionPhase.ABORT,
    (MissionPhase.SEARCH, "cmd_abort"): MissionPhase.ABORT,
    (MissionPhase.VERIFY, "cmd_abort"): MissionPhase.ABORT,
    (MissionPhase.RETURN, "cmd_abort"): MissionPhase.ABORT,
    (MissionPhase.DEGRADED, "cmd_abort"): MissionPhase.ABORT,
    (MissionPhase.RTL, "cmd_abort"): MissionPhase.ABORT,
    (MissionPhase.ABORT, "at_base"): MissionPhase.LAND,
}


@dataclass
class Transition:
    frm: MissionPhase
    event: str
    to: MissionPhase


class MissionFSM:
    """Strict FSM: illegal transitions raise ValueError."""

    def __init__(self, initial: MissionPhase = MissionPhase.INIT) -> None:
        self.phase = initial
        self.history: list[Transition] = []

    def trigger(self, event: str) -> MissionPhase:
        key = (self.phase, event)
        if key not in TRANSITIONS:
            raise ValueError(f"Illegal transition: {self.phase.name} + '{event}'")
        new = TRANSITIONS[key]
        self.history.append(Transition(self.phase, event, new))
        self.phase = new
        return new

    def can(self, event: str) -> bool:
        return (self.phase, event) in TRANSITIONS
