"""Structured JSONL event log with optional hash chain."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Event:
    t: float
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    prev_sha256: str = ""

    def to_line(self) -> str:
        d = {
            "t": self.t,
            "kind": self.kind,
            "payload": self.payload,
            "prev_sha256": self.prev_sha256,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":"))


class EventLog:
    """Append-only in-memory + optional file JSONL log with hash chain."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[Event] = []
        self._last_sha = "0" * 64

    def emit(self, t: float, kind: str, **payload: Any) -> Event:
        ev = Event(t=t, kind=kind, payload=payload, prev_sha256=self._last_sha)
        line = ev.to_line()
        self._last_sha = hashlib.sha256(line.encode("utf-8")).hexdigest()
        self.events.append(ev)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return ev

    def sha256(self) -> str:
        """Hash of the entire ordered event stream (for determinism tests)."""
        h = hashlib.sha256()
        for ev in self.events:
            h.update(ev.to_line().encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def verify_chain(self) -> bool:
        """Validate the hash chain of all events."""
        prev = "0" * 64
        for ev in self.events:
            if ev.prev_sha256 != prev:
                return False
            prev = hashlib.sha256(ev.to_line().encode("utf-8")).hexdigest()
        return True

    def clear(self) -> None:
        self.events.clear()
        self._last_sha = "0" * 64
        if self.path and self.path.exists():
            self.path.unlink()
