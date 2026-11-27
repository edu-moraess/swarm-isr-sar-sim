"""Network simulator: latency, loss, burst, jamming, partition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from numpy.random import Generator

from swarm_core.config import JammingConfig, NetworkConfig, NetworkProfile, QoSConfig


@dataclass
class Packet:
    src: int
    dst: int
    kind: str
    payload: Any
    t_send: float
    t_deliver: float = 0.0
    retries: int = 0
    seq: int = 0


@dataclass
class NetworkSim:
    profile: NetworkProfile
    qos: QoSConfig
    jamming: JammingConfig
    rng: Generator
    t: float = 0.0
    queue: list[Packet] = field(default_factory=list)
    delivered: list[Packet] = field(default_factory=list)
    drop_count: int = 0
    send_count: int = 0
    groups: list[set[int]] | None = None
    partition_active: bool = False
    jam_active: bool = False
    _in_burst: bool = False
    _burst_remaining: int = 0

    def set_time(self, t: float) -> None:
        self.t = t
        part = self.profile.partition
        if part:
            start = float(part.get("start_s", 0))
            dur = float(part.get("duration_s", 0))
            self.partition_active = start <= t < start + dur
            if self.partition_active and self.groups is None:
                raw = part.get("groups", [])
                self.groups = [set(g) for g in raw]
        else:
            self.partition_active = False

        j = self.jamming
        self.jam_active = j.start_s <= t < j.start_s + j.duration_s

    def _same_partition(self, a: int, b: int) -> bool:
        if not self.partition_active or not self.groups:
            return True
        return any(a in group and b in group for group in self.groups)

    def _loss_pct(self, src: int, dst: int, pos: tuple[float, float] | None = None) -> float:
        base = self.profile.loss_pct
        if self.jam_active and pos is not None:
            c = self.jamming.region.get("center", [100, 100])
            r = float(self.jamming.region.get("radius_m", 40))
            if (pos[0] - c[0]) ** 2 + (pos[1] - c[1]) ** 2 <= r * r:
                base = max(base, self.jamming.loss_pct)
        if self.profile.burst:
            if self._burst_remaining > 0:
                self._burst_remaining -= 1
                return min(95.0, base * 3)
            if self.rng.random() < 0.05:
                self._burst_remaining = int(self.rng.integers(3, 8))
        return base

    def send(
        self,
        src: int,
        dst: int,
        kind: str,
        payload: Any,
        t: float,
        pos: tuple[float, float] | None = None,
        seq: int = 0,
    ) -> bool:
        self.set_time(t)
        self.send_count += 1
        if dst != -1 and not self._same_partition(src, dst):
            self.drop_count += 1
            return False
        loss = self._loss_pct(src, dst, pos)
        if self.rng.random() * 100 < loss:
            self.drop_count += 1
            return False

        latency = self.profile.latency_ms / 1000.0
        if self.profile.jitter_ms > 0:
            latency += abs(self.rng.normal(0, self.profile.jitter_ms / 1000.0))
        pkt = Packet(
            src=src, dst=dst, kind=kind, payload=payload,
            t_send=t, t_deliver=t + latency, seq=seq,
        )
        self.queue.append(pkt)
        return True

    def broadcast(
        self,
        src: int,
        recipients: list[int],
        kind: str,
        payload: Any,
        t: float,
        pos: tuple[float, float] | None = None,
        seq: int = 0,
    ) -> int:
        n = 0
        for dst in recipients:
            if dst == src:
                continue
            if self.send(src, dst, kind, payload, t, pos=pos, seq=seq):
                n += 1
        return n

    def poll(self, t: float) -> list[Packet]:
        ready = [p for p in self.queue if p.t_deliver <= t]
        self.queue = [p for p in self.queue if p.t_deliver > t]
        self.delivered.extend(ready)
        return ready

    def loss_rate(self) -> float:
        if self.send_count == 0:
            return 0.0
        return self.drop_count / self.send_count


def make_network(cfg: NetworkConfig, profile_name: str, rng: Generator) -> NetworkSim:
    profile = cfg.profiles.get(profile_name, NetworkProfile())
    return NetworkSim(profile=profile, qos=cfg.qos, jamming=cfg.jamming, rng=rng)
