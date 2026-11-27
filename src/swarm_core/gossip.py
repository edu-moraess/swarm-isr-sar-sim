"""Gossip of ObservationBatches with epidemic flood and contiguous apply."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from swarm_core.belief import BeliefGrid
from swarm_core.messages import ObservationBatch
from swarm_core.network import NetworkSim


@dataclass
class GossipNode:
    drone_id: int
    belief: BeliefGrid
    store: dict[tuple[int, int], ObservationBatch] = field(default_factory=dict)
    applied_hwm: dict[int, int] = field(default_factory=dict)
    peer_watermarks: dict[int, dict[int, int]] = field(default_factory=dict)
    local_seq: int = 0

    def _apply_one(self, batch: ObservationBatch) -> int:
        for obs in batch.observations:
            s, f = obs.sensitivity, obs.false_alarm
            if obs.detected:
                delta = 10.0 if f <= 0 else math.log(max(s, 1e-9) / max(f, 1e-9))
            else:
                delta = math.log(max(1.0 - s, 1e-9) / max(1.0 - f, 1e-9))
            self.belief.log_odds[obs.cell_idx] += delta
        return len(batch.observations)

    def _try_apply(self, origin: int) -> int:
        hwm = self.applied_hwm.get(origin, 0)
        n = 0
        seq = hwm + 1
        while (origin, seq) in self.store:
            n += self._apply_one(self.store[(origin, seq)])
            self.applied_hwm[origin] = seq
            seq += 1
        return n

    def ingest(self, batch: ObservationBatch) -> int:
        key = (batch.drone_id, batch.seq)
        if key in self.store:
            return 0
        self.store[key] = batch
        if batch.drone_id == self.drone_id:
            self.local_seq = max(self.local_seq, batch.seq)
        return self._try_apply(batch.drone_id)

    def acknowledge_peer(self, peer_id: int, origin_id: int, seq: int) -> None:
        """Record the highest contiguous sequence acknowledged by a peer."""
        peer_hwm = self.peer_watermarks.setdefault(peer_id, {})
        peer_hwm[origin_id] = max(peer_hwm.get(origin_id, 0), seq)

    def batches_to_share(self) -> list[ObservationBatch]:
        return list(self.store.values())

    def record_local(self, batch: ObservationBatch) -> None:
        self.ingest(batch)

    def batches_for_peer(self, peer_id: int, max_batches: int = 10) -> list[ObservationBatch]:
        """Return batches this peer has not acknowledged, excluding its own origin."""
        if max_batches <= 0:
            return []
        peer_hwm = self.peer_watermarks.get(peer_id, {})
        pending = [
            batch
            for (origin_id, seq), batch in self.store.items()
            if origin_id != peer_id and seq > peer_hwm.get(origin_id, 0)
        ]
        pending.sort(key=lambda batch: (batch.seq, batch.drone_id))
        return pending[:max_batches]


@dataclass
class GossipSwarm:
    nodes: dict[int, GossipNode]
    net: NetworkSim
    drone_ids: list[int]

    def publish(self, batch: ObservationBatch, t: float, pos: tuple[float, float] | None = None) -> None:
        self.nodes[batch.drone_id].ingest(batch)
        self.net.broadcast(
            src=batch.drone_id,
            recipients=self.drone_ids,
            kind="obs_batch",
            payload=batch,
            t=t,
            pos=pos,
            seq=batch.seq,
        )

    def tick(self, t: float) -> dict[int, int]:
        self.net.set_time(t)
        updated: dict[int, int] = {drone_id: 0 for drone_id in self.drone_ids}
        for pkt in self.net.poll(t):
            if pkt.dst not in self.nodes:
                continue
            if pkt.kind == "obs_ack":
                for origin_id, seq in pkt.payload.items():
                    self.nodes[pkt.dst].acknowledge_peer(pkt.src, origin_id, seq)
                continue
            if pkt.kind != "obs_batch":
                continue

            n = self.nodes[pkt.dst].ingest(pkt.payload)
            updated[pkt.dst] = updated.get(pkt.dst, 0) + n
            self.net.send(
                src=pkt.dst,
                dst=pkt.src,
                kind="obs_ack",
                payload=dict(self.nodes[pkt.dst].applied_hwm),
                t=t,
            )
        return updated

    def retransmit(self, t: float) -> int:
        """Resend only batches beyond each recipient's acknowledged watermarks."""
        n_sent = 0
        for src_id, node in self.nodes.items():
            for peer_id in self.drone_ids:
                if peer_id == src_id:
                    continue
                for batch in node.batches_for_peer(peer_id):
                    if self.net.send(
                        src_id,
                        peer_id,
                        "obs_batch",
                        batch,
                        t,
                        seq=batch.seq,
                    ):
                        n_sent += 1
        return n_sent

    def max_divergence(self) -> float:
        ids = list(self.nodes.keys())
        if len(ids) < 2:
            return 0.0
        mx = 0.0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = self.nodes[ids[i]].belief.max_divergence(self.nodes[ids[j]].belief)
                if d > mx:
                    mx = d
        return mx

    def all_converged(self, tol: float = 1e-6) -> bool:
        return self.max_divergence() <= tol

    def store_sizes(self) -> dict[int, int]:
        return {drone_id: len(node.store) for drone_id, node in self.nodes.items()}


def make_gossip_swarm(
    drone_ids: list[int], n_cells: int, net: NetworkSim, prior: float = 0.1,
) -> GossipSwarm:
    nodes = {
        drone_id: GossipNode(drone_id=drone_id, belief=BeliefGrid(n_cells, prior=prior))
        for drone_id in drone_ids
    }
    return GossipSwarm(nodes=nodes, net=net, drone_ids=drone_ids)
