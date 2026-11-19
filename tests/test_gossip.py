"""Gossip: epidemic flood, harsh convergence, and partition reconvergence."""

from __future__ import annotations

from swarm_core.belief import BeliefGrid
from swarm_core.config import NetworkConfig
from swarm_core.gossip import GossipNode, make_gossip_swarm
from swarm_core.messages import Observation, ObservationBatch
from swarm_core.network import make_network
from swarm_core.rng import RNGManager


def _make_batch(drone_id: int, seq: int, cells: list[int], detected: bool = True) -> ObservationBatch:
    obs = [Observation(drone_id, seq, c, detected, 0.92, 0.02, float(seq)) for c in cells]
    return ObservationBatch(drone_id, seq, float(seq), obs)


def test_gossip_applies_remote_batch() -> None:
    rng = RNGManager(0).spawn("net")[0]
    cfg = NetworkConfig()
    net = make_network(cfg, "clean", rng)
    swarm = make_gossip_swarm([1, 2], n_cells=20, net=net, prior=0.2)
    swarm.publish(_make_batch(1, 1, [0, 1, 2]), t=0.0)
    swarm.tick(1.0)
    assert swarm.nodes[2].belief.p(0) != 0.2
    assert swarm.all_converged(tol=1e-5)


def test_store_and_share() -> None:
    node = GossipNode(drone_id=1, belief=BeliefGrid(10, prior=0.1))
    for seq in range(1, 6):
        node.record_local(_make_batch(1, seq, [seq % 10]))
    batches = node.batches_for_peer(2, max_batches=10)
    assert [batch.seq for batch in batches] == [1, 2, 3, 4, 5]
    assert node.applied_hwm[1] == 5


def test_batches_for_peer_uses_independent_watermarks() -> None:
    node = GossipNode(drone_id=1, belief=BeliefGrid(10, prior=0.1))
    for seq in range(1, 6):
        node.record_local(_make_batch(1, seq, [seq % 10]))

    node.acknowledge_peer(peer_id=2, origin_id=1, seq=3)

    assert [batch.seq for batch in node.batches_for_peer(2)] == [4, 5]
    assert [batch.seq for batch in node.batches_for_peer(3)] == [1, 2, 3, 4, 5]


def test_batches_for_peer_fairly_interleaves_origins() -> None:
    node = GossipNode(drone_id=1, belief=BeliefGrid(10, prior=0.1))
    for seq in range(1, 6):
        for origin_id in (1, 2, 3, 4):
            node.record_local(_make_batch(origin_id, seq, [seq % 10]))

    batches = node.batches_for_peer(4, max_batches=10)

    assert [(batch.drone_id, batch.seq) for batch in batches] == [
        (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3), (3, 3),
        (1, 4),
    ]


def test_acknowledged_batches_are_not_retransmitted() -> None:
    rng = RNGManager(11).spawn("net")[0]
    net = make_network(NetworkConfig(), "clean", rng)
    swarm = make_gossip_swarm([1, 2], n_cells=10, net=net)
    swarm.publish(_make_batch(1, 1, [0, 1]), t=0.0)

    swarm.tick(1.0)
    swarm.tick(2.0)

    assert swarm.nodes[1].peer_watermarks[2][1] == 1
    assert swarm.retransmit(2.0) == 0


def test_harsh_consensus_within_30s() -> None:
    rng = RNGManager(42).spawn("net")[0]
    cfg = NetworkConfig()
    cfg.profiles["harsh"].burst = False
    net = make_network(cfg, "harsh", rng)
    ids = [1, 2, 3, 4, 5]
    n_cells = 40
    swarm = make_gossip_swarm(ids, n_cells=n_cells, net=net, prior=0.15)
    dt = 0.1
    for step in range(300):
        t = step * dt
        if step < 150 and step % 10 == 0:
            for drone_id in ids:
                seq = step // 10 + 1
                cells = [(drone_id * 7 + seq + i) % n_cells for i in range(3)]
                swarm.publish(
                    _make_batch(drone_id, seq, cells, detected=(seq % 2 == 0)),
                    t=t,
                )
        swarm.tick(t)
        if step % 5 == 0:
            swarm.retransmit(t)
    assert swarm.all_converged(tol=1e-5), f"div={swarm.max_divergence()}"


def test_partition_reconvergence_under_5s() -> None:
    rng = RNGManager(7).spawn("net")[0]
    cfg = NetworkConfig()
    cfg.profiles["partition"].partition = {
        "start_s": 0.0,
        "duration_s": 5.0,
        "groups": [[1, 2], [3, 4]],
    }
    cfg.profiles["partition"].loss_pct = 0.0
    cfg.profiles["partition"].latency_ms = 5.0
    net = make_network(cfg, "partition", rng)
    ids = [1, 2, 3, 4]
    n_cells = 30
    swarm = make_gossip_swarm(ids, n_cells=n_cells, net=net, prior=0.2)
    dt = 0.1
    for step in range(50):
        t = step * dt
        for drone_id in ids:
            seq = step + 1
            cells = [(drone_id * 3 + step) % n_cells]
            swarm.publish(
                _make_batch(drone_id, seq, cells, detected=(drone_id <= 2)),
                t=t,
            )
        swarm.tick(t)
    assert swarm.max_divergence() > 1e-4
    t_heal = 5.0
    converged_at = None
    recovery_seq = 50
    for step in range(50, 120):
        t = step * dt
        swarm.tick(t)
        swarm.retransmit(t)
        if step % 5 == 0:
            recovery_seq += 1
            for drone_id in ids:
                swarm.publish(
                    _make_batch(drone_id, recovery_seq, [(drone_id + step) % n_cells]),
                    t=t,
                )
        if swarm.all_converged(tol=1e-5):
            converged_at = t - t_heal
            break
    assert converged_at is not None, f"no convergence, div={swarm.max_divergence()}"
    assert converged_at < 5.0, f"reconvergence took {converged_at:.2f}s >= 5s"

def test_retransmit_recovers_after_packet_loss() -> None:
    rng = RNGManager(19).spawn("net")[0]
    net = make_network(NetworkConfig(), "clean", rng)
    swarm = make_gossip_swarm([1, 2], n_cells=10, net=net)
    batch = _make_batch(1, 1, [0, 1])

    net.profile.loss_pct = 100.0
    swarm.publish(batch, t=0.0)
    swarm.tick(1.0)
    assert swarm.nodes[2].applied_hwm.get(1, 0) == 0

    net.profile.loss_pct = 0.0
    assert swarm.retransmit(1.0) == 1
    swarm.tick(2.0)
    swarm.tick(3.0)

    assert swarm.nodes[2].applied_hwm[1] == 1
    assert swarm.nodes[1].peer_watermarks[2][1] == 1
    assert swarm.retransmit(3.0) == 0
