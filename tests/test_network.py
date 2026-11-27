"""Network: loss rates, partition isolation, jamming region."""

from __future__ import annotations

from swarm_core.config import NetworkConfig
from swarm_core.network import make_network
from swarm_core.rng import RNGManager


def test_clean_no_loss() -> None:
    rng = RNGManager(0).spawn("net")[0]
    cfg = NetworkConfig()
    net = make_network(cfg, "clean", rng)
    for i in range(100):
        net.send(1, 2, "ping", i, t=float(i))
    delivered = net.poll(1000.0)
    assert len(delivered) == 100
    assert net.loss_rate() == 0.0


def test_harsh_has_loss() -> None:
    rng = RNGManager(1).spawn("net")[0]
    cfg = NetworkConfig()
    net = make_network(cfg, "harsh", rng)
    for i in range(200):
        net.send(1, 2, "ping", i, t=0.0)
    rate = net.loss_rate()
    assert 0.05 < rate < 0.50, f"loss_rate={rate}"


def test_partition_blocks_cross_group() -> None:
    rng = RNGManager(0).spawn("net")[0]
    cfg = NetworkConfig()
    cfg.profiles["partition"].partition = {
        "start_s": 0, "duration_s": 100, "groups": [[1, 2], [3, 4]],
    }
    net = make_network(cfg, "partition", rng)
    net.set_time(10.0)
    assert net.partition_active
    ok_cross = net.send(1, 3, "ping", "b", t=10.0)
    assert ok_cross is False


def test_send_uses_explicit_time_for_partition_state() -> None:
    rng = RNGManager(0).spawn("net")[0]
    cfg = NetworkConfig()
    cfg.profiles["partition"].partition = {
        "start_s": 1.0, "duration_s": 5.0, "groups": [[1], [2]],
    }
    net = make_network(cfg, "partition", rng)

    # No explicit set_time(): send() must honor its own timestamp.
    assert net.send(1, 2, "ping", "blocked", t=1.0) is False
    assert net.partition_active is True
    assert net.drop_count == 1
