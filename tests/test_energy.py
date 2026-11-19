"""Energy / RTL reserve."""

from swarm_core.config import SwarmConfig
from swarm_core.drone import create_drones
from swarm_core.energy import rtl_reserve_pct, should_rtl
from swarm_core.rng import RNGManager


def test_rtl_reserve_positive() -> None:
    rng = RNGManager(0).spawn("d")[0]
    s = SwarmConfig(n_drones=1)
    drones = create_drones(s, [0.0, 0.0], rng)
    d = drones[0]
    d.position[0] = 100.0
    res = rtl_reserve_pct(d, [0.0, 0.0], s.battery)
    assert res > s.battery.rtl_reserve_pct


def test_should_rtl_when_low() -> None:
    rng = RNGManager(0).spawn("d")[0]
    s = SwarmConfig(n_drones=1)
    drones = create_drones(s, [0.0, 0.0], rng)
    d = drones[0]
    d.battery_pct = 5.0
    assert should_rtl(d, [0.0, 0.0], s.battery)
