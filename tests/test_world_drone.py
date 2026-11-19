"""World construction and basic drone dynamics."""

from swarm_core.config import MissionConfig, SwarmConfig
from swarm_core.drone import create_drones, step_drone
from swarm_core.rng import RNGManager
from swarm_core.world import build_world


def test_world_dimensions() -> None:
    rng = RNGManager(0).spawn("w")[0]
    m = MissionConfig()
    w = build_world(m, rng)
    assert w.width == 40
    assert w.height == 40
    assert w.n_cells == 1600
    assert len(w.targets) == 8
    assert len(w.clutter) == 12


def test_drones_created() -> None:
    rng = RNGManager(0).spawn("d")[0]
    s = SwarmConfig(n_drones=5)
    drones = create_drones(s, [0.0, 0.0], rng)
    assert len(drones) == 5
    assert drones[0].id == 1
    assert drones[0].battery_pct == 100.0


def test_step_reduces_battery() -> None:
    rng = RNGManager(0)
    world_rng = rng.spawn("w")[0]
    drone_rng = rng.spawn("d")[0]
    m = MissionConfig()
    s = SwarmConfig(n_drones=1)
    w = build_world(m, world_rng)
    drones = create_drones(s, m.base, drone_rng)
    d = drones[0]
    bat0 = d.battery_pct
    step_drone(d, 0.1, s, s.battery, w, 1.5, drone_rng)
    assert d.battery_pct < bat0
