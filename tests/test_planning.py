"""Planning: boustrophedon, formation, sectors, frontier."""

from swarm_core.belief import BeliefGrid
from swarm_core.config import MissionConfig
from swarm_core.planning.boustrophedon import boustrophedon_waypoints, path_length
from swarm_core.planning.formation import formation_offsets
from swarm_core.planning.frontier import frontier_cells
from swarm_core.planning.sectors import n_sectors, sector_cells
from swarm_core.rng import RNGManager
from swarm_core.world import build_world


def test_boustrophedon_covers_many_cells() -> None:
    rng = RNGManager(0).spawn("w")[0]
    m = MissionConfig()
    w = build_world(m, rng)
    wps = boustrophedon_waypoints(w, altitude=30.0)
    assert len(wps) > 50
    assert path_length(wps) > 0


def test_formation_line() -> None:
    offs = formation_offsets(5, kind="line", spacing_m=15)
    assert len(offs) == 5
    assert offs[2][0] == 30.0


def test_sectors() -> None:
    rng = RNGManager(0).spawn("w")[0]
    m = MissionConfig()
    w = build_world(m, rng)
    sc = sector_cells(w, sector_m=20)
    assert len(sc) == n_sectors(w, 20)
    assert sum(len(v) for v in sc.values()) == len(w.free_cells())


def test_frontier_empty_when_nothing_searched() -> None:
    rng = RNGManager(0).spawn("w")[0]
    m = MissionConfig()
    w = build_world(m, rng)
    b = BeliefGrid(w.n_cells)
    fr = frontier_cells(w, b, searched=set())
    assert fr == []
