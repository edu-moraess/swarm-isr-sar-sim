"""swarm_core — Tier A pure-Python swarm intelligence for ISR/SAR simulation."""

from swarm_core.belief import BeliefGrid
from swarm_core.clock import SimClock
from swarm_core.config import FullConfig, MissionConfig, SwarmConfig
from swarm_core.events import EventLog
from swarm_core.fsm import MissionFSM
from swarm_core.rng import RNGManager

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "BeliefGrid",
    "SimClock",
    "FullConfig",
    "MissionConfig",
    "SwarmConfig",
    "EventLog",
    "MissionFSM",
    "RNGManager",
]
