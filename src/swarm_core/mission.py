"""Deterministic Tier A orchestration for a minimal synthetic SAR mission.

This v0 loop integrates the existing clock, world, point-mass drones, static
coverage planner, sensors, gossip network, mission FSM and hash-chained log.
It is an experimental baseline: task allocation, target-confirmation stop
criteria, collision avoidance and distributed leadership are out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from swarm_core.clock import SimClock
from swarm_core.contracts import AgentLocalState
from swarm_core.config import (
    MissionConfig,
    NetworkConfig,
    PerceptionConfig,
    SwarmConfig,
)
from swarm_core.drone import Drone, create_drones, step_drone
from swarm_core.events import EventLog
from swarm_core.fsm import MissionFSM
from swarm_core.gossip import GossipSwarm, make_gossip_swarm
from swarm_core.messages import MissionPhase
from swarm_core.network import make_network
from swarm_core.planning.boustrophedon import boustrophedon_waypoints
from swarm_core.rng import RNGManager
from swarm_core.sensors import sense
from swarm_core.telemetry import AgentTelemetryAdapter
from swarm_core.world import World, build_world


@dataclass(frozen=True)
class MissionRunResult:
    """Immutable summary of one run; completion means static routes were flown."""

    phase: MissionPhase
    completion_reason: str
    steps: int
    simulated_time_s: float
    coverage_pct: float
    observed_free_cells: int
    total_free_cells: int
    event_count: int
    event_sha256: str
    event_chain_valid: bool
    network_send_count: int
    network_drop_count: int

    @property
    def completed(self) -> bool:
        return self.phase == MissionPhase.COMPLETE


class MissionLoop:
    """Single-process deterministic mission orchestrator.

    The global MissionFSM is authoritative for mission lifecycle. Each Drone's
    phase mirrors that global phase in v0; per-drone task/health state machines
    are deliberately deferred.
    """

    def __init__(
        self,
        mission: MissionConfig,
        swarm: SwarmConfig,
        perception: PerceptionConfig,
        network: NetworkConfig | None = None,
        network_profile: str = "clean",
        dt: float = 0.1,
        sense_period_steps: int = 5,
        obstacle_fraction: float = 0.05,
    ) -> None:
        if sense_period_steps < 1:
            raise ValueError("sense_period_steps must be >= 1")
        if not 0.0 <= obstacle_fraction < 1.0:
            raise ValueError("obstacle_fraction must be in [0, 1)")

        self.mission_config = mission
        self.swarm_config = swarm
        self.perception_config = perception
        self.network_config = network or NetworkConfig()
        self.network_profile = network_profile
        if network_profile not in self.network_config.profiles:
            raise ValueError(f"Unknown network profile: {network_profile}")

        self.clock = SimClock(dt=dt)
        self.sense_period_steps = sense_period_steps
        self.rng = RNGManager(mission.seed)

        # Spawn order is fixed and documented to keep runs reproducible.
        world_rng = self.rng.spawn("world")[0]
        drone_rng = self.rng.spawn("drones")[0]
        network_rng = self.rng.spawn("network")[0]
        self.dynamics_rngs = self.rng.spawn("drone_dynamics", swarm.n_drones)
        self.sensor_rngs = self.rng.spawn("drone_sensors", swarm.n_drones)

        self.world: World = build_world(mission, world_rng, obstacle_fraction=obstacle_fraction)
        self.drones: list[Drone] = create_drones(swarm, mission.base, drone_rng)
        self.telemetry_adapters = {
            drone.id: AgentTelemetryAdapter(source_id=drone.id) for drone in self.drones
        }
        self.agent_local_states: dict[int, AgentLocalState] = {}
        self.network = make_network(self.network_config, network_profile, network_rng)
        self.gossip: GossipSwarm = make_gossip_swarm(
            [drone.id for drone in self.drones],
            self.world.n_cells,
            self.network,
        )
        self.fsm = MissionFSM()
        self.events = EventLog()
        self.observed_cells: set[int] = set()
        self.landing_positions: dict[int, np.ndarray] = {}
        self._started = False
        self._completion_reason = "not_started"

    @property
    def finished(self) -> bool:
        return self.fsm.phase == MissionPhase.COMPLETE

    def _capture_agent_local_states(self) -> None:
        """Capture agent-facing snapshots from reported telemetry only."""
        for drone in self.drones:
            self.agent_local_states[drone.id] = self.telemetry_adapters[drone.id].update(
                timestamp_s=self.clock.t,
                reported_position=drone.reported_position(),
                battery_pct=drone.battery_pct,
            )

    def _emit_phase(self, event: str) -> None:
        previous = self.fsm.phase
        current = self.fsm.trigger(event)
        for drone in self.drones:
            drone.phase = current
        self.events.emit(
            self.clock.t,
            "phase_transition",
            event=event,
            from_phase=previous.name,
            to_phase=current.name,
        )

    def _assign_waypoints(self, drone: Drone, points: list[np.ndarray]) -> None:
        drone.waypoints = [point.astype(float, copy=True) for point in points]
        drone.wp_idx = 0

    def _make_landing_positions(self) -> dict[int, np.ndarray]:
        """Create deterministic, separated landing pads inside the mission area."""
        count = len(self.drones)
        cols = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))
        spacing = max(3.0, min(6.0, self.swarm_config.formation.spacing_m * 0.25))
        bx, by = self.mission_config.base[:2]
        min_x = self.world.origin.x + 0.5
        min_y = self.world.origin.y + 0.5
        max_x = self.world.origin.x + self.mission_config.area.size_m[0] - (cols - 1) * spacing - 0.5
        max_y = self.world.origin.y + self.mission_config.area.size_m[1] - (rows - 1) * spacing - 0.5
        anchor_x = min(max(float(bx), min_x), max_x)
        anchor_y = min(max(float(by), min_y), max_y)
        positions: dict[int, np.ndarray] = {}
        for index, drone in enumerate(self.drones):
            row, col = divmod(index, cols)
            positions[drone.id] = np.array(
                [anchor_x + col * spacing, anchor_y + row * spacing, 0.0],
                dtype=float,
            )
        return positions

    def _all_drones_landed(self) -> bool:
        """Require pad proximity, near-zero altitude and low residual velocity."""
        for drone in self.drones:
            target = self.landing_positions.get(drone.id)
            if target is None or drone.wp_idx < len(drone.waypoints):
                return False
            horizontal_error = float(np.linalg.norm(drone.position[:2] - target[:2]))
            altitude_error = abs(float(drone.position[2] - target[2]))
            horizontal_speed = float(np.linalg.norm(drone.velocity[:2]))
            vertical_speed = abs(float(drone.velocity[2]))
            if (
                horizontal_error > 0.25
                or altitude_error > 0.20
                or horizontal_speed > 0.20
                or vertical_speed > 0.20
            ):
                return False
        return True

    def start(self) -> None:
        """Start the mission and command all drones to search altitude."""
        if self._started:
            raise RuntimeError("MissionLoop.start() may only be called once")
        self._started = True
        self._emit_phase("cmd_start")
        altitude = self.swarm_config.search_alt_m
        for drone in self.drones:
            self._assign_waypoints(
                drone,
                [np.array([drone.position[0], drone.position[1], altitude])],
            )
        self._capture_agent_local_states()
        self.events.emit(
            self.clock.t,
            "mission_started",
            mission_id=self.mission_config.mission_id,
            seed=self.mission_config.seed,
            drone_count=len(self.drones),
            network_profile=self.network_profile,
        )

    def _begin_search(self) -> None:
        self._emit_phase("altitude_reached")
        route = boustrophedon_waypoints(self.world, altitude=self.swarm_config.search_alt_m)
        if not route:
            raise RuntimeError("Coverage planner produced no free-cell waypoints")
        for index, drone in enumerate(self.drones):
            assigned = route[index :: len(self.drones)]
            if not assigned:
                assigned = [route[index % len(route)]]
            points = [np.array([point.x, point.y, point.z], dtype=float) for point in assigned]
            self._assign_waypoints(drone, points)
        self.events.emit(
            self.clock.t,
            "search_started",
            route_waypoints=len(route),
            assigned_waypoints=sum(len(d.waypoints) for d in self.drones),
        )

    def _sense_and_share(self) -> None:
        t = self.clock.t
        for index, drone in enumerate(self.drones):
            batch = sense(
                drone,
                self.world,
                self.perception_config,
                t,
                self.sensor_rngs[index],
            )
            self.observed_cells.update(observation.cell_idx for observation in batch.observations)
            self.gossip.publish(
                batch,
                t=t,
                pos=(float(drone.position[0]), float(drone.position[1])),
            )
            self.events.emit(
                t,
                "sensor_batch",
                drone_id=drone.id,
                seq=batch.seq,
                observation_count=len(batch.observations),
            )

    def _return_to_base(self) -> None:
        self._emit_phase("tasks_done")
        altitude = self.swarm_config.search_alt_m
        self.landing_positions = self._make_landing_positions()
        for drone in self.drones:
            pad = self.landing_positions[drone.id]
            staging_point = np.array([pad[0], pad[1], altitude], dtype=float)
            self._assign_waypoints(drone, [staging_point])
            self.events.emit(
                self.clock.t,
                "return_staging_assigned",
                drone_id=drone.id,
                staging_x_m=float(staging_point[0]),
                staging_y_m=float(staging_point[1]),
                staging_z_m=float(staging_point[2]),
            )
        self.events.emit(t=self.clock.t, kind="return_started")

    def _begin_landing(self) -> None:
        self._emit_phase("at_base")
        if not self.landing_positions:
            self.landing_positions = self._make_landing_positions()
        for drone in self.drones:
            pad = self.landing_positions[drone.id]
            self._assign_waypoints(drone, [pad])
            self.events.emit(
                self.clock.t,
                "landing_pad_assigned",
                drone_id=drone.id,
                pad_x_m=float(pad[0]),
                pad_y_m=float(pad[1]),
                pad_z_m=float(pad[2]),
            )
        self.events.emit(
            t=self.clock.t,
            kind="landing_started",
            pad_spacing_m=max(3.0, min(6.0, self.swarm_config.formation.spacing_m * 0.25)),
        )

    def step(self) -> MissionRunResult:
        """Advance exactly one fixed simulation step."""
        if not self._started:
            raise RuntimeError("Call start() or run() before step()")
        if self.finished:
            return self.result()

        t = self.clock.tick()
        self.world.update_wind(self.clock.dt)
        landing_phase = self.fsm.phase == MissionPhase.LAND
        for index, drone in enumerate(self.drones):
            step_drone(
                drone,
                self.clock.dt,
                self.swarm_config,
                self.swarm_config.battery,
                self.world,
                self.perception_config.gps_sigma_m,
                self.dynamics_rngs[index],
                tol=0.20 if landing_phase else 1.5,
                arrival_speed_mps=0.15 if landing_phase else 0.25,
                braking=landing_phase,
            )

        self._capture_agent_local_states()

        if self.fsm.phase == MissionPhase.TAKEOFF:
            if all(
                drone.position[2] >= self.swarm_config.search_alt_m - 1.5 for drone in self.drones
            ):
                self._begin_search()

        elif self.fsm.phase == MissionPhase.SEARCH:
            if self.clock.step % self.sense_period_steps == 0:
                self._sense_and_share()
            self.gossip.tick(t)
            if self.clock.step % self.sense_period_steps == 0:
                sent = self.gossip.retransmit(t)
                self.events.emit(t, "gossip_retransmit", sent=sent)
            if all(drone.wp_idx >= len(drone.waypoints) for drone in self.drones):
                self._return_to_base()

        elif self.fsm.phase == MissionPhase.RETURN:
            if all(drone.wp_idx >= len(drone.waypoints) for drone in self.drones):
                self._begin_landing()

        elif self.fsm.phase == MissionPhase.LAND and self._all_drones_landed():
            self._emit_phase("landed")
            self._completion_reason = "coverage_routes_completed"
            self.events.emit(
                t,
                "mission_completed",
                reason=self._completion_reason,
                coverage_pct=self._coverage_pct(),
            )

        self.events.emit(
            t,
            "tick",
            step=self.clock.step,
            phase=self.fsm.phase.name,
            observed_cells=len(self.observed_cells),
        )
        return self.result()

    def _coverage_pct(self) -> float:
        free_count = len(self.world.free_cells())
        if free_count == 0:
            return 0.0
        observed_free = len(self.observed_cells.intersection(self.world.free_cells()))
        return 100.0 * observed_free / free_count

    def result(self) -> MissionRunResult:
        free_count = len(self.world.free_cells())
        observed_free = len(self.observed_cells.intersection(self.world.free_cells()))
        return MissionRunResult(
            phase=self.fsm.phase,
            completion_reason=self._completion_reason,
            steps=self.clock.step,
            simulated_time_s=self.clock.t,
            coverage_pct=(100.0 * observed_free / free_count if free_count else 0.0),
            observed_free_cells=observed_free,
            total_free_cells=free_count,
            event_count=len(self.events.events),
            event_sha256=self.events.sha256(),
            event_chain_valid=self.events.verify_chain(),
            network_send_count=self.network.send_count,
            network_drop_count=self.network.drop_count,
        )

    def run(self, max_steps: int | None = None) -> MissionRunResult:
        """Run until route completion, mission time limit, or an explicit step cap."""
        if not self._started:
            self.start()
        if self.finished:
            return self.result()

        time_limit_steps = max(1, int(math.ceil(self.mission_config.time_limit_s / self.clock.dt)))
        limit = time_limit_steps
        if max_steps is not None:
            if max_steps < 1:
                raise ValueError("max_steps must be >= 1")
            limit = min(limit, max_steps)

        while not self.finished and self.clock.step < limit:
            self.step()

        if not self.finished:
            self._completion_reason = (
                "step_limit"
                if max_steps is not None and max_steps < time_limit_steps
                else "time_limit"
            )
            self.events.emit(
                self.clock.t,
                "mission_stopped",
                reason=self._completion_reason,
                phase=self.fsm.phase.name,
            )
        return self.result()
