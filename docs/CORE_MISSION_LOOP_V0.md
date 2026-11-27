# Core Mission Loop v0

## Purpose

The Tier A mission loop integrates the existing deterministic clock, synthetic
world, point-mass drone dynamics, static boustrophedon planner, sensor model,
NetworkSim/Gossip, MissionFSM and hash-chained EventLog into one reproducible
execution path.

## Lifecycle

`INIT → TAKEOFF → SEARCH → RETURN → LAND → COMPLETE`

- **TAKEOFF:** each drone climbs to the configured search altitude.
- **SEARCH:** the static route is distributed deterministically by waypoint
  index; sensor batches are published through Gossip at a fixed step interval.
- **RETURN:** each drone returns to the configured base at search altitude.
- **LAND:** each drone descends to ground level at base.
- **COMPLETE:** all static route waypoints have been consumed and all drones
  have completed the landing waypoint.

The global MissionFSM is authoritative in v0. Each drone's `phase` mirrors
the global phase. Per-drone health/task state machines are deferred.

## Reproducibility

The loop uses a fixed `dt`, deterministic drone-ID iteration, and a fixed RNG
spawn order: world, drone creation, network, per-drone dynamics, per-drone
sensors. The EventLog hash and hash-chain verifier are exposed in the run
result.

## Run result

`MissionRunResult` reports lifecycle phase, stop reason, simulated steps/time,
observed free-cell coverage, event count/hash/chain validity and network
send/drop counts.

A result with `completion_reason="coverage_routes_completed"` means the
static coverage routes were traversed and the drones landed. It does **not**
claim that configured SAR target-confirmation or coverage-percentage stop
criteria have been satisfied. Those acceptance criteria require a separate
mission-level evaluation layer.

## Explicit non-goals

- CBBA, auction, dynamic task allocation or retasking.
- Raft-like consensus, leader election or failure detection.
- Per-drone independent FSMs, collision avoidance or flight safety logic.
- Target confirmation as a mission stop condition.
- ROS 2, Gazebo, PX4, CLI, experiment runner or report generation.

This is a deterministic research baseline in a synthetic environment, not an
operational flight controller.
