"""Reproduce and quantify the MissionLoop V2 baseline without changing it.

The script records simulator truth for evaluation, and separately measures
what each Gossip node has incorporated into its local posterior. The local
coverage proxy counts free cells whose posterior differs from the prior.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from swarm_core.config import load_full
from swarm_core.mission import MissionLoop, MissionRunResult


def pairwise_minimum(positions: np.ndarray, dimensions: int) -> float:
    if len(positions) < 2:
        return math.nan
    best = math.inf
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = float(np.linalg.norm(positions[i, :dimensions] - positions[j, :dimensions]))
            best = min(best, distance)
    return best


def local_belief_coverage(loop: MissionLoop, free_indices: np.ndarray) -> list[int]:
    prior_log_odds = math.log(0.1 / 0.9)
    counts: list[int] = []
    for node in loop.gossip.nodes.values():
        posterior = node.belief.log_odds[free_indices]
        counts.append(int(np.count_nonzero(np.abs(posterior - prior_log_odds) > 1e-12)))
    return counts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No telemetry rows generated for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_capture(
    args: argparse.Namespace,
) -> tuple[MissionRunResult, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cfg = load_full(
        mission_path=args.config_dir / "mission.yaml",
        swarm_path=args.config_dir / "swarm.yaml",
        perception_path=args.config_dir / "perception.yaml",
        network_path=args.config_dir / "network.yaml",
    )
    mission_cfg = cfg.mission.model_copy(update={"seed": args.seed})
    loop = MissionLoop(
        mission=mission_cfg,
        swarm=cfg.swarm,
        perception=cfg.perception,
        network=cfg.network,
        network_profile=args.network_profile,
    )
    free_indices = np.asarray(loop.world.free_cells(), dtype=int)
    free_set = set(int(idx) for idx in free_indices)
    drone_rows: list[dict[str, Any]] = []
    mission_rows: list[dict[str, Any]] = []
    phase_minima: dict[str, dict[str, float]] = defaultdict(
        lambda: {"min_xy_sep_m": math.inf, "min_xyz_sep_m": math.inf}
    )
    takeoff_max_altitude = -math.inf
    final_states: dict[int, dict[str, Any]] = {}

    def record(integration_phase: str) -> None:
        nonlocal takeoff_max_altitude
        positions = np.asarray([drone.position.copy() for drone in loop.drones], dtype=float)
        velocities = np.asarray([drone.velocity.copy() for drone in loop.drones], dtype=float)
        min_xy = pairwise_minimum(positions, 2)
        min_xyz = pairwise_minimum(positions, 3)
        if integration_phase in {"TAKEOFF", "SEARCH", "RETURN", "LAND"}:
            phase_minima[integration_phase]["min_xy_sep_m"] = min(
                phase_minima[integration_phase]["min_xy_sep_m"], min_xy
            )
            phase_minima[integration_phase]["min_xyz_sep_m"] = min(
                phase_minima[integration_phase]["min_xyz_sep_m"], min_xyz
            )
        if integration_phase == "TAKEOFF":
            takeoff_max_altitude = max(takeoff_max_altitude, float(np.max(positions[:, 2])))

        central_cells = len(loop.observed_cells.intersection(free_set))
        local_counts = local_belief_coverage(loop, free_indices)
        local_min = min(local_counts) if local_counts else 0
        local_mean = mean(local_counts) if local_counts else 0.0
        local_max = max(local_counts) if local_counts else 0
        central_minus_mean = central_cells - local_mean

        for index, drone in enumerate(loop.drones):
            position = positions[index]
            velocity = velocities[index]
            pad = loop.landing_positions.get(drone.id)
            drone_rows.append({
                "step": loop.clock.step,
                "sim_time_s": round(loop.clock.t, 6),
                "phase": integration_phase,
                "phase_after_step": loop.fsm.phase.name,
                "drone_id": drone.id,
                "x_m": float(position[0]),
                "y_m": float(position[1]),
                "z_m": float(position[2]),
                "vx_mps": float(velocity[0]),
                "vy_mps": float(velocity[1]),
                "vz_mps": float(velocity[2]),
                "speed_3d_mps": float(np.linalg.norm(velocity)),
                "battery_pct": float(drone.battery_pct),
                "waypoint_index": int(drone.wp_idx),
                "waypoint_count": len(drone.waypoints),
                "pad_x_m": None if pad is None else float(pad[0]),
                "pad_y_m": None if pad is None else float(pad[1]),
                "pad_z_m": None if pad is None else float(pad[2]),
                "alive": bool(drone.alive),
            })

        mission_rows.append({
            "step": loop.clock.step,
            "sim_time_s": round(loop.clock.t, 6),
            "phase": integration_phase,
            "phase_after_step": loop.fsm.phase.name,
            "coverage_central_free_cells": central_cells,
            "coverage_central_pct": 100.0 * central_cells / max(1, len(free_indices)),
            "belief_coverage_local_min_free_cells": local_min,
            "belief_coverage_local_mean_free_cells": round(local_mean, 4),
            "belief_coverage_local_max_free_cells": local_max,
            "central_minus_local_mean_cells": round(central_minus_mean, 4),
            "network_send_count": loop.network.send_count,
            "network_drop_count": loop.network.drop_count,
            "jam_active": bool(loop.network.jam_active),
            "minimum_xy_separation_m": min_xy,
            "minimum_xyz_separation_m": min_xyz,
        })

        for drone in loop.drones:
            final_states[drone.id] = {
                "position_xyz_m": tuple(float(x) for x in drone.position),
                "velocity_xyz_mps": tuple(float(x) for x in drone.velocity),
                "battery_pct": float(drone.battery_pct),
            }

    loop.start()
    record("TAKEOFF")
    max_steps = max(1, math.ceil(mission_cfg.time_limit_s / loop.clock.dt))
    while not loop.finished and loop.clock.step < max_steps:
        integration_phase = loop.fsm.phase.name
        loop.step()
        record(integration_phase)

    result = loop.result()
    report: dict[str, Any] = {
        "mission_id": mission_cfg.mission_id,
        "seed": args.seed,
        "network_profile": args.network_profile,
        "completed": result.completed,
        "phase": result.phase.name,
        "completion_reason": result.completion_reason,
        "steps": result.steps,
        "simulated_time_s": result.simulated_time_s,
        "coverage_pct": result.coverage_pct,
        "event_count": result.event_count,
        "event_sha256": result.event_sha256,
        "event_chain_valid": result.event_chain_valid,
        "network_send_count": result.network_send_count,
        "network_drop_count": result.network_drop_count,
        "takeoff_max_altitude_m": takeoff_max_altitude,
        "takeoff_overshoot_m": takeoff_max_altitude - cfg.swarm.search_alt_m,
        "phase_minima": dict(phase_minima),
        "final_states": final_states,
    }
    return result, drone_rows, mission_rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/swarm_baseline_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--network-profile", default="clean")
    parser.add_argument("--skip-determinism-replay", action="store_true")
    args = parser.parse_args()

    result, drone_rows, mission_rows, report = run_capture(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "drone_telemetry_v2.csv", drone_rows)
    write_csv(args.output_dir / "mission_telemetry_v2.csv", mission_rows)

    if not args.skip_determinism_replay:
        cfg = load_full(
            mission_path=args.config_dir / "mission.yaml",
            swarm_path=args.config_dir / "swarm.yaml",
            perception_path=args.config_dir / "perception.yaml",
            network_path=args.config_dir / "network.yaml",
        )
        replay_cfg = cfg.mission.model_copy(update={"seed": args.seed})
        replay = MissionLoop(
            mission=replay_cfg,
            swarm=cfg.swarm,
            perception=cfg.perception,
            network=cfg.network,
            network_profile=args.network_profile,
        ).run()
        report["determinism_replay_sha256"] = replay.event_sha256
        report["determinism_replay_match"] = replay.event_sha256 == result.event_sha256

    window = [row for row in mission_rows if 120.0 <= float(row["sim_time_s"]) <= 180.0]
    if window:
        worst = max(window, key=lambda row: abs(float(row["central_minus_local_mean_cells"])))
        report["interference_window"] = {
            "start_s": 120.0,
            "end_s": 180.0,
            "samples": len(window),
            "start_sample": window[0],
            "end_sample": window[-1],
            "worst_divergence_sample": worst,
            "max_abs_central_local_gap_cells": max(
                abs(float(row["central_minus_local_mean_cells"])) for row in window
            ),
        }
    else:
        report["interference_window"] = {"start_s": 120.0, "end_s": 180.0, "samples": 0}

    post_window = [
        row for row in mission_rows if 180.0 < float(row["sim_time_s"]) <= 240.0
    ]
    if post_window:
        report["post_interference_window"] = {
            "start_s": 180.0,
            "end_s": 240.0,
            "samples": len(post_window),
            "start_sample": post_window[0],
            "end_sample": post_window[-1],
            "max_abs_central_local_gap_cells": max(
                abs(float(row["central_minus_local_mean_cells"])) for row in post_window
            ),
        }
    else:
        report["post_interference_window"] = {
            "start_s": 180.0, "end_s": 240.0, "samples": 0
        }

    print("=" * 76)
    print("SWARM ISR/SAR — BASELINE V2 / P2.0 STATE INTEGRITY")
    print("=" * 76)
    print(f"Mission: {report['mission_id']} | seed={args.seed} | profile={args.network_profile}")
    print(
        f"Result: {result.phase.name} | {result.completion_reason} | "
        f"steps={result.steps} | t={result.simulated_time_s:.1f}s"
    )
    print(f"Coverage central: {result.coverage_pct:.2f}% | Event chain: {result.event_chain_valid}")
    print(f"EVENT SHA256: {result.event_sha256}")
    print(f"TAKEOFF MAX ALT: {report['takeoff_max_altitude_m']:.4f} m")
    print(f"TAKEOFF OVERSHOOT: {report['takeoff_overshoot_m']:.4f} m")
    print("\nSEPARATION BY INTEGRATION PHASE")
    for phase in ("TAKEOFF", "SEARCH", "RETURN", "LAND"):
        values = report["phase_minima"].get(phase, {})
        print(
            f"{phase:<8} MIN XY SEP={values.get('min_xy_sep_m', math.nan):.6f} m | "
            f"MIN XYZ SEP={values.get('min_xyz_sep_m', math.nan):.6f} m"
        )
    print("\nINTERFERENCE WINDOW / LOCAL BELIEF PROPAGATION")
    iw = report["interference_window"]
    if iw["samples"]:
        for label in ("start_sample", "end_sample", "worst_divergence_sample"):
            sample = iw[label]
            print(
                f"{label}: t={sample['sim_time_s']}s central={sample['coverage_central_free_cells']} "
                f"local[min/mean/max]={sample['belief_coverage_local_min_free_cells']}/"
                f"{sample['belief_coverage_local_mean_free_cells']}/"
                f"{sample['belief_coverage_local_max_free_cells']} "
                f"gap={sample['central_minus_local_mean_cells']} cells"
            )
        print(f"MAX ABS GAP 120-180s: {iw['max_abs_central_local_gap_cells']:.4f} cells")
    else:
        print("No telemetry samples in the configured interference window.")

    print("\nPOST-INTERFERENCE RECOVERY WINDOW (180-240s)")
    post = report["post_interference_window"]
    if post["samples"]:
        for label in ("start_sample", "end_sample"):
            sample = post[label]
            print(
                f"{label}: t={sample['sim_time_s']}s central={sample['coverage_central_free_cells']} "
                f"local[min/mean/max]={sample['belief_coverage_local_min_free_cells']}/"
                f"{sample['belief_coverage_local_mean_free_cells']}/"
                f"{sample['belief_coverage_local_max_free_cells']} "
                f"gap={sample['central_minus_local_mean_cells']} cells"
            )
        print(f"MAX ABS GAP 180-240s: {post['max_abs_central_local_gap_cells']:.4f} cells")
    else:
        print("No telemetry samples in the post-interference window.")

    print("\nLANDING TERMINAL ERRORS")
    for drone_id, state in sorted(report["final_states"].items()):
        z_error = abs(float(state["position_xyz_m"][2]))
        vz_error = abs(float(state["velocity_xyz_mps"][2]))
        print(f"D{drone_id:02d}: |z|={z_error:.6f} m | |vz|={vz_error:.6f} m/s")
    if "determinism_replay_match" in report:
        print(
            f"DETERMINISM REPLAY: {report['determinism_replay_match']} | "
            f"SHA256={report['determinism_replay_sha256']}"
        )
    print("\n[ARTIFACTS]")
    print(args.output_dir / "drone_telemetry_v2.csv")
    print(args.output_dir / "mission_telemetry_v2.csv")
    print("[LIMIT] Local belief coverage counts posterior-changed free cells, not ground-truth detection.")
    print("[LIMIT] Separation is measured, not actively enforced; dynamics remain point-mass.")
    return 0 if result.completed and result.event_chain_valid and report.get("determinism_replay_match", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
