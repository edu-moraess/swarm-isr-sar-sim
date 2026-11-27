"""Read-only telemetry runner for the SWARM ISR/SAR MissionLoop baseline.

Runs the existing mission step-by-step and exports observed telemetry.
It does not modify MissionLoop or infer trajectories between simulation steps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swarm_core.config import load_full
from swarm_core.drone import min_separation
from swarm_core.mission import MissionLoop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--output-dir", type=Path, default=Path("swarm_observability_output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--network-profile", default="clean")
    parser.add_argument("--sample-every", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_every < 1:
        raise ValueError("--sample-every must be >= 1")

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

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    telemetry_path = out / "drone_telemetry.csv"
    mission_path = out / "mission_telemetry.csv"

    drone_rows: list[dict[str, Any]] = []
    mission_rows: list[dict[str, Any]] = []
    previous_positions = {
        drone.id: np.asarray(drone.position, dtype=float).copy()
        for drone in loop.drones
    }
    cumulative_distance = {drone.id: 0.0 for drone in loop.drones}
    last_sampled_step = -1

    def record() -> None:
        nonlocal last_sampled_step
        step = loop.clock.step
        if step != 0 and step % args.sample_every != 0 and not loop.finished:
            return
        if step == last_sampled_step:
            return
        last_sampled_step = step

        positions = []
        for drone in loop.drones:
            pos = np.asarray(drone.position, dtype=float).copy()
            vel = np.asarray(drone.velocity, dtype=float).copy()
            delta = pos - previous_positions[drone.id]
            cumulative_distance[drone.id] += float(np.linalg.norm(delta))
            previous_positions[drone.id] = pos.copy()
            positions.append(pos)

            drone_rows.append({
                "step": step,
                "sim_time_s": loop.clock.t,
                "phase": loop.fsm.phase.name,
                "drone_id": drone.id,
                "x_m": float(pos[0]),
                "y_m": float(pos[1]),
                "z_m": float(pos[2]),
                "vx_mps": float(vel[0]),
                "vy_mps": float(vel[1]),
                "vz_mps": float(vel[2]),
                "speed_3d_mps": float(np.linalg.norm(vel)),
                "battery_pct": float(drone.battery_pct),
                "distance_travelled_m": cumulative_distance[drone.id],
                "waypoint_index": int(drone.wp_idx),
                "waypoint_count": len(drone.waypoints),
                "alive": bool(drone.alive),
            })

        result = loop.result()
        mission_rows.append({
            "step": step,
            "sim_time_s": loop.clock.t,
            "phase": loop.fsm.phase.name,
            "observed_cells_total": len(loop.observed_cells),
            "coverage_pct": result.coverage_pct,
            "event_count": result.event_count,
            "network_send_count": result.network_send_count,
            "network_drop_count": result.network_drop_count,
            "minimum_pairwise_separation_m": min_separation(loop.drones),
        })

    loop.start()
    record()

    max_steps = max(1, math.ceil(mission_cfg.time_limit_s / loop.clock.dt))
    while not loop.finished and loop.clock.step < max_steps:
        loop.step()
        record()

    result = loop.result()

    if not drone_rows:
        raise RuntimeError("No drone telemetry was captured.")

    with telemetry_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(drone_rows[0]))
        writer.writeheader()
        writer.writerows(drone_rows)

    with mission_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mission_rows[0]))
        writer.writeheader()
        writer.writerows(mission_rows)

    final_by_drone = {}
    for drone in loop.drones:
        final_by_drone[str(drone.id)] = {
            "distance_travelled_m": cumulative_distance[drone.id],
            "final_position_xyz_m": [float(v) for v in drone.position],
            "final_battery_pct": float(drone.battery_pct),
            "alive": bool(drone.alive),
        }

    summary = {
        "experiment": "SWARM ISR/SAR Observability & Performance v1",
        "seed": args.seed,
        "network_profile": args.network_profile,
        "sample_every_steps": args.sample_every,
        "drone_count": len(loop.drones),
        "mission": {
            "completed": result.completed,
            "phase": result.phase.name,
            "completion_reason": result.completion_reason,
            "steps": result.steps,
            "simulated_time_s": result.simulated_time_s,
            "coverage_pct": result.coverage_pct,
            "observed_free_cells": result.observed_free_cells,
            "total_free_cells": result.total_free_cells,
            "event_count": result.event_count,
            "event_sha256": result.event_sha256,
            "event_chain_valid": result.event_chain_valid,
            "network_send_count": result.network_send_count,
            "network_drop_count": result.network_drop_count,
        },
        "minimum_observed_pairwise_separation_m": min(
            (r["minimum_pairwise_separation_m"] for r in mission_rows),
            default=None,
        ),
        "per_drone": final_by_drone,
        "interpretation_limits": [
            "Distance is accumulated from consecutive recorded true positions.",
            "No per-drone sensor coverage attribution exists in this baseline.",
            "Minimum separation is measured, not collision avoidance.",
            "Battery is the simulator's current point-mass battery model.",
            "No interpolation is used between simulation steps.",
        ],
    }
    summary_path = out / "observability_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Figure 1: real XY tracks from sampled telemetry.
    fig, ax = plt.subplots(figsize=(9, 8))
    for drone_id in sorted({r["drone_id"] for r in drone_rows}):
        rows = [r for r in drone_rows if r["drone_id"] == drone_id]
        ax.plot(
            [r["x_m"] for r in rows],
            [r["y_m"] for r in rows],
            linewidth=1.1,
            label=f"Drone {drone_id}",
        )
    ax.set(
        xlabel="X (m)", ylabel="Y (m)",
        title=f"Observed drone tracks — seed={args.seed}",
        aspect="equal",
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    tracks_path = out / "drone_tracks.png"
    fig.savefig(tracks_path, dpi=150)
    plt.close(fig)

    # Figure 2: altitude and battery over simulated time.
    fig, (ax_z, ax_b) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for drone_id in sorted({r["drone_id"] for r in drone_rows}):
        rows = [r for r in drone_rows if r["drone_id"] == drone_id]
        t = [r["sim_time_s"] for r in rows]
        ax_z.plot(t, [r["z_m"] for r in rows], label=f"D{drone_id}")
        ax_b.plot(t, [r["battery_pct"] for r in rows], label=f"D{drone_id}")
    ax_z.set_ylabel("Altitude (m)")
    ax_z.set_title("Altitude by drone")
    ax_b.set_ylabel("Battery (%)")
    ax_b.set_xlabel("Simulation time (s)")
    ax_b.set_title("Battery by drone")
    ax_z.grid(alpha=0.25)
    ax_b.grid(alpha=0.25)
    ax_b.legend(ncol=5, fontsize=7)
    fig.tight_layout()
    profile_path = out / "altitude_battery_profiles.png"
    fig.savefig(profile_path, dpi=150)
    plt.close(fig)

    # Figure 3: mission-level coverage and network counters.
    fig, ax = plt.subplots(figsize=(10, 5))
    t = [r["sim_time_s"] for r in mission_rows]
    ax.plot(t, [r["coverage_pct"] for r in mission_rows], label="Coverage (%)")
    ax.set(xlabel="Simulation time (s)", ylabel="Coverage (%)",
           title=f"Mission progress — seed={args.seed}")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(t, [r["network_drop_count"] for r in mission_rows],
             linestyle="--", label="Cumulative drops")
    ax2.set_ylabel("Dropped messages (cumulative)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    progress_path = out / "mission_progress.png"
    fig.savefig(progress_path, dpi=150)
    plt.close(fig)

    print("=" * 68)
    print("OBSERVABILITY V1 — RESULT")
    print("=" * 68)
    print("Seed:", args.seed)
    print("Network profile:", args.network_profile)
    print("Drones:", len(loop.drones))
    print("Mission completed:", result.completed)
    print("Completion reason:", result.completion_reason)
    print("Coverage:", f"{result.coverage_pct:.2f}%")
    print("Steps:", result.steps)
    print("Simulated time:", result.simulated_time_s)
    print("Event chain valid:", result.event_chain_valid)
    print("Telemetry rows:", len(drone_rows))
    print("Minimum pairwise separation (m):",
          summary["minimum_observed_pairwise_separation_m"])
    print("Output:", out.resolve())
    for path in (telemetry_path, mission_path, summary_path,
                 tracks_path, profile_path, progress_path):
        print("[ARTIFACT]", path.name, path.stat().st_size, "bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
