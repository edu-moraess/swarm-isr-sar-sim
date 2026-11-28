"""Technical 3D replay and engineering diagnostics for MissionLoop.

Captures true point-mass state at every simulation step. The video is a
visualization of this simulator's kinematic model, not a flight-physics model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from swarm_core.config import load_full
from swarm_core.drone import min_separation
from swarm_core.mission import MissionLoop


BG = "#07111D"
PANEL = "#0D1B2A"
GRID = "#26384A"
TEXT = "#E6EEF5"
MUTED = "#8EA4B8"
CYAN = "#39C6D8"
AMBER = "#F4B942"
RED = "#F06B6B"
GREEN = "#58C99A"
PALETTE = [
    "#39C6D8", "#F4B942", "#8D83F6", "#58C99A", "#F08A5D",
    "#5C9DED", "#D77BEA", "#B7D66B", "#E879A9", "#9CB5C9",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--output-dir", type=Path, default=Path("swarm_technical_replay_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--network-profile", default="clean")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--fps", type=int, default=24)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows captured for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase_spans(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    if not rows:
        return []
    spans: list[tuple[float, float, str]] = []
    start = float(rows[0]["sim_time_s"])
    current = str(rows[0]["phase"])
    for row in rows[1:]:
        phase = str(row["phase"])
        if phase != current:
            spans.append((start, float(row["sim_time_s"]), current))
            start = float(row["sim_time_s"])
            current = phase
    spans.append((start, float(rows[-1]["sim_time_s"]), current))
    return spans


def style_axes(ax: Any) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, alpha=0.55, linewidth=0.6)


def make_dashboard(
    out: Path,
    tracks: dict[int, dict[str, np.ndarray]],
    mission_rows: list[dict[str, Any]],
    landing_positions: dict[int, np.ndarray],
    seed: int,
    mission_id: str,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    fig.suptitle(
        f"SWARM ISR/SAR  /  MISSION ANALYTICS  /  SEED {seed}",
        color=TEXT, fontsize=17, fontweight="bold", x=0.06, ha="left", y=0.98,
    )
    ax_xy, ax_z, ax_b, ax_sep = axes.flat
    for ax in axes.flat:
        style_axes(ax)

    for idx, (drone_id, data) in enumerate(sorted(tracks.items())):
        color = PALETTE[idx % len(PALETTE)]
        ax_xy.plot(data["x"], data["y"], color=color, lw=1.25, alpha=0.88)
        ax_xy.scatter(data["x"][0], data["y"][0], color=color, marker="o", s=18)
        ax_xy.scatter(data["x"][-1], data["y"][-1], color=color, marker="x", s=28)
        ax_z.plot(data["time"], data["z"], color=color, lw=1.0, label=f"D{drone_id}")
        ax_b.plot(data["time"], data["battery"], color=color, lw=1.0, label=f"D{drone_id}")

    ax_xy.scatter([0], [0], marker="*", s=150, color=AMBER, label="Base", zorder=5)
    if landing_positions:
        pads = np.array(list(landing_positions.values()))
        ax_xy.scatter(pads[:, 0], pads[:, 1], marker="s", s=28, facecolors="none",
                      edgecolors=GREEN, linewidths=1.2, label="Landing pads")
    ax_xy.set_title("01  /  TRUE XY TRAJECTORIES", loc="left", fontweight="bold")
    ax_xy.set_xlabel("East / X (m)")
    ax_xy.set_ylabel("North / Y (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.legend(loc="upper right", fontsize=7, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)

    ax_z.set_title("02  /  ALTITUDE RESPONSE", loc="left", fontweight="bold")
    ax_z.set_xlabel("Simulation time (s)")
    ax_z.set_ylabel("Altitude Z (m)")
    ax_z.axhline(0.0, color=AMBER, lw=0.8, ls="--", alpha=0.8)
    ax_z.legend(ncol=5, fontsize=7, facecolor=PANEL, labelcolor=TEXT, framealpha=0.7)

    ax_b.set_title("03  /  BATTERY STATE", loc="left", fontweight="bold")
    ax_b.set_xlabel("Simulation time (s)")
    ax_b.set_ylabel("Remaining battery (%)")
    ax_b.set_ylim(0, 105)
    ax_b.legend(ncol=5, fontsize=7, facecolor=PANEL, labelcolor=TEXT, framealpha=0.7)

    mt = np.array([float(row["sim_time_s"]) for row in mission_rows])
    sep = np.array([float(row["minimum_horizontal_separation_m"]) for row in mission_rows])
    coverage = np.array([float(row["coverage_pct"]) for row in mission_rows])
    ax_sep.plot(mt, sep, color=CYAN, lw=1.2, label="Min horizontal separation")
    ax_sep.axhline(1.0, color=AMBER, ls="--", lw=0.9,
                   label="1 m visual reference (not a collision limit)")
    ax_sep.set_title("04  /  SWARM SPACING & COVERAGE", loc="left", fontweight="bold")
    ax_sep.set_xlabel("Simulation time (s)")
    ax_sep.set_ylabel("Horizontal separation (m)", color=CYAN)
    ax_sep.tick_params(axis="y", labelcolor=CYAN)
    ax_cov = ax_sep.twinx()
    ax_cov.plot(mt, coverage, color=GREEN, lw=1.0, alpha=0.85, label="Observed coverage")
    ax_cov.set_ylabel("Observed free-cell coverage (%)", color=GREEN)
    ax_cov.tick_params(axis="y", labelcolor=GREEN)
    ax_cov.set_ylim(0, 105)
    lines = ax_sep.get_lines() + ax_cov.get_lines()
    ax_sep.legend(lines, [line.get_label() for line in lines], fontsize=7,
                  loc="upper left", facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)

    fig.text(0.06, 0.015,
             f"Mission: {mission_id}   |   Data: per-step simulator state   |   "
             "Separation is a center-to-center measurement, not collision avoidance.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.94))
    path = out / "mission_technical_dashboard.png"
    fig.savefig(path, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def make_landing_diagnostics(
    out: Path,
    tracks: dict[int, dict[str, np.ndarray]],
    mission_rows: list[dict[str, Any]],
    landing_positions: dict[int, np.ndarray],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    fig.suptitle("LANDING PHASE  /  TERMINAL-STATE VERIFICATION",
                 color=TEXT, fontsize=16, fontweight="bold", x=0.06, ha="left", y=0.98)
    ax_z, ax_vz, ax_pad, ax_sep = axes.flat
    for ax in axes.flat:
        style_axes(ax)

    phase_rows = [r for r in mission_rows if r["phase"] in ("LAND", "COMPLETE")]
    if not phase_rows:
        phase_rows = mission_rows[-100:]
    t0 = max(0.0, float(phase_rows[0]["sim_time_s"]) - 10.0)
    for idx, (drone_id, data) in enumerate(sorted(tracks.items())):
        color = PALETTE[idx % len(PALETTE)]
        mask = data["time"] >= t0
        times = data["time"][mask]
        z = data["z"][mask]
        vz = data["vz"][mask]
        ax_z.plot(times, z, color=color, lw=1.15, label=f"D{drone_id}")
        ax_vz.plot(times, vz, color=color, lw=1.15, label=f"D{drone_id}")
        pad = landing_positions.get(drone_id)
        if pad is not None:
            pad_error = np.sqrt((data["x"][mask] - pad[0]) ** 2 +
                                (data["y"][mask] - pad[1]) ** 2)
            ax_pad.plot(times, pad_error, color=color, lw=1.15, label=f"D{drone_id}")

    ax_z.axhspan(-0.20, 0.20, color=GREEN, alpha=0.12)
    ax_z.axhline(0.0, color=AMBER, ls="--", lw=0.9)
    ax_z.set_title("01  /  ALTITUDE ERROR", loc="left", fontweight="bold")
    ax_z.set_ylabel("Z relative to ground (m)")
    ax_z.set_xlabel("Simulation time (s)")

    ax_vz.axhspan(-0.20, 0.20, color=GREEN, alpha=0.12)
    ax_vz.axhline(0.0, color=AMBER, ls="--", lw=0.9)
    ax_vz.set_title("02  /  VERTICAL VELOCITY", loc="left", fontweight="bold")
    ax_vz.set_ylabel("Vertical velocity (m/s)")
    ax_vz.set_xlabel("Simulation time (s)")

    ax_pad.axhspan(0.0, 0.25, color=GREEN, alpha=0.12)
    ax_pad.axhline(0.25, color=AMBER, ls="--", lw=0.9)
    ax_pad.set_title("03  /  HORIZONTAL PAD ERROR", loc="left", fontweight="bold")
    ax_pad.set_ylabel("Distance to assigned pad (m)")
    ax_pad.set_xlabel("Simulation time (s)")

    land_t = np.array([float(r["sim_time_s"]) for r in phase_rows])
    land_sep = np.array([float(r["minimum_horizontal_separation_m"]) for r in phase_rows])
    ax_sep.plot(land_t, land_sep, color=CYAN, lw=1.3)
    ax_sep.set_title("04  /  SWARM SEPARATION DURING LAND", loc="left", fontweight="bold")
    ax_sep.set_ylabel("Minimum horizontal separation (m)")
    ax_sep.set_xlabel("Simulation time (s)")
    for ax in (ax_z, ax_vz, ax_pad):
        ax.legend(ncol=5, fontsize=6, facecolor=PANEL, labelcolor=TEXT, framealpha=0.65)

    fig.text(0.06, 0.015,
             "Green bands show the configured terminal tolerances; they are simulator criteria, "
             "not certified flight-safety limits.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.94))
    path = out / "landing_diagnostics.png"
    fig.savefig(path, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def make_video(
    out: Path,
    tracks: dict[int, dict[str, np.ndarray]],
    mission_rows: list[dict[str, Any]],
    landing_positions: dict[int, np.ndarray],
    base: list[float],
    area_size: list[float],
    search_alt_m: float,
    seed: int,
    frames: int,
    fps: int,
    mission_id: str,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. In Colab run: !apt-get -qq install -y ffmpeg")

    n_samples = len(next(iter(tracks.values()))["time"])
    frame_ids = np.linspace(0, n_samples - 1, min(frames, n_samples), dtype=int)
    drone_ids = sorted(tracks)
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    gs = GridSpec(1, 5, figure=fig, width_ratios=[4.3, 0.03, 1.45, 0.03, 0.03])
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.set_facecolor(BG)
    ax.set_proj_type("persp")
    ax.view_init(elev=29, azim=-56)
    ax.set_xlim(0, float(area_size[0]))
    ax.set_ylim(0, float(area_size[1]))
    ax.set_zlim(-2, search_alt_m + 8)
    ax.set_box_aspect((1.0, 1.0, 0.36))
    ax.set_xlabel("X / East (m)", color=MUTED, labelpad=10)
    ax.set_ylabel("Y / North (m)", color=MUTED, labelpad=10)
    ax.set_zlabel("Altitude Z (m)", color=MUTED, labelpad=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.xaxis.pane.set_facecolor((0.03, 0.07, 0.11, 1.0))
    ax.yaxis.pane.set_facecolor((0.03, 0.07, 0.11, 1.0))
    ax.zaxis.pane.set_facecolor((0.03, 0.07, 0.11, 1.0))
    ax.xaxis.pane.set_edgecolor(GRID)
    ax.yaxis.pane.set_edgecolor(GRID)
    ax.zaxis.pane.set_edgecolor(GRID)
    ax.grid(color=GRID, alpha=0.45)

    # Ground reference grid and base marker.
    gx = np.linspace(0, float(area_size[0]), 9)
    gy = np.linspace(0, float(area_size[1]), 9)
    for x in gx:
        ax.plot([x, x], [0, area_size[1]], [0, 0], color=GRID, lw=0.35, alpha=0.5)
    for y in gy:
        ax.plot([0, area_size[0]], [y, y], [0, 0], color=GRID, lw=0.35, alpha=0.5)
    ax.scatter([base[0]], [base[1]], [0], marker="*", s=170, color=AMBER,
               edgecolors=TEXT, linewidths=0.5, label="Base")
    if landing_positions:
        pads = np.array(list(landing_positions.values()))
        ax.scatter(pads[:, 0], pads[:, 1], pads[:, 2], marker="s", s=45,
                   facecolors="none", edgecolors=GREEN, linewidths=1.2)

    trails = {}
    markers = {}
    velocity_lines = {}
    for idx, drone_id in enumerate(drone_ids):
        color = PALETTE[idx % len(PALETTE)]
        trails[drone_id], = ax.plot([], [], [], color=color, lw=1.5, alpha=0.85)
        markers[drone_id] = ax.scatter([], [], [], color=color, s=55,
                                       edgecolors=TEXT, linewidths=0.55, depthshade=False)
        velocity_lines[drone_id], = ax.plot([], [], [], color=color, lw=1.0, alpha=0.9)

    fig.text(0.045, 0.955, "SWARM ISR/SAR  /  TECHNICAL FLIGHT REPLAY",
             color=TEXT, fontsize=15, fontweight="bold")
    fig.text(0.045, 0.925,
             f"{mission_id}   |   SEED {seed}   |   POINT-MASS KINEMATICS   |   "
             f"{fps} FPS",
             color=MUTED, fontsize=8)

    panel = fig.add_subplot(gs[0, 2])
    panel.set_facecolor(PANEL)
    panel.set_xticks([])
    panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_color(GRID)
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel_text = panel.text(0.07, 0.94, "", va="top", ha="left", color=TEXT,
                            fontsize=9, family="DejaVu Sans Mono", linespacing=1.55)
    panel.text(0.07, 0.04, "SIMULATION TELEMETRY\nNot flight-certified",
               color=MUTED, fontsize=7, va="bottom")

    time_values = np.array([float(r["sim_time_s"]) for r in mission_rows])
    coverage_values = np.array([float(r["coverage_pct"]) for r in mission_rows])
    sep_values = np.array([float(r["minimum_horizontal_separation_m"]) for r in mission_rows])
    phase_values = [str(r["phase"]) for r in mission_rows]

    def update(frame_number: int) -> list[Any]:
        sample_idx = int(frame_ids[frame_number])
        for drone_id in drone_ids:
            data = tracks[drone_id]
            x = data["x"][:sample_idx + 1]
            y = data["y"][:sample_idx + 1]
            z = data["z"][:sample_idx + 1]
            trails[drone_id].set_data_3d(x, y, z)
            px, py, pz = float(x[-1]), float(y[-1]), float(z[-1])
            markers[drone_id]._offsets3d = ([px], [py], [pz])
            vx = float(data["vx"][sample_idx])
            vy = float(data["vy"][sample_idx])
            vz = float(data["vz"][sample_idx])
            vector_scale = 1.2
            velocity_lines[drone_id].set_data_3d(
                [px, px + vx * vector_scale],
                [py, py + vy * vector_scale],
                [pz, pz + vz * vector_scale],
            )

        sim_time = float(time_values[min(sample_idx, len(time_values) - 1)])
        mission_idx = min(sample_idx, len(mission_rows) - 1)
        phase = phase_values[mission_idx]
        coverage = float(coverage_values[mission_idx])
        separation = float(sep_values[mission_idx])
        battery_now = [float(tracks[d]["battery"][sample_idx]) for d in drone_ids]
        altitude_now = [float(tracks[d]["z"][sample_idx]) for d in drone_ids]
        speed_now = [float(np.linalg.norm([
            tracks[d]["vx"][sample_idx],
            tracks[d]["vy"][sample_idx],
            tracks[d]["vz"][sample_idx],
        ])) for d in drone_ids]
        panel_text.set_text(
            "MISSION STATUS\n"
            "────────────────────\n"
            f"PHASE       {phase:<10}\n"
            f"SIM TIME    {sim_time:>8.1f} s\n"
            f"STEP        {sample_idx:>8d}\n\n"
            "SWARM STATE\n"
            "────────────────────\n"
            f"AGENTS      {len(drone_ids):>8d}\n"
            f"MEAN ALT    {np.mean(altitude_now):>8.2f} m\n"
            f"MEAN SPEED  {np.mean(speed_now):>8.2f} m/s\n"
            f"MEAN BATT   {np.mean(battery_now):>8.2f} %\n\n"
            "MISSION METRICS\n"
            "────────────────────\n"
            f"COVERAGE    {coverage:>8.2f} %\n"
            f"MIN XY SEP  {separation:>8.4f} m\n\n"
            "MODEL SCOPE\n"
            "────────────────────\n"
            "3D position + velocity\n"
            "No collision avoidance\n"
            "No rotor / thrust model\n"
            "No terrain interaction"
        )
        return [*trails.values(), *markers.values(), *velocity_lines.values(), panel_text]

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / fps,
                              blit=False, repeat=False)
    path = out / "swarm_technical_replay_v2.mp4"
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=6500,
        metadata={"title": "SWARM ISR/SAR Technical Replay V2", "artist": "SWARM ISR/SAR"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    animation.save(path, writer=writer, dpi=120)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    if args.frames < 2 or args.fps < 1:
        raise ValueError("--frames must be >= 2 and --fps must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    drone_rows: list[dict[str, Any]] = []
    mission_rows: list[dict[str, Any]] = []
    tracks: dict[int, dict[str, list[float]]] = {
        drone.id: {
            "time": [], "x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": [],
            "battery": [], "distance": [],
        }
        for drone in loop.drones
    }
    previous = {drone.id: np.asarray(drone.position, dtype=float).copy() for drone in loop.drones}
    cumulative = {drone.id: 0.0 for drone in loop.drones}

    def record() -> None:
        positions = []
        for drone in loop.drones:
            pos = np.asarray(drone.position, dtype=float).copy()
            vel = np.asarray(drone.velocity, dtype=float).copy()
            cumulative[drone.id] += float(np.linalg.norm(pos - previous[drone.id]))
            previous[drone.id] = pos.copy()
            pad = loop.landing_positions.get(drone.id)
            tracks[drone.id]["time"].append(float(loop.clock.t))
            for key, value in zip(("x", "y", "z"), pos):
                tracks[drone.id][key].append(float(value))
            for key, value in zip(("vx", "vy", "vz"), vel):
                tracks[drone.id][key].append(float(value))
            tracks[drone.id]["battery"].append(float(drone.battery_pct))
            tracks[drone.id]["distance"].append(cumulative[drone.id])
            positions.append(pos)
            drone_rows.append({
                "step": loop.clock.step,
                "sim_time_s": loop.clock.t,
                "phase": loop.fsm.phase.name,
                "drone_id": drone.id,
                "x_m": float(pos[0]), "y_m": float(pos[1]), "z_m": float(pos[2]),
                "vx_mps": float(vel[0]), "vy_mps": float(vel[1]), "vz_mps": float(vel[2]),
                "speed_3d_mps": float(np.linalg.norm(vel)),
                "battery_pct": float(drone.battery_pct),
                "distance_travelled_m": cumulative[drone.id],
                "waypoint_index": int(drone.wp_idx),
                "waypoint_count": len(drone.waypoints),
                "pad_x_m": None if pad is None else float(pad[0]),
                "pad_y_m": None if pad is None else float(pad[1]),
                "pad_z_m": None if pad is None else float(pad[2]),
                "alive": bool(drone.alive),
            })

        mission_rows.append({
            "step": loop.clock.step,
            "sim_time_s": loop.clock.t,
            "phase": loop.fsm.phase.name,
            "coverage_pct": loop._coverage_pct(),
            "observed_free_cells": len(loop.observed_cells.intersection(loop.world.free_cells())),
            "event_count": len(loop.events.events),
            "network_send_count": loop.network.send_count,
            "network_drop_count": loop.network.drop_count,
            "minimum_horizontal_separation_m": min_separation(loop.drones),
        })

    loop.start()
    record()
    max_steps = max(1, math.ceil(mission_cfg.time_limit_s / loop.clock.dt))
    while not loop.finished and loop.clock.step < max_steps:
        loop.step()
        record()

    result = loop.result()
    tracks_np = {
        drone_id: {key: np.asarray(values, dtype=float) for key, values in data.items()}
        for drone_id, data in tracks.items()
    }
    write_csv(args.output_dir / "drone_telemetry_v2.csv", drone_rows)
    write_csv(args.output_dir / "mission_telemetry_v2.csv", mission_rows)

    summary = {
        "experiment": "SWARM ISR/SAR Technical Replay V2",
        "seed": args.seed,
        "network_profile": args.network_profile,
        "mission_id": mission_cfg.mission_id,
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
        "minimum_horizontal_separation_m": min(
            float(row["minimum_horizontal_separation_m"]) for row in mission_rows
        ),
        "per_drone": {
            str(drone.id): {
                "distance_travelled_m": cumulative[drone.id],
                "final_position_xyz_m": [float(value) for value in drone.position],
                "final_velocity_xyz_mps": [float(value) for value in drone.velocity],
                "final_battery_pct": float(drone.battery_pct),
                "landing_pad_xyz_m": (
                    [float(value) for value in loop.landing_positions[drone.id]]
                    if drone.id in loop.landing_positions else None
                ),
                "alive": bool(drone.alive),
            }
            for drone in loop.drones
        },
        "phase_transitions": phase_spans(mission_rows),
        "interpretation_limits": [
            "The replay visualizes the simulator's point-mass kinematics, not aerodynamic flight.",
            "Velocity vectors are scaled visual indicators, not force or thrust vectors.",
            "Horizontal separation is center-to-center measurement, not collision avoidance.",
            "Coverage is observed free-cell coverage; it is not target confirmation.",
            "Battery follows the current simplified simulator model.",
        ],
    }
    summary_path = args.output_dir / "technical_replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    dashboard = make_dashboard(
        args.output_dir, tracks_np, mission_rows, loop.landing_positions,
        args.seed, mission_cfg.mission_id,
    )
    landing_plot = make_landing_diagnostics(
        args.output_dir, tracks_np, mission_rows, loop.landing_positions,
    )
    video = make_video(
        args.output_dir, tracks_np, mission_rows, loop.landing_positions,
        mission_cfg.base, mission_cfg.area.size_m, cfg.swarm.search_alt_m,
        args.seed, args.frames, args.fps, mission_cfg.mission_id,
    )

    print("=" * 72)
    print("SWARM ISR/SAR — TECHNICAL REPLAY V2")
    print("=" * 72)
    print(f"Branch runtime seed: {args.seed} | network: {args.network_profile}")
    print(f"Mission: {mission_cfg.mission_id} | drones: {len(loop.drones)}")
    print(f"Phase: {result.phase.name} | reason: {result.completion_reason}")
    print(f"Steps: {result.steps} | simulated time: {result.simulated_time_s:.1f}s")
    print(f"Coverage: {result.coverage_pct:.2f}% | event chain: {result.event_chain_valid}")
    print(f"Minimum horizontal separation: {summary['minimum_horizontal_separation_m']:.4f}m")
    print("Final landing state:")
    for drone_id, data in summary["per_drone"].items():
        pos = data["final_position_xyz_m"]
        vel = data["final_velocity_xyz_mps"]
        print(f"  D{drone_id:>2}: xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m "
              f"v=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})m/s "
              f"battery={data['final_battery_pct']:.2f}%")
    print("\n[ARTIFACTS]")
    for path in (
        video, dashboard, landing_plot,
        args.output_dir / "drone_telemetry_v2.csv",
        args.output_dir / "mission_telemetry_v2.csv", summary_path,
    ):
        print(f"{path.name} | {path.stat().st_size:,} bytes")
    return 0 if result.completed and result.event_chain_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
