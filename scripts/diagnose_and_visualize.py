#!/usr/bin/env python3
"""Diagnóstico e visualização fiel de uma execução do MissionLoop v0.

AUDITORIA DO SNAPSHOT
---------------------
Ponto de entrada real: swarm_core.mission.MissionLoop.run(), que retorna
swarm_core.mission.MissionRunResult. As configurações públicas são carregadas
por swarm_core.config.load_full(); --seed substitui somente MissionConfig.seed.

Campos expostos em MissionRunResult:
phase, completion_reason, steps, simulated_time_s, coverage_pct,
observed_free_cells, total_free_cells, event_count, event_sha256,
event_chain_valid, network_send_count, network_drop_count e a propriedade
completed.

Dados internos acessíveis após a execução (não fazem parte do resultado):
MissionLoop.world (width, height, grid, cell_m, origin), observed_cells,
drones (posição atual), events.events, gossip.nodes[*].store e network.
As posições iniciais são capturadas antes de run() por leitura dos drones.
As observações por célula podem ser consultadas em observed_cells; o grid
contém FREE/OCCUPIED. Os batches de sensores permanecem nos stores do Gossip.

Limitações:
- MissionLoop não guarda histórico de posição por step. Portanto V2 não pode
  ser produzido honestamente; não há reconstrução de trajetórias.
- V1 usa posição inicial capturada antes da execução e posição final lida
  depois dela, além do grid e observed_cells reais.
- Não há métricas individuais consolidadas por drone; V5 é omitida.
- MissionRunResult não expõe eventos individuais, sensores, energia nem
  estados de crença. Esses objetos existem no MissionLoop, mas são internos
  ao resumo formal.
- O MissionLoop v0 termina quando as rotas estáticas são percorridas e os
  drones pousam. Isso não significa confirmação de alvo SAR nem atingimento
  do limiar de cobertura configurado.
- matplotlib já é dependência declarada em pyproject.toml (>=3.8).
- Os PNGs são renderizados duas vezes e seus SHA-256 são comparados. A
  comparação cobre o ambiente/versão local do Matplotlib; versões ou
  backends diferentes podem produzir bytes diferentes mesmo com dados iguais.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import sys
import tempfile
import traceback
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from swarm_core.config import load_full  # noqa: E402
from swarm_core.mission import MissionLoop, MissionRunResult  # noqa: E402
from swarm_core.world import CellState  # noqa: E402

EXPECTED_METRICS = (
    "coverage_pct",
    "observed_free_cells",
    "total_free_cells",
    "network_send_count",
    "network_drop_count",
    "event_count",
    "event_sha256",
    "event_chain_valid",
    "phase",
    "steps",
    "simulated_time_s",
    "completed",
)

PNG_NAMES = (
    "coverage_map",
    "trajectories",
    "metrics",
    "event_chain",
    "per_drone",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Seed da missão (default: 42)")
    parser.add_argument("--outdir", type=Path, default=Path("out/"), help="Diretório dos PNGs")
    return parser.parse_args()


def make_loop(seed: int) -> MissionLoop:
    cfg = load_full(
        mission_path=ROOT / "config/mission.yaml",
        swarm_path=ROOT / "config/swarm.yaml",
        perception_path=ROOT / "config/perception.yaml",
        network_path=ROOT / "config/network.yaml",
    )
    mission = cfg.mission.model_copy(update={"seed": seed})
    return MissionLoop(
        mission=mission,
        swarm=cfg.swarm,
        perception=cfg.perception,
        network=cfg.network,
    )


def run_once(seed: int) -> tuple[MissionLoop, MissionRunResult, dict[int, tuple[float, ...]]]:
    loop = make_loop(seed)
    initial = {drone.id: tuple(float(v) for v in drone.position) for drone in loop.drones}
    result = loop.run()
    if not isinstance(result, MissionRunResult):
        raise TypeError(f"MissionLoop.run() retornou tipo inesperado: {type(result)!r}")
    return loop, result, initial


def scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, Enum))


def printable(value: Any) -> str:
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name} ({value.value})"
    return repr(value)


def inspect_result(result: MissionRunResult) -> tuple[dict[str, Any], list[str]]:
    print("\n[A] TIPO DO RESULTADO + API PÚBLICA")
    print(f"Tipo: {type(result).__module__}.{type(result).__qualname__}")
    public = sorted(name for name in dir(result) if not name.startswith("_"))
    for name in public:
        try:
            member = getattr(result, name)
            if callable(member):
                try:
                    signature = str(inspect.signature(member))
                except (TypeError, ValueError):
                    signature = "(...)"
                print(f"  {name}{signature} [método]")
            else:
                print(f"  {name}: {type(member).__name__}")
        except Exception as exc:
            print(f"  {name}: <erro ao inspecionar: {exc}>")

    values: dict[str, Any] = {}
    if is_dataclass(result):
        for item in fields(result):
            values[item.name] = getattr(result, item.name)
    else:
        values.update(vars(result))

    print("\n[B] CAMPOS ESCALARES")
    for name, value in values.items():
        if scalar(value):
            print(f"  {name} = {printable(value)}")

    print("\n[C] CAMPOS COMPOSTOS")
    for name, value in values.items():
        if scalar(value):
            continue
        try:
            size = len(value)
            example = value[0] if size and hasattr(value, "__getitem__") else (
                next(iter(value)) if size else "<vazio>"
            )
            print(f"  {name}: {type(value).__name__} (len={size}); exemplo={printable(example)}")
        except (TypeError, KeyError, IndexError):
            print(f"  {name}: {type(value).__name__} (sem len/exemplo)")

    print("\n[D] MÉTRICAS-CHAVE")
    found = []
    for name in EXPECTED_METRICS:
        if hasattr(result, name):
            value = getattr(result, name)
            found.append(name)
            print(f"  {name} = {printable(value)}")
    print(f"  Encontradas: {len(found)}/{len(EXPECTED_METRICS)}")
    return values, found


def report_exposure(loop: MissionLoop, result: MissionRunResult) -> None:
    print("\n[E] CAMPOS ESPERADOS, EXPOSIÇÃO E AUSÊNCIAS")
    exposed = set(name for name in EXPECTED_METRICS if hasattr(result, name))
    internal = {
        "dimensões/grid": hasattr(loop, "world") and hasattr(loop.world, "grid"),
        "células observadas": hasattr(loop, "observed_cells"),
        "posições atuais dos drones": hasattr(loop, "drones"),
        "trilha de eventos": hasattr(loop, "events") and hasattr(loop.events, "events"),
        "batches dos sensores": hasattr(loop, "gossip") and hasattr(loop.gossip, "nodes"),
        "crenças por agente": hasattr(loop, "gossip") and hasattr(loop.gossip, "nodes"),
    }
    print("  Expostos no MissionRunResult:")
    for name in sorted(exposed):
        print(f"    - {name}")
    print("  Existem no MissionLoop, mas não são campos do resultado:")
    for name, exists in internal.items():
        print(f"    - {name}: {'SIM' if exists else 'NÃO'}")
    print("  Não existe como histórico neste snapshot:")
    print("    - posições dos drones por step: não há buffer/API de trajetória")
    print("    - métricas consolidadas individuais por drone: não há campo no resultado")
    print("  Não implementado como resultado de missão:")
    print("    - confirmação/contagem de alvos detectados pela missão")
    print("    - critério de parada por cobertura ou confirmação de todos os alvos")


def scalar_snapshot(result: MissionRunResult) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name in EXPECTED_METRICS:
        if hasattr(result, name):
            value = getattr(result, name)
            snapshot[name] = value.name if isinstance(value, Enum) else value
    return snapshot


def compare_runs(first: MissionRunResult, second: MissionRunResult) -> bool:
    print("\n[F] DETERMINISMO — DUAS EXECUÇÕES COM A MESMA SEED")
    a, b = scalar_snapshot(first), scalar_snapshot(second)
    differences = []
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            differences.append((key, a.get(key), b.get(key)))
    if differences:
        print("  DIVERGÊNCIAS:")
        for key, left, right in differences:
            print(f"    {key}: execução 1={left!r}; execução 2={right!r}")
        return False
    print("  OK — todos os campos escalares expostos coincidem.")
    return True


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", format="png", metadata={"Software": "SWARM diagnostic"})


def unavailable_axis(ax: plt.Axes, title: str, reason: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "NÃO DISPONÍVEL\n" + reason, ha="center", va="center",
            transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def render_all(
    loop: MissionLoop,
    result: MissionRunResult,
    initial: dict[int, tuple[float, ...]],
    seed: int,
    outdir: Path,
) -> tuple[list[Path], list[str]]:
    outdir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    skips: list[str] = []

    # V1 — mapa: usa o grid e observed_cells reais do World/MissionLoop.
    world = loop.world
    required_v1 = (
        hasattr(world, "width") and hasattr(world, "height")
        and hasattr(world, "grid") and hasattr(loop, "observed_cells")
        and bool(loop.drones)
    )
    if required_v1:
        fig, ax = plt.subplots(figsize=(9, 7))
        grid = np.asarray(world.grid).reshape((world.height, world.width))
        observed = np.zeros_like(grid, dtype=bool)
        for cell_idx in loop.observed_cells:
            if 0 <= int(cell_idx) < world.n_cells:
                row, col = divmod(int(cell_idx), world.width)
                observed[row, col] = True
        # Células ocupadas ficam em cinza; livres observadas/não observadas em dois tons.
        display = np.full(grid.shape, 0, dtype=np.int8)
        display[(grid == CellState.FREE) & observed] = 1
        display[grid == CellState.OCCUPIED] = 2
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#f4f5f6", "#8fa9b8", "#c7cbd0"])
        extent = (
            world.origin.x,
            world.origin.x + world.width * world.cell_m,
            world.origin.y,
            world.origin.y + world.height * world.cell_m,
        )
        ax.imshow(display, origin="lower", interpolation="nearest", extent=extent,
                  cmap=cmap, vmin=0, vmax=2, aspect="equal")
        ax.set_xticks(np.arange(extent[0], extent[1] + world.cell_m, world.cell_m), minor=True)
        ax.set_yticks(np.arange(extent[2], extent[3] + world.cell_m, world.cell_m), minor=True)
        ax.grid(which="minor", color="#d9dee2", linewidth=0.25)
        colors = plt.get_cmap("tab10")
        for index, drone in enumerate(loop.drones):
            color = colors(index % 10) if len(loop.drones) > 1 else "#245b78"
            start = initial.get(drone.id)
            end = tuple(float(v) for v in drone.position)
            if start is not None:
                ax.scatter(start[0], start[1], marker="o", facecolors="none",
                           edgecolors=[color], s=65, label=f"Drone {drone.id} — início")
            ax.scatter(end[0], end[1], marker="o", color=[color], s=34,
                       label=f"Drone {drone.id} — fim")
        from matplotlib.patches import Patch
        handles, labels = ax.get_legend_handles_labels()
        handles = [Patch(facecolor="#f4f5f6", edgecolor="#999", label="Livre não observada"),
                   Patch(facecolor="#8fa9b8", edgecolor="#999", label="Livre observada"),
                   Patch(facecolor="#c7cbd0", edgecolor="#999", label="Ocupada")] + handles
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        ax.set_title(f"Cobertura observada — seed={seed}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        fig.tight_layout()
        path = outdir / f"coverage_map_seed{seed}.png"
        save_figure(fig, path)
        plt.close(fig)
        generated.append(path)
    else:
        skips.append("[SKIP] coverage_map: faltam grid/dimensões, células observadas ou agentes acessíveis.")

    # V2 — não existe histórico de posições por step.
    skips.append("[SKIP] trajectories: MissionLoop não armazena posições por step; só há posições inicial/final.")

    # V3 — painéis para métricas existentes; ausências ficam explícitas.
    panels = [
        ("Cobertura observada (%)", "coverage_pct", "%"),
        ("Células livres observadas / total", "cell_ratio", ""),
        ("Comunicação", "network", ""),
        ("Eventos registrados", "event_count", ""),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    for ax, (title, key, unit) in zip(axes, panels, strict=True):
        if key == "cell_ratio":
            if hasattr(result, "observed_free_cells") and hasattr(result, "total_free_cells"):
                vals = [result.observed_free_cells, result.total_free_cells]
                ax.bar(["Observadas", "Total"], vals, color=["#6f91a5", "#c7cbd0"])
                ax.set_title(title)
                ax.set_ylabel("Células")
            else:
                unavailable_axis(ax, title, "contagem não exposta")
        elif key == "network":
            if hasattr(result, "network_send_count") and hasattr(result, "network_drop_count"):
                ax.bar(["Enviadas", "Descartadas"],
                       [result.network_send_count, result.network_drop_count],
                       color=["#6f91a5", "#c28b72"])
                ax.set_title(title)
                ax.set_ylabel("Mensagens")
            else:
                unavailable_axis(ax, title, "métricas de rede não expostas")
        elif hasattr(result, key):
            value = getattr(result, key)
            ax.bar([title], [float(value)], color="#6f91a5")
            ax.set_title(title)
            ax.set_ylabel(unit or "Quantidade")
            ax.tick_params(axis="x", labelrotation=15)
        else:
            unavailable_axis(ax, title, f"{key} ausente no resultado")
    fig.suptitle(f"Métricas da missão — seed={seed}")
    fig.tight_layout()
    path = outdir / f"metrics_seed{seed}.png"
    save_figure(fig, path)
    plt.close(fig)
    generated.append(path)

    # V4 — painel textual com dados reais do resultado.
    event_fields = ("event_count", "event_sha256", "event_chain_valid")
    if any(hasattr(result, field) for field in event_fields):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.axis("off")
        lines = ["TRILHA DE EVENTOS", f"Seed: {seed}"]
        for field in event_fields:
            if hasattr(result, field):
                value = getattr(result, field)
                if field == "event_chain_valid":
                    value = "SIM" if value else "NÃO"
                lines.append(f"{field}: {value}")
            else:
                lines.append(f"{field}: NÃO DISPONÍVEL")
        ax.text(0.06, 0.9, "\n\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=12)
        ax.set_title("Integridade da trilha de eventos")
        fig.tight_layout()
        path = outdir / f"event_chain_seed{seed}.png"
        save_figure(fig, path)
        plt.close(fig)
        generated.append(path)
    else:
        skips.append("[SKIP] event_chain: event_count, event_sha256 e event_chain_valid ausentes.")

    # V5 — não há métricas consolidadas por drone no MissionRunResult.
    skips.append("[SKIP] per_drone: não há métricas individuais por agente expostas no resultado.")

    return generated, skips


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    seed = args.seed
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    print("=== SWARM ISR/SAR — DIAGNÓSTICO E VISUALIZAÇÃO ===")
    print(f"Commit de referência solicitado: 81b4a4390d71bf60d0be37293775f71fafaea034")
    print(f"Seed: {seed}")
    print(f"Saída: {outdir}")

    try:
        loop, result, initial = run_once(seed)
        values, found = inspect_result(result)
        report_exposure(loop, result)

        second_loop, second_result, _ = run_once(seed)
        deterministic = compare_runs(result, second_result)

        generated, skips = render_all(loop, result, initial, seed, outdir)

        # Confirma determinismo byte a byte das figuras geradas, renderizando
        # novamente os mesmos dados para um diretório temporário.
        png_hashes_match = True
        with tempfile.TemporaryDirectory(prefix="swarm_png_check_") as tmp:
            temp_paths, _ = render_all(loop, result, initial, seed, Path(tmp))
            by_name = {p.name: p for p in temp_paths}
            for path in generated:
                other = by_name.get(path.name)
                if other is None or file_hash(path) != file_hash(other):
                    png_hashes_match = False
                    print(f"[PNG HASH DIVERGENCE] {path.name}")
        print(f"\nDeterminismo dos PNGs: {'OK' if png_hashes_match else 'DIVERGÊNCIA'}")

        print("\n=== VEREDITO ===")
        print(f"Rodou? SIM")
        print(f"Concluiu? {'SIM' if result.completed else 'NÃO'} ({result.completion_reason})")
        print(f"Determinístico? {'SIM' if deterministic else 'NÃO'}")
        print(f"Hashes PNG idênticos na re-renderização? {'SIM' if png_hashes_match else 'NÃO'}")
        print(f"Métricas-chave encontradas: {len(found)}/{len(EXPECTED_METRICS)}")
        print(f"Figuras geradas: {len(generated)}/5 tentadas")
        for reason in skips:
            print(reason)
        print("PNGs gerados:")
        for path in generated:
            print(f"  {path.resolve()}")
        print(f"Resultado escalar: {values}")
        return 0 if len(found) == len(EXPECTED_METRICS) and not skips and deterministic and png_hashes_match else 1

    except Exception:
        print("\n[FATAL] MissionLoop/diagnóstico falhou. Traceback completo:")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
