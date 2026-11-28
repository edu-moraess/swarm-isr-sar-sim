"""Architecture boundary tests for immutable Dual-World state contracts.

The allowlist is intentionally temporary technical debt. Each entry is
reviewable and must be removed as its migration is completed in P2.1/P2.2.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from swarm_core.contracts import AgentLocalState, BeliefSnapshot, PhysicalTruthState


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "swarm_core"

LEGACY_ALLOWLIST = {
    "mission.py": "P2.1/P2.2: orchestrator still reads World and true Drone state.",
    "drone.py": "P2.1: physical integrator owns true position/velocity and World access.",
    "sensors.py": "P2.2: legacy sensor model reads World and true Drone position.",
    "energy.py": "P2.1: RTL energy estimate still reads true Drone position.",
    "planning/boustrophedon.py": "P2.2: legacy coverage planner reads World grid and geometry.",
    "planning/frontier.py": "P2.2: legacy frontier planner reads World geometry.",
    "planning/sectors.py": "P2.2: legacy sector planner reads World geometry.",
}


class BoundaryVisitor(ast.NodeVisitor):
    """Find direct imports of physical World and direct drone.position reads."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "swarm_core.world":
                self.violations.append(f"line {node.lineno}: import swarm_core.world")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "swarm_core.world":
            imported = {alias.name for alias in node.names}
            if imported.intersection({"World", "CellState", "Target", "Clutter"}):
                self.violations.append(f"line {node.lineno}: import from swarm_core.world")
        if node.module == "swarm_core.contracts":
            imported = {alias.name for alias in node.names}
            if "PhysicalTruthState" in imported:
                self.violations.append(f"line {node.lineno}: agent imports PhysicalTruthState")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "position" and isinstance(node.value, ast.Name):
            if node.value.id in {"drone", "agent"}:
                self.violations.append(f"line {node.lineno}: direct true position access")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "World":
            self.violations.append(f"line {node.lineno}: World type/reference")
        self.generic_visit(node)


def test_state_contracts_are_frozen_dataclasses() -> None:
    for contract in (PhysicalTruthState, AgentLocalState, BeliefSnapshot):
        assert is_dataclass(contract)
        assert contract.__dataclass_params__.frozen is True
        assert hasattr(contract, "__slots__")


def test_contract_public_vectors_and_collections_are_immutable() -> None:
    physical = PhysicalTruthState(
        entity_id=1,
        timestamp_s=1.0,
        position_xyz_m=(1.0, 2.0, 3.0),
        velocity_xyz_mps=(0.1, 0.2, 0.3),
        terrain_elevation_m=0.0,
        cell_states=((4, 0),),
        target_states=((7, 4, False),),
    )
    local = AgentLocalState(
        source_id=1,
        timestamp_s=1.0,
        estimated_position_xyz_m=(1.1, 2.1, 3.1),
        estimated_velocity_xyz_mps=(0.0, 0.0, 0.0),
        battery_pct=95.0,
    )
    belief = BeliefSnapshot(
        source_id=1,
        timestamp_s=1.0,
        probabilities=(0.1, 0.8),
        applied_watermarks=((1, 3),),
    )

    with pytest.raises(FrozenInstanceError):
        physical.entity_id = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        local.battery_pct = 50.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        belief.probabilities = (0.0,)  # type: ignore[misc]

    with pytest.raises(ValueError):
        PhysicalTruthState(1, 0.0, [0.0, 0.0, 0.0], (0.0, 0.0, 0.0), 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BeliefSnapshot(1, 0.0, [0.1, 0.2])  # type: ignore[arg-type]


def test_contracts_reject_invalid_numeric_state() -> None:
    with pytest.raises(ValueError):
        AgentLocalState(1, 0.0, (0.0, 0.0, float("nan")), (0.0, 0.0, 0.0), 50.0)
    with pytest.raises(ValueError):
        BeliefSnapshot(1, 0.0, (0.1, 1.2))


def test_legacy_allowlist_is_explicit_and_reviewable() -> None:
    assert set(LEGACY_ALLOWLIST) == {
        "mission.py",
        "drone.py",
        "sensors.py",
        "energy.py",
        "planning/boustrophedon.py",
        "planning/frontier.py",
        "planning/sectors.py",
    }
    for relative_path, reason in LEGACY_ALLOWLIST.items():
        assert (SOURCE_ROOT / relative_path).is_file()
        assert "P2." in reason


def test_no_new_module_couples_agent_code_to_physical_world() -> None:
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        if relative in LEGACY_ALLOWLIST or relative == "world.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        visitor = BoundaryVisitor()
        visitor.visit(tree)
        violations.extend(f"{relative}:{item}" for item in visitor.violations)

    assert not violations, (
        "New code must consume AgentLocalState/BeliefSnapshot, not PhysicalTruthState, "
        "World or true drone.position. Remove legacy allowlist entries only with migration.\n"
        + "\n".join(violations)
    )
