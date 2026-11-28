"""Architecture boundary tests for immutable Dual-World state contracts.

The legacy allowlist exempts only documented violation categories, never whole
files. New access patterns in legacy modules remain visible to the AST scanner.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from swarm_core.contracts import AgentLocalState, BeliefSnapshot, PhysicalTruthState


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "swarm_core"

# Per-file/category exemptions: files are still scanned for every other pattern.
LEGACY_ALLOWLIST = {
    "mission.py": {
        "reason": "P2.1/P2.2: orchestrator still reads World and true Drone state.",
        "categories": {"physical_world_import", "physical_world_reference", "physical_world_access", "true_position_access"},
    },
    "drone.py": {
        "reason": "P2.1: physical integrator owns true position/velocity and World access.",
        "categories": {"physical_world_import", "physical_world_reference", "physical_world_access", "true_position_access"},
    },
    "sensors.py": {
        "reason": "P2.2: legacy sensor model reads World and true Drone position.",
        "categories": {"physical_world_import", "physical_world_reference", "physical_world_access", "true_position_access"},
    },
    "energy.py": {
        "reason": "P2.1: energy estimate still reads true Drone position.",
        "categories": {"physical_world_import", "physical_world_reference", "physical_world_access", "true_position_access"},
    },
    "planning/boustrophedon.py": {
        "reason": "P2.2: legacy coverage planner reads World grid and geometry.",
        "categories": {"physical_world_import", "physical_world_reference", "physical_world_access", "true_position_access"},
    },
    "planning/frontier.py": {
        "reason": "P2.2: legacy frontier planner reads World geometry.",
        "categories": {"physical_world_import", "physical_world_reference", "physical_world_access", "true_position_access"},
    },
    "planning/sectors.py": {
        "reason": "P2.2: legacy sector planner reads World geometry.",
        "categories": {"physical_world_import", "physical_world_reference", "physical_world_access", "true_position_access"},
    },
}


def _attribute_chain(node: ast.AST) -> set[str]:
    names: set[str] = set()
    current = node
    while isinstance(current, ast.Attribute):
        names.add(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        names.add(current.id)
    return names


class BoundaryVisitor(ast.NodeVisitor):
    """Detect physical World imports/access and true drone-position reads."""

    def __init__(self) -> None:
        self.violations: list[tuple[str, int, str]] = []
        self.agent_aliases = {"drone", "agent"}

    def _add(self, category: str, node: ast.AST, detail: str) -> None:
        self.violations.append((category, getattr(node, "lineno", 0), detail))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "swarm_core.world" or alias.name.startswith("swarm_core.world."):
                self._add("physical_world_import", node, f"import {alias.name} as {alias.asname or alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "swarm_core.world":
            self._add("physical_world_import", node, "import from swarm_core.world")
        if node.module == "swarm_core.contracts":
            for alias in node.names:
                if alias.name == "PhysicalTruthState":
                    self._add("physical_truth_contract_import", node, "agent imports PhysicalTruthState")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        value_names = _attribute_chain(node.value)
        if value_names.intersection(self.agent_aliases):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.agent_aliases.add(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and _attribute_chain(node.value).intersection(self.agent_aliases):
            if isinstance(node.target, ast.Name):
                self.agent_aliases.add(node.target.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _attribute_chain(node)
        if node.attr == "world":
            self._add("physical_world_access", node, "attribute access to .world")
        if node.attr == "position" and chain.intersection(self.agent_aliases):
            self._add("true_position_access", node, "agent/drone true position access")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "World":
            self._add("physical_world_reference", node, "World name/reference")
        self.generic_visit(node)


def _scan_source(path: Path, relative: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    visitor = BoundaryVisitor()
    visitor.visit(tree)
    allowed = LEGACY_ALLOWLIST.get(relative, {}).get("categories", set())
    return [
        f"{relative}:line {line}: {detail}"
        for category, line, detail in visitor.violations
        if category not in allowed
    ]


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


def test_contracts_reject_mutable_or_malformed_nested_payloads() -> None:
    valid_vector = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        PhysicalTruthState(1, 0.0, valid_vector, valid_vector, 0.0, cell_states=([1, 2],))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PhysicalTruthState(1, 0.0, valid_vector, valid_vector, 0.0, target_states=((1, 2, 1),))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BeliefSnapshot(1, 0.0, (0.1,), applied_watermarks=([1, 2],))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BeliefSnapshot(1, 0.0, (0.1, [0.2]))  # type: ignore[arg-type]


def test_contracts_reject_invalid_numeric_state() -> None:
    with pytest.raises(ValueError):
        AgentLocalState(1, 0.0, (0.0, 0.0, float("nan")), (0.0, 0.0, 0.0), 50.0)
    with pytest.raises(ValueError):
        BeliefSnapshot(1, 0.0, (0.1, 1.2))
    with pytest.raises(ValueError):
        PhysicalTruthState(True, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)  # type: ignore[arg-type]


def test_legacy_allowlist_is_explicit_and_category_scoped() -> None:
    expected = {
        "mission.py", "drone.py", "sensors.py", "energy.py",
        "planning/boustrophedon.py", "planning/frontier.py", "planning/sectors.py",
    }
    assert set(LEGACY_ALLOWLIST) == expected
    for relative_path, entry in LEGACY_ALLOWLIST.items():
        assert (SOURCE_ROOT / relative_path).is_file()
        assert "P2." in entry["reason"]
        assert entry["categories"]
        assert entry["categories"] <= {
            "physical_world_import", "physical_world_reference",
            "physical_world_access", "true_position_access",
        }


@pytest.mark.parametrize(
    "source, expected_category",
    [
        ("import swarm_core.world as physical_world", "physical_world_import"),
        ("from swarm_core.world import World as PhysicalWorld", "physical_world_import"),
        ("loop.world.grid", "physical_world_access"),
        ("self.drone.position", "true_position_access"),
        ("agent.drone.position", "true_position_access"),
        ("uav = drone\nuav.position", "true_position_access"),
        ("World()", "physical_world_reference"),
        ("from swarm_core.contracts import PhysicalTruthState as Truth", "physical_truth_contract_import"),
    ],
)
def test_ast_guard_detects_aliases_and_chained_access(source: str, expected_category: str) -> None:
    visitor = BoundaryVisitor()
    visitor.visit(ast.parse(source))
    assert expected_category in {category for category, _, _ in visitor.violations}


def test_no_unapproved_module_couples_agent_code_to_physical_world() -> None:
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        if relative == "world.py":
            continue
        violations.extend(_scan_source(path, relative))

    assert not violations, (
        "Agent-facing code must consume AgentLocalState/BeliefSnapshot, not PhysicalTruthState, "
        "World or true drone.position. Legacy exemptions are category-scoped.\n"
        + "\n".join(violations)
    )
