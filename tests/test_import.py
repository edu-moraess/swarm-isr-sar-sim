"""Smoke test: package is importable and exposes version."""

from swarm_core import __version__


def test_version_is_semver() -> None:
    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])


def test_import_swarm_core() -> None:
    import swarm_core

    assert hasattr(swarm_core, "__version__")
