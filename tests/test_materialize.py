"""Test that materialize.py can extract and write files correctly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize.py"


def _load_materialize():
    spec = importlib.util.spec_from_file_location("materialize", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["materialize"] = mod
    spec.loader.exec_module(mod)
    return mod


materialize_mod = _load_materialize()
extract_files = materialize_mod.extract_files
materialize = materialize_mod.materialize


SAMPLE = """\
### FILE: README.md
````markdown
# Hello World
This is a test.
````

### FILE: scripts/materialize.py
````python
print("self")
````
"""


def test_extract_files() -> None:
    files = extract_files(SAMPLE)
    assert len(files) == 2
    assert files[0][0] == "README.md"
    assert "# Hello World" in files[0][1]
    assert files[1][0] == "scripts/materialize.py"
    assert 'print("self")' in files[1][1]


def test_materialize_roundtrip(tmp_path: Path) -> None:
    n = materialize(SAMPLE, root=tmp_path)
    assert n == 2
    assert (tmp_path / "README.md").read_text().startswith("# Hello World")
    assert (tmp_path / "scripts" / "materialize.py").exists()


def test_ignores_file_header_inside_fence() -> None:
    nested = """\
### FILE: outer.py
````python
# ### FILE: should_be_ignored
print(1)
````
"""
    files = extract_files(nested)
    assert len(files) == 1
    assert files[0][0] == "outer.py"
    assert "### FILE: should_be_ignored" in files[0][1]
