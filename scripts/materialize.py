#!/usr/bin/env python3
"""Materialize the full repository tree from a continuous markdown dump.

Usage:
    python scripts/materialize.py < dump.md
    # or
    python scripts/materialize.py path/to/dump.md

The dump must contain blocks of the form:

### FILE: relative/path
````lang
content
````

Outer fences use four backticks so that nested triple-backtick code blocks
inside the content are preserved. Lines matching `### FILE:` that appear
*inside* a fenced block are ignored.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Match ### FILE: only when it appears at the beginning of a line *outside* fences.
FILE_HEADER_RE = re.compile(r"^### FILE:\s*(.+?)\s*$", re.MULTILINE)

# Four-backtick fence (outer) — language tag optional
OUTER_FENCE_RE = re.compile(
    r"^````(?:\w+)?\n(.*?)^````\s*$",
    re.DOTALL | re.MULTILINE,
)


def extract_files(text: str) -> list[tuple[str, str]]:
    """Return list of (relative_path, content).

    Ignores any `### FILE:` line that sits inside an outer four-backtick fence.
    """
    files: list[tuple[str, str]] = []
    pos = 0
    while True:
        m = FILE_HEADER_RE.search(text, pos)
        if not m:
            break
        path = m.group(1).strip()
        # Look for the next outer fence after the header
        fence = OUTER_FENCE_RE.search(text, m.end())
        if not fence:
            print(f"[WARN] No outer ```` fence found for {path}", file=sys.stderr)
            pos = m.end()
            continue
        content = fence.group(1)
        if not content.endswith("\n"):
            content += "\n"
        files.append((path, content))
        pos = fence.end()
    return files


def materialize(text: str, root: Path | None = None) -> int:
    """Write extracted files under root. Returns number of files written."""
    root = root or Path.cwd()
    files = extract_files(text)
    if not files:
        print("No ### FILE: blocks found.", file=sys.stderr)
        return 0
    created = 0
    for rel, content in files:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"  wrote {rel} ({len(content)} bytes)")
        created += 1
    print(f"\nMaterialized {created} files under {root}")
    return created


def main() -> int:
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    n = materialize(text)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
