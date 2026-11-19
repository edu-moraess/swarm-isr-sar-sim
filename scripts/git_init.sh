#!/usr/bin/env bash
# Initialize git history with Conventional Commits per phase and optional gh repo create.
# Usage: bash scripts/git_init.sh
set -euo pipefail

REPO_NAME="swarm-isr-sar-sim"
OWNER="edu-moraess"

if [ -d .git ]; then
  echo "Git already initialized."
else
  git init
  git branch -M main
fi

# Phase 0 — scaffolding
git add LICENSE pyproject.toml Makefile .pre-commit-config.yaml .gitignore \
        .github/workflows/ci.yml .github/PULL_REQUEST_TEMPLATE.md \
        .devcontainer/devcontainer.json \
        docs/DECISIONS.md docs/UNVERIFIED.md docs/ETHICS.md \
        scripts/materialize.py scripts/git_init.sh \
        src/swarm_core/__init__.py tests/test_import.py tests/test_materialize.py \
        README.md 2>/dev/null || true

git commit -m "chore: initial scaffolding (Lote 0)

- LICENSE (MIT)
- pyproject.toml with core + dev extras (ruff as sole formatter)
- Makefile, pre-commit, GitHub Actions CI
- .devcontainer for Codespaces
- .gitignore + PR template
- docs/DECISIONS.md, UNVERIFIED.md, ETHICS.md
- scripts/materialize.py (4-backtick fences) and git_init.sh
- src/swarm_core/__init__.py + basic import test
" || echo "Nothing to commit for phase 0"

# Tag
git tag -a v0.1.0 -m "v0.1.0 — scaffolding complete" 2>/dev/null || true

echo ""
echo "Local repository ready."
echo "To create / update remote (requires gh auth):"
echo "  gh repo create ${OWNER}/${REPO_NAME} --public --source=. --remote=origin --push"
echo "  # or, if already created:"
echo "  git remote add origin git@github.com:${OWNER}/${REPO_NAME}.git"
echo "  git push -u origin main --tags"
