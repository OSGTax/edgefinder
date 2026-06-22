#!/usr/bin/env bash
# Idempotent environment bootstrap for the EdgeFinder agent.
#
# Fresh Claude-Code-on-the-web containers don't run the devcontainer's
# postCreateCommand, so the package (and pydantic, sqlalchemy, …) isn't
# installed and `python -m agent.*` fails at import. Point the Routine
# environment's setup script at this file — or rely on the SessionStart hook in
# .claude/settings.json — so every session starts ready. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

# Install the package if its deps aren't importable yet (fast no-op otherwise).
if ! python -c "import pydantic, sqlalchemy, pandas" >/dev/null 2>&1; then
  echo "[bootstrap] installing edgefinder (editable)…"
  pip install -e . -q
else
  echo "[bootstrap] dependencies already present"
fi

# Surface readiness (transport + DB reachability + data freshness). Never fail
# the bootstrap on a red preflight — the cycle itself decides what to do.
python -m agent.preflight || true
