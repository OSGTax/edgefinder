#!/usr/bin/env bash
# Idempotent environment bootstrap for the EdgeFinder agent AND its Routines
# (trading-agent, data-refresh, app-evolver, reflection-agent).
#
# Fresh Claude-Code-on-the-web containers don't run the devcontainer's
# postCreateCommand, so the package — and pytest, which app-evolver's test gate
# REQUIRES — isn't installed, and the Routine fails at import or at the gate.
# Point EACH Routine environment's setup script at this file (setup command:
# `bash scripts/bootstrap.sh`), or rely on the SessionStart hook in
# .claude/settings.json. Safe to re-run.
#
# HARDENED (routine startup fix): this script must NEVER fail a session or
# environment start — a degraded start beats a dead one, and the run itself
# decides what to do about missing pieces. No `set -e`; every step tolerant;
# retries around pip (the managed agent proxy occasionally resets a TLS
# handshake); always exit 0.
set -u
cd "$(dirname "$0")/.." || { echo "[bootstrap] WARN: could not cd to repo root"; exit 0; }

# Install the package + DEV extras (pytest/ruff — app-evolver's gate needs them)
# whenever anything is missing. Fast no-op when already present. `.[dev]` is
# installed for ALL Routines so the one setup path is uniform; the extras are
# small and only fetched once per container.
if python -c "import pydantic, sqlalchemy, pandas, pytest" >/dev/null 2>&1; then
  echo "[bootstrap] dependencies already present"
else
  echo "[bootstrap] installing edgefinder + dev extras (editable)…"
  installed=""
  for attempt in 1 2 3; do
    if pip install -e ".[dev]" -q; then installed="yes"; break; fi
    echo "[bootstrap] pip attempt ${attempt} failed (transient network?), retrying…" >&2
    sleep $((attempt * 3))
  done
  if [ -z "${installed}" ]; then
    echo "[bootstrap] WARN: [dev] install failed after 3 tries — retrying plain" >&2
    pip install -e . -q || echo "[bootstrap] WARN: pip install failed; imports may be missing" >&2
  fi
fi

# alpaca-py powers live quotes/fills; it lives in the [live] extra, not [dev],
# so install it standalone if still absent.
python -c "import alpaca" >/dev/null 2>&1 || \
  pip install alpaca-py -q 2>/dev/null || \
  echo "[bootstrap] WARN: alpaca-py unavailable — live-quote tools will degrade" >&2

# Surface readiness (transport + DB reachability + data freshness). Never fail
# the bootstrap on a red preflight — a Routine that can't reach the DB should
# still start and report the problem itself, not die in setup with an opaque
# "environment setup failed". If this prints ok:false, the environment is
# missing a secret (SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY for the DB,
# EDGEFINDER_ALPACA_* for quotes, R2_* for the archive).
python -m agent.preflight || true

echo "[bootstrap] done"
exit 0
