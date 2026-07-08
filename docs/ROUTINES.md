# EdgeFinder Routines — setup & troubleshooting

EdgeFinder runs entirely on **Claude Code Routines** (claude.ai/code/routines),
not GitHub Actions or an in-process scheduler. Each Routine is a scheduled
Claude session on this repo that runs one skill. There are four:

| Routine | Skill | Cron (local) | What it does |
|---|---|---|---|
| Trading cycle | `trading-agent` | `~7 */2 * * 1-5` (every ~2h, market hours) | Observe + trade the $100k paper book at live quotes |
| Nightly data | `data-refresh` | `~7 6 * * *` (after the close) | Full-market ingest — keeps the top-N fresh set current |
| Desk evolution | `app-evolver` | end-of-day | One small, tested, additive `/desk` improvement |
| Weekly reflection | `reflection-agent` | `~5 17 * * 5` (Fri after close) | Grade the week, curate the lessons wiki |

`data-refresh` + `app-evolver` can share one end-of-day Routine (refresh data
first, then evolve the desk).

## The one setup every Routine needs

A Routine session must (1) install the package and (2) reach the DB + market
data. Both are handled by **`scripts/bootstrap.sh`** — point every Routine
environment's **setup script** at it:

```
bash scripts/bootstrap.sh
```

It installs the package **with dev extras** (`pip install -e ".[dev]"` — pytest
is REQUIRED by app-evolver's test gate), retries on a transient network failure,
and runs `agent.preflight`. It's also wired as a `SessionStart` hook in
`.claude/settings.json`, so it runs automatically when `CLAUDE_CODE_REMOTE=true`
— but the Routine environment's own setup script is the reliable place for it.

### Required secrets (set on each Routine's environment)
- **Database (Supabase, REST transport over 443):** `SUPABASE_URL` +
  `SUPABASE_SERVICE_ROLE_KEY`. (Or `DATABASE_URL` for the direct-Postgres
  transport where TCP is allowed.)
- **Live quotes / market data (Alpaca):** `EDGEFINDER_ALPACA_API_KEY` +
  `EDGEFINDER_ALPACA_API_SECRET`.
- **Deep archive (Cloudflare R2):** `R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`, `R2_BUCKET`.

`agent.preflight` prints `ok:true` only when the DB is reachable and bars are
fresh — a healthy setup ends with that.

## Troubleshooting: a Routine fails at "environment setup"

The platform runs the environment's setup script before the skill; if it exits
non-zero you get an opaque "environment setup failed / edit your setup" error.
Checklist, in order:

1. **Setup script points at bootstrap.** The environment's setup command should
   be exactly `bash scripts/bootstrap.sh`. A bare `pip install -e .` (no
   `[dev]`) leaves pytest missing and **app-evolver dies at its test gate** —
   this was the original failure.
2. **Secrets are present.** Run `python -m agent.preflight` in the session. If
   it prints `ok:false` with a DB error, the environment is missing
   `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY`. Missing Alpaca or R2 secrets
   degrade quotes/archive but don't block startup.
3. **Transient network.** The managed proxy occasionally resets a TLS
   handshake, which can fail a single `pip install`. `bootstrap.sh` already
   retries 3× — if setup still fails, just re-run the Routine; it's idempotent.
4. **Per-Routine environments.** Each Routine can point at its own environment.
   If the trading Routine works but app-evolver / reflection don't, their
   environments were created without the setup script or secrets above — copy
   the working environment's config.

App-evolver additionally needs **git push** to origin/main (it ships desk
changes) and reflection-agent needs only DB access (it writes the wiki via
`agent.brain`, no file commits).
