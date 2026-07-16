# EdgeFinder Routines — setup & troubleshooting

EdgeFinder runs entirely on **Claude Code Routines** (claude.ai/code/routines),
not GitHub Actions or an in-process scheduler. Each Routine is a scheduled
Claude session on this repo that runs one skill (the loop monitor is
prompt-only). There are six —
**crons in the Routine UI are UTC** (ET shown in parens):

| Routine | Skill | Cron (UTC) | What it does |
|---|---|---|---|
| Trading brain | `trading-agent` | **GitHub Actions, not this table** (since v9.11.0; all-day chain since v9.12.0) — the streamer dispatches every due wake/trip; the workflow's own `0,30 12-21 * * 1-5` half-hour floor restarts a dropped chain. The claude.ai Routine (daily `0 13 * * *`) is a transition fallback only | At the desk all day: prep cycle ~9:00 ET, rolling 15–60-min wake chain with a rotating study block, wrap cycle post-close |
| Nightly data | `data-refresh` | `45 0 * * 2-6` (8:45 PM ET Mon–Fri) | Full-market bar ingest (top-1000 + R2 sync) **+ SEC EDGAR fundamentals ingest** + builds the research brief |
| Strategy Lab | `strategy-lab` | `0 2 * * 2-6` (10:00 PM ET, post-ingest) | Sweeps the 168-combo grid (incl. `value_momentum` since v9.4.0) over 21y, split-sample scored; rebuilds the brief with tonight's board |
| Weekly reflection | `reflection-agent` | `30 22 * * 5` (6:30 PM ET Fri) | Grade the week mechanically (predictions, alpha, rejected list, fundamentals citations), curate the lessons wiki |
| Desk evolution | `app-evolver` | `0 15 * * 6` (11:00 AM ET Sat) | One small, tested, additive `/desk` improvement, announced on What's New |
| Loop monitor | (prompt-only, read-only) | `0 13-21 * * 1-5` (hourly, 9 AM–5 PM ET) | Fresh-session digest of the autonomy loop — reads `desk_dispatches`, wakes, journal FAILED notes, latest decision/fills, ledger state; **push + email notification** to the owner each hour of the trading day. Never trades, never writes. Created 2026-07-16 (`trig_01H4n1qmGrWxPp5KTc5BhYCn`) |

**The autonomy loop (v9.11.0, all-day chain v9.12.0):** the trading brain
no longer depends on a human finger — it is at the desk all day. Every
market-hours cycle ends by planning the next wake 15–60 minutes out (a
rolling chain from a ~9:00 AM ET prep cycle to a post-close wrap, with a
rotating study block each cycle); the always-on Render streamer watches
`desk_wakes` and tripped tripwires and fires the repo's
`trading-agent.yml` GitHub Actions workflow (`workflow_dispatch`, PAT with
Actions:write only) within ~1 minute of either. The workflow's half-hour
cron floor is a CHAIN-RESTARTER: its gate (`scripts/wake_gate.py`) runs a
scheduled slot only when a wake is due OR it's desk hours (9:00–4:30 ET)
with no cycle in the last 25 minutes — a healthy chain makes floor slots
free. Cycles authenticate with the owner's Max subscription
(`CLAUDE_CODE_OAUTH_TOKEN`, from `claude setup-token`). Guards: >=5-min
dispatch gap + 45/ET-day cap (DB-enforced, `desk_dispatches`), 3 attempts
per wake then terminal, wake budget 30/ET-day. Expected volume: ~15–25
cycles/trading day. Failures journal loudly to the desk. The claude.ai
trading Routine remains as a fallback during transition and should be
retired after a week of proven autonomous fires.

Routine prompts are thin pointers ("Run the X skill.") — behavior lives in
`.claude/skills/*/SKILL.md`, which every firing loads fresh from `main`, so
skill updates need **no Routine changes**. Fundamentals context:
`docs/fundamentals-sources.md` (why EDGAR) and
`docs/fundamentals-validation.md` (the accuracy gate the data passed).
EDGAR needs **no new secrets** — its authentication is the declared
User-Agent (`settings.edgar_user_agent`).

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

## Tripwires between runs: alerts vs hard stops

Because the trading brain is agent-paced, a **tripped alert wire**
(`above`/`below` in `desk_watch`) does nothing on its own — the streamer
only flips its status, and the trip is handled at the next owner-fired run.
The one exception is the opt-in **`hard_stop`** kind: it is the only wire
that acts by itself. When its level trips, the always-on streamer sells the
whole position through the ledger's normal fill gates (one attempt; a gated
rejection is recorded as `exec_failed` for the next run to handle). Hard
stops protect long **equity** positions only — the sweep watches the equity
SIP tape, crypto quotes never enter it, so `watch-set --hard` refuses to
arm on a crypto pair (protection that cannot trip must not arm). The
brain arms one per position explicitly via `agent.brain watch-set --hard`;
nothing else can ever trade between runs.
