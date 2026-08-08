# EdgeFinder Routines — setup & troubleshooting (REBUILD-V4)

EdgeFinder runs entirely on **Claude Code Routines** (claude.ai/code/
routines). GitHub Actions and the Render wake-dispatcher are GONE: the
trading brain schedules itself. Each Routine is a scheduled Claude session
on this repo that runs one skill. **Crons in the Routine UI are UTC** (ET
in parens); all fire fresh sessions with completion notifications
(push + email) ON — the notification IS the owner's per-run report.

| Routine | Skill / prompt | Cron (UTC) | What it does |
|---|---|---|---|
| Chain wakes | "Run the trading-agent skill." | **one-shot triggers, self-created** — every market-hours cycle arms its own next wake 15–60 min out (`create_trigger`, `run_once_at`, fresh session), gated by the `desk_wakes` budget (40/ET-day, 15-min floor) | The rolling chain: prep ~9:00 ET → session → wrap post-close |
| Chain restarter (floor) | "Run the trading-agent skill." | `0 13-20 * * 1-5` (hourly, 9a–4p EDT) | The FLOOR: `agent.brain chain-health` makes it a cheap early exit while the chain is healthy; it runs a full cycle only when a wake is due or the chain went quiet (no cycle in 25 min during desk hours) |
| Nightly data | `data-refresh` | `45 0 * * 2-6` (8:45 PM ET Mon–Fri) | Full-market ingest + EDGAR + brief **+ V4 duties: Alpaca mirror sync, portfolio snapshot, split guard, R2 knowledge backup, DB size check** |
| Strategy Lab | `strategy-lab` | `0 2 * * 2-6` (10 PM ET) | 21y split-sample sweep → leaderboard → brief |
| Weekly reflection | `reflection-agent` | `30 22 * * 5` (6:30 PM ET Fri) | Grade the week (via `agent.grade`), curate the wiki, lint the claims registry |
| Desk evolution | `app-evolver` | `0 15 * * 6` (11 AM ET Sat) | One small, tested, announced `/desk` improvement |

Retired with V4: the GitHub Actions `trading-agent.yml` workflow +
`wake_gate.py` gate, the Render wake-dispatcher + tripwire sweep +
hard-stop executor (protective stops now REST ON ALPACA'S BOOK as real
GTC orders — they fire with nothing of ours running), the SMTP
`cycle_report.py` email (Routine completion notifications replace it),
and the hourly loop-monitor Routine (same reason).

**The autonomy loop (V4):** every market-hours trading cycle ends with
`agent.brain wake-plan` (the budget gate — 40/ET-day, ≥15-min gap,
DB-enforced in `desk_wakes`) and then arms the one-shot trigger itself
via the claude-code-remote MCP (`create_trigger` with `run_once_at`,
`create_new_session_on_fire=true`, prompt "Run the trading-agent
skill."). The hourly floor Routine restarts a dropped chain; its
`chain-health` early exit makes healthy-hour firings nearly free.
Expected volume: ~15–25 cycles/trading day, all billed to the owner's
Claude subscription — no runner minutes, no dispatcher PAT.

Routine prompts are thin pointers ("Run the X skill.") — behavior lives
in `.claude/skills/*/SKILL.md`, which every firing loads fresh from
`main`, so skill updates need **no Routine changes**.

## The one setup every Routine needs

Point every Routine environment's **setup script** at:

```
bash scripts/bootstrap.sh
```

It installs the package with dev extras (pytest is REQUIRED by
app-evolver's test gate), retries transient network failures, and runs
`agent.preflight`.

### Required secrets (set on each Routine's environment)
- **Database (Supabase, REST transport over 443):** `SUPABASE_URL` +
  `SUPABASE_SERVICE_ROLE_KEY` (or `DATABASE_URL` where TCP is allowed).
- **Market DATA (Alpaca, ATP subscription):** `EDGEFINDER_ALPACA_API_KEY`
  + `EDGEFINDER_ALPACA_API_SECRET`.
- **The PAPER ACCOUNT (the book of record — trading Routines only):**
  `EDGEFINDER_ALPACA_TRADE_KEY` + `EDGEFINDER_ALPACA_TRADE_SECRET`.
  Paper keys cannot authenticate against the live API — live trading is
  unreachable by construction.
- **Deep archive + backups (Cloudflare R2):** `R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`, `R2_BUCKET`.

`agent.preflight` prints `ok:true` only when the DB is reachable and bars
are fresh; its `paper_account` check reports the trade-key health.

## Troubleshooting: a Routine fails at "environment setup"

1. **Setup script points at bootstrap** — exactly
   `bash scripts/bootstrap.sh`.
2. **Secrets are present** — run `python -m agent.preflight` in a
   session; `ok:false` with a DB error means the Supabase pair is
   missing. Missing Alpaca/R2 secrets degrade but don't block startup.
3. **Transient network** — bootstrap retries 3×; re-run the Routine.
4. **Per-Routine environments** — copy the working environment's config
   when one Routine works and another doesn't.

App-evolver additionally needs **git push** to origin/main; the
reflection needs only DB access.

## Protection between cycles

There is no in-house watcher anymore, on purpose: a protective exit is a
REAL GTC stop order resting on Alpaca's book (`agent.trade arm-stop`),
which executes whether or not any EdgeFinder process is up. Two caveats
the skill already knows: Alpaca auto-cancels GTC orders at 90 days
(`trade reconcile` warns from day 80), and stops exist for equities only
— options and crypto exits are managed at cycle cadence, eyes open.
