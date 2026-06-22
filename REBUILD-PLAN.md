# REBUILD PLAN — the autonomous AI trading agent (greenfield)

> **READ THIS FIRST on resume.** This supersedes the old EdgeFinder v2 direction.
> The project is being rebuilt greenfield around ONE thing: a self-directed AI
> trading agent that runs on **Claude Code Routines** (not GitHub Actions),
> builds and evolves its own strategy, trades a paper book with full discretion,
> and explains itself on a rich web "trading desk" page. Conversation history is
> gone after the context reset — this file is the source of truth.

## ⚡ EXECUTION AUTHORIZATION — one-shot; do NOT ask again
The owner **pre-authorized this entire build, including the destructive cutover,
in advance (2026-06-22).** When the owner says "go" (any kickoff such as "read
REBUILD-PLAN.md and go" / "start the build"), execute the FULL sequence below
end-to-end, autonomously — scaffold, build, verify, AND the cutover that drops
the old trading tables, deletes the old code, and deploys the new agent.
**Do NOT stop to re-confirm the deletion or ask "are you sure" — that decision is
final and was made in chat.** Report progress as you go; never pause for
permission. Drive it all the way to a deployed, running agent.

This means "don't ask," NOT "be reckless." Three NON-NEGOTIABLE engineering rails
remain — they protect the owner's data and guarantee a working result, so they
are correctness invariants, not permission gates:
1. **NEVER drop/clear the market-data tables or the R2 archive.** The wipe is
   ONLY the trading/accounts/old-app tables + the old code. The data is sacred
   and irreplaceable.
2. **Prove the new system works BEFORE cutover** — R2 reads confirmed, the agent
   loop dry-runs cleanly, the page renders — THEN wipe the old. Never delete the
   working old path before the new one is proven.
3. **Keep everything git-recoverable** — work on the branch, old code stays in
   history, no force-push / history rewrite.

## The pivot (why everything changes)
The owner concluded the prior build (a heavyweight strategy research platform —
hunt/validation, a 12-strategy validated fleet, a fixed-rule "analyst", a
real-money Robinhood path) was the wrong product. The real goal: **a fun,
self-explaining, autonomous AI stock picker + paper trader** that the owner
watches on a webpage, updated several times a day. We keep the proven *data
asset* and rebuild the rest from scratch.

## Hard decisions (locked with the owner)
- **Greenfield code.** Remove the entire old application/agent/research codebase.
- **Keep the market data** — it was audited and verified sound (see below). Keep
  the Supabase tables + the R2 archive + a thin, clean data-access/backtest layer.
- **Run on Claude Code Routines** (claude.ai/code/routines), NOT GitHub Actions
  (the owner couldn't get Actions working; Routines avoid them entirely).
  Target **~8 runs/day** (cron, 1-hour min interval; e.g. every 2h through the
  trading day), more on demand. The Routine session **is** the agent — Claude
  runs the skill each time with full tool access (Bash, MCP, web). No `claude -p`
  subprocess, no CLAUDE_CODE_OAUTH_TOKEN-in-Actions problem.
- **Full agent discretion.** No limits on number of stocks, position sizing, or
  per-stock decisions — every trade decision is the agent's. (Long-only, paper
  only; keep a fill-sanity guard so a bad quote can't book a wrong basis.)
- **The agent builds & evolves its own strategy** — it maintains a strategy-state
  + journal, backtests its own ideas to ground them, and pivots/tweaks when
  something stalls. Dynamic, not a fixed rule set.
- **Reuse the Supabase DB + Render app.** WIPE the old paper accounts + trading
  tables; KEEP the data tables. Re-point Render at the new code.
- **Paper only.** No real-money/broker execution in this build.

## Data audit results (2026-06-22, verified — KEEP the data)
Read-only audit via the Supabase MCP. The data is real (prices match the real
market to the penny — confirmed vs live quotes for MU/SOXL), clean, and valuable.
- `daily_bars`: 307,765 rows, 3,262 symbols, **2005-01-03 → 2026-06-18**; current
  top-1000/day hot set (1,006 on last day). **0 bad prices, 0 broken OHLC.**
- `dividends` 167k rows / 3,443 syms · `ticker_splits` 9,891 / 6,848 ·
  `fundamentals_snapshots` (PIT) 128,854 / 7,702 · `ticker_news` 15,236 / 2,004 ·
  `index_daily` 1,196 / 4 (SPY/QQQ/IWM/DIA).
- **Structure:** `daily_bars` stores RAW bars; split adjustment is applied at load;
  dividends/splits tables enable total-return. The DB is the *operational hot set*
  (most of the 3,262 symbols have <200 bars here by design — they rotate through
  the top-1000); the **deep 21-year per-symbol history lives in the R2 archive.**
- **MUST VERIFY AT BUILD TIME:** the R2 archive (needs the R2_* credentials, which
  are NOT in the sandbox — present on Render / as Actions secrets). Confirm R2
  reads work and the loader's split/dividend adjustment once more when wiring the
  new backtest tool.

## Target architecture
**KEEP:** Supabase DB + the data tables above; the R2 parquet archive; the Render
web app (rebuild its pages); a thin clean data-access + backtest layer (reading
the same tables/R2).

**REMOVE:** the hunt/validate/walkforward/promote/live engine; all strategies;
the v1 analyst (engine/analyst_strategy, agents/analyst, the Picks page, the
agent_decisions table); real-money (engine/live_ticket, REAL-MONEY-RUNBOOK);
watchdog/alerts; the old dashboard pages/routers; the old trading ORM tables
(promoted_strategies, trades, strategy_accounts, strategy_snapshots,
validation_runs, agent_decisions, dividend_credits, llm_decision_*); the
scheduler + all GitHub Actions workflows; the old scheduled jobs in services.py.

**BUILD:**
1. **Agent charter** — a skill in `.claude/skills/` (the prompt + decision
   contract + guardrails) that the Routine runs each cycle.
2. **Tool layer** (clean Python the agent calls via Bash): read market data
   (DB hot set + R2 deep history, split/div-adjusted); compute signals/indicators;
   `run_backtest(...)` over arbitrary symbols/windows; read news; read/write the
   agent's **strategy-state + journal + thinking-log**; a **ledger** (record
   trades, recompute the paper portfolio + equity).
3. **Clean schema** (new tables in the same DB): portfolio/positions, trades,
   strategy_state, agent_journal (pivots + why), thinking_log (per-run narration),
   decisions, backtests_run.
4. **Trading-desk page** on the Render app showing all four panels the owner
   chose: (a) chart-forward picks/holdings with why-now + rationale + news,
   (b) portfolio + equity curve + market-regime header, (c) live "thinking" feed
   (evolving thesis + pivots through the day + near-miss watchlist),
   (d) backtest evidence (the backtests it ran + on-demand). Reuse the vendored
   `lightweight-charts` + CSS design tokens + `dashboard/static/js/core/*`
   (charts.js, dom.js, fmt.js, net.js) — those are good; rebuild the pages.
5. **The Routine** — create at claude.ai/code/routines: this repo, cron ~every 2h
   during market hours (8×/day), running the agent skill, with the Supabase +
   Polygon + R2 config available to the session.

## Build sequence (branch-first; prod untouched until cutover)
Dev branch: `claude/handoff-doc-review-176vbl` (merges to main deploy to Render).
1. Scaffold the new structure on the branch; clean data-access + backtest tool
   layer; **verify R2 reads** end-to-end. New schema (additive; don't drop yet).
2. Agent charter/skill + the tool layer it calls; dry-run the agent loop locally
   (no trades) and confirm decisions + journaling + backtests work.
3. Trading-desk page (reads the new tables).
4. Wire + test the Routine (one-off manual run first).
5. **CUTOVER (PRE-AUTHORIZED — execute without asking; obey the 3 rails above):**
   stop old scheduled jobs, drop/clear old trading tables + accounts (KEEP data
   tables), delete old code, deploy new code to Render, enable the Routine.

## Environment / infra facts
- Repo: `OSGTax/edgefinder`. Render app (real URL): `https://edgefinder-pm8h.onrender.com`
  — **never probe `edgefinder.onrender.com`** (someone else's). Render build
  `pip install -e .`, start `python scripts/render_start.py` (it applies schema
  via idempotent DDL, NOT alembic — keep that pattern for new tables).
- Supabase Postgres via `DATABASE_URL` (pooler). Sandbox blocks direct Postgres
  TCP → use the **supabase MCP** (`execute_sql`) for DB work, or run in a Routine.
- R2 archive: Cloudflare R2 via boto3, env `R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY
  / R2_ENDPOINT / R2_BUCKET` (on Render + Actions; not in sandbox).
- Polygon: `EDGEFINDER_POLYGON_API_KEY`.
- Claude Routines: claude.ai/code/routines; cron min 1h; full tool access;
  reuses the environment's MCP/env/network; daily run cap drawn from the
  subscription (one-off runs don't count). Docs: code.claude.com/docs/en/routines
- Tests gate: `DATABASE_URL= python -m pytest tests/ -q -m "not integration"
  --ignore=tests/test_market.py` (will shrink as the old suite is removed).

## Working constraints (carry forward)
- Develop on `claude/handoff-doc-review-176vbl`; never push elsewhere without
  explicit permission. Do NOT open a PR unless asked. Commit + push when changes
  are complete; version-bump `dashboard/app.py __version__` on functional merges.
- Never put the model identifier in commits/PRs/code (chat only).
- Bump `dashboard/app.py __version__` on functional merges to main.
- The destructive cutover is PRE-AUTHORIZED (see EXECUTION AUTHORIZATION at top):
  one-shot it on "go" — do not ask again; just obey the 3 engineering rails.

## STATUS at handoff
Plan approved; data verified & kept; **nothing removed or wiped yet.** Old system
is still live on Render (the `ai_analyst` v1 + 13 other paper strategies are
running; harmless to leave until cutover). On "go": run the FULL sequence
(steps 1→5) autonomously without pausing for permission — start at Build step 1
(scaffold + clean data/backtest tool layer + R2 verification) and drive straight
through the pre-authorized cutover to a deployed, running agent.
