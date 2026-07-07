# REBUILD V3 — the live desk (source of truth; supersedes all prior plans)

> **Read this first on resume.** All earlier plans (the MOC daily-bar model,
> the broker-of-record LIVE-REBUILD plans v1/v2) are retired and deleted.
> This is the whole design, and it is deliberately small.

## What this is

An **autonomous AI paper-trading desk**, private to the owner:

- **Our own paper book** ($100k, in `desk_*` tables) — not a mirror of
  Alpaca's account. `desk_trades` is append-only and authoritative for cash.
- **Priced by live Alpaca SIP data** (Algo Trader Plus): a Render-hosted
  WebSocket streams real-time quotes to the `/desk` page and to the tools.
- **An hourly AI brain** (Claude Code Routine running
  `.claude/skills/trading-agent/SKILL.md`) that observes, researches,
  decides, and books fills into the ledger **at the live quote**.
- **Self-evolving**: the agent owns its strategy (versioned in
  `desk_strategy_state`, every pivot journaled) and a separate end-of-day
  routine (`app-evolver`) ships one small announced UI improvement per day.

## The honesty contract (why this is trustworthy)

1. **Every fill prices off the live quote at the decision moment** — buys at
   the live ask, sells at the live bid (±1 bp slippage), with the quote
   snapshot `{bid, ask, ts}` stamped on the fill row. Never a daily close,
   never a past price, never an invented price.
2. **Verifiable in real time**: the owner can cross-check any quote on
   `/desk` and any fill price against Alpaca's own dashboard or any ticker.
3. **Paper only. Long only. No leverage.** The ledger enforces no-overdraw
   and fill-sanity; a rejection is final.
4. **Alpaca is DATA ONLY** — `agent/broker.py` has no write methods, by
   design. Orders are never submitted to Alpaca; the book is ours.

## Runtime layout

| Piece | Where | Job |
|---|---|---|
| Quote streamer | Render (always-on) | One Alpaca SIP WebSocket → in-memory cache → SSE to `/desk` (`/api/desk/stream`) |
| Desk page | Render | Live ticks, book, thinking feed, decisions, journal, What's New |
| Trading brain | Claude Code Routine, hourly in market hours | Runs the trading-agent skill; fills via `agent.ledger fill` |
| App evolver | Claude Code Routine, nightly | One small announced UI improvement (`app-evolver` skill) |
| Book + state | Supabase (`desk_*`) | Ledger (source of truth), strategy, journal, thinking, changelog |
| Deep history | Cloudflare R2 (21y parquet, frozen 2026-06-18) + `daily_bars` | Backtests/grounding via `agent.backtest_tool`; topped up from Alpaca daily bars for the active universe |

**Sacred, never drop/clear:** the market-data tables (`daily_bars`,
`dividends`, `ticker_splits`, `fundamentals_snapshots`, `ticker_news`,
`index_daily`, …) and the R2 archive.

## Data sources (current, all paid-for)

- **Alpaca Algo Trader Plus ($99/mo)** — live SIP quotes, market clock,
  historical daily bars (~2016+), Benzinga news, OPRA options (future phase).
- **R2 archive** — 21 years of daily bars for backtests (static asset).
- **Polygon/Massive — DISABLED.** No live dependency remains on it.

## Environment

`EDGEFINDER_ALPACA_API_KEY / _API_SECRET / _DATA_FEED=sip / _PAPER=true`
(live in Render; add to the Claude environment + Routine when created), plus
the existing `SUPABASE_*`, `R2_*`, `DATABASE_URL` / `EDGEFINDER_DB_TRANSPORT`.

## Build phases (verifier-gated: a fresh subagent must pass each phase)

- **P0 — cleanup** (this commit): old plans deleted, MOC code removed,
  broker demoted to data-reader, this charter written.
- **P1 — live pipe**: `agent/streamer.py` (SIP WS → cache, reconnect/backoff,
  REST warm) + `/api/desk/stream` SSE + EventSource ticks + staleness banner
  on `/desk`. Proof: live ticks on prod, cross-checked.
- **P2 — live fills**: fractional shares, `fill_quote` snapshot column,
  `agent.ledger fill` (prices the correct side from the live quote, rejects
  when closed/stale), mark-at-live-quotes, `daily_bars` top-up from Alpaca.
  Proof: a scratch fill books at the live quote and reads back exactly.
- **P3 — the brain**: trading-agent skill rewritten for the live model;
  strategy state bumped + pivot journaled; Routine config template ready.
  Proof: every command the skill uses works end-to-end.
- **P4 — handoff**: What's New announcement; owner gets `/desk` proof links
  + the one-click Routine config. The Routine is created LAST, by the owner.

## Rules for future sessions

- Never force-push `main`; never skip the test gate
  (`DATABASE_URL= python -m pytest tests/ -q -m "not integration"`).
- Version-bump `dashboard/app.py` on every functional merge.
- The trading skill may not touch UI files; the app-evolver may not touch
  the ledger, fill model, or any sacred table.
