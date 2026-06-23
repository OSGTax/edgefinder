# LIVE REBUILD PLAN — real-time, broker-backed paper trading

> **READ THIS FIRST on resume.** This supersedes the *trading model* of
> `REBUILD-PLAN.md` (the daily-bar / market-on-close / Routine-as-pricing design).
> The data asset, R2 archive, desk page shell, and backtest tooling are kept.
> The lagged daily-bar execution model is being **replaced** with live,
> broker-backed paper trading the owner can verify in real-time.

## Why (the flaw being fixed)
The previous build priced decisions, fills, and marks off the **latest daily
close** and ran the agent in an hourly, ephemeral Routine. That can never show
live prices or verifiable real-time suggestions — so it was unverifiable and,
to the owner, useless. The fix is to trade a **real brokerage paper account**
that fills against the **live market**, stream live quotes to the page, and
stamp every suggestion with the live price + time it was made.

## Locked decisions (owner, 2026-06-23)
- **Foundation = Alpaca + Polygon.**
  - **Alpaca paper account** is the account of record (positions, cash, fills,
    P&L) and the **real-time quote source** (paid **SIP** feed). Independently
    verifiable: the owner can log into Alpaca's dashboard and see the exact
    orders/positions the agent created, priced at the consolidated tape.
  - **Polygon** is kept for the *backing research*: fundamentals, news, and the
    21-year history / R2 archive for backtests. It no longer prices fills.
- **Runtime = split.**
  - **Render (always-on)** holds the Alpaca **WebSocket**, maintains a live quote
    cache, and **streams quotes to the desk page over SSE**. It also serves live
    positions/P&L pulled from Alpaca.
  - **Claude Code Routine (hourly)** is the agent's brain: reads Alpaca positions
    + live quotes + Polygon backing data over HTTPS, decides, and submits **paper
    orders via Alpaca REST**. Uses the owner's CC subscription (no API token cost).
- **Cadence = hourly** re-evaluation during market hours. Live prices stream
  continuously regardless; suggestions refresh hourly and are timestamped.

## Architecture
```
        Alpaca paper account  ───────────────┐  (orders fill at live market)
        Alpaca Market Data (SIP)             │
            │ WebSocket (quotes/trades)       │ REST (orders, positions, account)
            ▼                                 │
  Render (always-on FastAPI)                  │
   ├─ quote streamer → in-memory live cache   │
   ├─ /api/desk/stream (SSE) → page live ticks │
   └─ /api/desk/* live positions, P&L, fills ◄┘
            ▲
            │ HTTPS (read positions/quotes/data, submit orders)
  Claude Code Routine (hourly, market hours)  ── Polygon (fundamentals/news/history)
   └─ runs the trading skill: decide on LIVE prices → place paper orders →
      record suggestions (stamped price+time) + thinking/journal
```

## What gets nuked vs kept
**Nuke** (after the new path is proven — see rails):
- The daily-bar **market-on-close** model and the settle-gate.
- `agent.refresh` as the price path; the hand-rolled cash/position **ledger as
  source of truth** (Alpaca becomes source of truth; we mirror for display).
- The Routine-as-pricing assumption; the MOC rules in the trading skill.
- DB-over-HTTPS bar plumbing **only where it existed to feed pricing** (Alpaca
  replaces it; keep what still serves desk_* writes/backing data).

**Keep:**
- Polygon market-data asset + R2 archive (backing data + backtests).
- The desk page shell + design system + the thinking/journal/What's-New surfaces
  (repointed at the live engine).
- The backtest tool (grounds ideas against history).

## Phases (each ends green + verifiable; destructive steps last)
0. **Scaffold (non-destructive, behind `EDGEFINDER_LIVE=0` flag).**
   - `agent/broker.py` — Alpaca wrapper (account, positions, quotes, submit/cancel
     order, recent fills). Paper base URL; SIP feed. Unit-tested with mocks.
   - Config + secrets plumbing (no keys committed).
1. **Prove execution.** With the owner's paper keys: submit a tiny paper order,
   confirm it fills at the live price, read it back from Alpaca, cross-check the
   price against an independent quote. This is the go/no-go gate.
2. **Live data pipe.** Render quote streamer (Alpaca WS) → live cache →
   `/api/desk/stream` SSE → page shows live-ticking quotes + live position P&L.
3. **Agent loop on live prices.** Rework the trading skill: read live quotes +
   Alpaca positions + Polygon backing data, decide hourly, place paper orders,
   record each **suggestion with the live price + timestamp + evidence**.
4. **Desk page rebuild.** Live quotes, live positions/P&L from Alpaca, time-
   stamped suggestions with backing data, fills feed, thinking/journal.
5. **Cut over & retire.** Flip `EDGEFINDER_LIVE=1`, repoint the Routine to the new
   skill, retire the daily-bar/MOC code + obsolete tables. Old code stays in git.

## Non-negotiable rails (unchanged from the rebuild)
1. **Never drop/clear the market-data tables or the R2 archive** — sacred.
2. **Prove the live path works BEFORE retiring the old one** (Phase 1 gate).
3. **Git-recoverable** — branch work, old code in history, no force-push of main.

## Secrets / env needed (owner)
Free Alpaca **paper** account (alpaca.markets) + enable the real-time **SIP**
data plan, then add to **Render** and the **Routine** environment:
- `ALPACA_API_KEY`, `ALPACA_API_SECRET` (paper keys)
- `ALPACA_PAPER=true`
- `ALPACA_DATA_FEED=sip`
Polygon (`EDGEFINDER_POLYGON_API_KEY`) and R2_* stay as-is for backing data.

## Open items
- Order type policy (market vs marketable-limit; how the agent sizes).
- Universe the streamer subscribes to (held + watchlist + candidates, capped).
- How much of `desk_*` we keep as a display mirror vs reading Alpaca live.
