# EdgeFinder v2 — Redesign Plan

## Current Status: ALL PHASES COMPLETE

Phases 1-7 are built and tested (244 tests passing). Phase 8 (AI Meta-Strategy) is future work.

| Phase | Status | Tests | What was built |
|-------|--------|-------|----------------|
| 1 | COMPLETE | 89 | Foundation: config, models, DB (10 tables), Polygon.io, cache, streaming |
| 2 | COMPLETE | 62 | Scanner (Lynch/Burry scoring), signal engine (9 patterns), 3 strategies |
| 3 | COMPLETE | 28 | Trading engine: virtual accounts, executor, arena, journal |
| 4 | COMPLETE | 14 | Market snapshots (indices/VIX/sectors), index benchmarks |
| 5 | COMPLETE | 26 | Sentiment: Reddit, Twitter stub, News RSS, weighted aggregator |
| 6 | COMPLETE | 20 | Research service, dashboard (6 API routers, 18+ endpoints) |
| 7 | COMPLETE | 5 | Scheduler (APScheduler ET), scripts, Render deployment config |
| 8 | FUTURE | — | AI meta-strategy (interfaces designed, not built) |

---

## Design Decisions

- **Polygon.io only** — single data source, no fallback chain
- **$5,000 per strategy** — isolated virtual accounts, cash-only buying power
- **PDT per-strategy toggle** — each account independently controls day trading rules
- **Market snapshot on every trade** — SPY/QQQ/IWM/DIA/VIX/sectors captured and FK-linked
- **Strategy plugin architecture** — `@StrategyRegistry.register()` decorator, ANY-match qualification
- **Protocol-based interfaces** — DataProvider, StreamProvider, SentimentProvider for testability
- **SQLAlchemy 2.0 + Alembic** — proper ORM with migration support
- **pydantic-settings** — all config via `EDGEFINDER_` env vars
- **Pure pandas indicators** — no pandas-ta dependency
- **SHA-256 hash chain** — immutable audit trail on trades

## Future: Phase 8 — AI Meta-Strategy

Interfaces are already in place:
- `BaseStrategy.get_state()` — AI reads strategy internal state
- `BaseStrategy.apply_suggestion()` — AI tunes strategy parameters
- `AIAgentDataAccess` protocol designed (queries trades, snapshots, sentiment history)
- DB schema supports all needed queries

When built, `strategies/ai_agent.py` will be just another strategy plugin that:
- Learns from all other strategies' trade history
- Backtests against historical data + market snapshots
- Suggests parameter tweaks to other strategies

## Future: Additional Data Sources

The `DataProvider` protocol allows clean extension:
- **US options**: Polygon.io paid tier upgrade
- **International equities**: EODHD (60+ exchanges)
- **Twitter/X**: Real API integration (currently stub returning neutral)
