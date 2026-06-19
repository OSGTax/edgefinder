# AI Analyst — the research-agent paper account

A self-running paper account (like the 12 finalists) whose brain is a research
**agent** that each day uses the data, backtesting, and news it has to pick
forward-looking stocks and decide hold / add / trim / sell. Visualized on the
**Picks** page (chart-forward).

## How it works

The slow research is **decoupled** from the fast 9:45 execution cycle:

1. **9:15 ET — research job** (`agents/analyst.run_analyst`, scheduled via
   `services._analyst_job`): resolves a PIT universe (default `top:200`),
   builds a point-in-time context (holdings-aware), **screens** every name
   through four entry rules (trend, 52-week-high breakout, pullback-in-uptrend,
   6-month relative strength), ranks by composite, and selects with **entry/exit
   hysteresis** so the book turns over on signals — not daily reselection (the
   toll that sank the fast-cadence hunts). For each pick it backtests the firing
   rules (the **proof**), gathers **news**, and (when `CLAUDE_CODE_OAUTH_TOKEN`
   is set) writes a plain-English rationale. It persists one `agent_decisions`
   row: `target_weights` + per-pick dossier + summary.
2. **9:45 ET — execution** (`engine/live` cycle): `AnalystStrategy` (spec
   `ai_analyst`) reads that decision's target weights and trades them through
   the proven engine; hold/add/trim/sell fall out of the weight diff. If no
   fresh decision exists, it **holds the current book** — never liquidates.

The account is an **experimental, live-paper** strategy: its edge is proven
*forward* by its real track record, not by a sealed backtest (a backtest of
`ai_analyst` finds no decisions and holds cash — it is live-only by nature).

## Files
- `edgefinder/agents/analyst.py` — entry rules, screen, hysteresis selection,
  news, rule track-record backtests, `run_analyst` orchestration + persistence.
- `edgefinder/engine/analyst_strategy.py` — `AnalystStrategy` (reads the
  decision; hold-on-missing/stale fallback). Wired as spec `ai_analyst`.
- `edgefinder/db/models.py` `AgentDecision` (+ migration `b8d2f4e16a09`).
- `dashboard/routers/picks.py` + `templates/picks.html` + `static/js/pages/picks.js`
  — the Picks page (chart leads, then proof, why-now metrics, news, rationale).
- `dashboard/services.py` `run_analyst_job` / `_analyst_job`; scheduler 9:15 ET.

## Preview vs activation
- **Preview (safe now):** the Picks page + “Run research now” (`POST /api/picks/run`)
  produce and show decisions **without an account** — nothing trades. The 9:15
  scheduled job also produces decisions once deployed.
- **Activation (trades real paper $):** create the promoted account, then the
  9:45 cycle trades it daily:
  ```
  python -m edgefinder.engine.promote --spec ai_analyst --universe top:200 \
      --schedule daily --tier experimental --name ai_analyst
  ```
  **Staged — do NOT activate until** the branch is deployed AND a dry preview on
  the live data looks sane (run the research job, eyeball the Picks page). A
  daily `top:200` universe adds an R2 load to every cycle; start there and tune.

## Phase 2 (later)
Broaden the universe via R2, deepen the proof to walk-forward grade, richer
visuals, and an “approve a pick → Robinhood ticket” path (reusing
`engine/live_ticket`) when the real-money connect comes off hold.
