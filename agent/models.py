"""The autonomous trading agent's own tables (greenfield rebuild).

A clean namespace (``desk_*``) that lives in the SAME database as the kept
market-data tables but shares none of the retired trading/app schema. The
agent reads and writes only these; the market-data tables (daily_bars,
dividends, ticker_splits, fundamentals_snapshots, ticker_news, index_daily)
are read-only inputs reached through the kept data-access layer.

Source-of-truth rule (mirrors the old engine's discipline): ``desk_trades``
is append-only and authoritative for cash. The portfolio/positions/equity
rows are recomputable projections — ``agent.ledger`` rebuilds them from the
trade ledger on every mark, so a corrupted projection can always be healed.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from edgefinder.db.engine import Base

# The single paper book the agent runs. One account, full discretion.
ACCOUNT = "agent"
STARTING_CAPITAL = 100_000.0


# ── desk_trades — append-only fill ledger (source of truth for cash) ──


class DeskTrade(Base):
    """One executed paper fill. Append-only; never updated or deleted.

    Cash is always ``STARTING_CAPITAL + Σ(SELL dollars) - Σ(BUY dollars)``
    over this table, so the ledger is auditable and self-correcting.
    """

    __tablename__ = "desk_trades"
    __table_args__ = (Index("idx_desk_trades_account_ts", "account", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY | SELL
    shares: Mapped[float] = mapped_column(Float)  # fractional shares (v7)
    price: Mapped[float] = mapped_column(Float)  # fill price after costs
    # signed-agnostic cash moved = shares*price*mult ± option fees (option
    # fills carry a flat per-contract fee inside dollars; fill_quote.fee is
    # the receipt — see agent.ledger.record_trade)
    dollars: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text)
    # The live-quote snapshot the fill priced off: {bid, ask, mid, t, src}.
    # The honesty contract's receipt — every live fill carries one.
    fill_quote: Mapped[dict | None] = mapped_column(JSON)


# ── desk_positions — open lots (projection, rebuilt from the ledger) ──


class DeskPosition(Base):
    """Current open lot per symbol. Rebuilt from desk_trades on every mark."""

    __tablename__ = "desk_positions"
    __table_args__ = (
        UniqueConstraint("account", "symbol", name="uq_desk_pos_account_symbol"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    shares: Mapped[float] = mapped_column(Float)  # fractional shares (v7)
    avg_price: Mapped[float] = mapped_column(Float)  # cost basis per share
    last_price: Mapped[float | None] = mapped_column(Float)  # latest mark
    opened_at: Mapped[datetime | None] = mapped_column(DateTime)
    marked_at: Mapped[datetime | None] = mapped_column(DateTime)


# ── desk_equity — equity-curve time series (one row per mark) ──


class DeskEquity(Base):
    """Account equity snapshot for the curve. Append-only series."""

    __tablename__ = "desk_equity"
    __table_args__ = (Index("idx_desk_equity_account_ts", "account", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)
    return_pct: Mapped[float] = mapped_column(Float)
    # Mark provenance: {"sources": {live, close, cost}, "cost_marked": [...],
    # "cost_marked_value_pct": x, "degraded": true?} — which pricing tier
    # marked each position, so a snapshot written during a quote/data outage
    # (cost-basis marks = fake-flat P&L) is visibly flagged, never silently
    # embedded in the curve forever. NOTE: a dev database created before this
    # column will error reading desk_equity via the ORM — rerun
    # scripts/setup_db.py (prod self-heals via the idempotent ALTER in
    # DESK_TABLE_DDL on deploy).
    mark_meta: Mapped[dict | None] = mapped_column(JSON)


# ── desk_strategy_state — the agent's current, evolving strategy ──


class DeskStrategyState(Base):
    """The agent's living strategy. One row per version; latest is current.

    The agent rewrites this when it adopts/evolves an approach. ``version``
    increments on a real pivot (a journal entry should accompany it).
    """

    __tablename__ = "desk_strategy_state"
    __table_args__ = (Index("idx_desk_state_account_ver", "account", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    name: Mapped[str] = mapped_column(String(80))
    thesis: Mapped[str | None] = mapped_column(Text)  # plain-English approach
    rules: Mapped[dict | None] = mapped_column(JSON)   # structured selection rules
    params: Mapped[dict | None] = mapped_column(JSON)  # knobs the agent tunes
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


# ── desk_journal — pivots, tweaks, and notes (the agent's diary) ──


class DeskJournal(Base):
    """Why the strategy changed. The narrative audit of the agent's evolution."""

    __tablename__ = "desk_journal"
    __table_args__ = (Index("idx_desk_journal_account_ts", "account", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # pivot | tweak | note
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str | None] = mapped_column(Text)
    version_from: Mapped[int | None] = mapped_column(Integer)
    version_to: Mapped[int | None] = mapped_column(Integer)


# ── desk_thinking — per-run narration feed (the "live thinking") ──


class DeskThinking(Base):
    """Streamed reasoning lines for one run. Powers the live thinking panel."""

    __tablename__ = "desk_thinking"
    __table_args__ = (Index("idx_desk_thinking_run_ts", "run_id", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    phase: Mapped[str | None] = mapped_column(String(30))  # observe|research|decide|execute
    text: Mapped[str] = mapped_column(Text)


# ── desk_decisions — one decision record per run ──


class DeskDecision(Base):
    """The agent's decision for a run: regime, picks, target book, watchlist.

    ``picks`` is a list of per-name dossiers (symbol, action, why_now,
    rationale, evidence, news) — the chart-forward holdings panel reads it.
    """

    __tablename__ = "desk_decisions"
    __table_args__ = (
        UniqueConstraint("account", "run_id", name="uq_desk_decision_run"),
        Index("idx_desk_decision_account_ts", "account", "ts"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    decision_date: Mapped[date | None] = mapped_column(Date)
    regime: Mapped[str | None] = mapped_column(String(40))
    summary: Mapped[str | None] = mapped_column(Text)
    target_weights: Mapped[dict | None] = mapped_column(JSON)
    picks: Mapped[list | None] = mapped_column(JSON)
    watchlist: Mapped[list | None] = mapped_column(JSON)
    # Candidates that LOST the slot this run: [{symbol, why_not}]. Graded by
    # the weekly reflection alongside the picks — "the thing I didn't buy did
    # X" doubles the learning signal at zero risk. NOTE: a dev database
    # created before v8.15 lacks this column and the ORM will error reading
    # desk_decisions — rerun scripts/setup_db.py (prod self-heals via the
    # idempotent ALTER in DESK_TABLE_DDL on deploy).
    rejected: Mapped[list | None] = mapped_column(JSON)
    strategy_version: Mapped[int | None] = mapped_column(Integer)


# ── desk_backtests — grounding evidence the agent ran ──


class DeskBacktest(Base):
    """A backtest the agent ran to ground an idea. Evidence panel reads these."""

    __tablename__ = "desk_backtests"
    __table_args__ = (Index("idx_desk_backtest_account_ts", "account", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    label: Mapped[str] = mapped_column(String(120))
    spec: Mapped[dict | None] = mapped_column(JSON)    # symbols/rule/schedule/window
    result: Mapped[dict | None] = mapped_column(JSON)  # return/sharpe/dd/excess vs SPY


# ── desk_changelog — what the app-evolution routine shipped ("What's New") ──


class DeskChangelog(Base):
    """One user-facing improvement the agent made to the dashboard itself.

    The end-of-day app-evolution routine, when it ships a genuinely useful
    change to what /desk shows, records a row here: a short ``title`` and a
    plain-English ``detail`` explaining the feature and why it helps. The page
    lights a "NEW" badge for entries inside the spotlight window and lists them
    in the "What's New" panel — so users (and the owner) can see how the app is
    growing and read what each addition does.
    """

    __tablename__ = "desk_changelog"
    __table_args__ = (Index("idx_desk_changelog_ts", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    kind: Mapped[str] = mapped_column(String(20), default="feature")  # feature|improvement|data|disclaimer|fix
    title: Mapped[str] = mapped_column(String(160))
    detail: Mapped[str | None] = mapped_column(Text)  # the explanation users read
    version: Mapped[str | None] = mapped_column(String(20))  # app version at ship time
    run_id: Mapped[str | None] = mapped_column(String(40))


# ── desk_options_snap — the options IV data bank (one row per underlying/day) ──


class DeskOptionsSnap(Base):
    """Daily options snapshot per underlying: ATM IV, straddle-implied expected
    move, 25-delta skew. Written once/day by the agent's refresh — accumulates
    into the IV history the charts plot and the agent reasons over (IV rank
    becomes computable as the bank grows)."""

    __tablename__ = "desk_options_snap"
    __table_args__ = (
        UniqueConstraint("symbol", "snap_date", name="uq_desk_optsnap_sym_date"),
        Index("idx_desk_optsnap_sym_date", "symbol", "snap_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    snap_date: Mapped[date] = mapped_column(Date)
    spot: Mapped[float | None] = mapped_column(Float)
    atm_iv: Mapped[float | None] = mapped_column(Float)
    expected_move_pct: Mapped[float | None] = mapped_column(Float)
    skew_25d: Mapped[float | None] = mapped_column(Float)
    dte: Mapped[int | None] = mapped_column(Integer)
    expiry: Mapped[str | None] = mapped_column(String(10))
    # When the snapshot was actually taken (UTC). The row's identity stays
    # (symbol, snap_date) — one canonical row per day — but captured_at is the
    # receipt proving it was a regular-hours read (the refresh's session gate),
    # not crossed pre-open OPRA marks locked in as the day's history. NOTE: a
    # dev database created before this column will error reading
    # desk_options_snap via the ORM — rerun scripts/setup_db.py (prod
    # self-heals via the idempotent ALTER in DESK_TABLE_DDL on deploy).
    captured_at: Mapped[datetime | None] = mapped_column(DateTime)


# ── desk_wiki — the agent's self-curated lessons wiki (system-prompt learning) ──


class DeskWiki(Base):
    """One curated page of the agent's lessons wiki.

    Karpathy-style "system prompt learning": a small, size-capped set of pages
    (playbook / setups / lessons / mistakes / postmortems / market-notes) the
    agent READS at the start of every cycle and REVISES from measured outcomes
    — knowledge accumulating in curated context, not weights. Pages are edited
    IN PLACE (fixed slugs are the curation constraint; no append-only sprawl);
    every edit writes a desk_journal note (kind="wiki") AND banks the outgoing
    body as a desk_wiki_history revision — so curation is aggressive without
    being destructive. Caps enforced by agent.brain (the only writer).
    """

    __tablename__ = "desk_wiki"
    __table_args__ = (
        UniqueConstraint("account", "slug", name="uq_desk_wiki_account_slug"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    # playbook|setups|lessons|mistakes|postmortems|market-notes
    slug: Mapped[str] = mapped_column(String(40), index=True)
    title: Mapped[str | None] = mapped_column(String(80))
    body: Mapped[str] = mapped_column(Text)  # markdown-lite, hard-capped by the tool
    revision: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())
    updated_run_id: Mapped[str | None] = mapped_column(String(40))


class DeskWikiHistory(Base):
    """One archived wiki revision — the OUTGOING body ``agent.brain set_wiki``
    banks immediately before every in-place rewrite (a page's first-ever write
    archives nothing: there is no prior). Append-only: pruning a lesson stops
    destroying its evidence, because the pruned text is one
    ``brain wiki-history`` read away. Written only by set_wiki; ``updated_at``
    / ``updated_run_id`` carry over from the revision being replaced (when it
    was written, and by which run)."""

    __tablename__ = "desk_wiki_history"
    __table_args__ = (
        Index("idx_desk_wiki_hist_slug", "account", "slug", "revision"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    slug: Mapped[str] = mapped_column(String(40), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(String(80))
    body: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)
    updated_run_id: Mapped[str | None] = mapped_column(String(40))


# ── desk_briefs — the nightly research pack ──


class DeskBrief(Base):
    """One nightly research pack, precomputed while the whole-market data is
    already in hand (the data-refresh routine builds it right after the
    ingest). The hourly trading cycle reads ONE dense payload — regime,
    ranked universe, movers, trend roster, headlines, data-coverage verdict —
    instead of re-deriving it with a dozen exploratory scans, so its context
    goes to deciding, not gathering. One row per (account, brief_date),
    rebuilt in place; written only by ``agent.market brief-build``.
    """

    __tablename__ = "desk_briefs"
    __table_args__ = (
        UniqueConstraint("account", "brief_date", name="uq_desk_brief_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    brief_date: Mapped[date] = mapped_column(Date, index=True)
    built_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    payload: Mapped[dict] = mapped_column(JSON)


# ── the attention system: tripwires + planned wakes ──


class DeskWatch(Base):
    """One standing tripwire the brain armed on its way out.

    The always-on streamer sweeps armed wires against the live tape every few
    seconds and marks trips; the brain reads them FIRST at its next wake.
    Cheap code watches continuously so expensive judgment only shows up when
    something it named actually happened. Written only by ``agent.brain
    watch-set`` / ``watch-clear``; the streamer writes the trip — and, for
    the opt-in ``hard_stop`` kind ONLY, executes a full-position sell through
    the ledger's normal live-fill gates (plain above/below wires stay
    advisory, always).
    """

    __tablename__ = "desk_watch"

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str | None] = mapped_column(String(40))
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    kind: Mapped[str] = mapped_column(String(10))  # above | below | hard_stop
    level: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    armed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    until: Mapped[datetime | None] = mapped_column(DateTime)  # expiry (UTC)
    # armed | tripped | expired | disarmed
    #   | executing | executed | exec_failed | stale   (hard_stop lifecycle:
    # 'executing' is the atomic claim — exactly one writer wins it; a claim
    # orphaned by a crash is flagged exec_failed, never auto-retried)
    status: Mapped[str] = mapped_column(String(12), default="armed", index=True)
    tripped_at: Mapped[datetime | None] = mapped_column(DateTime)
    tripped_price: Mapped[float | None] = mapped_column(Float)
    # hard_stop execution receipt: the fill's run_id ("hardstop:{id}") on
    # success, and a human-readable outcome (fill summary / gate rejection).
    honored_run_id: Mapped[str | None] = mapped_column(String(40))
    result: Mapped[str | None] = mapped_column(Text)


class DeskWake(Base):
    """One self-scheduled check-in the brain planned (and why).

    The budget ledger for the attention system: ``agent.brain wake-plan``
    enforces the per-day cap and minimum gap, and the desk shows the owner
    when the trader plans to look next. Routine-spawned sessions have no
    scheduler MCP (probed 2026-07-13), so a plan is a PROMISE the next
    heartbeat honors: the first cycle at/after ``at`` runs it as a focused
    wake and stamps ``honored_run_id``. Rows are otherwise append-only.
    """

    __tablename__ = "desk_wakes"

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str | None] = mapped_column(String(40))
    at: Mapped[datetime] = mapped_column(DateTime, index=True)  # UTC fire time
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    honored_run_id: Mapped[str | None] = mapped_column(String(40))
    # machine-fired autonomy (v9.11.0): how many workflow dispatches this
    # wake has triggered; at DISPATCH_MAX_PER_WAKE the dispatcher stamps it
    # honored_run_id='missed:auto' so no wake can loop forever
    dispatch_count: Mapped[int] = mapped_column(Integer, default=0)


class DeskDispatch(Base):
    """One GitHub workflow_dispatch fired (or attempted) by the streamer.

    The autonomy loop's at-most-once ledger: ``bucket`` (epoch // gap) is
    UNIQUE per account, so sibling streamer instances during a Render
    deploy overlap CAS-race on the insert and exactly one wins the window
    (the ``claim_watch`` idiom). Also the debounce clock and the per-ET-day
    dispatch cap — DB state, never process memory."""

    __tablename__ = "desk_dispatches"
    __table_args__ = (
        UniqueConstraint("account", "bucket", name="uq_desk_dispatch_bucket"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    bucket: Mapped[int] = mapped_column(Integer)   # epoch // (min-gap secs)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    reason: Mapped[str] = mapped_column(Text)
    wake_ids: Mapped[list | None] = mapped_column(JSON)
    watch_ids: Mapped[list | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(12), default="claimed")
    http_status: Mapped[int | None] = mapped_column(Integer)


# ── desk_outcomes — machine-graded pick facts (the learning loop's ledger) ──


class DeskOutcome(Base):
    """One pick's machine-graded outcome facts — the durable scoreboard the
    reflection agent grades FROM instead of re-deriving (or vibing) each week.

    Written by ``agent.ledger grade``: one row per (account, run_id, symbol),
    UPDATED IN PLACE on each grading pass (``grade_date`` tracks the latest)
    — machine facts only. The two judgment columns — ``verdict``
    (TRUE|FALSE|NOT_YET) and ``verdict_note`` — are filled ONLY by the
    reflection agent via ``agent.brain verdict`` and survive re-grading
    (grade never touches them). BOOK stances and picks with no entry (BUY)
    fills are never graded here (no per-pick entry to grade) — but a pick
    CLOSED by fills outside its own run (a hard stop, a later run's exit,
    expiry settlement) IS graded: grade reconstructs the exit from the
    closing sell fills and stamps ``exit_kind`` (same_run | cross_run |
    hardstop | settlement, by the dominant closing run_id), ``exit_avg_px``
    (current share basis; fee-net for options) and ``realized_pnl``.
    ``degraded`` marks a row whose mark-derived facts were nulled because
    the latest equity snapshot priced the symbol at cost basis (mark_meta) —
    a later clean re-grade overwrites it.
    """

    __tablename__ = "desk_outcomes"
    __table_args__ = (
        UniqueConstraint("account", "run_id", "symbol",
                         name="uq_desk_outcome_pick"),
        Index("idx_desk_outcomes_run", "run_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    grade_date: Mapped[date] = mapped_column(Date)  # latest grading pass (ET)
    entry_avg_px: Mapped[float | None] = mapped_column(Float)
    mark_px: Mapped[float | None] = mapped_column(Float)
    mark_basis: Mapped[str | None] = mapped_column(String(12))  # mark | exit
    since_pct: Mapped[float | None] = mapped_column(Float)
    spy_pct: Mapped[float | None] = mapped_column(Float)   # TR SPY, same window
    alpha_pct: Mapped[float | None] = mapped_column(Float)  # null for options
    horizon_days: Mapped[int | None] = mapped_column(Integer)
    horizon_elapsed: Mapped[bool | None] = mapped_column(Boolean)  # in sessions
    kill_level: Mapped[float | None] = mapped_column(Float)  # null: free text
    kill_breached: Mapped[bool | None] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(8))  # open | closed
    # How a closed pick actually exited (null while open / pre-migration):
    # same_run | cross_run | hardstop | settlement, by dominant closing run_id
    exit_kind: Mapped[str | None] = mapped_column(String(12))
    exit_avg_px: Mapped[float | None] = mapped_column(Float)  # current basis
    realized_pnl: Mapped[float | None] = mapped_column(Float)  # entry→flat, per symbol
    # True when mark-derived facts were nulled: the latest equity snapshot
    # priced this symbol at COST BASIS (desk_equity.mark_meta) and a fake-flat
    # mark must not grade a pick. Clean re-grades overwrite to False.
    degraded: Mapped[bool | None] = mapped_column(Boolean)
    verdict: Mapped[str | None] = mapped_column(String(12))  # reflection only
    verdict_note: Mapped[str | None] = mapped_column(Text)   # reflection only
    graded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── the knowledge layer: claims, commitments, proposals (SCHEMA.md) ──


class DeskClaim(Base):
    """One behavior-influencing fact in the structured claims registry.

    The source of truth the wiki's prose must cite (``[C-<id>]`` tokens):
    prose can inform, only claims can justify. Tiers (observation → digest →
    candidate → established) carry pre-registered ``promotion_criteria`` —
    written at candidate creation, BEFORE results — and promotion is refused
    in code unless stats recomputed from ``desk_outcomes`` meet them. No
    confidence floats anywhere: ``stats`` holds recorded sample sizes.
    Supersession, never deletion: status flips, ``superseded_by`` links, and
    every transition lands in ``desk_claim_events``. ``decay_class`` defaults
    are forced by ``kclass`` (risk_rule→never, system_mechanics→stable,
    market_strategy→regime_conditional). Written only by ``agent.knowledge``.
    """

    __tablename__ = "desk_claims"

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    # market_strategy | system_mechanics | operational | risk_rule
    kclass: Mapped[str] = mapped_column(String(20))
    # observation | digest | candidate | established
    tier: Mapped[str] = mapped_column(String(16))
    # candidate flagged to influence decisions under the exposure caps
    experimental: Mapped[bool] = mapped_column(Boolean, default=False)
    # active | superseded | retired | quarantined
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    statement: Mapped[str] = mapped_column(Text)  # one falsifiable sentence, tool-capped
    # {"account":"paper", "universe":..., "regimes":[...], "strategy_versions":[...]}
    scope: Mapped[dict | None] = mapped_column(JSON)
    # typed machine-resolvable refs: outcome/decision/trade/backtest/wiki_history/probe
    evidence: Mapped[list | None] = mapped_column(JSON)
    # {"n":..,"wins":..,"losses":..,"avg_alpha_pct":..,"span":[..],"regimes":{..},"symbols":[..]}
    stats: Mapped[dict | None] = mapped_column(JSON)
    # thresholds registered at candidate creation; promotion refused without them
    promotion_criteria: Mapped[dict | None] = mapped_column(JSON)
    # regime_conditional | stable | never
    decay_class: Mapped[str] = mapped_column(String(20))
    expires_at: Mapped[date | None] = mapped_column(Date)   # required for regime_conditional
    review_after: Mapped[date | None] = mapped_column(Date)
    supersedes: Mapped[int | None] = mapped_column(Integer)
    superseded_by: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    created_run_id: Mapped[str | None] = mapped_column(String(40))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())
    updated_run_id: Mapped[str | None] = mapped_column(String(40))


class DeskClaimEvent(Base):
    """One append-only lifecycle event on a claim — the typed counterpart to
    the prose journal. Every created/promoted/demoted/superseded/retired/
    quarantined/expired transition (plus evidence adds and proposal links)
    lands here with a detail snapshot, so the traceable path from any
    behavior-influencing fact to what happened to it is queryable without
    prose archaeology. Written only by ``agent.knowledge``."""

    __tablename__ = "desk_claim_events"
    __table_args__ = (Index("idx_desk_claim_events_claim", "claim_id", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    claim_id: Mapped[int] = mapped_column(Integer, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    run_id: Mapped[str | None] = mapped_column(String(40))
    # created | evidence_added | promoted | demoted | superseded | retired |
    # quarantined | expired | proposal_linked
    event: Mapped[str] = mapped_column(String(20))
    detail: Mapped[dict | None] = mapped_column(JSON)


class DeskCommitment(Base):
    """One structured falsification clause carried by a trim/exit/hold pick —
    the fix for free-text promises ("re-add if it reclaims $X") that escape
    the buy/add prediction registry and go silently unchecked (the AAPL
    ~$500 lesson). Materialized by ``agent.brain decision`` from a pick's
    ``commitment`` object; machine-checked by ``agent.ledger grade`` against
    stored closes (same split-aware touch semantics as kill breaches);
    fired-and-unhonored rows surface in ``brain context`` as obligations
    until a later decision stamps ``honored_run_id`` — even when the honest
    answer is "standing down, because Y"."""

    __tablename__ = "desk_commitments"
    __table_args__ = (Index("idx_desk_commit_status", "account", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)  # creating decision
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    kind: Mapped[str] = mapped_column(String(16))       # reentry | stop | review
    direction: Mapped[str | None] = mapped_column(String(6))  # above | below
    level: Mapped[float | None] = mapped_column(Float)
    until: Mapped[date | None] = mapped_column(Date)
    text: Mapped[str] = mapped_column(Text)             # the clause, verbatim
    # open | fired | honored | expired | withdrawn
    status: Mapped[str] = mapped_column(String(12), default="open")
    fired_date: Mapped[date | None] = mapped_column(Date)
    fired_close: Mapped[float | None] = mapped_column(Float)
    honored_run_id: Mapped[str | None] = mapped_column(String(40))
    watch_id: Mapped[int | None] = mapped_column(Integer)  # linked advisory tripwire
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DeskProposal(Base):
    """One owner-approval request for a trading-behavior change derived from
    learned facts (strategy pivots, cap raises, setup adoption). The agent
    proposes with the justifying ``claim_ids`` and the exact intended
    ``payload``; the owner approves out-of-band (GitHub issue comment with
    verifiable authorship — `PROPOSAL-<id>` — or the weaker CLI fallback,
    recorded in ``decided_via``). ``agent.brain state-set --bump`` requires an
    approved proposal id or an audited ``--no-learned-basis`` escape hatch.
    Written only by ``agent.knowledge``."""

    __tablename__ = "desk_proposals"
    __table_args__ = (Index("idx_desk_proposals_status", "account", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    run_id: Mapped[str | None] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(160))
    body: Mapped[str] = mapped_column(Text)             # plain-English what/why
    claim_ids: Mapped[list | None] = mapped_column(JSON)
    # params | rules | caps | setup_adoption
    change_kind: Mapped[str] = mapped_column(String(16))
    payload: Mapped[dict | None] = mapped_column(JSON)  # exact intended diff
    # pending | approved | rejected | expired | applied
    status: Mapped[str] = mapped_column(String(12), default="pending")
    decided_at: Mapped[datetime | None] = mapped_column(DateTime)
    decided_by: Mapped[str | None] = mapped_column(String(60))
    decided_via: Mapped[str | None] = mapped_column(String(12))  # github | cli
    applied_run_id: Mapped[str | None] = mapped_column(String(40))
    expires_at: Mapped[date | None] = mapped_column(Date)


# Idempotent CREATE TABLE IF NOT EXISTS DDL for render_start.py (Render skips
# create_all). Postgres-flavored; SQLite ignores the JSON type harmlessly.
DESK_TABLE_DDL: list[str] = [
    """CREATE TABLE IF NOT EXISTS desk_trades (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        ts TIMESTAMP DEFAULT NOW(),
        run_id VARCHAR(40),
        symbol VARCHAR(24) NOT NULL,
        side VARCHAR(4) NOT NULL,
        shares FLOAT NOT NULL,
        price FLOAT NOT NULL,
        dollars FLOAT NOT NULL,
        rationale TEXT,
        fill_quote JSON
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_trades_account_ts ON desk_trades (account, ts)",
    "CREATE INDEX IF NOT EXISTS idx_desk_trades_run ON desk_trades (run_id)",
    """CREATE TABLE IF NOT EXISTS desk_positions (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        symbol VARCHAR(24) NOT NULL,
        shares FLOAT NOT NULL,
        avg_price FLOAT NOT NULL,
        last_price FLOAT,
        opened_at TIMESTAMP,
        marked_at TIMESTAMP,
        CONSTRAINT uq_desk_pos_account_symbol UNIQUE (account, symbol)
    )""",
    """CREATE TABLE IF NOT EXISTS desk_equity (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        ts TIMESTAMP DEFAULT NOW(),
        cash FLOAT NOT NULL,
        positions_value FLOAT NOT NULL,
        equity FLOAT NOT NULL,
        return_pct FLOAT NOT NULL,
        mark_meta JSON
    )""",
    # Additive upgrade for desk_equity tables created before mark provenance.
    "ALTER TABLE desk_equity ADD COLUMN IF NOT EXISTS mark_meta JSON",
    "CREATE INDEX IF NOT EXISTS idx_desk_equity_account_ts ON desk_equity (account, ts)",
    """CREATE TABLE IF NOT EXISTS desk_strategy_state (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        version INTEGER DEFAULT 1,
        name VARCHAR(80) NOT NULL,
        thesis TEXT,
        rules JSON,
        params JSON,
        updated_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_state_account_ver ON desk_strategy_state (account, version)",
    """CREATE TABLE IF NOT EXISTS desk_journal (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        ts TIMESTAMP DEFAULT NOW(),
        kind VARCHAR(20) NOT NULL,
        title VARCHAR(200) NOT NULL,
        body TEXT,
        version_from INTEGER,
        version_to INTEGER
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_journal_account_ts ON desk_journal (account, ts)",
    """CREATE TABLE IF NOT EXISTS desk_thinking (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40) NOT NULL,
        ts TIMESTAMP DEFAULT NOW(),
        phase VARCHAR(30),
        text TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_thinking_run_ts ON desk_thinking (run_id, ts)",
    """CREATE TABLE IF NOT EXISTS desk_decisions (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40) NOT NULL,
        ts TIMESTAMP DEFAULT NOW(),
        decision_date DATE,
        regime VARCHAR(40),
        summary TEXT,
        target_weights JSON,
        picks JSON,
        watchlist JSON,
        rejected JSON,
        strategy_version INTEGER,
        CONSTRAINT uq_desk_decision_run UNIQUE (account, run_id)
    )""",
    # Additive upgrade for desk_decisions tables created before v8.15.
    "ALTER TABLE desk_decisions ADD COLUMN IF NOT EXISTS rejected JSON",
    "CREATE INDEX IF NOT EXISTS idx_desk_decision_account_ts ON desk_decisions (account, ts)",
    """CREATE TABLE IF NOT EXISTS desk_backtests (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40),
        ts TIMESTAMP DEFAULT NOW(),
        label VARCHAR(120) NOT NULL,
        spec JSON,
        result JSON
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_backtest_account_ts ON desk_backtests (account, ts)",
    """CREATE TABLE IF NOT EXISTS desk_changelog (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        ts TIMESTAMP DEFAULT NOW(),
        kind VARCHAR(20) DEFAULT 'feature',
        title VARCHAR(160) NOT NULL,
        detail TEXT,
        version VARCHAR(20),
        run_id VARCHAR(40)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_changelog_ts ON desk_changelog (ts)",
    """CREATE TABLE IF NOT EXISTS desk_options_snap (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        snap_date DATE NOT NULL,
        spot FLOAT,
        atm_iv FLOAT,
        expected_move_pct FLOAT,
        skew_25d FLOAT,
        dte INTEGER,
        expiry VARCHAR(10),
        captured_at TIMESTAMP,
        CONSTRAINT uq_desk_optsnap_sym_date UNIQUE (symbol, snap_date)
    )""",
    # Additive upgrade for desk_options_snap tables created before the
    # capture-time receipt (the IV pass's RTH session gate).
    "ALTER TABLE desk_options_snap ADD COLUMN IF NOT EXISTS captured_at TIMESTAMP",
    "CREATE INDEX IF NOT EXISTS idx_desk_optsnap_sym_date ON desk_options_snap (symbol, snap_date)",
    """CREATE TABLE IF NOT EXISTS desk_wiki (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        slug VARCHAR(40) NOT NULL,
        title VARCHAR(80),
        body TEXT NOT NULL,
        revision INTEGER DEFAULT 1,
        updated_at TIMESTAMP DEFAULT NOW(),
        updated_run_id VARCHAR(40),
        CONSTRAINT uq_desk_wiki_account_slug UNIQUE (account, slug)
    )""",
    """CREATE TABLE IF NOT EXISTS desk_wiki_history (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        slug VARCHAR(40) NOT NULL,
        revision INTEGER NOT NULL,
        title VARCHAR(80),
        body TEXT NOT NULL,
        updated_at TIMESTAMP,
        updated_run_id VARCHAR(40)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_wiki_hist_slug ON desk_wiki_history (account, slug, revision)",
    # Same lockdown as every other desk_* table: RLS on, zero policies.
    "ALTER TABLE desk_wiki_history ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_briefs (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        brief_date DATE NOT NULL,
        built_at TIMESTAMP DEFAULT NOW(),
        payload JSON NOT NULL,
        CONSTRAINT uq_desk_brief_date UNIQUE (account, brief_date)
    )""",
    # Same lockdown as every other desk_* table (scripts/enable_rls.sql):
    # RLS on, zero policies — anon/authenticated denied; the owning postgres
    # role (Render/agent) bypasses. Without this a new public-schema table is
    # world-writable through the Supabase Data API. Idempotent.
    "ALTER TABLE desk_briefs ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_watch (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40),
        symbol VARCHAR(24) NOT NULL,
        kind VARCHAR(10) NOT NULL,
        level FLOAT NOT NULL,
        reason TEXT NOT NULL,
        armed_at TIMESTAMP DEFAULT NOW(),
        until TIMESTAMP,
        status VARCHAR(12) DEFAULT 'armed',
        tripped_at TIMESTAMP,
        tripped_price FLOAT,
        honored_run_id VARCHAR(40),
        result TEXT
    )""",
    # Additive upgrades for desk_watch tables created before hard stops:
    # 'exec_failed' is 11 chars, and the execution receipt needs two columns.
    "ALTER TABLE desk_watch ALTER COLUMN status TYPE VARCHAR(12)",
    "ALTER TABLE desk_watch ADD COLUMN IF NOT EXISTS honored_run_id VARCHAR(40)",
    "ALTER TABLE desk_watch ADD COLUMN IF NOT EXISTS result TEXT",
    "CREATE INDEX IF NOT EXISTS idx_desk_watch_status ON desk_watch (account, status)",
    "ALTER TABLE desk_watch ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_wakes (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40),
        at TIMESTAMP NOT NULL,
        reason TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        honored_run_id VARCHAR(40)
    )""",
    "ALTER TABLE desk_wakes ADD COLUMN IF NOT EXISTS honored_run_id VARCHAR(40)",
    "CREATE INDEX IF NOT EXISTS idx_desk_wakes_at ON desk_wakes (account, at)",
    "ALTER TABLE desk_wakes ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_outcomes (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40) NOT NULL,
        symbol VARCHAR(24) NOT NULL,
        grade_date DATE NOT NULL,
        entry_avg_px FLOAT,
        mark_px FLOAT,
        mark_basis VARCHAR(12),
        since_pct FLOAT,
        spy_pct FLOAT,
        alpha_pct FLOAT,
        horizon_days INTEGER,
        horizon_elapsed BOOLEAN,
        kill_level FLOAT,
        kill_breached BOOLEAN,
        status VARCHAR(8) NOT NULL,
        exit_kind VARCHAR(12),
        exit_avg_px FLOAT,
        realized_pnl FLOAT,
        degraded BOOLEAN,
        verdict VARCHAR(12),
        verdict_note TEXT,
        graded_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT uq_desk_outcome_pick UNIQUE (account, run_id, symbol)
    )""",
    # Additive upgrades for desk_outcomes tables created before exit
    # reconstruction / degraded-mark flagging (v9.8.x review fixes).
    "ALTER TABLE desk_outcomes ADD COLUMN IF NOT EXISTS exit_kind VARCHAR(12)",
    "ALTER TABLE desk_outcomes ADD COLUMN IF NOT EXISTS exit_avg_px FLOAT",
    "ALTER TABLE desk_outcomes ADD COLUMN IF NOT EXISTS realized_pnl FLOAT",
    "ALTER TABLE desk_outcomes ADD COLUMN IF NOT EXISTS degraded BOOLEAN",
    "CREATE INDEX IF NOT EXISTS idx_desk_outcomes_run ON desk_outcomes (run_id)",
    "ALTER TABLE desk_outcomes ENABLE ROW LEVEL SECURITY",
    # fundamentals_pit is a MARKET-DATA table (edgefinder/db/models.py), not a
    # desk_* one, but new tables reach prod through this idempotent list —
    # same precedent as desk_briefs. Written only by agent.edgar.
    """CREATE TABLE IF NOT EXISTS fundamentals_pit (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        cik INTEGER,
        filed DATE NOT NULL,
        period_end DATE,
        form VARCHAR(12),
        source VARCHAR(12) DEFAULT 'edgar',
        data JSON NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT uq_fund_pit_symbol_filed_period UNIQUE (symbol, filed, period_end)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_fund_pit_symbol_filed ON fundamentals_pit (symbol, filed)",
    "ALTER TABLE fundamentals_pit ENABLE ROW LEVEL SECURITY",
    # v9.11.0 autonomy loop: the dispatch ledger + per-wake attempt counter
    """CREATE TABLE IF NOT EXISTS desk_dispatches (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        bucket INTEGER NOT NULL,
        ts TIMESTAMP DEFAULT NOW(),
        reason TEXT NOT NULL,
        wake_ids JSON,
        watch_ids JSON,
        status VARCHAR(12) DEFAULT 'claimed',
        http_status INTEGER,
        CONSTRAINT uq_desk_dispatch_bucket UNIQUE (account, bucket)
    )""",
    "ALTER TABLE desk_dispatches ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE desk_wakes ADD COLUMN IF NOT EXISTS dispatch_count INTEGER DEFAULT 0",
    # v9.13.0 knowledge layer (SCHEMA.md): claims registry + lifecycle events +
    # commitments + owner-approval proposals. Same lockdown as every desk_*
    # table: RLS on, zero policies.
    """CREATE TABLE IF NOT EXISTS desk_claims (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        kclass VARCHAR(20) NOT NULL,
        tier VARCHAR(16) NOT NULL,
        experimental BOOLEAN DEFAULT FALSE,
        status VARCHAR(16) DEFAULT 'active',
        statement TEXT NOT NULL,
        scope JSON,
        evidence JSON,
        stats JSON,
        promotion_criteria JSON,
        decay_class VARCHAR(20) NOT NULL,
        expires_at DATE,
        review_after DATE,
        supersedes INTEGER,
        superseded_by INTEGER,
        created_at TIMESTAMP DEFAULT NOW(),
        created_run_id VARCHAR(40),
        updated_at TIMESTAMP DEFAULT NOW(),
        updated_run_id VARCHAR(40)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_claims_status ON desk_claims (account, status)",
    "ALTER TABLE desk_claims ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_claim_events (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        claim_id INTEGER NOT NULL,
        ts TIMESTAMP DEFAULT NOW(),
        run_id VARCHAR(40),
        event VARCHAR(20) NOT NULL,
        detail JSON
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_claim_events_claim ON desk_claim_events (claim_id, ts)",
    "ALTER TABLE desk_claim_events ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_commitments (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40) NOT NULL,
        symbol VARCHAR(24) NOT NULL,
        kind VARCHAR(16) NOT NULL,
        direction VARCHAR(6),
        level FLOAT,
        until DATE,
        text TEXT NOT NULL,
        status VARCHAR(12) DEFAULT 'open',
        fired_date DATE,
        fired_close FLOAT,
        honored_run_id VARCHAR(40),
        watch_id INTEGER,
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_commit_status ON desk_commitments (account, status)",
    "ALTER TABLE desk_commitments ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_proposals (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        created_at TIMESTAMP DEFAULT NOW(),
        run_id VARCHAR(40),
        title VARCHAR(160) NOT NULL,
        body TEXT NOT NULL,
        claim_ids JSON,
        change_kind VARCHAR(16) NOT NULL,
        payload JSON,
        status VARCHAR(12) DEFAULT 'pending',
        decided_at TIMESTAMP,
        decided_by VARCHAR(60),
        decided_via VARCHAR(12),
        applied_run_id VARCHAR(40),
        expires_at DATE
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_proposals_status ON desk_proposals (account, status)",
    "ALTER TABLE desk_proposals ENABLE ROW LEVEL SECURITY",
]
