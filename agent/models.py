"""The autonomous trading agent's own tables (greenfield rebuild).

A clean namespace (``desk_*``) that lives in the SAME database as the kept
market-data tables but shares none of the retired trading/app schema. The
agent reads and writes only these; the market-data tables (daily_bars,
dividends, ticker_splits, fundamentals_snapshots, ticker_news, index_daily)
are read-only inputs reached through the kept data-access layer.

Source-of-truth rule (REBUILD-V4): the Alpaca PAPER account is the book of
record. ``desk_orders`` / ``desk_activities`` / ``desk_portfolio_history``
mirror it locally (Alpaca's retention of closed history is undocumented);
the mirror is a cache re-synced every cycle, never the arbiter. The frozen
V3 book lives on as ``era1_trades``/``era1_positions``/``era1_equity``
(renamed at cutover — deliberately absent from this module so nothing can
write them).
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


# ── V4 mirror tables — Alpaca is the book of record; these are OUR copy ──
#
# REBUILD-V4: orders execute on Alpaca's paper account. Alpaca's retention of
# closed orders/activities is undocumented, so every order, fill activity, and
# daily portfolio snapshot is mirrored locally the moment we see it and
# archived to R2 nightly. The mirror is a cache/archive, never the arbiter —
# on any conflict, Alpaca wins and the mirror is re-synced.


class DeskOrder(Base):
    """One row per Alpaca order — and one per mleg LEG (legs carry
    ``parent_order_id`` and inherit the parent's run_id/seq).

    ``client_order_id`` is the attribution carrier: ``<run_id>:<seq>``.
    The knowledge loop's ``(run_id, symbol)`` joins recover the symbol from
    this row, never from the id string (OCC and BTC/USD need no escaping).
    Alpaca timestamps are stored as ISO-8601 text — lexicographic order IS
    chronological order, and both transports pass them through untouched.
    """

    __tablename__ = "desk_orders"
    __table_args__ = (
        UniqueConstraint("alpaca_order_id", name="uq_desk_orders_alpaca_id"),
        Index("idx_desk_orders_run", "account", "run_id"),
        Index("idx_desk_orders_symbol", "account", "symbol"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str | None] = mapped_column(String(48), index=True)
    seq: Mapped[int | None] = mapped_column(Integer)
    client_order_id: Mapped[str | None] = mapped_column(String(128))  # null on legs
    alpaca_order_id: Mapped[str] = mapped_column(String(64))
    parent_order_id: Mapped[str | None] = mapped_column(String(64))  # set on legs
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    asset_class: Mapped[str | None] = mapped_column(String(12))  # us_equity | us_option | crypto
    side: Mapped[str | None] = mapped_column(String(4))  # buy | sell
    kind: Mapped[str | None] = mapped_column(String(8))  # entry | exit | stop
    order_type: Mapped[str | None] = mapped_column(String(14))
    tif: Mapped[str | None] = mapped_column(String(6))
    order_class: Mapped[str | None] = mapped_column(String(10))
    limit_price: Mapped[float | None] = mapped_column(Float)
    stop_price: Mapped[float | None] = mapped_column(Float)
    qty: Mapped[float | None] = mapped_column(Float)
    notional: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str | None] = mapped_column(String(24))
    filled_qty: Mapped[float | None] = mapped_column(Float)
    filled_avg_price: Mapped[float | None] = mapped_column(Float)
    submitted_at: Mapped[str | None] = mapped_column(String(40))  # ISO text
    filled_at: Mapped[str | None] = mapped_column(String(40))
    canceled_at: Mapped[str | None] = mapped_column(String(40))
    raw: Mapped[dict | None] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())


class DeskActivity(Base):
    """Append-only mirror of Alpaca account activities (FILL, SSP splits,
    OPASN/OPEXP option events — both OPEXC and OPXRC exercise codes — CFEE,
    …), cursor-synced by unique ``alpaca_activity_id``. On paper, option
    non-trade activities land T+1 — grading never waits on same-day rows."""

    __tablename__ = "desk_activities"
    __table_args__ = (
        UniqueConstraint("alpaca_activity_id", name="uq_desk_activities_alpaca_id"),
        Index("idx_desk_activities_date", "account", "date"),
        Index("idx_desk_activities_symbol", "account", "symbol"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    alpaca_activity_id: Mapped[str] = mapped_column(String(64))
    activity_type: Mapped[str] = mapped_column(String(12), index=True)
    date: Mapped[str | None] = mapped_column(String(10))  # YYYY-MM-DD
    symbol: Mapped[str | None] = mapped_column(String(24))
    side: Mapped[str | None] = mapped_column(String(10))
    qty: Mapped[float | None] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float)
    net_amount: Mapped[float | None] = mapped_column(Float)
    alpaca_order_id: Mapped[str | None] = mapped_column(String(64))
    raw: Mapped[dict | None] = mapped_column(JSON)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DeskPortfolioSnapshot(Base):
    """One nightly snapshot per trading day: account equity + the positions
    map. Durability (portfolio-history survives nothing we don't copy) plus
    the split-reconciliation guard's baseline — yesterday's qty × split ratio
    must equal today's Alpaca qty, or the divergence is journaled loudly."""

    __tablename__ = "desk_portfolio_history"
    __table_args__ = (
        UniqueConstraint("account", "snap_date", name="uq_desk_pf_hist_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    snap_date: Mapped[str] = mapped_column(String(10))  # ET trading date
    equity: Mapped[float | None] = mapped_column(Float)
    cash: Mapped[float | None] = mapped_column(Float)
    profit_loss: Mapped[float | None] = mapped_column(Float)
    base_value: Mapped[float | None] = mapped_column(Float)
    positions: Mapped[dict | None] = mapped_column(JSON)  # symbol → {qty, avg_entry_price}
    captured_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# Idempotent CREATE TABLE IF NOT EXISTS DDL for render_start.py (Render skips
# create_all). Postgres-flavored; SQLite ignores the JSON type harmlessly.
DESK_TABLE_DDL: list[str] = [
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
    # v10.0.0 (REBUILD-V4): Alpaca-paper mirror — orders, activities, and the
    # nightly portfolio snapshot. Cache/archive of the broker's book of
    # record; Alpaca wins every conflict.
    """CREATE TABLE IF NOT EXISTS desk_orders (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(48),
        seq INTEGER,
        client_order_id VARCHAR(128),
        alpaca_order_id VARCHAR(64) NOT NULL,
        parent_order_id VARCHAR(64),
        symbol VARCHAR(24) NOT NULL,
        asset_class VARCHAR(12),
        side VARCHAR(4),
        kind VARCHAR(8),
        order_type VARCHAR(14),
        tif VARCHAR(6),
        order_class VARCHAR(10),
        limit_price FLOAT,
        stop_price FLOAT,
        qty FLOAT,
        notional FLOAT,
        status VARCHAR(24),
        filled_qty FLOAT,
        filled_avg_price FLOAT,
        submitted_at VARCHAR(40),
        filled_at VARCHAR(40),
        canceled_at VARCHAR(40),
        raw JSON,
        updated_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT uq_desk_orders_alpaca_id UNIQUE (alpaca_order_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_orders_run ON desk_orders (account, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_desk_orders_symbol ON desk_orders (account, symbol)",
    "ALTER TABLE desk_orders ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_activities (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        alpaca_activity_id VARCHAR(64) NOT NULL,
        activity_type VARCHAR(12) NOT NULL,
        date VARCHAR(10),
        symbol VARCHAR(24),
        side VARCHAR(10),
        qty FLOAT,
        price FLOAT,
        net_amount FLOAT,
        alpaca_order_id VARCHAR(64),
        raw JSON,
        ingested_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT uq_desk_activities_alpaca_id UNIQUE (alpaca_activity_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_activities_date ON desk_activities (account, date)",
    "CREATE INDEX IF NOT EXISTS idx_desk_activities_symbol ON desk_activities (account, symbol)",
    "ALTER TABLE desk_activities ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_portfolio_history (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        snap_date VARCHAR(10) NOT NULL,
        equity FLOAT,
        cash FLOAT,
        profit_loss FLOAT,
        base_value FLOAT,
        positions JSON,
        captured_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT uq_desk_pf_hist_date UNIQUE (account, snap_date)
    )""",
    "ALTER TABLE desk_portfolio_history ENABLE ROW LEVEL SECURITY",
    # agent.backup's size check on the REST lane — PostgREST cannot run
    # pg_database_size directly, so it calls this function via /rpc.
    """CREATE OR REPLACE FUNCTION edgefinder_db_size() RETURNS bigint
       LANGUAGE sql SECURITY DEFINER
       AS 'SELECT pg_database_size(current_database())'""",
]
