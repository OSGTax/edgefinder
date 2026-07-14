"""EdgeFinder — SQLAlchemy 2.0 ORM models.

Post-cutover (2026-06-22): only the tables the trading-desk runtime
actually reads or writes live here. The retired trading-workbench models
(TradeRecord, StrategyAccount, StrategySnapshot, ValidationRun,
PromotedStrategy, FundamentalsSnapshot, DividendCredit, SystemHeartbeat,
ManualInjection, StrategyParameterLog, TickerStrategyQualification,
TradeContext, AgentObservation, AgentAction, AgentMemory,
LLMDecisionCache, LLMDecisionLog, AgentDecision, MarketSnapshotRecord,
Ticker, Fundamental, TickerDividend) were dropped in the cutover — the
runtime never imported them again, and the tests that exercised them
were retired with them.

Surviving models — each is used by ``agent/`` or ``dashboard/`` on the
live path:

- ``DailyBar``           — historical OHLCV (Alpaca ingest, R2 archive,
                            engine backtests, chart page)
- ``IndexDaily``         — benchmark index history (SPY/QQQ/IWM/DIA)
- ``TickerNews``         — news feed (Alpaca news, agent.refresh writes,
                            /desk + /symbol read)
- ``TickerSplit``        — stock splits (agent.refresh writes,
                            engine.data adjusts for)
- ``DividendRecord``     — cash dividends (agent.refresh writes,
                            engine.data adjusts for, /desk shows on
                            symbol events)

The desk_* tables live in ``agent/models.py`` — that's where the
autonomous agent's own book, thinking, and decisions persist.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
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


class DailyBar(Base):
    """Daily OHLCV bars — the historical data asset.

    One row per (symbol, date). Populated by ``agent.refresh`` from Alpaca
    daily bars (source="alpaca_daily"); older rows may carry legacy
    ``source`` values from pre-cutover backfills. Bars are dividend/split
    UNadjusted; total-return and split adjustment are load-time transforms
    in ``edgefinder/engine/data.py`` so the raw record stays immutable.
    ``volume`` is Float because pre-cutover flat-file sources reported
    volume-weighted share counts.
    """

    __tablename__ = "daily_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_daily_bars_symbol_date"),
        Index("idx_daily_bars_symbol_date", "symbol", "date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    transactions: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(20), default="alpaca_daily")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class IndexDaily(Base):
    """Benchmark index history (SPY / QQQ / IWM / DIA) — the regime read
    on ``/desk`` and the comparison line on backtests. Sparse: only the
    close and day-over-day change_pct are stored."""

    __tablename__ = "index_daily"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_index_daily_symbol_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[datetime] = mapped_column(DateTime)
    close: Mapped[float] = mapped_column(Float)
    change_pct: Mapped[float] = mapped_column(Float)


class TickerNews(Base):
    """News articles for tickers, accumulated over time from Alpaca's
    Benzinga feed via ``agent.refresh``. One row per unique (symbol,
    title, published_utc). Rendered as chart-event markers on the symbol
    page and as the "why now" text on the agent's research pass."""

    __tablename__ = "ticker_news"
    __table_args__ = (
        UniqueConstraint("symbol", "title", "published_utc", name="uq_ticker_news"),
        Index("idx_ticker_news_symbol", "symbol"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10))
    title: Mapped[str] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(200))
    published_utc: Mapped[str | None] = mapped_column(String(30))
    article_url: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    publisher_name: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TickerSplit(Base):
    """Stock splits — written by ``agent.refresh`` from Alpaca corporate
    announcements, consumed by ``edgefinder.engine.data.adjust_for_splits``
    to keep pre-split bars in post-split units."""

    __tablename__ = "ticker_splits"
    __table_args__ = (
        UniqueConstraint("symbol", "execution_date", name="uq_ticker_split"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    execution_date: Mapped[str] = mapped_column(String(20))
    split_from: Mapped[int | None] = mapped_column(Integer)
    split_to: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FundamentalsSnapshot(Base):
    """FROZEN Polygon-era fundamentals snapshots (through 2026-06-10).

    ORM restored (it was dropped with the workbench purge) because the table
    is the VALIDATION REFERENCE for the EDGAR pipeline: 128k vendor-computed
    rows to cross-check our computed values against. Read-only forever; the
    vendor product that fed it was sunset 2026-06-22.
    """

    __tablename__ = "fundamentals_snapshots"
    __table_args__ = (
        UniqueConstraint("symbol", "as_of", name="uq_fund_snap_symbol_asof"),
        Index("idx_fund_snap_symbol_asof", "symbol", "as_of"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    data: Mapped[dict] = mapped_column(JSON)


class FundamentalsPit(Base):
    """Point-in-time fundamentals from SEC EDGAR — one row per FILING.

    ``data`` holds PRICE-FREE values knowable at ``filed`` (EPS-TTM, ROE,
    debt/equity, growth, plus `_`-prefixed ratio ingredients — shares, TTM
    revenue/income/FCF/EBITDA, book equity, debt, cash). Price-dependent
    ratios are computed at decision/brief time against a price the caller
    vouches for. Written only by ``agent.edgar`` (nightly via the refresh);
    the frozen Polygon-era ``fundamentals_snapshots`` table stays untouched
    as the validation reference. Public-domain source: display anything.
    """

    __tablename__ = "fundamentals_pit"
    __table_args__ = (
        UniqueConstraint("symbol", "filed", "period_end",
                         name="uq_fund_pit_symbol_filed_period"),
        Index("idx_fund_pit_symbol_filed", "symbol", "filed"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    cik: Mapped[int | None] = mapped_column(Integer)
    filed: Mapped[date] = mapped_column(Date, index=True)
    period_end: Mapped[date | None] = mapped_column(Date)
    form: Mapped[str | None] = mapped_column(String(12))
    source: Mapped[str] = mapped_column(String(12), default="edgar")
    data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DividendRecord(Base):
    """Cash dividends per (symbol, ex_date) — fuel for total-return
    adjustment and the desk/symbol page's dividend event markers. Written
    by ``agent.refresh`` from Alpaca corporate announcements; raw bars
    stay dividend-UNadjusted everywhere (DB and R2), so this row stays
    immutable and adjustments can be revised/audited freely."""

    __tablename__ = "dividends"
    __table_args__ = (
        UniqueConstraint("symbol", "ex_date", name="uq_dividends_symbol_exdate"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    ex_date: Mapped[date] = mapped_column(Date, index=True)
    cash_amount: Mapped[float] = mapped_column(Float)
