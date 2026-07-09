"""EdgeFinder — runtime configuration for the trading-desk agent.

Every value can be overridden via environment variable with the
``EDGEFINDER_`` prefix. Example::

    EDGEFINDER_STARTING_CAPITAL=100000

Post-cutover (2026-06-22): this file is a museum-free version of the
retired trading-workbench settings. The old Polygon knobs, scanner
filters, scheduler cron times, liveness watchdog, and cache TTLs are
gone with the code that read them. The surviving fields are what the
agent actually consults at runtime. Indicator parameters live next to
their consumer — ``edgefinder/data/indicator_engine.py`` — as module
constants, not settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EDGEFINDER_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── ACCOUNT ──────────────────────────────────────
    starting_capital: float = 100_000.00

    # ── LIVE BROKER (Alpaca paper + real-time data) ──
    # The account of record for live paper trading and the real-time quote
    # source. Keys are paper keys; data feed "sip" = full consolidated tape
    # (paid), "iex" = free single-exchange. broker.py also accepts the SDK's
    # native APCA_API_KEY_ID / APCA_API_SECRET_KEY env vars as a fallback.
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_paper: bool = True
    alpaca_data_feed: str = "sip"

    # ── LIVE QUOTE STREAM (Render → /desk SSE) ───────
    # Held names are added to this seed universe automatically at stream start.
    stream_symbols: str = "SPY,QQQ,IWM,NVDA,AAPL,MSFT,AMZN,GOOGL,META,TSLA"
    # A single live-quote consumer (the /desk header pill, SSE stream, WS
    # cache) reads this. 5 s means "a healthy tape refreshes at least once
    # every 5 seconds" — anything longer is a genuine feed hiccup, not
    # normal quiet-period pacing.
    stream_stale_secs: int = 5

    # ── DATABASE ─────────────────────────────────────
    database_url: str = "sqlite:///data/edgefinder.db"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_recycle: int = 300  # seconds; Supabase Supavisor retires idle connections
    db_connect_timeout: int = 8  # seconds; fail fast if the Postgres port is blocked
    # Persistence transport for the agent tools: "pg" (SQLAlchemy/Postgres),
    # "rest" (Supabase Data API over HTTPS — the web Routine sandbox blocks the
    # Postgres port), or "auto" (rest iff SUPABASE_URL + service-role key set).
    db_transport: str = "auto"

    # ── LOGGING ──────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "data/edgefinder.log"

    # ── MARKET SNAPSHOT SYMBOLS ──────────────────────
    # SPY/QQQ/IWM comparison series on the desk's regime read (agent.market).
    index_symbols: list[str] = Field(default=["SPY", "QQQ", "IWM", "DIA"])


# Singleton — import this everywhere
settings = Settings()
