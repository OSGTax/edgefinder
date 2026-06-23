"""EdgeFinder v2 — All tunable parameters.

Every value can be overridden via environment variable with EDGEFINDER_ prefix.
Example: EDGEFINDER_STARTING_CAPITAL=10000
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EDGEFINDER_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── ACCOUNT ──────────────────────────────────────
    starting_capital: float = 10_000.00

    # ── RISK ─────────────────────────────────────────
    # Drawdown level treated as account-critical by the watchdog checks.
    drawdown_circuit_breaker_pct: float = 0.20
    # Minimum target distance as a percentage of entry price. Floors the
    # ATR-based target so signal targets aim for substantive wins instead
    # of micro-moves driven by tiny intraday ATR (signals/engine.py).
    signal_min_target_pct_day: float = 0.01    # 1% min target for DAY trades
    signal_min_target_pct_swing: float = 0.02  # 2% min target for SWING trades

    # ── SCANNER FILTERS ──────────────────────────────
    scanner_min_market_cap: int = 300_000_000
    scanner_max_market_cap: int = 200_000_000_000
    scanner_min_avg_volume: int = 500_000
    scanner_min_price: float = 5.00
    scanner_max_price: float = 500.00
    scanner_excluded_sectors: list[str] = Field(default=["Utilities"])
    # Unified scanner pre-filter: top N stocks by yesterday's dollar volume
    # (volume * close). Cuts the universe from ~5000 common stocks to the
    # most liquid names that strategies actually trade.
    scanner_max_universe_size: int = 1000
    # Concurrent fetches in UnifiedScanner Pass 1. Polygon Starter plan
    # is "unlimited" calls so 10 workers ~10x the throughput of sequential.
    scanner_concurrent_workers: int = 10
    # Commit cadence for incremental persistence — survives mid-scan deploys.
    scanner_commit_batch_size: int = 100

    # ── SIGNALS / TECHNICAL ──────────────────────────
    signal_ema_fast_day: int = 9
    signal_ema_slow_day: int = 21
    signal_ema_fast_swing: int = 50
    signal_ema_slow_swing: int = 200
    signal_rsi_period: int = 14
    signal_rsi_oversold: int = 30
    signal_rsi_overbought: int = 70
    signal_macd_fast: int = 12
    signal_macd_slow: int = 26
    signal_macd_signal: int = 9
    signal_volume_spike_multiplier: float = 1.5
    signal_volume_avg_period: int = 20
    signal_bb_period: int = 20
    signal_bb_std: float = 2.0
    signal_bb_squeeze_threshold: float = 0.04
    signal_adx_period: int = 14
    signal_adx_strong_trend: float = 25.0
    signal_stoch_rsi_period: int = 14
    signal_stoch_rsi_k: int = 3
    signal_stoch_rsi_d: int = 3
    signal_williams_r_period: int = 14
    signal_atr_period: int = 14
    signal_confidence_low: int = 40
    signal_confidence_moderate: int = 60
    signal_confidence_high: int = 80
    signal_min_confidence_to_trade: int = 60

    # ── EARNINGS ─────────────────────────────────────
    earnings_blackout_days: int = 5

    # ── SCHEDULING (ET timezone) ─────────────────────
    scanner_run_time: str = "18:15"
    market_open_et: str = "09:30"
    market_close_et: str = "16:00"

    # ── LIVENESS WATCHDOG ────────────────────────────
    # Detect a stalled v2 portfolio cycle (daily cadence, 9:45 AM ET). The
    # cycle writes a heartbeat each run; check_cycle_liveness raises CRITICAL
    # on weekdays after 10:00 ET if the heartbeat is missing, errored, or
    # older than this many hours (26h tolerates daily jitter but catches a
    # fully missed day).
    liveness_stale_hours: int = 26
    liveness_enabled: bool = True

    # ── DATA SOURCE (Polygon.io) ─────────────────────
    polygon_api_key: str = ""
    polygon_base_url: str = "https://api.polygon.io"
    polygon_ws_url: str = "wss://socket.polygon.io/stocks"
    polygon_request_timeout: int = 15
    polygon_max_retries: int = 3
    polygon_retry_delay: float = 1.0

    # ── LIVE BROKER (Alpaca paper + real-time data) ──
    # The account of record for live paper trading and the real-time quote
    # source. Keys are paper keys; data feed "sip" = full consolidated tape
    # (paid), "iex" = free single-exchange. broker.py also accepts the SDK's
    # native APCA_API_KEY_ID / APCA_API_SECRET_KEY env vars as a fallback.
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_paper: bool = True
    alpaca_data_feed: str = "sip"
    live_trading_enabled: bool = False  # master flag for the live engine (Phase 5 flip)

    # ── FLAT FILES (Massive/Polygon S3) ──────────────
    # Bulk historical aggregates served as gzipped CSV over an S3-compatible
    # API. Used for daily-bar backfill and minute-bar backtesting — NOT for
    # live intraday data (files land T+1). Credentials via EDGEFINDER_POLYGON_S3_*.
    polygon_s3_endpoint_url: str = "https://files.massive.com"
    polygon_s3_access_key_id: str = ""
    polygon_s3_secret_access_key: str = ""
    polygon_s3_bucket: str = "flatfiles"
    flatfiles_stocks_prefix: str = "us_stocks_sip"

    # ── EOD TRIGGER ──────────────────────────────────
    # Shared secret for the POST /api/admin/run-eod endpoint, which an external
    # scheduler (GitHub Actions cron) hits after market close to run the nightly
    # scan + EOD jobs and wake the idle web service. Empty ⇒ endpoint disabled.
    # Set via EDGEFINDER_EOD_TRIGGER_TOKEN.
    eod_trigger_token: str = ""

    # ── CACHE ────────────────────────────────────────
    cache_dir: Path = Path("data/cache")
    # Intraday TTLs are deliberately shorter than the signal-check interval
    # (5 min) so each signal check fetches fresh bars instead of returning
    # the same parquet file barely-not-yet-expired from the previous cycle.
    cache_bars_ttl_minutes: dict[str, int] = Field(default={
        "1": 1, "5": 1, "15": 5, "60": 30, "day": 1080,
    })
    cache_fundamentals_ttl_hours: int = 24
    cache_profile_ttl_days: int = 7

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
    index_symbols: list[str] = Field(default=["SPY", "QQQ", "IWM", "DIA"])
    vix_symbol: str = "VIX"
    sector_etfs: list[str] = Field(default=[
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLRE", "XLC", "XLB",
    ])


# Singleton — import this everywhere
settings = Settings()
