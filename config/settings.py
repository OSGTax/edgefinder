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
    starting_capital: float = 5_000.00
    max_open_positions: int = 5
    max_portfolio_concentration_pct: float = 0.20

    # ── PDT (defaults; overridden per-strategy in DB) ─
    pdt_day_trade_limit: int = 3
    pdt_window_days: int = 5

    # ── RISK ─────────────────────────────────────────
    max_risk_per_trade_pct: float = 0.02
    drawdown_circuit_breaker_pct: float = 0.20
    min_reward_to_risk: float = 1.5
    revenge_trade_cooldown_minutes: int = 30
    max_same_sector_positions: int = 3
    trailing_stop_activation_r: float = 1.0
    trailing_stop_trail_r: float = 2.0
    # Per-ticker re-entry cooldown after closing a position. Prevents the
    # signal check from immediately reopening the same ticker after a close.
    ticker_reentry_cooldown_minutes: int = 30
    # Minimum target distance as a percentage of entry price. Floors the
    # ATR-based target so trades aim for substantive wins instead of
    # micro-moves driven by tiny intraday ATR.
    signal_min_target_pct_day: float = 0.01    # 1% min target for DAY trades
    signal_min_target_pct_swing: float = 0.02  # 2% min target for SWING trades

    # ── SCANNER FILTERS ──────────────────────────────
    scanner_min_market_cap: int = 300_000_000
    scanner_max_market_cap: int = 200_000_000_000
    scanner_min_avg_volume: int = 500_000
    scanner_min_price: float = 5.00
    scanner_max_price: float = 500.00
    scanner_excluded_sectors: list[str] = Field(default=["Utilities"])
    scanner_batch_count: int = 5  # one batch per weekday (Mon=0 ... Fri=4)
    scanner_full_universe: bool = True  # scan all tickers nightly (unlimited API plan)
    scanner_max_watchlist_per_strategy: int = 50  # top N scored stocks per strategy

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
    signal_check_interval_minutes: int = 5
    market_open_et: str = "07:00"
    market_close_et: str = "18:00"
    position_monitor_interval_minutes: int = 5

    # ── DATA SOURCE (Polygon.io) ─────────────────────
    polygon_api_key: str = ""
    polygon_base_url: str = "https://api.polygon.io"
    polygon_ws_url: str = "wss://socket.polygon.io/stocks"
    polygon_request_timeout: int = 15
    polygon_max_retries: int = 3
    polygon_retry_delay: float = 1.0

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

    # ── SLIPPAGE MODEL ───────────────────────────────
    slippage_base_rate: float = 0.0005
    slippage_volume_factor: float = 1.5
    slippage_min_avg_volume: int = 100_000

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
