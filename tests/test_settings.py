"""Tests for config/settings.py — kept to the fields the runtime actually reads.

Post-cutover: the settings surface was pruned from ~50 fields to ~15. The
retired Polygon knobs, scanner filters, scheduler cron times, liveness
watchdog, and cache TTL dict are gone with the code that read them.
Indicator parameters (rsi_period, macd_*, bb_*, etc.) moved into
``edgefinder.data.indicator_engine`` as module constants.
"""

from config.settings import Settings, settings


def test_defaults_load():
    """Settings loads with all defaults without error, and the values
    the ledger + broker rely on carry the right defaults."""
    s = Settings()
    assert s.starting_capital == 100_000.00
    assert s.alpaca_paper is True
    assert s.alpaca_data_feed == "sip"
    assert s.stream_stale_secs == 15


def test_env_override(monkeypatch):
    """Environment variables override defaults via EDGEFINDER_ prefix."""
    monkeypatch.setenv("EDGEFINDER_STARTING_CAPITAL", "250000")
    monkeypatch.setenv("EDGEFINDER_ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("EDGEFINDER_ALPACA_API_SECRET", "test_secret")
    s = Settings()
    assert s.starting_capital == 250_000.00
    assert s.alpaca_api_key == "test_key"
    assert s.alpaca_api_secret == "test_secret"


def test_type_coercion(monkeypatch):
    """String env vars are coerced to the declared type."""
    monkeypatch.setenv("EDGEFINDER_STREAM_STALE_SECS", "45")
    monkeypatch.setenv("EDGEFINDER_ALPACA_PAPER", "false")
    s = Settings()
    assert s.stream_stale_secs == 45 and isinstance(s.stream_stale_secs, int)
    assert s.alpaca_paper is False


def test_index_symbols_defaults():
    """Regime read on the desk uses this list — order matters."""
    assert settings.index_symbols[:3] == ["SPY", "QQQ", "IWM"]


def test_retired_fields_are_gone():
    """The Polygon-era knobs should not be reachable via Settings anymore —
    guards against a future revert that quietly resurrects them."""
    s = Settings()
    for retired in ("polygon_api_key", "scanner_max_universe_size",
                    "market_open_et", "liveness_stale_hours",
                    "signal_rsi_period", "cache_bars_ttl_minutes",
                    "vix_symbol", "eod_trigger_token", "live_trading_enabled"):
        assert not hasattr(s, retired), f"Settings still exposes retired field: {retired}"
