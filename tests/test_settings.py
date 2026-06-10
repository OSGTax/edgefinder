"""Tests for config/settings.py."""

import os
from config.settings import Settings


def test_defaults_load():
    """Settings loads with all defaults without error."""
    s = Settings(polygon_api_key="test")
    assert s.starting_capital == 10_000.00
    assert s.drawdown_circuit_breaker_pct == 0.20
    assert s.liveness_stale_hours == 26


def test_env_override(monkeypatch):
    """Environment variables override defaults via EDGEFINDER_ prefix."""
    monkeypatch.setenv("EDGEFINDER_STARTING_CAPITAL", "10000")
    monkeypatch.setenv("EDGEFINDER_POLYGON_API_KEY", "test_key")
    s = Settings()
    assert s.starting_capital == 10_000.00
    assert s.polygon_api_key == "test_key"


def test_type_coercion(monkeypatch):
    """String env vars are coerced to correct types."""
    monkeypatch.setenv("EDGEFINDER_LIVENESS_STALE_HOURS", "48")
    monkeypatch.setenv("EDGEFINDER_POLYGON_API_KEY", "test")
    s = Settings()
    assert s.liveness_stale_hours == 48
    assert isinstance(s.liveness_stale_hours, int)


def test_cache_ttl_defaults():
    """Cache TTL dict has expected timeframe keys."""
    s = Settings(polygon_api_key="test")
    assert "day" in s.cache_bars_ttl_minutes
    assert "1" in s.cache_bars_ttl_minutes
    assert s.cache_bars_ttl_minutes["day"] == 1080


def test_scanner_and_schedule_settings():
    from config.settings import settings
    assert settings.market_open_et == "09:30"
    assert settings.market_close_et == "16:00"
    assert settings.scanner_run_time == "18:15"
    assert settings.scanner_min_price == 5.00
    assert settings.scanner_max_price == 500.00
    assert settings.scanner_max_universe_size == 1000
