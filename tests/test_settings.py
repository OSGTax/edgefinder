"""Tests for config/settings.py."""

import os
from config.settings import Settings


def test_defaults_load():
    """Settings loads with all defaults without error."""
    s = Settings(polygon_api_key="test")
    assert s.starting_capital == 5_000.00
    assert s.max_open_positions == 5
    assert s.max_risk_per_trade_pct == 0.02
    assert s.pdt_day_trade_limit == 3


def test_env_override(monkeypatch):
    """Environment variables override defaults via EDGEFINDER_ prefix."""
    monkeypatch.setenv("EDGEFINDER_STARTING_CAPITAL", "10000")
    monkeypatch.setenv("EDGEFINDER_POLYGON_API_KEY", "test_key")
    s = Settings()
    assert s.starting_capital == 10_000.00
    assert s.polygon_api_key == "test_key"


def test_type_coercion(monkeypatch):
    """String env vars are coerced to correct types."""
    monkeypatch.setenv("EDGEFINDER_MAX_OPEN_POSITIONS", "10")
    monkeypatch.setenv("EDGEFINDER_POLYGON_API_KEY", "test")
    s = Settings()
    assert s.max_open_positions == 10
    assert isinstance(s.max_open_positions, int)


def test_signal_check_interval():
    """Signal check interval should be 5 minutes."""
    s = Settings(polygon_api_key="test")
    assert s.signal_check_interval_minutes == 5


def test_cache_ttl_defaults():
    """Cache TTL dict has expected timeframe keys."""
    s = Settings(polygon_api_key="test")
    assert "day" in s.cache_bars_ttl_minutes
    assert "1" in s.cache_bars_ttl_minutes
    assert s.cache_bars_ttl_minutes["day"] == 1080
