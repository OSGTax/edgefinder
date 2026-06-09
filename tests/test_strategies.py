"""Tests for strategy registry with new strategies (coward, gambler, degenerate)."""

import pytest

from edgefinder.strategies.base import BaseStrategy, StrategyRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is reset between tests, then reload strategies."""
    import importlib
    from edgefinder.strategies import coward, gambler, degenerate_v2

    # Setup: these tests assert exactly the 3 live strategies are registered.
    StrategyRegistry.clear()
    importlib.reload(coward)
    importlib.reload(gambler)
    importlib.reload(degenerate_v2)
    yield
    # Teardown: restore the FULL registry — research candidates included — or
    # later modules that backtest them in-process (walk-forward) hit an unknown
    # strategy. Reload every strategy submodule (robust to future additions).
    import pkgutil
    import edgefinder.strategies as pkg
    StrategyRegistry.clear()
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name in ("base", "strategy_interface"):
            continue
        importlib.reload(importlib.import_module(f"edgefinder.strategies.{m.name}"))


class TestStrategyRegistry:
    def test_register_and_get(self):
        assert StrategyRegistry.get("coward") is not None

    def test_get_all(self):
        all_strats = StrategyRegistry.get_all()
        assert "coward" in all_strats
        assert "gambler" in all_strats
        assert "degenerate" in all_strats

    def test_get_instances(self):
        instances = StrategyRegistry.get_instances()
        assert len(instances) == 3
        names = {s.name for s in instances}
        assert names == {"coward", "gambler", "degenerate"}

    def test_list_names(self):
        names = StrategyRegistry.list_names()
        assert "coward" in names

    def test_get_nonexistent(self):
        assert StrategyRegistry.get("nonexistent") is None

    def test_clear(self):
        StrategyRegistry.clear()
        assert len(StrategyRegistry.get_all()) == 0


class TestOptionalMethods:
    def test_defaults_dont_raise(self):
        s = StrategyRegistry.get("coward")()
        # SwingStrategy defaults
        assert s.watchlist_size == 50
        assert s.stop_pct == 0.20
