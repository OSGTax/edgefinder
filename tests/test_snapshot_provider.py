"""Tests for enriched snapshots (price + volume + OHLC for the intraday cycle).

The intraday signal cycle fetches a market-wide snapshot once per tick via
``provider.get_enriched_snapshots()``. The provider injected at runtime is a
``DataHub`` wrapping ``CachedDataProvider`` wrapping ``PolygonDataProvider`` —
only the innermost provider owns the REST ``_client``. These tests pin both the
parsing (on PolygonDataProvider) and the delegation through the wrappers, so the
call can never again reach for ``_client`` on a layer that doesn't have one.
"""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

from edgefinder.core.interfaces import DataHub
from edgefinder.data.cache import DataCache
from edgefinder.data.polygon import PolygonDataProvider
from edgefinder.data.provider import CachedDataProvider


def _provider_with_snapshots(snapshots) -> PolygonDataProvider:
    p = PolygonDataProvider(api_key="test")  # RESTClient construction is lazy — no network
    p._client = MagicMock()
    p._client.get_snapshot_all.return_value = snapshots
    return p


def _snapshot(ticker, *, day_close=None, day_volume=0, day_open=None,
              day_high=None, day_low=None, prev_close=None):
    s = MagicMock()
    s.ticker = ticker
    if day_close is None and day_open is None:
        s.day = None
    else:
        s.day = MagicMock()
        s.day.close = day_close
        s.day.volume = day_volume
        s.day.open = day_open
        s.day.high = day_high
        s.day.low = day_low
    if prev_close is None:
        s.prev_day = None
    else:
        s.prev_day = MagicMock()
        s.prev_day.close = prev_close
    return s


class TestPolygonEnrichedSnapshots:
    def test_returns_price_volume_and_ohlc(self):
        provider = _provider_with_snapshots([
            _snapshot("AAPL", day_close=150.0, day_volume=50_000_000,
                      day_open=148.0, day_high=151.0, day_low=147.0,
                      prev_close=148.0),
        ])
        result = provider.get_enriched_snapshots()
        assert result["AAPL"]["price"] == 150.0
        assert result["AAPL"]["volume"] == 50_000_000
        assert result["AAPL"]["open"] == 148.0
        assert result["AAPL"]["high"] == 151.0
        assert result["AAPL"]["low"] == 147.0

    def test_falls_back_to_prev_close_when_no_day_data(self):
        provider = _provider_with_snapshots([_snapshot("AAPL", prev_close=148.0)])
        result = provider.get_enriched_snapshots()
        assert result["AAPL"]["price"] == 148.0
        assert result["AAPL"]["volume"] == 0

    def test_returns_empty_on_api_failure(self):
        provider = _provider_with_snapshots(None)
        provider._max_retries = 1  # don't sleep/backoff in the test
        provider._client.get_snapshot_all.side_effect = Exception("API error")
        assert provider.get_enriched_snapshots() == {}


class TestEnrichedSnapshotDelegation:
    """Regression: the runtime provider is a DataHub with NO ``_client``.

    The original bug reached ``provider._client.get_snapshot_all`` on the
    DataHub, which raised AttributeError, was swallowed, and returned ``{}``
    every cycle — so no trade ever fired. These tests prove the call now
    flows through proper interface methods at every wrapper layer.
    """

    _SAMPLE = {"AAPL": {"price": 150.0, "volume": 1.0, "open": 149.0,
                        "high": 151.0, "low": 148.0}}

    def test_datahub_has_no_client_but_delegates(self):
        primary = MagicMock()
        primary.get_enriched_snapshots.return_value = self._SAMPLE
        hub = DataHub(primary)
        assert not hasattr(hub, "_client")
        assert hub.get_enriched_snapshots() == self._SAMPLE
        primary.get_enriched_snapshots.assert_called_once()

    def test_cached_provider_delegates_without_caching(self):
        inner = MagicMock()
        inner.get_enriched_snapshots.return_value = self._SAMPLE
        cached = CachedDataProvider(inner, MagicMock(spec=DataCache))
        assert cached.get_enriched_snapshots() == self._SAMPLE
        inner.get_enriched_snapshots.assert_called_once()

    def test_full_stack_returns_data(self):
        polygon = _provider_with_snapshots([
            _snapshot("AAPL", day_close=150.0, day_volume=10, day_open=149.0,
                      day_high=151.0, day_low=148.0),
        ])
        hub = DataHub(CachedDataProvider(polygon, MagicMock(spec=DataCache)))
        result = hub.get_enriched_snapshots()
        assert result["AAPL"]["price"] == 150.0
        assert result["AAPL"]["volume"] == 10


def _minute_bar(d: date, hour=15, minute=30, *, close, open_=None,
                high=None, low=None, volume=0):
    """A Polygon/Massive minute aggregate landing on calendar date ``d``.

    Timestamp built with the same local-clock basis ``_snapshot_from_minutes``
    uses to derive a bar's date, so the round-trip lands on ``d`` in CI.
    """
    a = MagicMock()
    a.timestamp = int(datetime(d.year, d.month, d.day, hour, minute).timestamp() * 1000)
    a.close = close
    a.open = open_ if open_ is not None else close
    a.high = high if high is not None else close
    a.low = low if low is not None else close
    a.volume = volume
    return a


class TestSnapshotFromMinutesFallback:
    """Per-ticker minute-agg fallback — the path the live system depends on
    because the bulk snapshot endpoint is empty on this data plan.

    Regression (v5.3.6 → v5.3.8): the fetch window was narrowed to today
    only, so any ticker without *today's* intraday bars (low liquidity,
    feed delay, an unpublished current session) returned no aggregates and
    was dropped — collapsing the watchlist to ``snapshot_count: 0`` and
    halting all trading. The window must look back far enough to capture a
    prior session so the most-recent bar can stand in.
    """

    def _provider(self, aggs):
        p = PolygonDataProvider(api_key="test")  # lazy client — no network
        p._client = MagicMock()
        p._client.get_aggs.return_value = aggs
        return p

    def test_fetch_window_spans_multiple_days(self):
        # The regression lives in the *requested window*, not the parsing:
        # assert the call looks back several days, ending today.
        p = self._provider([_minute_bar(date.today() - timedelta(days=1), close=100.0)])
        p._snapshot_from_minutes("DBC")
        _, kwargs = p._client.get_aggs.call_args
        frm = date.fromisoformat(kwargs["from_"])
        to = date.fromisoformat(kwargs["to"])
        assert to == date.today()
        assert (to - frm).days >= 3, (
            "snapshot window must look back several days so a ticker with no "
            "bars today falls back to a prior session instead of being "
            "dropped (v5.3.8 regression guard)"
        )

    def test_uses_most_recent_bar_when_no_today_bars(self):
        # Today empty; only prior-session bars exist → still produce a snapshot.
        yest = date.today() - timedelta(days=1)
        older = date.today() - timedelta(days=2)
        p = self._provider([
            _minute_bar(older, close=98.0),
            _minute_bar(yest, hour=15, minute=58, close=101.5, volume=10),
        ])
        snap = p._snapshot_from_minutes("DBC")
        assert snap is not None
        assert snap["price"] == 101.5

    def test_prefers_today_bars_when_present(self):
        today = date.today()
        yest = date.today() - timedelta(days=1)
        p = self._provider([
            _minute_bar(yest, close=200.0, volume=999),       # stale, must be ignored
            _minute_bar(today, hour=9, minute=30, close=150.0, volume=5),
            _minute_bar(today, hour=15, minute=59, close=152.0, volume=7),
        ])
        snap = p._snapshot_from_minutes("AAPL")
        assert snap["price"] == 152.0          # latest *today* close
        assert snap["volume"] == 12            # sum of today's bars only, not stale

    def test_returns_none_when_no_bars_at_all(self):
        p = self._provider([])
        assert p._snapshot_from_minutes("ZZZZ") is None
