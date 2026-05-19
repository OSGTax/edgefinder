"""Tests for the enriched snapshot provider."""

from unittest.mock import MagicMock
from edgefinder.data.snapshot_provider import get_enriched_snapshots


class TestEnrichedSnapshots:
    def test_returns_price_and_volume(self):
        mock_provider = MagicMock()
        snap = MagicMock()
        snap.ticker = "AAPL"
        snap.day = MagicMock()
        snap.day.close = 150.0
        snap.day.volume = 50_000_000
        snap.day.open = 148.0
        snap.day.high = 151.0
        snap.day.low = 147.0
        snap.prev_day = MagicMock()
        snap.prev_day.close = 148.0

        mock_provider._client.get_snapshot_all.return_value = [snap]

        result = get_enriched_snapshots(mock_provider)
        assert "AAPL" in result
        assert result["AAPL"]["price"] == 150.0
        assert result["AAPL"]["volume"] == 50_000_000

    def test_handles_missing_day_data(self):
        mock_provider = MagicMock()
        snap = MagicMock()
        snap.ticker = "AAPL"
        snap.day = None
        snap.prev_day = MagicMock()
        snap.prev_day.close = 148.0

        mock_provider._client.get_snapshot_all.return_value = [snap]

        result = get_enriched_snapshots(mock_provider)
        assert "AAPL" in result
        assert result["AAPL"]["price"] == 148.0
        assert result["AAPL"]["volume"] == 0

    def test_returns_empty_on_failure(self):
        mock_provider = MagicMock()
        mock_provider._client.get_snapshot_all.side_effect = Exception("API error")

        result = get_enriched_snapshots(mock_provider)
        assert result == {}
