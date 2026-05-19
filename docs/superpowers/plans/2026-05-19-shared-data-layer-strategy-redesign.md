# Shared Data Layer & Strategy Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the entire strategy layer with a shared data layer, new risk system, and three new strategies (Coward, Gambler, Degenerate) that write their own entry/exit logic against raw market data.

**Architecture:** A shared `MarketData` object is computed once per ticker (daily indicators recomputed with provisional close every 5 min). Strategies receive this object and return `TradeIntent` / `ExitIntent`. A centralized risk system handles sizing (per-strategy risk %), stops (fixed 20%), and targets. Old strategies are removed entirely.

**Tech Stack:** Python 3.11+, pandas, pytest, SQLAlchemy 2.0, existing Polygon/Massive REST API.

**Spec:** `docs/superpowers/specs/2026-05-19-shared-data-layer-strategy-redesign.md`

**Key reference files to read before starting any task:**
- `edgefinder/signals/engine.py` — existing indicator math (reuse `_ema`, `_rsi`, `_macd`, `_bollinger_bands`, `_atr`, `_adx`, `_stochastic_rsi`, `_williams_r`)
- `edgefinder/trading/account.py` — `Position` and `VirtualAccount` (modified in Task 5)
- `edgefinder/trading/executor.py` — current executor (replaced in Task 6)
- `edgefinder/trading/arena.py` — current arena (rewritten in Task 9)
- `dashboard/services.py` — service wiring (modified in Task 11)
- `config/settings.py` — settings (modified in Task 4)
- `edgefinder/core/models.py` — domain models (extended in Task 2)

---

## File Structure

### New files (create in order)

| File | Responsibility |
|------|---------------|
| `edgefinder/data/market_data.py` | `IndicatorSnapshot`, `IndicatorHistory` (30-day buffer), `MarketData` object, `MarketContext` |
| `edgefinder/data/indicator_engine.py` | `compute_indicators_from_bars()` — pure function, reuses math from `signals/engine.py` |
| `edgefinder/data/snapshot_provider.py` | `get_enriched_snapshots()` — returns `{ticker: {price, volume}}` from bulk API |
| `edgefinder/trading/risk.py` | `RiskManager` — sizing, stop/target computation, account checks |
| `edgefinder/strategies/strategy_interface.py` | `SwingStrategy` base class, `TradeIntent`, `ExitIntent` |
| `edgefinder/strategies/coward.py` | Coward strategy |
| `edgefinder/strategies/gambler.py` | Gambler strategy |
| `edgefinder/strategies/degenerate_v2.py` | Degenerate strategy |

### Modified files

| File | Changes |
|------|---------|
| `config/settings.py` | Add new risk/schedule settings, update market hours |
| `edgefinder/trading/account.py` | Remove `max_open_positions` hard limit, add PDT day-trade flagging |
| `edgefinder/trading/executor.py` | Accept `TradeIntent` + `RiskManager` instead of `Signal` |
| `edgefinder/core/models.py` | Add `TradeIntent`, `ExitIntent` domain models |
| `edgefinder/trading/arena.py` | Rewrite intraday/daily cycles using shared data |
| `edgefinder/trading/journal.py` | Add rich context fields (entry_reasoning, exit_reasoning, indicators_at_entry/exit, pdt_flag, hold_duration) |
| `edgefinder/db/models.py` | Add `entry_reasoning`, `exit_reasoning`, `indicators_at_entry`, `indicators_at_exit`, `pdt_flag`, `hold_duration_hours` columns to `TradeRecord` |
| `dashboard/services.py` | Wire new strategies, startup safeguard, new cycle logic |
| `edgefinder/scheduler/scheduler.py` | Add daily indicator job, restrict market hours to 9:30-16:00 |
| `edgefinder/strategies/__init__.py` | Register only new strategies |

### Removed files (Task 12)

`edgefinder/strategies/alpha.py`, `bravo.py`, `charlie.py`, `degenerate.py`, `echo.py`

---

### Task 1: MarketData objects and indicator history buffer

**Files:**
- Create: `edgefinder/data/market_data.py`
- Test: `tests/test_market_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_market_data.py`:

```python
"""Tests for the shared MarketData layer."""

from edgefinder.data.market_data import (
    IndicatorSnapshot,
    IndicatorHistory,
    MarketContext,
    MarketData,
)


class TestIndicatorSnapshot:
    def test_to_dict_excludes_none(self):
        snap = IndicatorSnapshot(close=100.0, rsi=45.0)
        d = snap.to_dict()
        assert d["close"] == 100.0
        assert d["rsi"] == 45.0
        assert "macd_line" not in d  # None fields excluded


class TestIndicatorHistory:
    def test_add_and_retrieve(self):
        hist = IndicatorHistory(max_days=30)
        snap1 = IndicatorSnapshot(close=100.0, rsi=40.0)
        snap2 = IndicatorSnapshot(close=105.0, rsi=55.0)
        hist.add(snap1)
        hist.add(snap2)
        assert len(hist) == 2
        assert hist.latest.close == 105.0

    def test_max_days_enforced(self):
        hist = IndicatorHistory(max_days=3)
        for i in range(5):
            hist.add(IndicatorSnapshot(close=float(100 + i)))
        assert len(hist) == 3
        assert hist.latest.close == 104.0  # most recent
        assert hist.oldest.close == 102.0  # oldest kept

    def test_get_field_series(self):
        hist = IndicatorHistory(max_days=30)
        hist.add(IndicatorSnapshot(close=100.0, rsi=30.0))
        hist.add(IndicatorSnapshot(close=105.0, rsi=45.0))
        hist.add(IndicatorSnapshot(close=110.0, rsi=60.0))
        rsi_series = hist.get_field_series("rsi")
        assert rsi_series == [30.0, 45.0, 60.0]

    def test_previous_returns_day_before_latest(self):
        hist = IndicatorHistory(max_days=30)
        hist.add(IndicatorSnapshot(close=100.0, macd_histogram=-0.5))
        hist.add(IndicatorSnapshot(close=105.0, macd_histogram=0.3))
        assert hist.previous.macd_histogram == -0.5


class TestMarketData:
    def test_construction(self):
        snap = IndicatorSnapshot(close=100.0, rsi=45.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext(
            spy_price=450.0, spy_change_pct=0.5,
            qqq_price=380.0, qqq_change_pct=0.3,
            iwm_price=200.0, iwm_change_pct=-0.1,
            dia_price=350.0, dia_change_pct=0.2,
            vix_level=18.0, market_regime="bull",
        )
        md = MarketData(
            ticker="AAPL",
            current=snap,
            history=hist,
            fundamentals=None,
            context=ctx,
            current_price=150.0,
            today_volume=5_000_000,
            avg_daily_volume=3_000_000,
            volume_ratio=2.5,
        )
        assert md.ticker == "AAPL"
        assert md.current.rsi == 45.0
        assert md.volume_ratio == 2.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_market_data.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `edgefinder/data/market_data.py`**

```python
"""Shared market data objects for strategy consumption.

MarketData bundles current indicators, 30-day history, fundamentals,
and market context into a single object that all strategies receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from edgefinder.core.models import TickerFundamentals


@dataclass
class IndicatorSnapshot:
    """All indicator values for a single daily bar (or provisional bar)."""

    close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    ema_9: float | None = None
    ema_21: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    rsi: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None
    atr: float | None = None
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    stoch_rsi_k: float | None = None
    stoch_rsi_d: float | None = None
    williams_r: float | None = None
    volume_avg: float | None = None
    volume_ratio: float | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class IndicatorHistory:
    """Rolling buffer of daily IndicatorSnapshots (max N days)."""

    def __init__(self, max_days: int = 30) -> None:
        self._max_days = max_days
        self._snapshots: list[IndicatorSnapshot] = []

    def add(self, snapshot: IndicatorSnapshot) -> None:
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_days:
            self._snapshots = self._snapshots[-self._max_days:]

    @property
    def latest(self) -> IndicatorSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    @property
    def previous(self) -> IndicatorSnapshot | None:
        return self._snapshots[-2] if len(self._snapshots) >= 2 else None

    @property
    def oldest(self) -> IndicatorSnapshot | None:
        return self._snapshots[0] if self._snapshots else None

    def get_field_series(self, field_name: str) -> list[float | None]:
        """Get a list of values for a specific field across all snapshots."""
        return [getattr(s, field_name, None) for s in self._snapshots]

    def __len__(self) -> int:
        return len(self._snapshots)


@dataclass
class MarketContext:
    """Broad market state shared across all tickers."""

    spy_price: float = 0.0
    spy_change_pct: float = 0.0
    qqq_price: float = 0.0
    qqq_change_pct: float = 0.0
    iwm_price: float = 0.0
    iwm_change_pct: float = 0.0
    dia_price: float = 0.0
    dia_change_pct: float = 0.0
    vix_level: float = 0.0
    market_regime: str = "sideways"
    sector_performance: dict = field(default_factory=dict)


@dataclass
class MarketData:
    """Everything a strategy needs to make a decision for one ticker.

    Assembled once per ticker per intraday cycle. Passed to every strategy.
    """

    ticker: str
    current: IndicatorSnapshot
    history: IndicatorHistory
    fundamentals: TickerFundamentals | None
    context: MarketContext
    current_price: float = 0.0
    today_volume: float = 0.0
    avg_daily_volume: float = 0.0
    volume_ratio: float = 0.0  # time-normalized
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_market_data.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add edgefinder/data/market_data.py tests/test_market_data.py
git commit -m "feat: add MarketData objects and indicator history buffer"
```

---

### Task 2: TradeIntent and ExitIntent domain models

**Files:**
- Modify: `edgefinder/core/models.py`
- Test: `tests/test_core_models.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_core_models.py`:

```python
from edgefinder.core.models import TradeIntent, ExitIntent


class TestTradeIntent:
    def test_creation(self):
        intent = TradeIntent(
            ticker="AAPL",
            direction="LONG",
            reasoning="RSI oversold at BB lower",
            strategy_name="coward",
            indicators_snapshot={"rsi": 28.0, "close": 150.0},
        )
        assert intent.ticker == "AAPL"
        assert intent.direction == "LONG"
        assert intent.reasoning == "RSI oversold at BB lower"

    def test_defaults(self):
        intent = TradeIntent(
            ticker="AAPL", direction="LONG",
            reasoning="test", strategy_name="coward",
        )
        assert intent.indicators_snapshot == {}


class TestExitIntent:
    def test_creation(self):
        intent = ExitIntent(
            ticker="AAPL",
            reasoning="RSI overbought",
            indicators_snapshot={"rsi": 72.0},
        )
        assert intent.ticker == "AAPL"
        assert intent.reasoning == "RSI overbought"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core_models.py::TestTradeIntent tests/test_core_models.py::TestExitIntent -v`
Expected: FAIL — import error

- [ ] **Step 3: Add models to `edgefinder/core/models.py`**

Add at the end of the file:

```python
class TradeIntent(BaseModel):
    """A strategy's decision to enter a trade. The risk system handles sizing/stops."""

    ticker: str
    direction: str  # "LONG"
    reasoning: str  # human-readable explanation of why
    strategy_name: str
    indicators_snapshot: dict = Field(default_factory=dict)
    fundamentals_snapshot: dict = Field(default_factory=dict)
    market_context_snapshot: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExitIntent(BaseModel):
    """A strategy's decision to exit an open position."""

    ticker: str
    reasoning: str
    indicators_snapshot: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core_models.py::TestTradeIntent tests/test_core_models.py::TestExitIntent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add edgefinder/core/models.py tests/test_core_models.py
git commit -m "feat: add TradeIntent and ExitIntent domain models"
```

---

### Task 3: Indicator computation engine (refactored from signals/engine.py)

**Files:**
- Create: `edgefinder/data/indicator_engine.py`
- Test: `tests/test_indicator_engine.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_indicator_engine.py`:

```python
"""Tests for the shared indicator computation engine."""

import numpy as np
import pandas as pd
import pytest

from edgefinder.data.indicator_engine import compute_indicators_from_bars
from edgefinder.data.market_data import IndicatorSnapshot


class TestComputeIndicators:
    @pytest.fixture
    def daily_bars(self):
        """Generate 60 days of realistic daily OHLCV bars."""
        np.random.seed(42)
        n = 60
        close = 100 + np.random.normal(0, 1.5, n).cumsum()
        df = pd.DataFrame({
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        }, index=pd.date_range("2024-01-01", periods=n, freq="B", name="timestamp"))
        return df

    def test_returns_indicator_snapshot(self, daily_bars):
        result = compute_indicators_from_bars(daily_bars)
        assert isinstance(result, IndicatorSnapshot)
        assert result.close > 0
        assert result.rsi is not None
        assert result.ema_9 is not None
        assert result.ema_21 is not None
        assert result.macd_line is not None
        assert result.bb_upper is not None
        assert result.atr is not None

    def test_returns_none_on_insufficient_data(self):
        df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.0], "volume": [1000000.0],
        }, index=pd.date_range("2024-01-01", periods=1, freq="B", name="timestamp"))
        result = compute_indicators_from_bars(df)
        assert result is None

    def test_includes_ohlcv_in_snapshot(self, daily_bars):
        result = compute_indicators_from_bars(daily_bars)
        assert result.open > 0
        assert result.high > 0
        assert result.low > 0
        assert result.volume > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_indicator_engine.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `edgefinder/data/indicator_engine.py`**

```python
"""Shared indicator computation — pure functions, no signal detection.

Reuses the math from edgefinder/signals/engine.py but returns an
IndicatorSnapshot instead of detecting patterns. Called once per ticker
per cycle by the arena.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import settings
from edgefinder.data.market_data import IndicatorSnapshot

logger = logging.getLogger(__name__)


# ── Pure pandas indicator functions (from signals/engine.py) ──


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)
    rsi = rsi.where(avg_gain > 0, 0)
    return rsi


def _macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    close: pd.Series, period: int, std: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return _ema(tr, period)


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm_smooth / atr_smooth.replace(0, float("nan")))
    minus_di = 100 * (minus_dm_smooth / atr_smooth.replace(0, float("nan")))
    di_sum = plus_di + minus_di
    dx = 100 * ((plus_di - minus_di).abs() / di_sum.replace(0, float("nan")))
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def _stochastic_rsi(
    close: pd.Series, rsi_period: int, k_period: int, d_period: int
) -> tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, rsi_period)
    rsi_min = rsi.rolling(window=rsi_period).min()
    rsi_max = rsi.rolling(window=rsi_period).max()
    rsi_range = rsi_max - rsi_min
    stoch_rsi = ((rsi - rsi_min) / rsi_range.replace(0, float("nan"))) * 100
    k = stoch_rsi.rolling(window=k_period).mean()
    d = k.rolling(window=d_period).mean()
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    wr = -100 * ((highest - close) / (highest - lowest).replace(0, float("nan")))
    return wr


# ── Main computation ──────────────────────────────────


MIN_BARS = 30  # need at least 30 bars for meaningful indicators


def compute_indicators_from_bars(df: pd.DataFrame) -> IndicatorSnapshot | None:
    """Compute all technical indicators on daily OHLCV bars.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].
            Needs 30+ rows for meaningful results.

    Returns:
        IndicatorSnapshot with latest-bar values, or None if insufficient data.
    """
    if len(df) < MIN_BARS:
        logger.debug("Insufficient data: %d rows (need %d)", len(df), MIN_BARS)
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMAs
    ema_9 = _ema(close, 9)
    ema_21 = _ema(close, 21)
    ema_50 = _ema(close, 50) if len(df) >= 50 else pd.Series([None] * len(df))
    ema_200 = _ema(close, 200) if len(df) >= 200 else pd.Series([None] * len(df))

    # RSI
    rsi = _rsi(close, settings.signal_rsi_period)

    # MACD
    macd_line, macd_signal, macd_hist = _macd(
        close, settings.signal_macd_fast, settings.signal_macd_slow, settings.signal_macd_signal
    )

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = _bollinger_bands(
        close, settings.signal_bb_period, settings.signal_bb_std
    )
    bb_width_series = (bb_upper - bb_lower) / bb_middle.replace(0, float("nan"))

    # ATR
    atr = _atr(high, low, close, settings.signal_atr_period)

    # ADX
    adx_series, plus_di_series, minus_di_series = _adx(
        high, low, close, settings.signal_adx_period
    )

    # Stochastic RSI
    stoch_k, stoch_d = _stochastic_rsi(
        close, settings.signal_stoch_rsi_period,
        settings.signal_stoch_rsi_k, settings.signal_stoch_rsi_d,
    )

    # Williams %R
    williams = _williams_r(high, low, close, settings.signal_williams_r_period)

    # Volume
    vol_avg = volume.rolling(window=settings.signal_volume_avg_period).mean()
    vol_ratio = volume / vol_avg.replace(0, float("nan"))

    def _safe_float(series, idx=-1):
        try:
            val = series.iloc[idx]
            return float(val) if not pd.isna(val) else None
        except (IndexError, TypeError):
            return None

    return IndicatorSnapshot(
        close=float(close.iloc[-1]),
        open=float(df["open"].iloc[-1]),
        high=float(high.iloc[-1]),
        low=float(low.iloc[-1]),
        volume=float(volume.iloc[-1]),
        ema_9=_safe_float(ema_9),
        ema_21=_safe_float(ema_21),
        ema_50=_safe_float(ema_50),
        ema_200=_safe_float(ema_200),
        rsi=_safe_float(rsi),
        macd_line=_safe_float(macd_line),
        macd_signal=_safe_float(macd_signal),
        macd_histogram=_safe_float(macd_hist),
        bb_upper=_safe_float(bb_upper),
        bb_middle=_safe_float(bb_middle),
        bb_lower=_safe_float(bb_lower),
        bb_width=_safe_float(bb_width_series),
        atr=_safe_float(atr),
        adx=_safe_float(adx_series),
        plus_di=_safe_float(plus_di_series),
        minus_di=_safe_float(minus_di_series),
        stoch_rsi_k=_safe_float(stoch_k),
        stoch_rsi_d=_safe_float(stoch_d),
        williams_r=_safe_float(williams),
        volume_avg=_safe_float(vol_avg),
        volume_ratio=_safe_float(vol_ratio),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_indicator_engine.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add edgefinder/data/indicator_engine.py tests/test_indicator_engine.py
git commit -m "feat: add shared indicator computation engine"
```

---

### Task 4: Update settings for new risk system and market hours

**Files:**
- Modify: `config/settings.py`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_settings.py`:

```python
def test_new_risk_settings():
    from config.settings import settings
    assert settings.stop_loss_pct == 0.20
    assert settings.profit_target_coward == 0.15
    assert settings.profit_target_gambler == 0.25
    assert settings.profit_target_degenerate == 0.50
    assert settings.risk_pct_coward == 0.05
    assert settings.risk_pct_gambler == 0.10
    assert settings.risk_pct_degenerate == 0.20
    assert settings.market_open_et == "09:30"
    assert settings.market_close_et == "16:00"
    assert settings.indicator_history_days == 30
    assert settings.volume_anomaly_threshold == 3.0
    assert settings.scanner_max_universe_size == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_settings.py::test_new_risk_settings -v`
Expected: FAIL — attributes don't exist

- [ ] **Step 3: Update `config/settings.py`**

Add these new settings and modify existing values. Read the file first, then add/change:

```python
    # ── NEW RISK SYSTEM ───────────────────────────────
    stop_loss_pct: float = 0.20  # fixed 20% stop for all strategies
    risk_pct_coward: float = 0.05  # max loss per trade as % of equity
    risk_pct_gambler: float = 0.10
    risk_pct_degenerate: float = 0.20
    profit_target_coward: float = 0.15
    profit_target_gambler: float = 0.25
    profit_target_degenerate: float = 0.50

    # ── INDICATOR HISTORY ─────────────────────────────
    indicator_history_days: int = 30

    # ── VOLUME ANOMALY (Degenerate dynamic watchlist) ──
    volume_anomaly_threshold: float = 3.0  # 3x time-normalized volume
```

Change existing market hours:
```python
    market_open_et: str = "09:30"   # was "07:00"
    market_close_et: str = "16:00"  # was "18:00"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_settings.py::test_new_risk_settings -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/settings.py tests/test_settings.py
git commit -m "feat: add risk system settings, fix market hours to 9:30-16:00"
```

---

### Task 5: Risk manager

**Files:**
- Create: `edgefinder/trading/risk.py`
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_risk.py`:

```python
"""Tests for the centralized risk manager."""

import pytest
from edgefinder.trading.risk import RiskManager
from edgefinder.trading.account import VirtualAccount


class TestRiskManager:
    def test_compute_stop(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        stop = rm.compute_stop(entry_price=200.0)
        assert stop == 160.0  # 200 * (1 - 0.20)

    def test_compute_target(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        target = rm.compute_target(entry_price=200.0)
        assert target == 250.0  # 200 * (1 + 0.25)

    def test_compute_shares(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        acct = VirtualAccount("gambler", starting_capital=5000.0)
        shares = rm.compute_shares(entry_price=200.0, equity=5000.0)
        # max_loss = 5000 * 0.10 = 500
        # stop_distance = 200 * 0.20 = 40
        # shares = 500 / 40 = 12
        assert shares == 12

    def test_compute_shares_limited_by_cash(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        # Only $300 cash — can afford 1 share at $200
        shares = rm.compute_shares(entry_price=200.0, equity=5000.0, available_cash=300.0)
        assert shares == 1

    def test_compute_shares_zero_when_no_cash(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        shares = rm.compute_shares(entry_price=200.0, equity=5000.0, available_cash=50.0)
        assert shares == 0

    def test_should_stop_out(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        assert rm.should_stop_out(entry_price=200.0, current_price=159.0) is True
        assert rm.should_stop_out(entry_price=200.0, current_price=161.0) is False

    def test_should_take_profit(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        assert rm.should_take_profit(entry_price=200.0, current_price=251.0) is True
        assert rm.should_take_profit(entry_price=200.0, current_price=249.0) is False

    def test_coward_risk_profile(self):
        rm = RiskManager(risk_pct=0.05, stop_pct=0.20, target_pct=0.15)
        shares = rm.compute_shares(entry_price=200.0, equity=5000.0)
        # max_loss = 250, stop_distance = 40, shares = 6
        assert shares == 6

    def test_degenerate_risk_profile(self):
        rm = RiskManager(risk_pct=0.20, stop_pct=0.20, target_pct=0.50)
        shares = rm.compute_shares(entry_price=200.0, equity=5000.0)
        # max_loss = 1000, stop_distance = 40, shares = 25
        assert shares == 25
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_risk.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `edgefinder/trading/risk.py`**

```python
"""Centralized risk manager — sizing, stops, targets.

Each strategy gets its own RiskManager instance configured with
its risk percentage, stop percentage, and target percentage.
The account system handles cash constraints; this handles the math.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Computes position size, stop loss, and profit target."""

    def __init__(
        self,
        risk_pct: float,
        stop_pct: float = 0.20,
        target_pct: float = 0.25,
    ) -> None:
        self.risk_pct = risk_pct      # max loss per trade as fraction of equity
        self.stop_pct = stop_pct      # stop distance as fraction of entry price
        self.target_pct = target_pct  # target distance as fraction of entry price

    def compute_stop(self, entry_price: float) -> float:
        """Stop loss = entry * (1 - stop_pct)."""
        return round(entry_price * (1 - self.stop_pct), 2)

    def compute_target(self, entry_price: float) -> float:
        """Profit target = entry * (1 + target_pct)."""
        return round(entry_price * (1 + self.target_pct), 2)

    def compute_shares(
        self,
        entry_price: float,
        equity: float,
        available_cash: float | None = None,
    ) -> int:
        """Position size based on risk budget and stop distance.

        shares = max_loss / stop_distance, capped by available cash.
        """
        max_loss = equity * self.risk_pct
        stop_distance = entry_price * self.stop_pct
        if stop_distance <= 0:
            return 0

        shares = int(max_loss / stop_distance)

        # Cap by available cash
        if available_cash is not None:
            max_by_cash = int(available_cash / entry_price)
            shares = min(shares, max_by_cash)

        return max(shares, 0)

    def should_stop_out(self, entry_price: float, current_price: float) -> bool:
        """Check if current price has hit the 20% stop."""
        stop = self.compute_stop(entry_price)
        return current_price <= stop

    def should_take_profit(self, entry_price: float, current_price: float) -> bool:
        """Check if current price has hit the profit target."""
        target = self.compute_target(entry_price)
        return current_price >= target
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_risk.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add edgefinder/trading/risk.py tests/test_risk.py
git commit -m "feat: add centralized risk manager"
```

---

### Task 6: Strategy interface (SwingStrategy base class)

**Files:**
- Create: `edgefinder/strategies/strategy_interface.py`
- Test: `tests/test_strategy_interface.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_strategy_interface.py`:

```python
"""Tests for the new strategy interface."""

from edgefinder.data.market_data import (
    IndicatorHistory, IndicatorSnapshot, MarketContext, MarketData,
)
from edgefinder.strategies.strategy_interface import SwingStrategy


class FakeStrategy(SwingStrategy):
    name = "fake"
    risk_pct = 0.10
    target_pct = 0.25
    watchlist_size = 50

    def qualifies_stock(self, fundamentals):
        return True

    def evaluate(self, ticker, data):
        if data.current.rsi and data.current.rsi < 30:
            return self.make_intent(ticker, data, "RSI oversold")
        return None

    def should_exit(self, ticker, data, entry_price):
        if data.current.rsi and data.current.rsi > 70:
            return self.make_exit(ticker, data, "RSI overbought")
        return None


class TestSwingStrategy:
    def _make_data(self, rsi=50.0):
        snap = IndicatorSnapshot(close=100.0, rsi=rsi)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        return MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx, current_price=100.0,
        )

    def test_evaluate_returns_intent_when_conditions_met(self):
        s = FakeStrategy()
        data = self._make_data(rsi=25.0)
        intent = s.evaluate("AAPL", data)
        assert intent is not None
        assert intent.ticker == "AAPL"
        assert intent.direction == "LONG"
        assert "RSI oversold" in intent.reasoning

    def test_evaluate_returns_none_when_no_signal(self):
        s = FakeStrategy()
        data = self._make_data(rsi=50.0)
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_should_exit_returns_intent(self):
        s = FakeStrategy()
        data = self._make_data(rsi=75.0)
        exit_intent = s.should_exit("AAPL", data, entry_price=90.0)
        assert exit_intent is not None
        assert "RSI overbought" in exit_intent.reasoning

    def test_should_exit_returns_none(self):
        s = FakeStrategy()
        data = self._make_data(rsi=50.0)
        exit_intent = s.should_exit("AAPL", data, entry_price=90.0)
        assert exit_intent is None

    def test_stop_pct_is_always_20(self):
        s = FakeStrategy()
        assert s.stop_pct == 0.20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategy_interface.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `edgefinder/strategies/strategy_interface.py`**

```python
"""New strategy interface — strategies evaluate raw data and return intents.

All strategies are swing-oriented. Day trades only happen as damage control
(stop hit on entry day). Each strategy defines its own entry/exit logic
using the full MarketData object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData


class SwingStrategy(ABC):
    """Base class for all new strategies.

    Subclasses define:
    - name: unique identifier
    - risk_pct: fraction of equity to risk per trade
    - target_pct: profit target as fraction of entry price
    - watchlist_size: max tickers to watch
    - qualifies_stock(): fundamental watchlist filter
    - evaluate(): entry decision
    - should_exit(): exit decision
    """

    stop_pct: float = 0.20  # fixed for all strategies, non-negotiable

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def risk_pct(self) -> float: ...

    @property
    @abstractmethod
    def target_pct(self) -> float: ...

    @property
    def watchlist_size(self) -> int:
        return 50

    @abstractmethod
    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        """Watchlist filter — applied during nightly scan."""
        ...

    @abstractmethod
    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        """Decide whether to enter a trade. Return TradeIntent or None."""
        ...

    @abstractmethod
    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        """Decide whether to exit an open position. Return ExitIntent or None."""
        ...

    # ── Helpers for subclasses ──

    def make_intent(
        self, ticker: str, data: MarketData, reasoning: str
    ) -> TradeIntent:
        return TradeIntent(
            ticker=ticker,
            direction="LONG",
            reasoning=reasoning,
            strategy_name=self.name,
            indicators_snapshot=data.current.to_dict(),
            fundamentals_snapshot=(
                data.fundamentals.model_dump(exclude_none=True)
                if data.fundamentals else {}
            ),
            market_context_snapshot={
                "spy_price": data.context.spy_price,
                "spy_change_pct": data.context.spy_change_pct,
                "vix_level": data.context.vix_level,
                "market_regime": data.context.market_regime,
            },
        )

    def make_exit(
        self, ticker: str, data: MarketData, reasoning: str
    ) -> ExitIntent:
        return ExitIntent(
            ticker=ticker,
            reasoning=reasoning,
            indicators_snapshot=data.current.to_dict(),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strategy_interface.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add edgefinder/strategies/strategy_interface.py tests/test_strategy_interface.py
git commit -m "feat: add SwingStrategy base class with evaluate/should_exit interface"
```

---

### Task 7: Coward strategy

**Files:**
- Create: `edgefinder/strategies/coward.py`
- Test: `tests/test_strategies_new.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_strategies_new.py`:

```python
"""Tests for the three new strategies: Coward, Gambler, Degenerate."""

import pytest
from edgefinder.core.models import TickerFundamentals
from edgefinder.data.market_data import (
    IndicatorHistory, IndicatorSnapshot, MarketContext, MarketData,
)
from edgefinder.strategies.coward import CowardStrategy


def _make_data(ticker="AAPL", rsi=50.0, bb_lower=95.0, close=100.0, **kwargs):
    snap = IndicatorSnapshot(close=close, rsi=rsi, bb_lower=bb_lower, **kwargs)
    hist = IndicatorHistory(max_days=30)
    hist.add(snap)
    ctx = MarketContext()
    return MarketData(
        ticker=ticker, current=snap, history=hist,
        fundamentals=None, context=ctx, current_price=close,
    )


class TestCowardStrategy:
    def test_name_and_risk(self):
        s = CowardStrategy()
        assert s.name == "coward"
        assert s.risk_pct == 0.05
        assert s.target_pct == 0.15
        assert s.stop_pct == 0.20
        assert s.watchlist_size == 50

    def test_qualifies_stock_passes(self):
        s = CowardStrategy()
        fund = TickerFundamentals(
            symbol="AAPL", earnings_growth=0.15, current_ratio=2.0,
        )
        assert s.qualifies_stock(fund) is True

    def test_qualifies_stock_fails_no_earnings(self):
        s = CowardStrategy()
        fund = TickerFundamentals(
            symbol="AAPL", earnings_growth=-0.05, current_ratio=2.0,
        )
        assert s.qualifies_stock(fund) is False

    def test_qualifies_stock_fails_low_current_ratio(self):
        s = CowardStrategy()
        fund = TickerFundamentals(
            symbol="AAPL", earnings_growth=0.10, current_ratio=1.2,
        )
        assert s.qualifies_stock(fund) is False

    def test_entry_on_rsi_oversold(self):
        s = CowardStrategy()
        data = _make_data(rsi=30.0)
        intent = s.evaluate("AAPL", data)
        assert intent is not None
        assert "RSI" in intent.reasoning

    def test_entry_on_bb_lower_touch(self):
        s = CowardStrategy()
        # Price within 1% of BB lower: close=100, bb_lower=99.5
        data = _make_data(rsi=50.0, bb_lower=99.5, close=100.0)
        intent = s.evaluate("AAPL", data)
        assert intent is not None
        assert "BB" in intent.reasoning or "Bollinger" in intent.reasoning

    def test_no_entry_when_no_conditions_met(self):
        s = CowardStrategy()
        data = _make_data(rsi=50.0, bb_lower=80.0, close=100.0)
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_exit_on_rsi_overbought(self):
        s = CowardStrategy()
        data = _make_data(rsi=72.0)
        exit_intent = s.should_exit("AAPL", data, entry_price=90.0)
        assert exit_intent is not None
        assert "RSI" in exit_intent.reasoning

    def test_no_exit_when_rsi_normal(self):
        s = CowardStrategy()
        data = _make_data(rsi=55.0)
        exit_intent = s.should_exit("AAPL", data, entry_price=90.0)
        assert exit_intent is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategies_new.py::TestCowardStrategy -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `edgefinder/strategies/coward.py`**

```python
"""Coward — conservative swing trading.

Watches quality stocks (positive earnings, strong balance sheet).
Enters on oversold dips. Exits early at first sign of a top.
Wins often, wins small.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.strategy_interface import SwingStrategy


class CowardStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "coward"

    @property
    def risk_pct(self) -> float:
        return 0.05

    @property
    def target_pct(self) -> float:
        return 0.15

    @property
    def watchlist_size(self) -> int:
        return 50

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        eg = fundamentals.earnings_growth
        if eg is None or eg <= 0:
            return False
        cr = fundamentals.current_ratio
        if cr is None or cr <= 1.5:
            return False
        return True

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current

        # Entry condition 1: RSI below 35
        if ind.rsi is not None and ind.rsi < 35:
            return self.make_intent(
                ticker, data,
                f"RSI oversold at {ind.rsi:.1f} (threshold: 35)",
            )

        # Entry condition 2: Price within 1% of BB lower band
        if (
            ind.bb_lower is not None
            and ind.close > 0
            and abs(ind.close - ind.bb_lower) / ind.close <= 0.01
        ):
            return self.make_intent(
                ticker, data,
                f"Price ${ind.close:.2f} near BB lower ${ind.bb_lower:.2f} (within 1%)",
            )

        return None

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current

        # Exit when RSI crosses above 70
        if ind.rsi is not None and ind.rsi > 70:
            return self.make_exit(
                ticker, data,
                f"RSI overbought at {ind.rsi:.1f} (threshold: 70)",
            )

        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strategies_new.py::TestCowardStrategy -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add edgefinder/strategies/coward.py tests/test_strategies_new.py
git commit -m "feat: add Coward strategy — conservative swing trading"
```

---

### Task 8: Gambler and Degenerate strategies

**Files:**
- Create: `edgefinder/strategies/gambler.py`
- Create: `edgefinder/strategies/degenerate_v2.py`
- Modify: `tests/test_strategies_new.py`

- [ ] **Step 1: Write failing tests for Gambler**

Add to `tests/test_strategies_new.py`:

```python
from edgefinder.strategies.gambler import GamblerStrategy


class TestGamblerStrategy:
    def test_name_and_risk(self):
        s = GamblerStrategy()
        assert s.name == "gambler"
        assert s.risk_pct == 0.10
        assert s.target_pct == 0.25
        assert s.watchlist_size == 100

    def test_qualifies_with_earnings_growth(self):
        s = GamblerStrategy()
        fund = TickerFundamentals(symbol="AAPL", earnings_growth=0.10)
        assert s.qualifies_stock(fund) is True

    def test_qualifies_with_revenue_growth_only(self):
        s = GamblerStrategy()
        fund = TickerFundamentals(symbol="AAPL", revenue_growth=0.05)
        assert s.qualifies_stock(fund) is True

    def test_rejects_no_growth(self):
        s = GamblerStrategy()
        fund = TickerFundamentals(symbol="AAPL")
        assert s.qualifies_stock(fund) is False

    def test_entry_on_macd_cross_and_rsi_midrange(self):
        s = GamblerStrategy()
        # Current histogram positive, need previous negative for cross
        snap_prev = IndicatorSnapshot(close=98.0, macd_histogram=-0.5, rsi=50.0)
        snap_curr = IndicatorSnapshot(close=100.0, macd_histogram=0.3, rsi=50.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap_prev)
        hist.add(snap_curr)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap_curr, history=hist,
            fundamentals=None, context=ctx, current_price=100.0,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is not None
        assert "MACD" in intent.reasoning

    def test_no_entry_without_macd_cross(self):
        s = GamblerStrategy()
        # Both positive — no cross
        snap_prev = IndicatorSnapshot(close=98.0, macd_histogram=0.2, rsi=50.0)
        snap_curr = IndicatorSnapshot(close=100.0, macd_histogram=0.5, rsi=50.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap_prev)
        hist.add(snap_curr)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap_curr, history=hist,
            fundamentals=None, context=ctx, current_price=100.0,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_no_entry_rsi_out_of_range(self):
        s = GamblerStrategy()
        # MACD crosses but RSI too high
        snap_prev = IndicatorSnapshot(close=98.0, macd_histogram=-0.5, rsi=65.0)
        snap_curr = IndicatorSnapshot(close=100.0, macd_histogram=0.3, rsi=65.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap_prev)
        hist.add(snap_curr)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap_curr, history=hist,
            fundamentals=None, context=ctx, current_price=100.0,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_exit_on_macd_negative_cross(self):
        s = GamblerStrategy()
        snap_prev = IndicatorSnapshot(close=102.0, macd_histogram=0.5)
        snap_curr = IndicatorSnapshot(close=100.0, macd_histogram=-0.2)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap_prev)
        hist.add(snap_curr)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap_curr, history=hist,
            fundamentals=None, context=ctx, current_price=100.0,
        )
        exit_intent = s.should_exit("AAPL", data, entry_price=95.0)
        assert exit_intent is not None
        assert "MACD" in exit_intent.reasoning
```

- [ ] **Step 2: Write failing tests for Degenerate**

Add to `tests/test_strategies_new.py`:

```python
from edgefinder.strategies.degenerate_v2 import DegenerateStrategy


class TestDegenerateStrategy:
    def test_name_and_risk(self):
        s = DegenerateStrategy()
        assert s.name == "degenerate"
        assert s.risk_pct == 0.20
        assert s.target_pct == 0.50
        assert s.watchlist_size == 200

    def test_qualifies_anything_with_data(self):
        s = DegenerateStrategy()
        fund = TickerFundamentals(symbol="AAPL", market_cap=500_000_000)
        assert s.qualifies_stock(fund) is True

    def test_entry_on_volume_spike_and_momentum(self):
        s = DegenerateStrategy()
        snap = IndicatorSnapshot(close=100.0, rsi=55.0, ema_21=98.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx,
            current_price=100.0, volume_ratio=2.5,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is not None
        assert "volume" in intent.reasoning.lower()

    def test_no_entry_without_volume_spike(self):
        s = DegenerateStrategy()
        snap = IndicatorSnapshot(close=100.0, rsi=55.0, ema_21=98.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx,
            current_price=100.0, volume_ratio=1.2,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_no_entry_without_momentum(self):
        s = DegenerateStrategy()
        # Volume spike but RSI below 50
        snap = IndicatorSnapshot(close=100.0, rsi=40.0, ema_21=102.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx,
            current_price=100.0, volume_ratio=2.5,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_exit_on_volume_fade_and_overbought(self):
        s = DegenerateStrategy()
        snap = IndicatorSnapshot(close=150.0, rsi=82.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx,
            current_price=150.0, volume_ratio=0.8,
        )
        exit_intent = s.should_exit("AAPL", data, entry_price=100.0)
        assert exit_intent is not None

    def test_no_exit_if_volume_still_high(self):
        s = DegenerateStrategy()
        # RSI overbought but volume still high
        snap = IndicatorSnapshot(close=150.0, rsi=82.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx,
            current_price=150.0, volume_ratio=1.5,
        )
        exit_intent = s.should_exit("AAPL", data, entry_price=100.0)
        assert exit_intent is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_strategies_new.py::TestGamblerStrategy tests/test_strategies_new.py::TestDegenerateStrategy -v`
Expected: FAIL — modules not found

- [ ] **Step 4: Implement `edgefinder/strategies/gambler.py`**

```python
"""Gambler — balanced swing trading.

Rides momentum in the middle of moves. Enters when MACD crosses
bullish with RSI in neutral territory. Exits when momentum fades.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.strategy_interface import SwingStrategy


class GamblerStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "gambler"

    @property
    def risk_pct(self) -> float:
        return 0.10

    @property
    def target_pct(self) -> float:
        return 0.25

    @property
    def watchlist_size(self) -> int:
        return 100

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        eg = fundamentals.earnings_growth
        rg = fundamentals.revenue_growth
        has_earnings = eg is not None and eg > 0
        has_revenue = rg is not None and rg > 0
        return has_earnings or has_revenue

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current
        prev = data.history.previous

        if prev is None or ind.macd_histogram is None or ind.rsi is None:
            return None

        # MACD histogram crosses from negative to positive
        prev_hist = prev.macd_histogram
        if prev_hist is None:
            return None

        macd_crossed = prev_hist < 0 and ind.macd_histogram >= 0

        # RSI in mid-range (40-60)
        rsi_midrange = 40 <= ind.rsi <= 60

        if macd_crossed and rsi_midrange:
            return self.make_intent(
                ticker, data,
                f"MACD histogram crossed positive ({prev_hist:.3f} -> {ind.macd_histogram:.3f}), "
                f"RSI neutral at {ind.rsi:.1f}",
            )

        return None

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current
        prev = data.history.previous

        if prev is None or ind.macd_histogram is None:
            return None

        prev_hist = prev.macd_histogram
        if prev_hist is None:
            return None

        # MACD histogram crosses from positive to negative
        if prev_hist > 0 and ind.macd_histogram <= 0:
            return self.make_exit(
                ticker, data,
                f"MACD histogram crossed negative ({prev_hist:.3f} -> {ind.macd_histogram:.3f})",
            )

        return None
```

- [ ] **Step 5: Implement `edgefinder/strategies/degenerate_v2.py`**

```python
"""Degenerate — aggressive swing trading.

Jumps into volume spikes with bullish momentum. Rides until the
hype dies (volume fades + overbought). Lives or dies on single trades.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.strategy_interface import SwingStrategy


class DegenerateStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "degenerate"

    @property
    def risk_pct(self) -> float:
        return 0.20

    @property
    def target_pct(self) -> float:
        return 0.50

    @property
    def watchlist_size(self) -> int:
        return 200

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        # Anything with data — no fundamental requirements
        return fundamentals.market_cap is not None

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current

        if ind.rsi is None or ind.ema_21 is None:
            return None

        # Volume spike: time-normalized ratio > 2x
        volume_spike = data.volume_ratio > 2.0

        # Bullish momentum: RSI > 50 and price above EMA 21
        bullish = ind.rsi > 50 and ind.close > ind.ema_21

        if volume_spike and bullish:
            return self.make_intent(
                ticker, data,
                f"Volume spike {data.volume_ratio:.1f}x with bullish momentum "
                f"(RSI {ind.rsi:.1f}, price ${ind.close:.2f} > EMA21 ${ind.ema_21:.2f})",
            )

        return None

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current

        if ind.rsi is None:
            return None

        # Both conditions required: volume fading AND overbought
        volume_faded = data.volume_ratio < 1.0
        overbought = ind.rsi > 80

        if volume_faded and overbought:
            return self.make_exit(
                ticker, data,
                f"Volume faded ({data.volume_ratio:.1f}x) and overbought (RSI {ind.rsi:.1f})",
            )

        return None
```

- [ ] **Step 6: Run all strategy tests**

Run: `pytest tests/test_strategies_new.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add edgefinder/strategies/gambler.py edgefinder/strategies/degenerate_v2.py tests/test_strategies_new.py
git commit -m "feat: add Gambler and Degenerate strategies"
```

---

### Task 9: Enriched snapshot provider

**Files:**
- Create: `edgefinder/data/snapshot_provider.py`
- Test: `tests/test_snapshot_provider.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_snapshot_provider.py`:

```python
"""Tests for the enriched snapshot provider."""

from unittest.mock import MagicMock
from edgefinder.data.snapshot_provider import get_enriched_snapshots


class TestEnrichedSnapshots:
    def test_returns_price_and_volume(self):
        mock_provider = MagicMock()
        # Simulate Polygon snapshot objects
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_snapshot_provider.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `edgefinder/data/snapshot_provider.py`**

```python
"""Enriched snapshot provider — price + volume for all tickers in one call.

Uses Polygon's get_snapshot_all which returns today's OHLCV for every stock.
This replaces individual get_latest_price calls in the intraday cycle.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_enriched_snapshots(provider) -> dict[str, dict]:
    """Fetch price + volume for all tickers in one API call.

    Returns: {ticker: {"price": float, "volume": float, "open": float,
                        "high": float, "low": float}}
    """
    try:
        snapshots = provider._client.get_snapshot_all("stocks")
    except Exception:
        logger.exception("get_enriched_snapshots failed")
        return {}

    if not snapshots:
        return {}

    result: dict[str, dict] = {}
    for s in snapshots:
        ticker = getattr(s, "ticker", None)
        if not ticker:
            continue

        price = None
        volume = 0.0
        open_price = 0.0
        high = 0.0
        low = 0.0

        if s.day:
            price = float(s.day.close) if s.day.close else None
            volume = float(s.day.volume) if s.day.volume else 0.0
            open_price = float(s.day.open) if getattr(s.day, "open", None) else 0.0
            high = float(s.day.high) if getattr(s.day, "high", None) else 0.0
            low = float(s.day.low) if getattr(s.day, "low", None) else 0.0

        if price is None and s.prev_day and s.prev_day.close:
            price = float(s.prev_day.close)

        if price:
            result[ticker] = {
                "price": price,
                "volume": volume,
                "open": open_price,
                "high": high,
                "low": low,
            }

    logger.info("Enriched snapshots: %d tickers", len(result))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_snapshot_provider.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add edgefinder/data/snapshot_provider.py tests/test_snapshot_provider.py
git commit -m "feat: add enriched snapshot provider with price + volume"
```

---

### Task 10: Remove old strategies and update registry

**Files:**
- Delete: `edgefinder/strategies/alpha.py`, `bravo.py`, `charlie.py`, `degenerate.py`, `echo.py`
- Modify: `edgefinder/strategies/__init__.py`
- Modify: `edgefinder/strategies/base.py` (keep StrategyRegistry for backward compat with scanner)

- [ ] **Step 1: Update `edgefinder/strategies/__init__.py`**

Replace the contents:

```python
# Register new strategies — old ones (alpha, bravo, charlie, degenerate, echo) removed
from edgefinder.strategies import coward, gambler, degenerate_v2  # noqa: F401
```

- [ ] **Step 2: Delete old strategy files**

```bash
rm edgefinder/strategies/alpha.py
rm edgefinder/strategies/bravo.py
rm edgefinder/strategies/charlie.py
rm edgefinder/strategies/degenerate.py
rm edgefinder/strategies/echo.py
```

- [ ] **Step 3: Register new strategies with StrategyRegistry**

Add registration decorators to each new strategy file. In `coward.py`, add at the top after imports:

```python
from edgefinder.strategies.base import StrategyRegistry
```

And add decorator to the class:

```python
@StrategyRegistry.register("coward")
class CowardStrategy(SwingStrategy):
```

Do the same for `gambler.py`:

```python
from edgefinder.strategies.base import StrategyRegistry

@StrategyRegistry.register("gambler")
class GamblerStrategy(SwingStrategy):
```

And `degenerate_v2.py`:

```python
from edgefinder.strategies.base import StrategyRegistry

@StrategyRegistry.register("degenerate")
class DegenerateStrategy(SwingStrategy):
```

- [ ] **Step 4: Verify new strategies are registered**

```bash
python -c "
from edgefinder.strategies.base import StrategyRegistry
import edgefinder.strategies  # triggers registration
names = StrategyRegistry.list_names()
print('Registered:', names)
assert 'coward' in names
assert 'gambler' in names
assert 'degenerate' in names
assert 'alpha' not in names
print('OK')
"
```

- [ ] **Step 5: Run tests that don't depend on old strategies**

Run: `pytest tests/test_strategies_new.py tests/test_market_data.py tests/test_indicator_engine.py tests/test_risk.py tests/test_strategy_interface.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: remove old strategies, register Coward/Gambler/Degenerate"
```

---

### Task 11: Add trade logging columns to DB

**Files:**
- Modify: `edgefinder/db/models.py`
- Modify: `edgefinder/trading/journal.py`
- Modify: `scripts/render_start.py` (add migration for new columns)

- [ ] **Step 1: Add columns to TradeRecord**

In `edgefinder/db/models.py`, add to the `TradeRecord` class after `integrity_hash`:

```python
    # Rich trade context (new)
    entry_reasoning: Mapped[str | None] = mapped_column(Text)
    exit_reasoning: Mapped[str | None] = mapped_column(Text)
    indicators_at_entry: Mapped[dict | None] = mapped_column(JSON)
    indicators_at_exit: Mapped[dict | None] = mapped_column(JSON)
    fundamentals_at_entry: Mapped[dict | None] = mapped_column(JSON)
    market_context_at_entry: Mapped[dict | None] = mapped_column(JSON)
    pdt_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    hold_duration_hours: Mapped[float | None] = mapped_column(Float)
```

- [ ] **Step 2: Add migration entries to `scripts/render_start.py`**

In the `migrations` list in `init_database()`, add:

```python
        ("trades", "entry_reasoning", "TEXT"),
        ("trades", "exit_reasoning", "TEXT"),
        ("trades", "indicators_at_entry", "TEXT"),  # JSON stored as TEXT in SQLite
        ("trades", "indicators_at_exit", "TEXT"),
        ("trades", "fundamentals_at_entry", "TEXT"),
        ("trades", "market_context_at_entry", "TEXT"),
        ("trades", "pdt_flag", "BOOLEAN DEFAULT 0"),
        ("trades", "hold_duration_hours", "FLOAT"),
```

- [ ] **Step 3: Update `TradeJournal.log_trade` to save new fields**

In `edgefinder/trading/journal.py`, in the `log_trade` method, add to the `else` branch (new record creation):

```python
                entry_reasoning=getattr(trade, 'entry_reasoning', None),
                indicators_at_entry=getattr(trade, 'indicators_at_entry', None),
                fundamentals_at_entry=getattr(trade, 'fundamentals_at_entry', None),
                market_context_at_entry=getattr(trade, 'market_context_at_entry', None),
```

And in the `if existing` branch (updating on close):

```python
            existing.exit_reasoning = getattr(trade, 'exit_reasoning', None)
            existing.indicators_at_exit = getattr(trade, 'indicators_at_exit', None)
            existing.pdt_flag = getattr(trade, 'pdt_flag', False)
            existing.hold_duration_hours = getattr(trade, 'hold_duration_hours', None)
```

- [ ] **Step 4: Add new fields to the Trade domain model**

In `edgefinder/core/models.py`, add to the `Trade` class:

```python
    entry_reasoning: Optional[str] = None
    exit_reasoning: Optional[str] = None
    indicators_at_entry: Optional[dict] = None
    indicators_at_exit: Optional[dict] = None
    fundamentals_at_entry: Optional[dict] = None
    market_context_at_entry: Optional[dict] = None
    pdt_flag: bool = False
    hold_duration_hours: Optional[float] = None
```

- [ ] **Step 5: Run existing journal tests (should still pass)**

Run: `pytest tests/test_trading.py::TestTradeJournal -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add edgefinder/db/models.py edgefinder/trading/journal.py edgefinder/core/models.py scripts/render_start.py
git commit -m "feat: add rich trade logging columns for entry/exit context"
```

---

### Task 12: Rewrite arena for new architecture

This is the largest task — the arena orchestrates the entire intraday and daily cycle.

**Files:**
- Modify: `edgefinder/trading/arena.py`
- Test: `tests/test_arena_new.py`

**Note to implementer:** This task is complex. Read the spec carefully (sections: Shared Data Layer, Risk System, System Flow). Read the existing `arena.py` for patterns. The new arena replaces `run_signal_check()` and `check_positions()` with methods that use `MarketData`, `SwingStrategy`, and `RiskManager`.

- [ ] **Step 1: Write tests for the new arena**

Create `tests/test_arena_new.py`:

```python
"""Tests for the redesigned arena with shared data layer."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from edgefinder.core.models import TradeIntent
from edgefinder.data.market_data import (
    IndicatorHistory, IndicatorSnapshot, MarketContext, MarketData,
)
from edgefinder.trading.account import Position, VirtualAccount
from edgefinder.trading.arena import ArenaEngine
from edgefinder.trading.risk import RiskManager


class TestNewArena:
    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        # Return daily bars
        np.random.seed(42)
        n = 60
        close = 100 + np.random.normal(0, 1.5, n).cumsum()
        df = pd.DataFrame({
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        }, index=pd.date_range("2024-01-01", periods=n, freq="B", name="timestamp"))
        provider.get_bars.return_value = df
        provider.get_latest_price.return_value = 100.0
        return provider

    def test_stop_loss_fires_at_20_pct(self):
        """Verify the 20% stop is non-negotiable."""
        acct = VirtualAccount("gambler", starting_capital=5000.0)
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)

        # Open a position at $100
        pos = Position(
            symbol="AAPL", shares=12, entry_price=100.0,
            stop_loss=rm.compute_stop(100.0),
            target=rm.compute_target(100.0),
            direction="LONG", trade_type="SWING",
            trade_id="test-stop-1",
        )
        acct.open_position(pos)

        # Price drops to $79 — below 20% stop ($80)
        assert rm.should_stop_out(100.0, 79.0) is True
        assert rm.should_stop_out(100.0, 81.0) is False

    def test_profit_target_fires(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        assert rm.should_take_profit(100.0, 126.0) is True
        assert rm.should_take_profit(100.0, 124.0) is False

    def test_position_sizing_per_strategy(self):
        """Each strategy gets different position sizes."""
        rm_coward = RiskManager(risk_pct=0.05, stop_pct=0.20, target_pct=0.15)
        rm_gambler = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        rm_degen = RiskManager(risk_pct=0.20, stop_pct=0.20, target_pct=0.50)

        equity = 5000.0
        price = 200.0

        # Coward: 250/40 = 6 shares
        assert rm_coward.compute_shares(price, equity) == 6
        # Gambler: 500/40 = 12 shares
        assert rm_gambler.compute_shares(price, equity) == 12
        # Degenerate: 1000/40 = 25 shares
        assert rm_degen.compute_shares(price, equity) == 25
```

- [ ] **Step 2: Run tests to verify they pass** (these test RiskManager + Account, already implemented)

Run: `pytest tests/test_arena_new.py -v`
Expected: ALL PASS

- [ ] **Step 3: Rewrite `edgefinder/trading/arena.py`**

This is a major rewrite. The implementer should read the existing `arena.py` and the spec, then rewrite to:

1. Store `IndicatorHistory` per ticker (30-day buffer)
2. Accept new `SwingStrategy` instances instead of old `BaseStrategy`
3. Use `RiskManager` per strategy for sizing/stops/targets
4. New `run_intraday_cycle(snapshot_data, market_context)` method that:
   - Builds `MarketData` per ticker with provisional indicators
   - Runs entry evaluation for tickers without positions
   - Runs exit evaluation + stop/target checks for open positions
   - Handles trade opening/closing through the existing account/executor
5. New `run_daily_cycle()` method that saves indicator snapshots to history
6. Keep `get_account()`, `get_all_accounts()`, etc. for dashboard compatibility

The arena should import and use:
- `edgefinder.data.market_data.MarketData`, `IndicatorHistory`, `IndicatorSnapshot`, `MarketContext`
- `edgefinder.data.indicator_engine.compute_indicators_from_bars`
- `edgefinder.trading.risk.RiskManager`
- `edgefinder.strategies.strategy_interface.SwingStrategy`
- `edgefinder.core.models.TradeIntent`, `ExitIntent`

**The implementer should write the full arena rewrite.** This is a judgment task — use the most capable model available. The key constraint: preserve backward compatibility with `dashboard/services.py` (which calls `load_strategies()`, `get_account()`, `get_all_accounts()`, `get_all_open_positions()`).

- [ ] **Step 4: Run all new tests**

Run: `pytest tests/test_arena_new.py tests/test_strategies_new.py tests/test_risk.py tests/test_market_data.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add edgefinder/trading/arena.py tests/test_arena_new.py
git commit -m "feat: rewrite arena for shared data layer and new strategy interface"
```

---

### Task 13: Wire new system in services.py

**Files:**
- Modify: `dashboard/services.py`
- Modify: `edgefinder/scheduler/scheduler.py`
- Modify: `edgefinder/trading/account.py` (remove hard position limit)

**Note to implementer:** This task wires everything together. Read `dashboard/services.py` fully before starting. The key changes:

1. **Account changes:** In `edgefinder/trading/account.py`, in `can_open_position()`, remove the `max_open_positions` check (line ~135-136). Cash is now the only constraint. Keep the drawdown circuit breaker, revenge cooldown, and re-entry cooldown.

2. **Market hours:** Update the scheduler to use 9:30-16:00 ET instead of 7:00-18:00. The settings were already updated in Task 4.

3. **Startup safeguard:** In `init_services()`, check if market opens within 2 hours. If so, skip the initial scan.

4. **Intraday cycle:** The `_signal_check_job()` and `_position_monitor_job()` should be replaced with a single `_intraday_cycle_job()` that:
   - Calls `get_enriched_snapshots()` once
   - Builds `MarketContext`
   - Calls `arena.run_intraday_cycle()`
   - Persists account state

5. **Daily cycle:** Add a `_daily_indicator_job()` that runs at 4:30 PM ET to save daily indicators to the history buffer.

6. **PDT tracking:** When closing a trade, check if `entry_time.date() == exit_time.date()`. If so, set `pdt_flag = True` on the trade.

**This is a wiring task — follow existing patterns in `services.py`.** The implementer should read the full file and modify in place.

- [ ] **Step 1: Make account changes**

Remove the hard position limit from `VirtualAccount.can_open_position()` in `edgefinder/trading/account.py`.

- [ ] **Step 2: Wire services.py**

Update `dashboard/services.py` to use new strategies, arena methods, and cycle logic.

- [ ] **Step 3: Update scheduler**

The scheduler already reads `market_open_et` and `market_close_et` from settings. Since Task 4 changed these to "09:30" and "16:00", the scheduler automatically picks up the new hours. Add a daily indicator job at 16:30.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v -m "not integration" --tb=short 2>&1 | tail -20`
Expected: ALL PASS (some old tests may need updates due to removed strategies — fix them)

- [ ] **Step 5: Commit**

```bash
git add dashboard/services.py edgefinder/trading/account.py edgefinder/scheduler/scheduler.py
git commit -m "feat: wire new shared data layer and strategies into services"
```

---

### Task 14: Update old tests and final verification

**Files:**
- Modify: `tests/test_trading.py` (fix tests that reference old strategies)
- Modify: `tests/test_strategies.py` (remove or rewrite for new strategies)

- [ ] **Step 1: Fix test_trading.py arena tests**

The `TestArena` class in `test_trading.py` imports and reloads `alpha`, `bravo`, `charlie`. These are deleted. Either:
- Remove these tests (they test old behavior)
- Or rewrite them to use the new strategies

Recommended: Remove the old `TestArena` class entirely. The new arena is tested by `test_arena_new.py`.

- [ ] **Step 2: Fix test_strategies.py**

This file tests old strategy qualification/signal logic. Remove it or rewrite for new strategies.

- [ ] **Step 3: Fix any other broken tests**

Run: `pytest tests/ -v -m "not integration" --tb=short`

Fix any failures. Common issues:
- Tests importing old strategy modules
- Tests referencing `max_open_positions` behavior
- Tests using old `Signal` flow

- [ ] **Step 4: Run full suite — must be green**

Run: `pytest tests/ -v -m "not integration" --tb=short`
Expected: ALL PASS — zero failures

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "fix: update tests for new strategy architecture"
```

---

### Task 15: Version bump and final commit

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: Bump version**

Change `__version__` in `dashboard/app.py` from `"4.7.0"` to `"5.0.0"` — this is a major architectural change.

- [ ] **Step 2: Run full test suite one final time**

Run: `pytest tests/ -v -m "not integration" --tb=short`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "[v5.0.0] shared data layer, new risk system, Coward/Gambler/Degenerate strategies"
```
