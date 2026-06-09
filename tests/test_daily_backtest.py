"""Daily backtester — drives the real Arena cycle over synthetic daily bars.

No network/S3: bars are built in-memory, so this exercises the actual
strategy + executor + account/risk code paths end to end.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.backtest.daily_backtest import run_daily_backtest
from edgefinder.core.models import TradeIntent
from edgefinder.strategies.base import BaseStrategy, StrategyRegistry


def _series(prices: list[float], symbol: str = "TEST") -> pd.DataFrame:
    start = date(2024, 1, 1)
    rows = []
    for i, p in enumerate(prices):
        rows.append({
            "date": start + timedelta(days=i),
            "open": p, "high": p * 1.01, "low": p * 0.99,
            "close": p, "volume": 1_000_000.0,
        })
    return pd.DataFrame(rows)


def _decline_then_rally() -> list[float]:
    prices = [120.0]
    for _ in range(40):           # sustained decline -> RSI < 35 (entry)
        prices.append(prices[-1] * 0.988)
    for _ in range(25):           # sharp rally -> hits target / RSI > 70 (exit)
        prices.append(prices[-1] * 1.02)
    return prices


def test_backtest_runs_real_engine_and_trades():
    bars = {"TEST": _series(_decline_then_rally())}
    result = run_daily_backtest("coward", bars, starting_cash=10_000.0)

    # Equity curve only starts once there's enough history for indicators
    assert result["equity_curve"], "expected a populated equity curve"
    assert result["stats"]["days"] > 0
    # Coward should enter on the oversold dip and exit on the rally
    assert result["stats"]["num_closed_trades"] >= 1
    trade = result["trades"][0]
    assert trade["exit_price"] is not None
    assert trade["pnl_dollars"] is not None
    assert any(
        trade["exit_reason"].startswith(p)
        for p in ("STOP_LOSS", "TARGET_HIT", "STRATEGY_EXIT",
                  "TRAILING_STOP", "TIME_EXIT")
    )
    # Final equity is internally consistent: cash + marked positions.
    assert result["final_equity"] > 0
    # Rich metrics are all present.
    s = result["stats"]
    for key in ("cagr_pct", "sharpe", "profit_factor", "avg_win",
                "avg_loss", "exposure_pct", "max_drawdown_pct"):
        assert key in s


def test_backtest_benchmark_excess_return():
    bars = {"TEST": _series(_decline_then_rally())}
    bench = {"symbol": "SPY", "return_pct": 3.0, "period": "2023-05-30..2026-05-26"}
    result = run_daily_backtest("coward", bars, starting_cash=10_000.0, benchmark=bench)
    s = result["stats"]
    assert s["benchmark_symbol"] == "SPY"
    assert s["benchmark_return_pct"] == 3.0
    assert s["excess_return_pct"] == round(s["return_pct"] - 3.0, 2)


def test_backtest_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        run_daily_backtest("nope", {"TEST": _series([100.0] * 40)})


def test_precompute_matches_live_indicator_engine():
    """The vectorised precompute must equal the live compute_indicators_from_bars
    on each prefix — that equivalence is what keeps the fast path faithful."""
    from edgefinder.backtest.daily_backtest import precompute_snapshots
    from edgefinder.data.indicator_engine import compute_indicators_from_bars

    df = _series(_decline_then_rally())
    snaps = precompute_snapshots(df)
    for i in (35, 45, 60, len(df) - 1):
        ref = compute_indicators_from_bars(df.iloc[: i + 1])
        got = snaps[i]
        for field in ("close", "rsi", "macd_line", "macd_signal", "bb_lower",
                      "ema_21", "atr", "volume_avg", "volume_ratio"):
            r, g = getattr(ref, field), getattr(got, field)
            if r is None:
                assert g is None, f"{field}@{i}: expected None, got {g}"
            else:
                assert g is not None and abs(g - r) < 1e-6, f"{field}@{i}: {g} vs {r}"


def test_backtest_does_not_touch_global_event_bus():
    """Backtest trades must never reach the live event bus (which persists to
    the trades table) — that leak corrupted live account balances."""
    from edgefinder.core.events import event_bus

    seen = []
    def _open(t): seen.append(("open", t))
    def _close(t): seen.append(("close", t))
    event_bus.subscribe("trade.opened", _open)
    event_bus.subscribe("trade.closed", _close)
    try:
        result = run_daily_backtest("coward", {"TEST": _series(_decline_then_rally())})
        assert result["stats"]["num_closed_trades"] >= 1  # trades really happened
        assert seen == []                                  # but none leaked out
    finally:
        event_bus.unsubscribe("trade.opened", _open)
        event_bus.unsubscribe("trade.closed", _close)


def test_backtest_steady_uptrend_no_entry():
    # A steady uptrend keeps RSI high and price near the upper band, so
    # coward's oversold / BB-lower entries never fire -> no trades, flat equity.
    bars = {"TEST": _series([100.0 * (1.004 ** i) for i in range(60)])}
    result = run_daily_backtest("coward", bars, starting_cash=10_000.0)
    assert result["stats"]["num_closed_trades"] == 0
    assert result["stats"]["num_open_positions"] == 0
    assert result["final_equity"] == 10_000.0


# ── look-ahead regression: entries fill at next-day open, not signal close ──


class _OneShotBuy(BaseStrategy):
    """Probe strategy: buys the first bar whose close >= 100, exactly once,
    and never exits. Lets a test pin the exact signal bar so the fill price
    is unambiguous."""

    # Risk knobs the arena's RiskManager/account read off the strategy.
    risk_pct = 0.02
    stop_pct = 0.08
    target_pct = 0.15
    max_concentration_pct = 0.20  # so sizing caps within the account's cap

    @property
    def name(self) -> str:
        return "_lookahead_probe"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def preferred_signals(self) -> list[str]:
        return []

    def init(self) -> None:
        pass

    def generate_signals(self, ticker, bars):
        return []

    def qualifies_stock(self, fundamentals) -> bool:
        return True

    def on_trade_executed(self, notification) -> None:
        pass

    def evaluate(self, ticker, data):
        if not getattr(self, "_fired", False) and data.current_price >= 100.0:
            self._fired = True
            return TradeIntent(
                ticker=ticker, direction="LONG",
                reasoning="probe", strategy_name=self.name,
            )
        return None


@pytest.fixture
def probe_strategy():
    StrategyRegistry._strategies["_lookahead_probe"] = _OneShotBuy
    try:
        yield
    finally:
        StrategyRegistry._strategies.pop("_lookahead_probe", None)


def _gap_series() -> pd.DataFrame:
    """12 warm-up days (close < 100, no fire), a signal bar (close == 100),
    then a gap-up open of 112 on the next bar where the fill must land."""
    start = date(2024, 1, 1)
    rows = []
    for i in range(12):                       # close 85..96, all < 100
        p = 85.0 + i
        rows.append({"date": start + timedelta(days=i),
                     "open": p, "high": p * 1.01, "low": p * 0.99,
                     "close": p, "volume": 1_000_000.0})
    # signal bar: close == 100 (probe fires). Its OWN open is 98.
    rows.append({"date": start + timedelta(days=12),
                 "open": 98.0, "high": 101.0, "low": 97.0,
                 "close": 100.0, "volume": 1_000_000.0})
    # fill bar: open gaps to 112 — the fill must land here, not at the 100 close.
    for j in range(13, 18):
        rows.append({"date": start + timedelta(days=j),
                     "open": 112.0, "high": 112.5, "low": 111.5,
                     "close": 112.0, "volume": 1_000_000.0})
    return pd.DataFrame(rows)


def test_entry_fills_at_next_day_open_not_signal_close(probe_strategy):
    # The probe fires on the close=100 bar; with look-ahead removed, the fill
    # must be the NEXT bar's open (112) — not the 100 close that generated the
    # signal, and not that bar's own 98 open.
    result = run_daily_backtest("_lookahead_probe", {"TEST": _gap_series()},
                                starting_cash=10_000.0)
    positions = result["open_positions"]
    assert len(positions) == 1, result
    entry = positions[0]["entry_price"]
    assert 112.0 <= entry <= 114.0, (
        f"entry {entry} must track the 112 next-day open (got look-ahead?)"
    )
    assert entry > 105.0  # decisively NOT the 100.0 signal-day close
    assert result["stats"]["num_closed_trades"] == 0


# ── warmup regression: folds must not run indicator-cold ────


def _gap_event_series(n_warm: int = 250) -> pd.DataFrame:
    """Flat-ish warmup, then a held 7% gap-up on heavy volume (gap_drift bait)."""
    start = date(2024, 1, 1)
    rows = []
    for i in range(n_warm):
        p = 100.0 + (i % 5) * 0.3  # mild noise so indicators are sane
        rows.append({"date": start + timedelta(days=i),
                     "open": p, "high": p * 1.01, "low": p * 0.99,
                     "close": p, "volume": 1_000_000.0})
    # gap day: opens +7%, holds into a strong close, 3x volume
    rows.append({"date": start + timedelta(days=n_warm),
                 "open": 107.0, "high": 108.2, "low": 106.8,
                 "close": 108.0, "volume": 3_000_000.0})
    # fill day + a few quiet days after
    for j in range(n_warm + 1, n_warm + 6):
        rows.append({"date": start + timedelta(days=j),
                     "open": 108.0, "high": 109.0, "low": 107.0,
                     "close": 108.0, "volume": 1_000_000.0})
    return pd.DataFrame(rows)


def test_fold_without_warmup_is_indicator_cold():
    # A 63-bar cold slice cannot satisfy gap_drift's ema_200 trend gate —
    # the bug that zeroed every fold before the warmup fix.
    df = _gap_event_series()
    tail = df.iloc[-63:].reset_index(drop=True)
    res = run_daily_backtest("gap_drift", {"TEST": tail}, starting_cash=10_000.0)
    assert res["stats"]["num_closed_trades"] == 0
    assert res["stats"]["num_open_positions"] == 0


def test_trade_start_warmup_enables_indicators():
    # Same window, but with warmup bars + trade_start: ema_200 is live and
    # the held gap fires; stats cover only the scored region.
    df = _gap_event_series()
    scored_start = df.iloc[-63]["date"]
    res = run_daily_backtest("gap_drift", {"TEST": df}, starting_cash=10_000.0,
                             trade_start=scored_start)
    opened = res["stats"]["num_closed_trades"] + res["stats"]["num_open_positions"]
    assert opened >= 1, res["stats"]
    assert res["stats"]["days"] <= 63  # equity curve covers scored days only


# ── market context: real per-day SPY state reaches strategies ────


class _ContextProbe(BaseStrategy):
    """Records the MarketContext it sees each day; never trades."""

    risk_pct = 0.02
    stop_pct = 0.20
    target_pct = 0.15
    max_concentration_pct = 0.20
    seen: list = []

    @property
    def name(self): return "_context_probe"
    @property
    def version(self): return "1.0"
    @property
    def preferred_signals(self): return []
    def init(self): pass
    def generate_signals(self, ticker, bars): return []
    def qualifies_stock(self, fundamentals): return True
    def on_trade_executed(self, n): pass

    def evaluate(self, ticker, data):
        type(self).seen.append(
            (data.context.spy_price, data.context.spy_sma_200,
             data.context.spy_uptrend))
        return None


def test_per_day_spy_context_reaches_strategies():
    StrategyRegistry._strategies["_context_probe"] = _ContextProbe
    _ContextProbe.seen = []
    try:
        bars = {"TEST": _series([100.0 + i * 0.1 for i in range(260)])}
        spy = _series([400.0 + i * 0.5 for i in range(260)])  # rising SPY
        run_daily_backtest("_context_probe", {"TEST": bars["TEST"]},
                           starting_cash=10_000.0, spy_bars=spy)
        assert _ContextProbe.seen, "probe never evaluated"
        # Every day saw a real SPY price (not the neutral 0.0)
        assert all(p > 0 for p, _, _ in _ContextProbe.seen)
        # Early days: sma200 unknown -> spy_uptrend is None (no information)
        assert _ContextProbe.seen[0][2] is None
        # Late days (>200 bars): sma200 known and rising SPY => uptrend True
        assert _ContextProbe.seen[-1][1] > 0
        assert _ContextProbe.seen[-1][2] is True
    finally:
        StrategyRegistry._strategies.pop("_context_probe", None)


# ── SPY cash-overlay: idle cash parked in SPY (zero-knob accounting layer) ────


def _spy_series(n: int, *, start_price: float = 100.0, daily_ret: float = 0.0,
                start: date = date(2024, 1, 1)) -> pd.DataFrame:
    rows, p = [], start_price
    for i in range(n):
        rows.append({"date": start + timedelta(days=i), "close": p})
        p *= (1 + daily_ret)
    return pd.DataFrame(rows)


def test_cash_overlay_idle_cash_earns_spy():
    # Steady uptrend → coward never enters → 100% idle cash the whole run.
    prices = [100.0 * (1.002 ** i) for i in range(60)]
    bars = {"TEST": _series(prices)}
    spy = _spy_series(60, daily_ret=0.01)  # SPY grinds up 1%/day

    plain = run_daily_backtest("coward", bars, starting_cash=10_000.0,
                               spy_bars=spy)
    overlay = run_daily_backtest("coward", bars, starting_cash=10_000.0,
                                 spy_bars=spy, cash_overlay=True)

    assert plain["stats"]["num_closed_trades"] == 0
    assert plain["final_equity"] == 10_000.0  # idle cash earns nothing
    assert overlay["stats"]["cash_overlay"] is True
    assert "cash_overlay" not in plain["stats"]

    # Overlay equity = initial sweep (5 bps fee) compounding at SPY's return
    # from the FIRST SCORED day's close to the last.
    first = date.fromisoformat(overlay["equity_curve"][0]["date"])
    days_held = (date(2024, 1, 1) + timedelta(days=59) - first).days
    expected = 10_000.0 * (1 - 0.0005) * (1.01 ** days_held)
    assert overlay["final_equity"] == pytest.approx(expected, rel=1e-3)
    assert overlay["final_equity"] > plain["final_equity"]


def test_cash_overlay_requires_spy_bars():
    bars = {"TEST": _series([100.0] * 40)}
    with pytest.raises(ValueError, match="spy_bars"):
        run_daily_backtest("coward", bars, starting_cash=10_000.0,
                           cash_overlay=True)


def test_cash_overlay_flat_spy_only_costs_fees(probe_strategy):
    # Flat SPY → the sweep adds no yield; the only difference vs the plain
    # run is conversion friction. The probe buys once (entries consume cash
    # mid-run) so this exercises the flow accounting, not just the accrual.
    bars = {"TEST": _gap_series()}
    spy = _spy_series(len(_gap_series()), daily_ret=0.0)

    plain = run_daily_backtest("_lookahead_probe", bars,
                               starting_cash=10_000.0, spy_bars=spy)
    overlay = run_daily_backtest("_lookahead_probe", bars,
                                 starting_cash=10_000.0, spy_bars=spy,
                                 cash_overlay=True)

    assert len(overlay["open_positions"]) == 1  # trades unchanged by overlay
    assert overlay["open_positions"] == plain["open_positions"]
    diff = plain["final_equity"] - overlay["final_equity"]
    # initial sweep fee (~$5) + sell-to-fund fee (~$1) — small and positive
    assert 0 < diff < 25.0, f"flat-SPY overlay drift should be fees only: {diff}"


# ── microcap cost model wiring (spread + impact + participation cap) ────


def test_cost_model_untradeable_thin_name_opens_nothing(probe_strategy):
    # With a min-ADV floor above the name's dollar volume, the liquidity cap
    # forces 0 shares — no position opens — while the no-cost path trades.
    from edgefinder.backtest.costs import CostModel
    bars = {"TEST": _gap_series()}
    no_cost = run_daily_backtest("_lookahead_probe", bars, starting_cash=10_000.0)
    capped = run_daily_backtest("_lookahead_probe", bars, starting_cash=10_000.0,
                                cost_model=CostModel(min_adv_dollars=1e12))
    assert len(no_cost["open_positions"]) == 1
    assert len(capped["open_positions"]) == 0


def test_cost_model_makes_entry_fill_worse(probe_strategy):
    # Same signal, same fill bar — the cost model pays the spread (+impact) so
    # the entry lands HIGHER than the flat-5bps fill.
    from edgefinder.backtest.costs import CostModel
    bars = {"TEST": _gap_series()}
    no_cost = run_daily_backtest("_lookahead_probe", bars, starting_cash=10_000.0)
    costed = run_daily_backtest("_lookahead_probe", bars, starting_cash=10_000.0,
                                cost_model=CostModel())
    assert len(costed["open_positions"]) == 1
    assert costed["open_positions"][0]["entry_price"] > \
        no_cost["open_positions"][0]["entry_price"]


def test_cost_model_only_reduces_pnl_end_to_end():
    # High-volume name (cap never binds), so the cost model changes ONLY fill
    # prices — round-trips can therefore only cost money, never make it.
    from edgefinder.backtest.costs import CostModel
    bars = {"TEST": _series(_decline_then_rally())}  # volume 1e6 → ADV huge
    no_cost = run_daily_backtest("coward", bars, starting_cash=10_000.0)
    costed = run_daily_backtest("coward", bars, starting_cash=10_000.0,
                                cost_model=CostModel())
    assert costed["stats"]["num_closed_trades"] == no_cost["stats"]["num_closed_trades"]
    assert costed["final_equity"] <= no_cost["final_equity"]


def test_delisted_position_is_force_closed_not_frozen(probe_strategy):
    # DEADCO trades for 15 days then its data ends (delisting) while ALIVE keeps
    # the calendar running to day 30. The held DEADCO position must close at its
    # last price with reason DELISTED — never freeze open to run-end.
    bars = {
        "ALIVE": _series([50.0] * 30),    # never triggers the probe (<100)
        "DEADCO": _series([105.0] * 15),  # triggers buy, then delists
    }
    result = run_daily_backtest("_lookahead_probe", bars, starting_cash=10_000.0)
    delisted = [t for t in result["trades"]
                if (t["exit_reason"] or "").startswith("DELISTED")]
    assert len(delisted) == 1, result["trades"]
    assert delisted[0]["symbol"] == "DEADCO"
    assert delisted[0]["exit_price"] == pytest.approx(105.0, rel=0.01)
    assert all(p["symbol"] != "DEADCO" for p in result["open_positions"])
