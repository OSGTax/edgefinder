"""Phase 1 mechanics: risk caps that let accounts diversify, and exits that
recycle capital so trades actually complete (the live stall: 0 closed trades).
"""

from datetime import datetime, timedelta, timezone

from config.settings import settings
from edgefinder.trading.account import Position, VirtualAccount
from edgefinder.trading.arena import ArenaEngine
from edgefinder.trading.risk import RiskManager


class _DummyProvider:
    def get_bars(self, *a, **k):
        return None

    def get_latest_price(self, *a, **k):
        return None


# ── Position sizing: concentration cap ──────────────────────────────────


def test_compute_shares_uncapped_can_consume_whole_account():
    # risk_pct/stop_pct == 1.0 -> 100% of equity in one name (the degenerate bug)
    rm = RiskManager(risk_pct=0.20, stop_pct=0.20)
    assert rm.compute_shares(100.0, 10_000.0) == 100  # 100 * $100 = full $10k


def test_compute_shares_respects_concentration_cap():
    rm = RiskManager(risk_pct=0.20, stop_pct=0.20)
    # 20% of $10k / $100 = 20 shares max
    assert rm.compute_shares(100.0, 10_000.0, max_concentration_pct=0.20) == 20


# ── can_open_position: count + concentration gates ──────────────────────


def _pos(sym: str) -> Position:
    return Position(
        symbol=sym, shares=1, entry_price=1.0, stop_loss=0.8, target=1.2,
        direction="LONG", trade_type="SWING",
    )


def test_rejects_beyond_max_open_positions():
    acct = VirtualAccount("x", starting_capital=100_000.0)
    for i in range(settings.max_open_positions):
        acct.positions.append(_pos(f"S{i}"))
    allowed, reason = acct.can_open_position(10.0, symbol="NEW")
    assert not allowed
    assert "Max open positions" in reason


def test_rejects_over_concentration():
    acct = VirtualAccount("x", starting_capital=10_000.0, max_concentration_pct=0.20)
    allowed, reason = acct.can_open_position(3_000.0, symbol="BIG")  # >20% of 10k
    assert not allowed
    assert "concentration" in reason.lower()


def test_affordability_checked_before_concentration():
    acct = VirtualAccount("x", starting_capital=100.0)
    allowed, reason = acct.can_open_position(500.0)
    assert not allowed
    assert "Insufficient" in reason


# ── Exits that recycle capital ──────────────────────────────────────────


def _arena_with_position(*, entry, stop, target, entry_time, peak):
    arena = ArenaEngine(_DummyProvider())
    arena.load_strategies()
    slot = arena._slots["coward"]
    pos = Position(
        symbol="ABC", shares=10, entry_price=entry, stop_loss=stop, target=target,
        direction="LONG", trade_type="SWING", entry_time=entry_time,
        trade_id="t1", market_price=entry, peak_price=peak,
    )
    slot.account.positions.append(pos)
    slot.account.cash -= pos.cost_basis
    return arena, slot


def test_time_exit_closes_stale_position():
    # Held past max_hold_days, price between stop and target -> neither fires.
    old = datetime.now(timezone.utc) - timedelta(days=settings_max_hold() + 5)
    arena, slot = _arena_with_position(
        entry=100.0, stop=80.0, target=115.0, entry_time=old, peak=100.0
    )
    closed = arena._check_exits(slot, {}, {"ABC": {"price": 100.0}})
    assert len(closed) == 1
    assert closed[0].exit_reason == "TIME_EXIT"
    assert len(slot.account.positions) == 0


def test_trailing_stop_closes_after_pullback_from_peak():
    # Up >=1R (peak 125 vs entry 100, risk 20 -> arms at 120), then pulls back
    # below peak*(1-0.10)=112.5, while still under the 115 target.
    now = datetime.now(timezone.utc)
    arena, slot = _arena_with_position(
        entry=100.0, stop=80.0, target=115.0, entry_time=now, peak=125.0
    )
    closed = arena._check_exits(slot, {}, {"ABC": {"price": 112.0}})
    assert len(closed) == 1
    assert closed[0].exit_reason == "TRAILING_STOP"


def test_trailing_stop_does_not_arm_before_1R():
    # Only up ~5% (no peak above +1R) -> trailing must NOT fire on a dip.
    now = datetime.now(timezone.utc)
    arena, slot = _arena_with_position(
        entry=100.0, stop=80.0, target=115.0, entry_time=now, peak=105.0
    )
    closed = arena._check_exits(slot, {}, {"ABC": {"price": 101.0}})
    assert closed == []
    assert len(slot.account.positions) == 1


def settings_max_hold() -> int:
    from edgefinder.strategies.base import StrategyRegistry
    coward = next(s for s in StrategyRegistry.get_instances() if s.name == "coward")
    return coward.max_hold_days
