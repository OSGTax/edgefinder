"""Event-driven minute-bar backtester.

Feeds historical minute bars (long-format DataFrame, one row per
symbol-minute) to a strategy in timestamp order and simulates a single
cash account with market-on-close fills, proportional slippage, and a
flat per-trade commission. Tracks positions, realized P&L, and an equity
curve marked to the latest seen price for every held symbol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)

_BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Order:
    symbol: str
    side: str  # "BUY" | "SELL"
    quantity: int


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float


@dataclass
class Fill:
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float


@runtime_checkable
class BacktestStrategy(Protocol):
    def on_bar(self, bar: Bar, ctx: "BacktestContext") -> list[Order] | None:
        """Return orders to submit at this bar's close (or None for no-op)."""
        ...


@dataclass
class BacktestContext:
    """Read-only view of account state handed to the strategy each bar."""

    engine: "BacktestEngine"
    bar: Bar

    @property
    def cash(self) -> float:
        return self.engine.cash

    def position(self, symbol: str) -> Position | None:
        return self.engine.positions.get(symbol)

    def price(self, symbol: str) -> float | None:
        return self.engine.last_prices.get(symbol)

    def equity(self) -> float:
        return self.engine.equity()


@dataclass
class BacktestResult:
    starting_cash: float
    final_equity: float
    realized_pnl: float
    fills: list[Fill]
    equity_curve: list[tuple[datetime, float]]

    @property
    def return_pct(self) -> float:
        if self.starting_cash == 0:
            return 0.0
        return (self.final_equity - self.starting_cash) / self.starting_cash * 100.0

    @property
    def num_fills(self) -> int:
        return len(self.fills)

    def equity_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])


class BacktestEngine:
    def __init__(
        self,
        starting_cash: float = 10_000.0,
        *,
        slippage: float = 0.0005,
        commission: float = 0.0,
    ) -> None:
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.slippage = slippage
        self.commission = commission
        self.positions: dict[str, Position] = {}
        self.last_prices: dict[str, float] = {}
        self.fills: list[Fill] = []
        self.realized_pnl = 0.0

    def equity(self) -> float:
        held = sum(p.quantity * self.last_prices.get(s, p.avg_price) for s, p in self.positions.items())
        return self.cash + held

    def _execute(self, order: Order, bar: Bar) -> None:
        if order.quantity <= 0:
            return
        if order.side == "BUY":
            price = bar.close * (1 + self.slippage)
            cost = order.quantity * price + self.commission
            if cost > self.cash:
                affordable = int((self.cash - self.commission) // price)
                if affordable <= 0:
                    logger.debug("Skip BUY %s: insufficient cash", order.symbol)
                    return
                order = Order(order.symbol, "BUY", affordable)
                cost = order.quantity * price + self.commission
            self.cash -= cost
            pos = self.positions.get(order.symbol)
            if pos:
                total_qty = pos.quantity + order.quantity
                pos.avg_price = (pos.avg_price * pos.quantity + price * order.quantity) / total_qty
                pos.quantity = total_qty
            else:
                self.positions[order.symbol] = Position(order.symbol, order.quantity, price)
            self.fills.append(Fill(bar.timestamp, order.symbol, "BUY", order.quantity, price, self.commission))
        elif order.side == "SELL":
            pos = self.positions.get(order.symbol)
            if not pos or pos.quantity <= 0:
                return
            qty = min(order.quantity, pos.quantity)
            price = bar.close * (1 - self.slippage)
            self.cash += qty * price - self.commission
            self.realized_pnl += (price - pos.avg_price) * qty - self.commission
            pos.quantity -= qty
            if pos.quantity == 0:
                del self.positions[order.symbol]
            self.fills.append(Fill(bar.timestamp, order.symbol, "SELL", qty, price, self.commission))
        else:
            logger.warning("Unknown order side: %s", order.side)

    def run(self, bars: pd.DataFrame, strategy: BacktestStrategy) -> BacktestResult:
        """Run the strategy over a long-format minute-bar DataFrame."""
        missing = [c for c in _BAR_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(f"bars missing required columns: {missing}")
        if bars.empty:
            return BacktestResult(self.starting_cash, self.cash, 0.0, [], [])

        ordered = bars.sort_values(["timestamp", "symbol"]).itertuples(index=False)
        equity_by_ts: dict[datetime, float] = {}

        for row in ordered:
            bar = Bar(
                symbol=str(row.symbol),
                timestamp=row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            self.last_prices[bar.symbol] = bar.close
            orders = strategy.on_bar(bar, BacktestContext(self, bar)) or []
            for order in orders:
                self._execute(order, bar)
            equity_by_ts[bar.timestamp] = self.equity()

        equity_curve = sorted(equity_by_ts.items())
        return BacktestResult(
            starting_cash=self.starting_cash,
            final_equity=self.equity(),
            realized_pnl=self.realized_pnl,
            fills=self.fills,
            equity_curve=equity_curve,
        )


def load_minute_bars(client, days, symbols: list[str]) -> pd.DataFrame:
    """Load minute bars for the given days/symbols into a long DataFrame.

    Returns columns: symbol, timestamp, open, high, low, close, volume.
    """
    frames: list[pd.DataFrame] = []
    for day in days:
        df = client.read_minute_aggs(day, symbols=symbols)
        if df.empty:
            continue
        frames.append(
            df[["ticker", "timestamp", "open", "high", "low", "close", "volume"]].rename(
                columns={"ticker": "symbol"}
            )
        )
    if not frames:
        return pd.DataFrame(columns=_BAR_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str)
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
