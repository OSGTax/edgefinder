"""OCC option-symbol helpers — pure functions, no I/O.

OCC format: ``{ROOT}{YYMMDD}{C|P}{STRIKE*1000 :08d}`` — e.g. NVDA Jan-16-2027
$200 call = ``NVDA270116C00200000``. The ledger stores contracts under their
OCC symbol in the same ``symbol`` column as equities; ``is_option`` is how
every layer tells the two apart.
"""

from __future__ import annotations

import re
from datetime import date

_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def is_option(symbol: str) -> bool:
    return bool(_OCC_RE.match((symbol or "").strip().upper()))


def parse(symbol: str) -> dict:
    """OCC symbol → {underlying, expiry(date), type 'C'|'P', strike(float)}.
    Raises ValueError on a non-OCC symbol."""
    m = _OCC_RE.match((symbol or "").strip().upper())
    if not m:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    root, ymd, cp, strike = m.groups()
    return {"underlying": root,
            "expiry": date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6])),
            "type": cp,
            "strike": int(strike) / 1000.0}


def build(underlying: str, expiry: date, type_: str, strike: float) -> str:
    """Compose an OCC symbol (the inverse of parse)."""
    type_ = type_.upper().strip()
    if type_ not in ("C", "P"):
        raise ValueError(f"option type must be C or P, got {type_!r}")
    return (f"{underlying.upper().strip()}{expiry.strftime('%y%m%d')}"
            f"{type_}{int(round(strike * 1000)):08d}")


def describe(symbol: str) -> str:
    """Human-readable: NVDA270116C00200000 → 'NVDA $200C 2027-01-16'."""
    p = parse(symbol)
    strike = f"{p['strike']:g}"
    return f"{p['underlying']} ${strike}{p['type']} {p['expiry'].isoformat()}"


def intrinsic(symbol: str, underlying_price: float) -> float:
    """Per-share intrinsic value at the given underlying price (≥ 0)."""
    p = parse(symbol)
    if p["type"] == "C":
        return max(0.0, float(underlying_price) - p["strike"])
    return max(0.0, p["strike"] - float(underlying_price))
