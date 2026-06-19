"""The research-agent account's daily brain (the "AI analyst").

Runs as its OWN daily job, decoupled from the 9:45 live cycle: it screens a
universe through a set of entry rules, applies entry/exit hysteresis so the
book turns over on signals (not by daily reselection — the toll that sank the
fast-cadence hunt rounds), gathers news, and writes a decision to
``agent_decisions``. The live cycle's AnalystStrategy then trades that
decision; hold / add / trim / sell fall out of the engine diffing the target
against the current book.

Phase 1 is deterministic-first (the entry rules + their evidence ARE the
signal); an optional LLM step (``use_llm``) writes the plain-English rationale
on top. The deterministic path always produces the executable target, so the
account trades even with no model token, and tests run offline.

Look-ahead honesty: the decision is built from a point-in-time
RebalanceContext (data only through the decision date). The agent's edge is
proven FORWARD by the account's real track record — this is a live strategy,
not a sealed-backtest one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from edgefinder.engine.hunt_r1 import _high_ratio
from edgefinder.engine.strategy import AssetView, RebalanceContext

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
DEFAULT_NAME = "ai_analyst"
STARTING_CAPITAL = 100_000.0
HISTORY_DAYS = 450          # calendar days of bars for indicator warmup

# selection geometry — a narrow ENTER band inside a wide EXIT band keeps a held
# name until it clearly breaks down (the hysteresis turnover-killer)
DEFAULT_CAP = 12
DEFAULT_ENTER_RANK = 12
DEFAULT_EXIT_RANK = 25


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ── entry rules (pure functions of one point-in-time AssetView) ──────────


@dataclass
class RuleHit:
    rule: str
    score: float          # normalized to [0, 1]
    detail: str           # human-readable "why"


def score_trend(a: AssetView) -> RuleHit | None:
    """Confirmed uptrend: price above a rising 200-EMA."""
    ind = a.indicators
    if not ind.ema_200 or a.price <= 0:
        return None
    above = a.price / ind.ema_200 - 1.0
    if above <= 0:
        return None
    if ind.ema_50 and ind.ema_50 < ind.ema_200:   # structure must not be broken
        return None
    return RuleHit("trend", _clamp01(above / 0.30),
                   f"{above * 100:.1f}% above its 200-day average")


def score_breakout(a: AssetView) -> RuleHit | None:
    """At or near the trailing 52-week high."""
    hr = _high_ratio(a, 252)
    if hr is None or hr < 0.97:
        return None
    return RuleHit("breakout", _clamp01((hr - 0.90) / 0.10),
                   f"within {(1 - hr) * 100:.1f}% of its 52-week high")


def score_pullback(a: AssetView) -> RuleHit | None:
    """A dip (RSI 30-50) inside an established uptrend — buy the pullback."""
    ind = a.indicators
    if not ind.ema_200 or a.price <= ind.ema_200:
        return None
    if ind.rsi is None or not (30.0 <= ind.rsi <= 50.0):
        return None
    return RuleHit("pullback", _clamp01((50.0 - ind.rsi) / 20.0),
                   f"RSI {ind.rsi:.0f} pullback within an uptrend")


def score_rs_momentum(a: AssetView) -> RuleHit | None:
    """Strong trailing 6-month relative strength."""
    r = a.ret(126)
    if r is None or r <= 0:
        return None
    return RuleHit("rs_momentum", _clamp01(r / 0.50),
                   f"6-month return {r * 100:.0f}%")


RULE_BY_NAME = {
    "trend": score_trend,
    "breakout": score_breakout,
    "pullback": score_pullback,
    "rs_momentum": score_rs_momentum,
}
RULES = tuple(RULE_BY_NAME.values())


@dataclass
class Candidate:
    symbol: str
    composite: float
    hits: list[RuleHit]
    metrics: dict = field(default_factory=dict)


def screen(ctx: RebalanceContext) -> list[Candidate]:
    """Score every asset across the entry rules; best composite first.

    Composite = sum of the (normalized) scores of the rules that fired. A name
    that fires no rule is not a candidate."""
    out: list[Candidate] = []
    for sym, a in ctx.assets.items():
        hits = [h for rule in RULES if (h := rule(a)) is not None]
        if not hits:
            continue
        metrics = {
            "price": round(a.price, 4) if a.price else None,
            "ret_20": a.ret(20),
            "ret_126": a.ret(126),
            "rsi": round(a.indicators.rsi, 1) if a.indicators.rsi is not None else None,
            "high_ratio": _high_ratio(a, 252),
            "price_over_ema200": (round(a.price / a.indicators.ema_200, 4)
                                  if a.indicators.ema_200 else None),
        }
        out.append(Candidate(sym, round(sum(h.score for h in hits), 4), hits, metrics))
    out.sort(key=lambda c: (-c.composite, c.symbol))
    return out


def select_with_hysteresis(
    ranked: list[str], held: list[str], *,
    cap: int = DEFAULT_CAP, enter_rank: int = DEFAULT_ENTER_RANK,
    exit_rank: int = DEFAULT_EXIT_RANK,
) -> list[str]:
    """KEEP held names still inside the wide exit band; ADD new names from the
    narrow entry band up to ``cap``. Held names that fell out of the ranking
    entirely (no longer pass any rule) are dropped — that is the exit signal."""
    rank = {s: i for i, s in enumerate(ranked)}
    keep = [s for s in held if s in rank and rank[s] < exit_rank]
    book, seen = list(keep), set(keep)
    for s in ranked:
        if len(book) >= cap:
            break
        if s in seen:
            continue
        if rank[s] < enter_rank:
            book.append(s)
            seen.add(s)
    return book[:cap]


# ── evidence: news ───────────────────────────────────────────────────────


def gather_news(session, symbols: list[str], *, per_symbol: int = 3) -> dict[str, list]:
    """Most recent headlines per symbol from the ticker_news table (best-effort)."""
    try:
        from edgefinder.db.models import TickerNews
    except Exception:  # pragma: no cover
        return {}
    out: dict[str, list] = {}
    for s in symbols:
        try:
            rows = (session.query(TickerNews.title, TickerNews.published_utc)
                    .filter(TickerNews.symbol == s)
                    .order_by(TickerNews.published_utc.desc())
                    .limit(per_symbol).all())
        except Exception:
            session.rollback()
            continue
        if rows:
            out[s] = [{"title": t, "published": p} for t, p in rows]
    return out


# ── decision assembly ────────────────────────────────────────────────────


def _action(current_w: float, target_w: float) -> str:
    if current_w <= 0:
        return "buy"
    if target_w <= 0:
        return "sell"
    if target_w > current_w * 1.05:
        return "add"
    if target_w < current_w * 0.95:
        return "trim"
    return "hold"


def build_decision(
    selected: list[str], candidates: list[Candidate],
    holdings: dict[str, float], news: dict[str, list],
) -> tuple[dict[str, float], list[dict]]:
    """Equal-weight the selected book; build the per-name dossier list.

    Returns ``(target_weights, picks)``. ``picks`` also lists currently-held
    names that were NOT selected (action 'sell', weight 0) so the report
    explains every exit."""
    cand_by_sym = {c.symbol: c for c in candidates}
    weight = round(1.0 / len(selected), 6) if selected else 0.0
    target = {s: weight for s in selected}

    picks: list[dict] = []
    for s in selected:
        c = cand_by_sym.get(s)
        cur = holdings.get(s, 0.0)
        hits = c.hits if c else []
        rationale = ("; ".join(h.detail for h in hits)
                     or "held from a prior decision")
        picks.append({
            "symbol": s,
            "target_weight": weight,
            "current_weight": round(cur, 6),
            "action": _action(cur, weight),
            "composite": c.composite if c else None,
            "rules": [h.rule for h in hits],
            "metrics": c.metrics if c else {},
            "news": news.get(s, []),
            "rationale": rationale,
        })
    # exits: held names that didn't make the cut
    for s, cur in holdings.items():
        if cur > 0 and s not in target:
            picks.append({
                "symbol": s, "target_weight": 0.0,
                "current_weight": round(cur, 6), "action": "sell",
                "composite": None, "rules": [], "metrics": {},
                "news": news.get(s, []),
                "rationale": "no longer passes the entry screen — exit",
            })
    return target, picks


class _RuleStrategy:
    """A rule expressed as a portfolio strategy: equal-weight every name that
    currently fires the rule. Used only to BACKTEST the rule's track record
    (the "proof") — never traded live."""

    def __init__(self, rule_name: str, rule_fn) -> None:
        self._name = f"rule_{rule_name}"
        self._rule = rule_fn

    @property
    def name(self) -> str:
        return self._name

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        fired = [s for s, a in ctx.assets.items() if self._rule(a) is not None]
        if not fired:
            return {}
        w = 1.0 / len(fired)
        return {s: w for s in fired}


def rule_track_records(bars: dict, rule_names: list[str]) -> dict[str, dict]:
    """Backtest each named entry rule over ``bars`` -> compact track-record stats.

    The honesty backbone of a pick: "buy the names this rule fires on, hold,
    rebalance monthly — here's the historical result." Stats only (no curve);
    the dossier's chart is the ticker's own price history (fetched live)."""
    from edgefinder.engine.backtest import run_backtest

    out: dict[str, dict] = {}
    for name in rule_names:
        fn = RULE_BY_NAME.get(name)
        if fn is None:
            continue
        try:
            res = run_backtest(bars, _RuleStrategy(name, fn),
                               start_cash=100_000.0, schedule="monthly",
                               warmup_days=210)
            st = res.stats
            out[name] = {
                "return_pct": st.get("return_pct"),
                "sharpe": st.get("sharpe"),
                "max_drawdown_pct": st.get("max_drawdown_pct"),
                "excess_vs_spy_pct": st.get("excess_return_pct"),
                "trades": st.get("num_trades"),
            }
        except Exception:
            logger.warning("rule track-record backtest failed for %s", name,
                           exc_info=True)
    return out


def _deterministic_summary(picks: list[dict]) -> str:
    buys = [p["symbol"] for p in picks if p["action"] in ("buy", "add")]
    sells = [p["symbol"] for p in picks if p["action"] == "sell"]
    holds = [p["symbol"] for p in picks if p["action"] in ("hold", "trim")]
    parts = [f"{len(holds)} held"]
    if buys:
        parts.append(f"entering/adding {', '.join(buys)}")
    if sells:
        parts.append(f"exiting {', '.join(sells)}")
    return "Daily screen: " + "; ".join(parts) + "."


def _llm_summary(picks: list[dict], model: str, runner: Any | None) -> str | None:
    """Optional plain-English narration of the day's decision (graceful)."""
    import json

    from edgefinder.agents.reasoning import call_claude_json
    payload = [{k: p[k] for k in ("symbol", "action", "rules", "metrics", "rationale")}
               for p in picks]
    prompt = (
        "You are an equity analyst explaining today's trade decisions to a "
        "non-expert, in 3-5 plain sentences. Each item lists a ticker, the "
        "action, which entry rules fired, and key metrics. Explain simply why "
        "this basket makes sense today. Return ONLY a JSON object "
        '{"summary": "..."}.\n\n' + json.dumps(payload, default=str))
    try:
        raw = call_claude_json(prompt, model, runner)
        return str(json.loads(raw).get("summary") or "").strip() or None
    except Exception:
        logger.warning("analyst LLM summary failed — using deterministic", exc_info=True)
        return None


# ── orchestration ────────────────────────────────────────────────────────


def run_analyst(
    session_factory, *,
    strategy_name: str = DEFAULT_NAME,
    today: date | None = None,
    symbols: list[str] | None = None,
    universe_spec: str | None = None,
    rank_window: int = 126,
    cap: int = DEFAULT_CAP,
    enter_rank: int = DEFAULT_ENTER_RANK,
    exit_rank: int = DEFAULT_EXIT_RANK,
    use_llm: bool = False,
    with_proof: bool = True,
    model: str = "claude-haiku-4-5-20251001",
    runner: Any | None = None,
) -> int | None:
    """Run one day of research and persist the decision; return its row id.

    Holdings-aware (reads the account's open lots) so the hysteresis can hold
    existing positions. Returns None if there are no bars / no decision date
    yet (e.g. an empty dev DB)."""
    from edgefinder.engine.backtest import _build_context, prepare_bars
    from edgefinder.engine.data import load_bars
    from edgefinder.db.models import AgentDecision, DailyBar
    from edgefinder.trading.journal import TradeJournal

    today = today or datetime.now(ET).date()
    session = session_factory()
    try:
        decision_date_row = (session.query(DailyBar.date)
                             .filter(DailyBar.date < today)
                             .order_by(DailyBar.date.desc()).first())
        if decision_date_row is None:
            logger.warning("analyst: no completed bars before %s", today)
            return None
        decision_date = decision_date_row[0]

        # current book (shares + cash) so the agent can hold/add/trim/sell
        journal = TradeJournal(session)
        open_lots = journal.get_open_trades(strategy_name)
        holdings_shares: dict[str, int] = {}
        for t in open_lots:
            holdings_shares[t.symbol] = holdings_shares.get(t.symbol, 0) + t.shares
        cash = _recalc_cash(session, journal, strategy_name)

        universe = _resolve_universe(session, symbols, universe_spec,
                                     decision_date, rank_window)
        load_syms = sorted(set(universe) | set(holdings_shares))
        bars = load_bars(session, load_syms,
                         start=decision_date - timedelta(days=HISTORY_DAYS))
        bars = {s: df for s, df in bars.items() if len(df)}
        if not bars:
            logger.warning("analyst: no bars loaded for %d symbols", len(load_syms))
            return None
        prep, _ = prepare_bars(bars)
        universe_set = set(universe) | set(holdings_shares)
        ctx = _build_context({s: p for s, p in prep.items() if s in universe_set},
                             decision_date, None,
                             holdings_shares=holdings_shares, cash=cash)

        candidates = screen(ctx)
        ranked = [c.symbol for c in candidates]
        selected = select_with_hysteresis(
            ranked, ctx.held_symbols(), cap=cap,
            enter_rank=enter_rank, exit_rank=exit_rank)
        news = gather_news(session, selected)
        target, picks = build_decision(selected, candidates, ctx.holdings, news)

        # attach the backtested track record of each pick's firing rules
        if with_proof and picks:
            fired_rules = sorted({r for p in picks for r in p.get("rules", [])})
            tracks = rule_track_records(bars, fired_rules) if fired_rules else {}
            for p in picks:
                p["proof"] = {r: tracks[r] for r in p.get("rules", []) if r in tracks}

        summary = (_llm_summary(picks, model, runner) if use_llm else None) \
            or _deterministic_summary(picks)

        row = (session.query(AgentDecision)
               .filter_by(strategy_name=strategy_name, decision_date=decision_date)
               .one_or_none())
        if row is None:
            row = AgentDecision(strategy_name=strategy_name,
                                decision_date=decision_date)
            session.add(row)
        row.target_weights = target
        row.picks = picks
        row.summary = summary
        row.model = model if use_llm else None
        session.commit()
        logger.info("analyst %s @ %s: %d picks, %d sells",
                    strategy_name, decision_date, len(selected),
                    sum(1 for p in picks if p["action"] == "sell"))
        return row.id
    finally:
        session.close()


def _recalc_cash(session, journal, strategy_name: str) -> float:
    """Account cash from the trades table + dividend credits (reuses the live
    engine's canonical formula; STARTING_CAPITAL for a fresh account)."""
    try:
        from edgefinder.engine.live import _recalc_cash as live_recalc
        return live_recalc(session, journal, strategy_name)
    except Exception:
        logger.warning("analyst: cash recalc fell back to starting capital",
                       exc_info=True)
        return STARTING_CAPITAL


def _resolve_universe(session, symbols, universe_spec, decision_date, rank_window):
    if symbols:
        return [s.strip().upper() for s in symbols if s and s.strip()]
    if not universe_spec:
        raise ValueError("run_analyst needs symbols or universe_spec")
    from edgefinder.db.models import DailyBar
    from edgefinder.engine.data import (
        parse_universe_spec, resolve_universe, trailing_rank_start,
    )
    top_n, offset = parse_universe_spec(universe_spec)
    calendar = [r[0] for r in session.query(DailyBar.date)
                .filter(DailyBar.symbol == "SPY", DailyBar.date <= decision_date)
                .order_by(DailyBar.date).all()]
    rank_start = (trailing_rank_start(calendar, decision_date, rank_window)
                  if calendar else None)
    return resolve_universe(session, "top", [], top_n, as_of=decision_date,
                            rank_offset=offset, rank_start=rank_start)


def main(argv: list[str] | None = None) -> None:
    """Manual run: ``python -m edgefinder.agents.analyst --symbols A,B,C``."""
    import argparse
    import json

    from edgefinder.db.engine import get_engine, get_session_factory

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--symbols", default=None, help="comma-separated fixed list")
    p.add_argument("--universe", default=None, metavar="top:N")
    p.add_argument("--date", default=None, help="cycle date (YYYY-MM-DD)")
    p.add_argument("--use-llm", action="store_true")
    args = p.parse_args(argv)

    syms = [s for s in (args.symbols.split(",") if args.symbols else [])]
    rid = run_analyst(
        get_session_factory(get_engine()),
        strategy_name=args.name,
        today=date.fromisoformat(args.date) if args.date else None,
        symbols=syms or None, universe_spec=args.universe, use_llm=args.use_llm)
    print(json.dumps({"decision_id": rid}, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
