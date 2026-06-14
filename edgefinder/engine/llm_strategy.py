"""Blind LLM judgment strategy — trade off model judgment, not just technicals.

A strategy whose signal is an LLM's buy/sell/hold judgment over the
point-in-time context. ONE prompt per rebalance (the whole universe), back
comes a {label: weight} dict, normalized long-only (sum ≤ 1).

THE BLINDNESS PROBLEM (handled honestly). An LLM has MEMORIZED real market
history through its training cutoff. If the prompt named "AAPL" on
"2026-04-15" it could recall what happened next — look-ahead via training
data, the subtlest leak. The engine guarantees the DATA is point-in-time,
but not the model's memory. So the BACKTEST payload is ANONYMIZED:
- NO date (only an ordinal ``day_index``);
- tickers RELABELED "A", "B", "C", … (a private, decision-local map; the
  real symbol is NEVER sent);
- prices normalized (last-20 closes indexed to 100.0) + indicator values +
  trailing returns.
This REDUCES but cannot ELIMINATE memorization, so the backtest is
explicitly non-authoritative. Every prompt + raw response is logged to
``llm_decision_log``; decisions are cached by sha256(anonymized payload) in
``llm_decision_cache`` so a re-run reproduces identical trades (preserving
the reproducible-scorecard / re-check discipline). LIVE paper trading is the
real, uncompromised test: the future genuinely does not exist, so the agent
is blind by construction.

DEGRADES SAFELY: if CLAUDE_CODE_OAUTH_TOKEN is absent, or the call/parse
fails, ``rebalance`` returns {} (all cash) and logs — it NEVER crashes a
backtest or live cycle.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Protocol

from edgefinder.agents.reasoning import call_claude_json
from edgefinder.engine.strategy import AssetView, RebalanceContext

logger = logging.getLogger(__name__)

# A CHEAP model for the (non-authoritative) backtest by default.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM = (
    "You are a portfolio strategist. You are given an anonymized cross-section "
    "of assets, each identified ONLY by an opaque label (A, B, C, ...). For "
    "each asset you see a recent normalized price path (most recent 20 closes "
    "indexed so the last value is 100.0), a few technical indicator values, "
    "and trailing returns. You have NO ticker names and NO dates — judge only "
    "from the numbers in front of you.\n\n"
    "Decide a long-only target portfolio: pick the assets you would hold and "
    "assign each a weight in [0, 1]. Weights must sum to at most 1.0 (the rest "
    "is cash). It is fine to hold cash (return {}) when nothing looks good.\n\n"
    "Return ONLY a JSON object mapping label -> weight, e.g. "
    '{"A": 0.5, "C": 0.25}. No prose, no markdown fences.'
)


# ── pluggable cache (in-process default; DB-backed optional) ─────────────


class DecisionCache(Protocol):
    def get(self, context_hash: str) -> dict[str, float] | None: ...

    def put(self, context_hash: str, weights: dict[str, float], model: str) -> None: ...

    def log(self, context_hash: str, model: str, prompt: str,
            response: str | None, weights: dict[str, float] | None) -> None: ...


class InProcessCache:
    """Default cache — a plain dict, no DB. Logging is a no-op."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, float]] = {}

    def get(self, context_hash: str) -> dict[str, float] | None:
        return self._store.get(context_hash)

    def put(self, context_hash: str, weights: dict[str, float], model: str) -> None:
        self._store[context_hash] = dict(weights)

    def log(self, context_hash: str, model: str, prompt: str,
            response: str | None, weights: dict[str, float] | None) -> None:
        return None


class DBDecisionCache:
    """DB-backed cache + append-only log over ``llm_decision_cache`` /
    ``llm_decision_log``. Opens its own short-lived session per op (the
    Strategy protocol has no session seam). Any DB error degrades to a
    miss / no-op so a backtest or live cycle never crashes."""

    def __init__(self, session_factory: Any | None = None) -> None:
        self._sf = session_factory

    def _factory(self):
        if self._sf is not None:
            return self._sf
        from edgefinder.db.engine import get_engine, get_session_factory
        self._sf = get_session_factory(get_engine())
        return self._sf

    def get(self, context_hash: str) -> dict[str, float] | None:
        try:
            from edgefinder.db.models import LLMDecisionCache
            session = self._factory()()
            try:
                row = session.get(LLMDecisionCache, context_hash)
                return dict(row.weights_json) if row and row.weights_json else (
                    {} if row else None)
            finally:
                session.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning("llm cache get failed", exc_info=True)
            return None

    def put(self, context_hash: str, weights: dict[str, float], model: str) -> None:
        try:
            from edgefinder.db.models import LLMDecisionCache
            session = self._factory()()
            try:
                if session.get(LLMDecisionCache, context_hash) is None:
                    session.add(LLMDecisionCache(
                        context_hash=context_hash, weights_json=dict(weights),
                        model=model))
                    session.commit()
            finally:
                session.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning("llm cache put failed", exc_info=True)

    def log(self, context_hash: str, model: str, prompt: str,
            response: str | None, weights: dict[str, float] | None) -> None:
        try:
            from edgefinder.db.models import LLMDecisionLog
            session = self._factory()()
            try:
                session.add(LLMDecisionLog(
                    context_hash=context_hash, model=model, prompt=prompt,
                    response=response,
                    weights_json=dict(weights) if weights is not None else None))
                session.commit()
            finally:
                session.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning("llm decision log failed", exc_info=True)


# ── the strategy ─────────────────────────────────────────────────────────


class LLMStrategy:
    """Long-only LLM judgment over an anonymized, point-in-time universe."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        runner: Any | None = None,
        cache: DecisionCache | None = None,
    ) -> None:
        self.model = model
        self._runner = runner
        self._cache: DecisionCache = cache if cache is not None else InProcessCache()

    @property
    def name(self) -> str:
        return f"llm_blind_{self.model}"

    # -- payload construction (anonymized, dateless) --

    @staticmethod
    def _norm_path(a: AssetView, n: int = 20) -> list[float] | None:
        c = a.history["close"].iloc[-n:]
        if len(c) < 2:
            return None
        last = float(c.iloc[-1])
        if last <= 0:
            return None
        return [round(float(x) / last * 100.0, 4) for x in c]

    @staticmethod
    def _indicator_block(a: AssetView) -> dict[str, float]:
        ind = a.indicators
        out: dict[str, float] = {}
        if ind.rsi is not None:
            out["rsi"] = round(float(ind.rsi), 3)
        if ind.bb_width is not None:
            out["bb_width"] = round(float(ind.bb_width), 5)
        if ind.ema_50 and a.price:
            out["price_over_ema50"] = round(a.price / ind.ema_50, 4)
        if ind.ema_200 and a.price:
            out["price_over_ema200"] = round(a.price / ind.ema_200, 4)
        return out

    def _build_payload(self, ctx: RebalanceContext) -> tuple[dict, dict[str, str]]:
        """Return (anonymized payload, label->real-symbol map). Ordering is
        deterministic by anonymized label, so the hash is reproducible."""
        real_syms = sorted(ctx.assets)
        label_to_symbol: dict[str, str] = {}
        assets_block: dict[str, dict] = {}
        for i, sym in enumerate(real_syms):
            label = _label(i)
            label_to_symbol[label] = sym
            a = ctx.assets[sym]
            path = self._norm_path(a)
            if path is None:
                continue
            feats: dict[str, Any] = {"path": path}
            ind = self._indicator_block(a)
            if ind:
                feats["indicators"] = ind
            rets = {}
            for n in (5, 20, 60):
                r = a.ret(n)
                if r is not None:
                    rets[f"ret_{n}"] = round(r, 5)
            if rets:
                feats["returns"] = rets
            assets_block[label] = feats
        # day_index is an ORDINAL only — no calendar date is ever exposed.
        payload = {
            "schema": "llm_blind_v1",
            "day_index": len(ctx.assets.get(real_syms[0]).history) if real_syms else 0,
            "assets": assets_block,
        }
        return payload, label_to_symbol

    @staticmethod
    def _canonical(payload: dict) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    # -- response handling --

    @staticmethod
    def _map_and_normalize(
        raw_weights: dict, label_to_symbol: dict[str, str]
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for label, w in raw_weights.items():
            sym = label_to_symbol.get(str(label))
            if sym is None:
                continue  # unknown label dropped
            try:
                val = float(w)
            except (TypeError, ValueError):
                continue
            if val <= 0:  # clamp negatives / zeros out
                continue
            out[sym] = out.get(sym, 0.0) + val
        total = sum(out.values())
        if total > 1.0:
            out = {s: w / total for s, w in out.items()}  # scale to 1.0
        return out

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        if not ctx.assets:
            return {}
        payload, label_to_symbol = self._build_payload(ctx)
        if not payload["assets"]:
            return {}
        canonical = self._canonical(payload)
        context_hash = hashlib.sha256(canonical.encode()).hexdigest()

        cached = self._cache.get(context_hash)
        if cached is not None:
            # cache stores anonymized labels -> weight; remap to live symbols
            return self._map_and_normalize(cached, label_to_symbol)

        prompt = f"{_SYSTEM}\n\n## Anonymized universe\n\n{canonical}\n"
        try:
            raw_text = call_claude_json(prompt, self.model, self._runner)
        except Exception as exc:
            logger.warning("LLMStrategy call failed (%s) — holding cash", exc)
            self._cache.log(context_hash, self.model, prompt, None, None)
            return {}

        try:
            raw_weights = json.loads(raw_text)
            if not isinstance(raw_weights, dict):
                raise ValueError("response is not a JSON object")
        except Exception as exc:
            logger.warning("LLMStrategy parse failed (%s) — holding cash", exc)
            self._cache.log(context_hash, self.model, prompt, raw_text, None)
            return {}

        # cache the ANONYMIZED weights so a re-run on the same context (which
        # produces the same labels) reproduces identical trades.
        anon_clean = {str(k): v for k, v in raw_weights.items()
                      if str(k) in label_to_symbol}
        self._cache.put(context_hash, anon_clean, self.model)
        self._cache.log(context_hash, self.model, prompt, raw_text, anon_clean)
        return self._map_and_normalize(raw_weights, label_to_symbol)


def _label(i: int) -> str:
    """0->A, 25->Z, 26->AA, ... (base-26, deterministic)."""
    s = ""
    i += 1
    while i > 0:
        i, rem = divmod(i - 1, 26)
        s = chr(65 + rem) + s
    return s


def make_llm_strategy_factory(
    model: str | None = None,
    runner: Any | None = None,
    cache: DecisionCache | None = None,
):
    """Spec factory for ``llm_blind`` / ``llm_blind:<model>``.

    Live/backtest use a DB-backed cache+log by default (degrades to a miss
    on any DB error); tests inject a stub runner and an InProcessCache.
    """
    effective_model = model or DEFAULT_MODEL
    effective_cache = cache if cache is not None else DBDecisionCache()

    def _factory() -> LLMStrategy:
        return LLMStrategy(model=effective_model, runner=runner,
                           cache=effective_cache)

    return _factory
