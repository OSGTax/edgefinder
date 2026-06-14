"""Tests for the blind LLM judgment strategy (engine/llm_strategy.py).

No test invokes a real `claude -p` — the subprocess runner is always stubbed
(mirroring tests/test_reasoning.py's _RecordingRunner). These prove the
anonymization hides ticker + date, weights map back and normalize, the cache
makes a re-run skip the call, and every failure mode degrades to all-cash.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.data.market_data import IndicatorSnapshot
from edgefinder.engine.llm_strategy import (
    InProcessCache,
    LLMStrategy,
    make_llm_strategy_factory,
)
from edgefinder.engine.strategy import AssetView, RebalanceContext


# ── stubbed claude -p runner (no real call, ever) ──────────────────────────


class _FakeCompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _RecordingRunner:
    def __init__(self, result: _FakeCompletedProcess):
        self.result = result
        self.calls: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, **kwargs})
        return self.result


def _envelope(payload: dict) -> str:
    return json.dumps({"type": "result", "result": json.dumps(payload)})


def _runner_for(weights: dict) -> _RecordingRunner:
    return _RecordingRunner(_FakeCompletedProcess(stdout=_envelope(weights)))


# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fake_oauth_token(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-token-for-tests")


def _bars(closes, start=date(2024, 1, 2)):
    rows = []
    d = start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        rows.append({"date": d, "open": float(c), "high": float(c) * 1.01,
                     "low": float(c) * 0.99, "close": float(c),
                     "volume": 1_000_000.0})
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def _asset(symbol, price=100.0, n=120):
    closes = [price * 1.001 ** i for i in range(n)]
    df = _bars(closes)
    ind = IndicatorSnapshot(close=closes[-1], ema_50=closes[-1] * 0.98,
                            ema_200=closes[-1] * 0.9, rsi=55.0, bb_width=0.04)
    return AssetView(symbol=symbol, price=closes[-1], indicators=ind, history=df)


def _ctx():
    # AAPL/MSFT — real-looking tickers we assert never reach the prompt
    assets = {"AAPL": _asset("AAPL", 150.0), "MSFT": _asset("MSFT", 300.0)}
    return RebalanceContext(date=date(2024, 6, 3), assets=assets)


def _captured_prompt(runner: _RecordingRunner) -> str:
    return runner.calls[0]["input"]


# ── (a) anonymization: no real ticker, no date in the prompt ───────────────


def test_prompt_is_anonymized_no_ticker_no_date():
    runner = _runner_for({"A": 0.5, "B": 0.5})
    strat = LLMStrategy(runner=runner, cache=InProcessCache())
    strat.rebalance(_ctx())
    prompt = _captured_prompt(runner)
    assert "AAPL" not in prompt
    assert "MSFT" not in prompt
    assert "2024" not in prompt  # no year / date leaks
    assert "2024-06-03" not in prompt
    # opaque labels are present
    assert '"A"' in prompt and '"B"' in prompt


# ── (b) labels map back to the right real symbols, sum <= 1 ────────────────


def test_labels_map_back_to_real_symbols():
    runner = _runner_for({"A": 0.5, "B": 0.5})  # A=AAPL, B=MSFT (sorted)
    strat = LLMStrategy(runner=runner, cache=InProcessCache())
    w = strat.rebalance(_ctx())
    assert set(w) == {"AAPL", "MSFT"}
    assert abs(w["AAPL"] - 0.5) < 1e-9 and abs(w["MSFT"] - 0.5) < 1e-9
    assert sum(w.values()) <= 1.0 + 1e-9


# ── (c) negative / oversized weights are clamped + scaled ──────────────────


def test_negative_and_oversized_weights_clamped_and_scaled():
    runner = _runner_for({"A": -0.3, "B": 3.0})  # negative dropped, rest scaled
    strat = LLMStrategy(runner=runner, cache=InProcessCache())
    w = strat.rebalance(_ctx())
    assert "AAPL" not in w  # negative dropped
    assert abs(w["MSFT"] - 1.0) < 1e-9  # 3.0 scaled down to full 1.0
    assert sum(w.values()) <= 1.0 + 1e-9


def test_unknown_label_is_dropped():
    runner = _runner_for({"A": 0.4, "ZZZ": 0.4})
    strat = LLMStrategy(runner=runner, cache=InProcessCache())
    w = strat.rebalance(_ctx())
    assert set(w) == {"AAPL"}


# ── (d) cache hit avoids a second runner call ──────────────────────────────


def test_cache_hit_avoids_second_call():
    runner = _runner_for({"A": 0.5, "B": 0.5})
    cache = InProcessCache()
    strat = LLMStrategy(runner=runner, cache=cache)
    ctx = _ctx()
    w1 = strat.rebalance(ctx)
    w2 = strat.rebalance(ctx)  # identical context -> cache hit
    assert w1 == w2
    assert len(runner.calls) == 1  # second decision served from cache


# ── (e) missing token -> {} not crash ──────────────────────────────────────


def test_missing_token_holds_cash(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    runner = _runner_for({"A": 0.5})
    strat = LLMStrategy(runner=runner, cache=InProcessCache())
    assert strat.rebalance(_ctx()) == {}
    assert runner.calls == []  # never reached the subprocess


# ── (f) malformed response -> {} not crash ─────────────────────────────────


def test_malformed_response_holds_cash():
    runner = _RecordingRunner(_FakeCompletedProcess(
        stdout=json.dumps({"type": "result", "result": "not json at all {{"})))
    strat = LLMStrategy(runner=runner, cache=InProcessCache())
    assert strat.rebalance(_ctx()) == {}


def test_non_dict_response_holds_cash():
    runner = _RecordingRunner(_FakeCompletedProcess(stdout=_envelope([1, 2, 3])))
    strat = LLMStrategy(runner=runner, cache=InProcessCache())
    assert strat.rebalance(_ctx()) == {}


def test_runner_nonzero_exit_holds_cash():
    runner = _RecordingRunner(_FakeCompletedProcess(returncode=1, stderr="boom"))
    strat = LLMStrategy(runner=runner, cache=InProcessCache())
    assert strat.rebalance(_ctx()) == {}


def test_empty_universe_holds_cash():
    runner = _runner_for({"A": 1.0})
    strat = LLMStrategy(runner=runner, cache=InProcessCache())
    assert strat.rebalance(RebalanceContext(date=date(2024, 6, 3), assets={})) == {}
    assert runner.calls == []


# ── factory wiring ─────────────────────────────────────────────────────────


def test_factory_builds_with_injected_runner_and_cache():
    runner = _runner_for({"A": 0.5, "B": 0.5})
    factory = make_llm_strategy_factory(
        model="claude-haiku-4-5-20251001", runner=runner, cache=InProcessCache())
    strat = factory()
    assert strat.name == "llm_blind_claude-haiku-4-5-20251001"
    w = strat.rebalance(_ctx())
    assert set(w) == {"AAPL", "MSFT"}
