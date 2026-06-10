"""Lab explorer API + the phase-4 strategies/ops/research additions."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from edgefinder.db.models import (
    AgentAction,
    AgentObservation,
    DailyBar,
    DividendCredit,
    PromotedStrategy,
    StrategyAccount,
    StrategyParameterLog,
    StrategySnapshot,
    TradeRecord,
    ValidationRun,
)


@pytest.fixture
def client(db_engine, db_session):
    from dashboard.app import app
    from dashboard.dependencies import get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with patch("dashboard.services.init_services"), \
         patch("dashboard.services.shutdown_services"):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


def _run(name, universe, *, all_met=False, holdout=None, excess=1.0,
         folds=None, hours_ago=0):
    return ValidationRun(
        strategy_name=name, universe=universe, git_sha="abc1234",
        run_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        config={"engine": "v2", "schedule": "monthly", "cost_bps": 2.0,
                "num_folds": 6, "costed": True},
        oos={"mean_excess_vs_spy_pct": excess, "folds_beating_spy": "4/6",
             "mean_excess_sharpe": -0.1, "mean_sharpe": 1.0,
             "total_trades": 500,
             **({"folds": folds} if folds else {})},
        criteria={"mode": "total_return", "all_met": all_met},
        holdout=holdout, verdict="PASS" if all_met else "FAIL",
    )


class TestLabRuns:
    def _seed(self, s):
        s.add(_run("xsec_mom_12_1", "hunt-r1:top500", all_met=True, excess=9.17,
                   folds=[{"window": "a..b", "regime": "bull_calm",
                           "excess_vs_spy_pct": 5.0, "trades": 100}]))
        s.add(_run("deep_value_pe10", "hunt-r1:top500-pit", all_met=True,
                   excess=8.71, hours_ago=1))
        s.add(_run("low_vol_50", "hunt-r1:top500", excess=-6.7, hours_ago=2))
        s.add(_run("equal_weight", "etf7+v2", excess=-0.8, hours_ago=3,
                   holdout={"window": "w", "passes": False}))
        s.commit()

    def test_list_filters_and_pagination(self, client, db_session):
        self._seed(db_session)
        r = client.get("/api/lab/runs").json()
        assert r["total"] == 4 and len(r["runs"]) == 4
        assert r["runs"][0]["strategy_name"] == "xsec_mom_12_1"  # newest first

        assert client.get("/api/lab/runs?label=hunt-r1").json()["total"] == 3
        assert client.get("/api/lab/runs?strategy=value").json()["total"] == 1
        assert client.get("/api/lab/runs?verdict=PASS").json()["total"] == 2
        assert client.get("/api/lab/runs?holdout=fail").json()["total"] == 1
        page = client.get("/api/lab/runs?limit=2&offset=2").json()
        assert page["total"] == 4 and len(page["runs"]) == 2

    def test_detail_includes_folds_and_disclosure(self, client, db_session):
        self._seed(db_session)
        run_id = client.get("/api/lab/runs?strategy=xsec").json()["runs"][0]["id"]
        d = client.get(f"/api/lab/runs/{run_id}").json()
        assert d["folds"][0]["regime"] == "bull_calm"
        assert "folds" not in d["oos"]            # popped out of oos
        assert d["cost_disclosure"]["lab_default_cost_bps"] == 2.0
        # legacy row without folds renders None
        rid2 = client.get("/api/lab/runs?strategy=low_vol").json()["runs"][0]["id"]
        assert client.get(f"/api/lab/runs/{rid2}").json()["folds"] is None

    def test_detail_404(self, client):
        assert client.get("/api/lab/runs/99999").status_code == 404

    def test_scoreboard(self, client, db_session):
        self._seed(db_session)
        db_session.add(PromotedStrategy(strategy_name="xsec_mom_12_1",
                                        spec="xsec_mom_12_1", symbols=["A"],
                                        schedule="monthly", tier="experimental",
                                        active=True))
        db_session.commit()
        sb = client.get("/api/lab/scoreboard").json()
        assert sb["target"] == 10
        names = {f["strategy_name"] for f in sb["finalists"]}
        assert names == {"xsec_mom_12_1", "deep_value_pe10"}
        assert sb["counts"]["criteria_passing"] == 2
        assert sb["counts"]["promoted"] == 1
        xsec = next(f for f in sb["finalists"] if f["strategy_name"] == "xsec_mom_12_1")
        assert xsec["promoted"] is True and xsec["tier"] == "experimental"

    def test_labels(self, client, db_session):
        self._seed(db_session)
        lbl = client.get("/api/lab/labels").json()
        assert "hunt-r1" in lbl["prefixes"]
        assert "etf7+v2" in lbl["universes"]


class TestRecordPersistsFolds:
    def test_folds_and_regimes_ride_in_oos(self, db_session):
        from edgefinder.engine.record import record_validation_run

        rid = record_validation_run(db_session, {
            "strategy": "x", "config": {}, "oos": {"mean_sharpe": 1.0},
            "criteria": {"all_met": False}, "holdout": None, "verdict": "FAIL",
            "folds": [{"window": "a..b"}], "by_regime": {"bull_calm": {"folds": 1}},
        }, universe="t")
        row = db_session.get(ValidationRun, rid)
        assert row.oos["folds"] == [{"window": "a..b"}]
        assert row.oos["by_regime"]["bull_calm"]["folds"] == 1


class TestStrategiesAdditions:
    def _seed(self, s):
        s.add(StrategyAccount(strategy_name="coward", starting_capital=5000,
                              cash_balance=4000, open_positions_value=1500,
                              total_equity=5500, peak_equity=5600,
                              drawdown_pct=1.0, realized_pnl=500))
        s.add(StrategyAccount(strategy_name="equal_weight",
                              starting_capital=100000, cash_balance=20000,
                              open_positions_value=85000, total_equity=105000,
                              peak_equity=106000, drawdown_pct=0.9,
                              realized_pnl=5000))
        s.add(PromotedStrategy(strategy_name="equal_weight", spec="equal_weight",
                               symbols=["SPY"], schedule="monthly",
                               tier="experimental", active=True))
        s.add(StrategySnapshot(strategy_name="coward",
                               timestamp=datetime.now(timezone.utc) - timedelta(days=1),
                               cash=4000, positions_value=1400, total_equity=5400,
                               drawdown_pct=0, total_return_pct=8.0))
        s.add(TradeRecord(trade_id="w1", strategy_name="coward", symbol="A",
                          direction="LONG", trade_type="SWING", entry_price=10,
                          shares=1, stop_loss=9, target=12, confidence=50,
                          entry_time=datetime.now(timezone.utc), status="CLOSED",
                          pnl_dollars=10))
        s.add(DividendCredit(strategy_name="equal_weight", symbol="SPY",
                             ex_date=datetime.now(timezone.utc).date(),
                             shares=10, amount=6.5))
        s.add(StrategyParameterLog(strategy_name="coward", param_name="rsi_min",
                                   old_value="30", new_value="35",
                                   changed_by="coach"))
        s.commit()

    def test_accounts_have_lanes(self, client, db_session):
        # every account reads lane "v2" now — the arena lane was retired
        # (the field is kept for API-shape stability)
        self._seed(db_session)
        rows = client.get("/api/strategies/accounts").json()
        lanes = {r["strategy_name"]: r["lane"] for r in rows}
        assert lanes["coward"] == "v2"
        assert lanes["equal_weight"] == "v2"

    def test_summary_lane_rollups(self, client, db_session):
        # keys stay {arena, v2, all} for API stability; arena is all zeros
        self._seed(db_session)
        s = client.get("/api/strategies/summary").json()
        assert s["arena"]["starting_capital"] == 0
        assert s["arena"]["strategies"] == 0
        assert s["v2"]["starting_capital"] == 105000
        assert s["all"]["total_equity"] == 110500
        assert s["all"]["total_pnl"] == 5500
        assert s["v2"]["day_pnl"] == pytest.approx(100.0)  # 5500 - 5400
        assert s["v2"]["win_rate"] == 100.0

    def test_promoted_dividends_params_meta(self, client, db_session):
        self._seed(db_session)
        promos = client.get("/api/strategies/promoted").json()
        assert promos[0]["strategy_name"] == "equal_weight"
        assert promos[0]["total_equity"] == 105000

        divs = client.get("/api/strategies/dividends?strategy=equal_weight").json()
        assert divs[0]["amount"] == 6.5

        params = client.get("/api/strategies/params").json()
        assert params[0]["param_name"] == "rsi_min"

        meta = client.get("/api/strategies/meta").json()
        assert any(m["lane"] == "v2" and m["name"] == "equal_weight" for m in meta)
        assert all("color_slot" in m for m in meta)


class TestOpsAdditions:
    def test_activity_merges_obs_and_actions(self, client, db_session):
        db_session.add(AgentObservation(agent_name="watchdog",
                                        timestamp=datetime.now(timezone.utc),
                                        severity="WARN", category="cash_drift",
                                        message="drift"))
        db_session.add(AgentAction(agent_name="coach",
                                   timestamp=datetime.now(timezone.utc),
                                   action_type="propose_param",
                                   summary="tune rsi", status="merged"))
        db_session.commit()
        items = client.get("/api/ops/activity").json()["items"]
        kinds = {i["kind"] for i in items}
        assert kinds == {"observation", "action"}

    def test_storage_db_side_without_r2(self, client, db_session, monkeypatch):
        import dashboard.routers.ops as ops_mod

        monkeypatch.setattr(ops_mod, "_storage_cache", None)
        db_session.add(DailyBar(symbol="AAA",
                                date=datetime.now(timezone.utc).date(),
                                open=1, high=1, low=1, close=1, volume=1,
                                source="t"))
        db_session.commit()
        out = client.get("/api/ops/storage").json()
        assert out["db"]["symbols"] == 1 and out["db"]["rows"] == 1
        assert out["r2"] is None     # no R2 env in tests
