"""Plan items D1 (bearer auth) and D2 (tight CORS)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import dashboard.app as app_module
    return TestClient(app_module.app)


class TestBearerAuth:
    def test_no_token_env_means_open_dashboard(self, client, monkeypatch):
        monkeypatch.delenv("EDGEFINDER_DASHBOARD_TOKEN", raising=False)
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_token_required_when_set(self, client, monkeypatch):
        monkeypatch.setenv("EDGEFINDER_DASHBOARD_TOKEN", "s3cret")
        r = client.get("/api/strategies")
        assert r.status_code == 401

    def test_valid_token_accepted(self, client, monkeypatch):
        monkeypatch.setenv("EDGEFINDER_DASHBOARD_TOKEN", "s3cret")
        r = client.get("/api/strategies", headers={"authorization": "Bearer s3cret"})
        # Route may return any non-401 — we just want the auth check to pass.
        assert r.status_code != 401

    def test_wrong_token_rejected(self, client, monkeypatch):
        monkeypatch.setenv("EDGEFINDER_DASHBOARD_TOKEN", "s3cret")
        r = client.get("/api/strategies", headers={"authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_health_exempt_even_with_token(self, client, monkeypatch):
        monkeypatch.setenv("EDGEFINDER_DASHBOARD_TOKEN", "s3cret")
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_root_dashboard_exempt(self, client, monkeypatch):
        monkeypatch.setenv("EDGEFINDER_DASHBOARD_TOKEN", "s3cret")
        r = client.get("/")
        # Page is template-rendered; auth middleware should let it through.
        assert r.status_code == 200

    def test_missing_bearer_prefix_rejected(self, client, monkeypatch):
        monkeypatch.setenv("EDGEFINDER_DASHBOARD_TOKEN", "s3cret")
        r = client.get("/api/strategies", headers={"authorization": "s3cret"})
        assert r.status_code == 401


class TestCorsOrigins:
    def test_default_cors_is_localhost_only(self, monkeypatch):
        monkeypatch.delenv("EDGEFINDER_CORS_ORIGINS", raising=False)
        import dashboard.app as app_module
        origins = app_module._cors_origins()
        assert "http://localhost:8000" in origins
        assert "*" not in origins

    def test_configured_origins_parse(self, monkeypatch):
        monkeypatch.setenv(
            "EDGEFINDER_CORS_ORIGINS", "https://a.example,https://b.example",
        )
        import dashboard.app as app_module
        origins = app_module._cors_origins()
        assert origins == ["https://a.example", "https://b.example"]

    def test_single_wildcard_parse(self, monkeypatch):
        monkeypatch.setenv("EDGEFINDER_CORS_ORIGINS", "*")
        import dashboard.app as app_module
        origins = app_module._cors_origins()
        assert origins == ["*"]
