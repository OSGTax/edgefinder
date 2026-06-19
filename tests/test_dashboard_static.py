"""Static asset + template hygiene guards for the dashboard redesign.

Grows with the redesign phases: vendored assets must actually be served
(a 404 here means broken charts in production with no test failure
anywhere else), and — from phase 1 on — templates must carry zero inline
styles (they break the runtime theme toggle; tokens/utility classes only).
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

TEMPLATES = Path(__file__).parent.parent / "dashboard" / "templates"


@pytest.fixture()
def client():
    with patch("dashboard.services.init_services"), \
         patch("dashboard.services.shutdown_services"):
        from dashboard.app import app

        with TestClient(app) as c:
            yield c


def test_vendored_chart_library_is_served(client):
    r = client.get("/static/vendor/lightweight-charts.standalone.production.js")
    assert r.status_code == 200
    assert "Lightweight Charts" in r.text[:300]
    assert "v4.1.0" in r.text[:300]


NEW_ASSETS = [
    "/static/css/tokens.css",
    "/static/css/base.css",
    "/static/css/components.css",
    "/static/css/charts.css",
    "/static/js/core/net.js",
    "/static/js/core/fmt.js",
    "/static/js/core/dom.js",
    "/static/js/core/theme.js",
    "/static/js/core/charts.js",
    "/static/js/core/nav.js",
    "/static/js/core/poll.js",
    "/static/js/components/sparkline.js",
    "/static/js/components/heatmap.js",
    "/static/js/pages/symbol.js",
    "/static/js/pages/lab.js",
    "/static/js/pages/picks.js",
    "/static/js/pages/portfolio.js",
    "/static/js/pages/trades.js",
    "/static/js/pages/strategies.js",
    "/static/js/pages/screener.js",
    "/static/js/components/treemap.js",
    "/static/js/pages/ops.js",
]


@pytest.mark.parametrize("path", NEW_ASSETS)
def test_redesign_assets_are_served(client, path):
    r = client.get(path)
    assert r.status_code == 200, path
    assert len(r.content) > 100, path


def test_base_template_loads_vendored_not_cdn_for_lightweight_charts():
    html = (TEMPLATES / "base.html").read_text()
    assert "/static/vendor/lightweight-charts" in html
    assert "cdn.jsdelivr.net/npm/lightweight-charts" not in html


def _inline_styles(name: str) -> list[str]:
    return re.findall(r'style="[^"]*"', (TEMPLATES / name).read_text())


# End state (redesign phase 9): EVERY template is inline-style-free.
STYLE_FREE_TEMPLATES: list[str] = sorted(
    p.name for p in TEMPLATES.glob("*.html"))


@pytest.mark.parametrize("name", STYLE_FREE_TEMPLATES)
def test_templates_have_no_inline_styles(name):
    assert _inline_styles(name) == [], f"{name} has inline styles"


def test_symbol_page_routes(client):
    assert client.get("/symbol").status_code == 200
    assert client.get("/symbol/AAPL").status_code == 200


def test_research_redirects_to_symbol(client):
    r = client.get("/research", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/symbol"
    r = client.get("/research?ticker=aapl", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/symbol/AAPL"


def test_lab_page_and_backtest_redirect(client):
    assert client.get("/lab").status_code == 200
    r = client.get("/backtest", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/lab"


def test_no_cdn_script_tags_remain():
    html = (TEMPLATES / "base.html").read_text()
    assert "cdn.jsdelivr.net" not in html
    assert "unpkg.com" not in html


def test_ops_page(client):
    assert client.get("/ops").status_code == 200


def test_picks_page(client):
    assert client.get("/picks").status_code == 200


def test_legacy_assets_are_gone(client):
    assert client.get("/static/js/common.js").status_code == 404
    assert client.get("/static/css/theme.css").status_code == 404
