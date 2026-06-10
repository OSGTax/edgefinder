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


def test_base_template_loads_vendored_not_cdn_for_lightweight_charts():
    html = (TEMPLATES / "base.html").read_text()
    assert "/static/vendor/lightweight-charts" in html
    assert "cdn.jsdelivr.net/npm/lightweight-charts" not in html


def _inline_styles(name: str) -> list[str]:
    return re.findall(r'style="[^"]*"', (TEMPLATES / name).read_text())


# Coverage expands as each page is rebuilt (redesign phases 1-9). The end
# state is every template in dashboard/templates/ inline-style-free.
STYLE_FREE_TEMPLATES: list[str] = []


@pytest.mark.parametrize("name", STYLE_FREE_TEMPLATES or ["__none__"])
def test_templates_have_no_inline_styles(name):
    if name == "__none__":
        pytest.skip("no templates migrated yet (redesign phase 0)")
    assert _inline_styles(name) == [], f"{name} has inline styles"
