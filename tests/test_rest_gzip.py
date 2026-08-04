"""The REST transport asks for gzip and decodes it — Supabase bills wire bytes.

The stdlib urllib client does not negotiate compression on its own; before
this, every PostgREST JSON payload crossed the wire uncompressed and counted
fully against the Supabase egress quota (the proximate cause of the 2026-08
exceed_egress_quota outage). These tests pin: the Accept-Encoding header is
sent, gzip bodies decode, plain bodies still work, and a gzipped ERROR body
still yields a readable RestError.
"""

import gzip
import io
import json
import urllib.error

import pytest

from agent.rest import Rest, RestError


class _FakeResponse:
    def __init__(self, body: bytes, *, gzipped: bool, status: int = 200):
        self._body = body
        self.status = status
        self.headers = {"Content-Encoding": "gzip"} if gzipped else {}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def client():
    return Rest(url="https://example.supabase.co", key="test-key")


def test_accept_encoding_header_sent(client, monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["headers"] = dict(req.header_items())
        return _FakeResponse(b"[]", gzipped=False)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client.select("desk_trades", limit=1)
    # urllib title-cases header names on the wire
    assert seen["headers"].get("Accept-encoding") == "gzip"


def test_gzipped_body_decodes(client, monkeypatch):
    rows = [{"symbol": "SPY", "close": 500.0}]
    body = gzip.compress(json.dumps(rows).encode())
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeResponse(body, gzipped=True))
    assert client.select("daily_bars", limit=1) == rows


def test_plain_body_still_decodes(client, monkeypatch):
    rows = [{"symbol": "QQQ"}]
    body = json.dumps(rows).encode()
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeResponse(body, gzipped=False))
    assert client.select("daily_bars", limit=1) == rows


def test_gzipped_error_body_readable(client, monkeypatch):
    err_body = gzip.compress(b'{"message":"quota exceeded"}')

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 402, "Payment Required",
            {"Content-Encoding": "gzip"}, io.BytesIO(err_body))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RestError) as exc:
        client.select("daily_bars", limit=1)
    assert exc.value.status == 402
    assert "quota exceeded" in exc.value.body
