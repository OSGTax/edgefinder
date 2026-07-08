"""Lessons-wiki tool tests: fixed slugs, in-place curation, caps, audit trail."""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'wiki.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def test_empty_wiki(store):
    from agent.brain import get_wiki
    w = get_wiki(store)
    assert w["pages"] == [] and w["total_chars"] == 0
    assert w["caps"] == {"page": 4000, "total": 12000}
    assert w["slugs"][0] == "playbook"


def test_create_then_edit_in_place_with_journal(store):
    from agent.brain import get_wiki, set_wiki

    r1 = set_wiki(store, slug="lessons", body="Momentum works in risk-on.",
                  title="Lessons", reason="seed", run_id="R1")
    assert r1["ok"] and r1["revision"] == 1
    r2 = set_wiki(store, slug="lessons",
                  body="Momentum works in risk-on; fade it near earnings.",
                  reason="NVDA +3.1% exit confirmed the earnings fade", run_id="R2")
    assert r2["ok"] and r2["revision"] == 2

    rows = store.select("desk_wiki", filters={"slug": "lessons"})
    assert len(rows) == 1                       # edited IN PLACE, one row
    assert "fade it near earnings" in rows[0]["body"]
    assert rows[0]["revision"] == 2 and rows[0]["updated_run_id"] == "R2"
    assert rows[0]["title"] == "Lessons"        # title survives an edit without --title

    journal = store.select("desk_journal", filters={"kind": "wiki"})
    assert len(journal) == 2                    # one audit note per edit
    assert "wiki/lessons r2" in journal[1]["title"]
    assert "NVDA +3.1%" in journal[1]["title"]

    w = get_wiki(store, slug="lessons")
    assert len(w["pages"]) == 1 and w["pages"][0]["revision"] == 2


def test_unknown_slug_rejected(store):
    from agent.brain import set_wiki
    r = set_wiki(store, slug="secret-plans", body="x")
    assert not r["ok"] and "unknown wiki slug" in r["error"]
    assert store.select("desk_wiki") == []


def test_page_and_total_caps(store):
    from agent.brain import (WIKI_PAGE_MAX_CHARS, WIKI_TOTAL_MAX_CHARS,
                             set_wiki)

    over = set_wiki(store, slug="playbook", body="x" * (WIKI_PAGE_MAX_CHARS + 1))
    assert not over["ok"] and "size cap" in over["error"]
    assert store.select("desk_wiki") == []
    assert store.select("desk_journal", filters={"kind": "wiki"}) == []

    # fill three pages to the max, then the fourth must respect the TOTAL cap
    for slug in ("playbook", "lessons", "mistakes"):
        assert set_wiki(store, slug=slug, body="x" * WIKI_PAGE_MAX_CHARS)["ok"]
    room = WIKI_TOTAL_MAX_CHARS - 3 * WIKI_PAGE_MAX_CHARS  # == 0 with defaults
    blocked = set_wiki(store, slug="market-notes", body="x" * (room + 1))
    assert not blocked["ok"] and "prune another page" in blocked["error"]
    # a rewrite that SHRINKS a page is always allowed
    assert set_wiki(store, slug="playbook", body="short now")["ok"]
    assert set_wiki(store, slug="market-notes", body="fits now")["ok"]


def test_wiki_pages_canonical_order(store):
    from agent.brain import get_wiki, set_wiki
    set_wiki(store, slug="market-notes", body="c")
    set_wiki(store, slug="playbook", body="a")
    set_wiki(store, slug="mistakes", body="b")
    assert [p["slug"] for p in get_wiki(store)["pages"]] == \
        ["playbook", "mistakes", "market-notes"]
