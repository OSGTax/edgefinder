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
    assert w["caps"] == {"page": 8000, "total": 40000}
    assert w["slugs"] == ["playbook", "setups", "lessons", "mistakes",
                          "postmortems", "market-notes"]


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

    assert WIKI_PAGE_MAX_CHARS == 8000 and WIKI_TOTAL_MAX_CHARS == 40000
    over = set_wiki(store, slug="playbook", body="x" * (WIKI_PAGE_MAX_CHARS + 1))
    assert not over["ok"] and "size cap" in over["error"]
    assert store.select("desk_wiki") == []
    assert store.select("desk_journal", filters={"kind": "wiki"}) == []

    # fill five pages to within 100 chars of the total, then the sixth must
    # respect the TOTAL cap (its body alone is comfortably under the page cap)
    for slug in ("playbook", "setups", "lessons", "mistakes"):
        assert set_wiki(store, slug=slug, body="x" * WIKI_PAGE_MAX_CHARS)["ok"]
    room = 100
    assert set_wiki(store, slug="postmortems",
                    body="x" * (WIKI_TOTAL_MAX_CHARS
                                - 4 * WIKI_PAGE_MAX_CHARS - room))["ok"]
    blocked = set_wiki(store, slug="market-notes", body="x" * (room + 1))
    assert not blocked["ok"] and "prune another page" in blocked["error"]
    # a rewrite that SHRINKS a page is always allowed
    assert set_wiki(store, slug="playbook", body="short now")["ok"]
    assert set_wiki(store, slug="market-notes", body="fits now")["ok"]


def test_total_cap_math_across_all_six_pages(store):
    """The total cap holds across the full six-slug wiki: five maxed pages fit
    exactly; the sixth then has zero headroom until another page shrinks."""
    from agent.brain import (WIKI_PAGE_MAX_CHARS, WIKI_SLUGS,
                             WIKI_TOTAL_MAX_CHARS, get_wiki, set_wiki)

    assert len(WIKI_SLUGS) == 6
    assert 5 * WIKI_PAGE_MAX_CHARS == WIKI_TOTAL_MAX_CHARS
    for slug in WIKI_SLUGS[:5]:
        assert set_wiki(store, slug=slug, body="x" * WIKI_PAGE_MAX_CHARS)["ok"]
    assert get_wiki(store)["total_chars"] == WIKI_TOTAL_MAX_CHARS
    blocked = set_wiki(store, slug=WIKI_SLUGS[5], body="x")
    assert not blocked["ok"] and "prune another page" in blocked["error"]
    # shrink one page → the sixth fits, and the total stays under the cap
    assert set_wiki(store, slug=WIKI_SLUGS[0], body="lean")["ok"]
    fits = set_wiki(store, slug=WIKI_SLUGS[5], body="y" * 100)
    assert fits["ok"] and fits["total_chars"] <= WIKI_TOTAL_MAX_CHARS


def test_wiki_pages_canonical_order(store):
    from agent.brain import get_wiki, set_wiki
    set_wiki(store, slug="market-notes", body="c")
    set_wiki(store, slug="postmortems", body="d")
    set_wiki(store, slug="playbook", body="a")
    set_wiki(store, slug="setups", body="e")
    set_wiki(store, slug="mistakes", body="b")
    assert [p["slug"] for p in get_wiki(store)["pages"]] == \
        ["playbook", "setups", "mistakes", "postmortems", "market-notes"]


def test_new_slugs_accepted_and_validation_intact(store):
    from agent.brain import set_wiki

    assert set_wiki(store, slug="setups", body="breakout-pullback: 3 graded "
                    "instances, +1.2% avg alpha")["ok"]
    assert set_wiki(store, slug="postmortems",
                    body="2026-07-10 NVDA: +2.1%, alpha +0.8%")["ok"]
    bad = set_wiki(store, slug="secret-plans", body="x")
    assert not bad["ok"] and "unknown wiki slug" in bad["error"]
    assert len(store.select("desk_wiki")) == 2


# ── revision history (the wiki's memory) ──


def test_history_banked_on_edit_not_on_first_write(store):
    from agent.brain import set_wiki

    set_wiki(store, slug="lessons", body="v1 body", title="Lessons",
             run_id="R1")
    assert store.select("desk_wiki_history") == []   # nothing prior to bank

    set_wiki(store, slug="lessons", body="v2 body", run_id="R2")
    hist = store.select("desk_wiki_history", filters={"slug": "lessons"})
    assert len(hist) == 1
    assert hist[0]["revision"] == 1 and hist[0]["body"] == "v1 body"
    assert hist[0]["title"] == "Lessons"
    assert hist[0]["updated_run_id"] == "R1"         # who wrote the banked rev

    set_wiki(store, slug="lessons", body="v3 body", run_id="R3")
    hist = store.select("desk_wiki_history", filters={"slug": "lessons"},
                        order=[("revision", "asc")])
    assert [h["revision"] for h in hist] == [1, 2]
    assert hist[1]["body"] == "v2 body" and hist[1]["updated_run_id"] == "R2"
    # the live page is untouched by the archive
    live = store.select("desk_wiki", filters={"slug": "lessons"})
    assert len(live) == 1 and live[0]["body"] == "v3 body"
    assert live[0]["revision"] == 3


def test_wiki_history_read_path(store):
    from agent.brain import get_wiki_history, set_wiki

    set_wiki(store, slug="mistakes", body="first " * 100, run_id="R1")
    set_wiki(store, slug="mistakes", body="second", run_id="R2")
    set_wiki(store, slug="mistakes", body="third", run_id="R3")

    listing = get_wiki_history(store, slug="mistakes")
    assert listing["ok"] and listing["slug"] == "mistakes"
    assert [r["revision"] for r in listing["revisions"]] == [2, 1]  # newest first
    assert listing["revisions"][1]["chars"] == 600
    assert len(listing["revisions"][1]["preview"]) <= 200  # finding aid, clipped

    one = get_wiki_history(store, slug="mistakes", revision=1)
    assert one["ok"] and one["body"] == "first " * 100  # full body on demand
    assert one["updated_run_id"] == "R1"

    missing = get_wiki_history(store, slug="mistakes", revision=99)
    assert not missing["ok"] and "no archived revision" in missing["error"]
    bad = get_wiki_history(store, slug="nope")
    assert not bad["ok"] and "unknown wiki slug" in bad["error"]


class _WrapStore:
    """Delegating store wrapper for fault injection."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_history_bank_not_duplicated_on_retry(store, caplog):
    """L3: bank succeeds → page write fails → the RETRY must not bank the same
    (slug, revision) again — the newest archived row already carrying this
    revision skips the re-bank."""
    import logging

    from agent.brain import set_wiki

    class _FailPageWriteOnce(_WrapStore):
        tripped = False

        def update(self, table, filters, values, *, returning=True):
            if table == "desk_wiki" and not self.tripped:
                self.tripped = True
                raise ConnectionError("reset after the history bank")
            return self._inner.update(table, filters, values,
                                      returning=returning)

    assert set_wiki(store, slug="lessons", body="v1", run_id="R1")["ok"]
    flaky = _FailPageWriteOnce(store)
    with pytest.raises(ConnectionError):
        set_wiki(flaky, slug="lessons", body="v2", run_id="R2")
    hist = store.select("desk_wiki_history", filters={"slug": "lessons"})
    assert [h["revision"] for h in hist] == [1]      # banked exactly once

    with caplog.at_level(logging.INFO, logger="agent.brain"):
        r = set_wiki(flaky, slug="lessons", body="v2", run_id="R2")
    assert r["ok"] and r["revision"] == 2
    assert "already banked" in caplog.text
    hist = store.select("desk_wiki_history", filters={"slug": "lessons"})
    assert [h["revision"] for h in hist] == [1]      # STILL exactly once
    live = store.select("desk_wiki", filters={"slug": "lessons"})
    assert live[0]["body"] == "v2" and live[0]["revision"] == 2

    # the dedup never suppresses a REAL subsequent edit's bank
    assert set_wiki(store, slug="lessons", body="v3", run_id="R3")["ok"]
    hist = store.select("desk_wiki_history", filters={"slug": "lessons"},
                        order=[("revision", "asc")])
    assert [h["revision"] for h in hist] == [1, 2]


def test_set_wiki_missing_history_table_exits_actionably(store):
    """Pre-deploy grace: a DB without desk_wiki_history gets an actionable
    message, not a stack trace — now classified by type/code, not str()."""
    from agent.brain import set_wiki
    from edgefinder.db.engine import Base, get_engine

    assert set_wiki(store, slug="lessons", body="v1")["ok"]
    Base.metadata.tables["desk_wiki_history"].drop(get_engine())
    r = set_wiki(store, slug="lessons", body="v2")
    assert not r["ok"] and "not migrated" in r["error"]


def test_set_wiki_transient_history_error_reraises(store):
    """M3: a transient error that merely MENTIONS desk_wiki_history (SQLAlchemy
    embeds the SQL in str(exc)) must re-raise, not be misdiagnosed as an
    unmigrated schema."""
    from agent.brain import set_wiki

    assert set_wiki(store, slug="lessons", body="v1")["ok"]

    class _Blip(_WrapStore):
        def select(self, table, **kw):
            if table == "desk_wiki_history":
                raise RuntimeError('connection reset during "SELECT revision '
                                   'FROM desk_wiki_history"')
            return self._inner.select(table, **kw)

    with pytest.raises(RuntimeError, match="connection reset"):
        set_wiki(_Blip(store), slug="lessons", body="v2")
