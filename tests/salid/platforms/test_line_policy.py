"""LINE User Data Policy: 3.2.3 (no stored friend / group information) and 3.2.5 (no linking)."""

import pytest
from conftest import FixedClockSettings
from platform_fakes import Clock, inc, make_assistant, stored
from test_line import BOT, FakeLineApi, mev

from salid.platforms import core, line
from salid.platforms.line import LineAdapter, LinePort


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


@pytest.fixture
def fx(s, tmp_path):
    api = FakeLineApi()
    clock = Clock(500.0)
    a = make_assistant(s, tmp_path)
    port = LinePort(api, a.ps, clock=clock)
    return api, a, port, LineAdapter(a, port, BOT, clock=clock), clock


async def test_a_stored_group_message_has_neither_the_group_name_nor_a_display_name(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(mev("hello team", mid="m1"), BOT)
    await a.drain()
    group = inc(platform="line", tenant=BOT, room="Cgroup")
    rows = stored(a, group)
    assert [r.text for r in rows] == ["hello team"]
    dump = " ".join(f"{r.author_name} {r.text}" for r in rows)
    assert "Alice" not in dump and "Ops team" not in dump
    assert rows[0].author_name.startswith("LINE user")
    # the name the summaries are built under is an id-based label too
    assert "Ops team" not in a.names.values() and "LINE group" in a.names.values()


async def test_a_stored_one_to_one_message_keeps_no_display_name(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(mev("hi", src="user", mid="m2"), BOT)
    await a.drain()
    rows = stored(a, inc(platform="line", tenant=BOT, room="Ualice"))
    assert rows and all("Alice" not in r.author_name for r in rows)
    assert "LINE 1:1" in a.names.values()


async def test_a_ledger_receipt_from_line_carries_an_id_label_not_the_display_name(fx, monkeypatch):
    api, a, port, adapter, clock = fx
    seen = {}
    monkeypatch.setattr(core, "ingest_attachment", lambda *args, **kw: seen.update(kw))
    msg = inc(platform="line", tenant=BOT, room="Cgroup", user="Ualice", user_name="Alice")
    actions = a._file_actions(msg, 1, 2, "en")
    from salid.qa import FileInput

    actions.record_receipt(FileInput(index=0, data=b"x", filename="a.png", mime="image/png"), "")
    assert "Alice" not in seen["submitter_name"] and seen["submitter_name"].startswith("LINE user")


async def test_other_platforms_keep_their_names(fx, monkeypatch):
    api, a, port, adapter, clock = fx
    seen = {}
    monkeypatch.setattr(core, "ingest_attachment", lambda *args, **kw: seen.update(kw))
    msg = inc(platform="slack", tenant="T1", room="C1", user="U1", user_name="Ann")
    from salid.qa import FileInput

    a._file_actions(msg, 1, 2, "en").record_receipt(
        FileInput(index=0, data=b"x", filename="a.png", mime="image/png"), ""
    )
    assert seen["submitter_name"] == "Ann"


async def test_name_caches_expire_within_24_hours_and_are_pruned(fx):
    api, a, port, adapter, clock = fx
    assert line.NAME_TTL_S <= 24 * 3600
    await port.display_name("group", "Cgroup", "Ualice")
    await port.group_name("group", "Cgroup")
    assert len(port._names) == 2
    api.names["Ualice"] = "Alice2"
    clock.advance(line.NAME_TTL_S + 1)
    assert await port.display_name("group", "Cgroup", "Ualice") == "Alice2"  # looked up again
    await port.display_name("group", "Cgroup", "Ubob")  # a write drops the expired entries
    assert all(clock() - at < line.NAME_TTL_S for at, _ in port._names.values())


def test_expired_name_entries_are_gone_even_with_no_later_write(fx):
    import asyncio

    api, a, port, adapter, clock = fx
    asyncio.run(port.display_name("group", "Cgroup", "Ualice"))
    assert len(port._names) == 1
    clock.advance(25 * 3600)
    assert port._lookup(("group", "Cgroup", "Ualice")) is None  # a read of an expired entry
    assert port._names == {}


async def test_no_fetched_line_name_reaches_the_model_or_a_persisted_answer(fx):
    """Two addressed LINE group turns: the model's history and the stored Salid answer carry only
    the id-based labels, never the display name or the group name."""
    from platform_fakes import FakeAsk

    from salid.qa import Answer

    api, a, port, adapter, clock = fx
    seen: list[str] = []

    class Echo(FakeAsk):
        def __call__(self, settings, store, question, **kw):
            seen.append(f"{question}\n{kw.get('history', '')}")
            return Answer(f"you said: {kw.get('history', '')}", 0, 0.0, [])

    a.ask_fn = Echo()
    mention = [{"index": 0, "length": 6, "type": "user", "isSelf": True}]
    await adapter.on_event(mev("hello team", mid="m1", token="t1"), BOT)
    await adapter.on_event(
        mev("@Salid what did I say", mid="m2", token="t2", mentionees=mention), BOT
    )
    await a.drain()
    await adapter.on_event(mev("@Salid and now?", mid="m3", token="t3", mentionees=mention), BOT)
    await a.drain()
    assert len(seen) == 2, seen
    group = inc(platform="line", tenant=BOT, room="Cgroup")
    dump = " ".join(seen) + " ".join(f"{r.author_name} {r.text}" for r in stored(a, group))
    assert "Alice" not in dump and "Ops team" not in dump
    assert "LINE user" in seen[0]
    assert api.replies  # the answer was actually delivered
    assert all("Alice" not in t and "Ops team" not in t for _tok, ts in api.replies for t in ts)
