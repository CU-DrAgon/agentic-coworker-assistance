"""The Workspace interface: button ids, capability names and the recording fake."""

import pytest

from salid import workspace as w
from salid.workspace import (
    Button,
    FakeWorkspace,
    FileOut,
    Person,
    Room,
    RoomView,
    WorkspaceError,
    button_id,
    parse_button_id,
)


def test_button_id_round_trip():
    bid = button_id("task", "done", 123, "t-42")
    assert bid == "salid:task:done:123:t-42"
    got = parse_button_id(bid)
    assert (got.feature, got.action, got.tenant, got.obj) == ("task", "done", 123, "t-42")


def test_button_id_object_may_contain_colons():
    got = parse_button_id(button_id("vote", "pick", 1, "poll9:opt:2"))
    assert got.obj == "poll9:opt:2"


@pytest.mark.parametrize(
    "bad",
    ["", "x", "salid:a:b:c", "other:a:b:1:x", "salid:a:b:notnum:x", "salid::b:1:x", "salid:a:b:1:"],
)
def test_parse_rejects_foreign_ids(bad):
    assert parse_button_id(bad) is None


def test_button_id_validates():
    with pytest.raises(ValueError):
        button_id("a:b", "x", 1, "o")
    with pytest.raises(ValueError):
        button_id("a", "x", 1, "o" * 100)
    with pytest.raises(ValueError):
        button_id("a", "x", 1, "")


def test_capabilities_are_distinct_names():
    assert len(set(w.CAPABILITIES)) == len(w.CAPABILITIES) == 8
    assert {w.VOICE, w.BUTTONS, w.EVENTS, w.THREADS, w.EDIT, w.DM, w.FILES, w.HISTORY} == set(
        w.CAPABILITIES
    )


async def test_fake_records_posts_dms_edits():
    ana = Person(1, "Ana", "u1", "ana@x.co")
    bot = Person(2, "Bot", "u2", is_bot=True)
    ws = FakeWorkspace(
        people=[ana, bot],
        rooms=[Room(10, "general", "c10", private=False)],
        execs=[1],
        ranks={1: 5},
    )
    assert await ws.people() == [ana]
    assert await ws.person("@ana") == ana and await ws.person("ana@x.co") == ana
    assert await ws.person("nobody") is None
    assert await ws.is_exec(1) and not await ws.is_exec(2)
    assert await ws.rank(1) == 5 and await ws.rank(9) == 0
    assert (await ws.room_view(10)) == RoomView(None, [])  # unknown unless set

    btn = Button(button_id("task", "done", 1, "t1"), "Done", "success")
    ref = await ws.post(10, "hi", files=[FileOut("a.txt", b"x", "text/plain")], buttons=[btn])
    assert ws.posts[0].ref == ref and ws.posts[0].buttons == (btn,)
    dm = await ws.dm(1, "psst")
    assert dm is not None and ws.dms[0].user == 1
    await ws.edit(ref, "hi again")
    assert ws.edits == [(ref, "hi again", ())]
    assert ws.mention(1) == "@Ana"


async def test_fake_dm_can_fail_and_rejects_bad_buttons():
    ws = FakeWorkspace(can_dm=lambda u: False)
    assert await ws.dm(5, "x") is None
    with pytest.raises(WorkspaceError):
        await ws.post(1, "x", buttons=[Button("id", "L", "weird")])
    with pytest.raises(WorkspaceError):
        await ws.post(1, "x", buttons=[Button("id", "L")] * 26)


def test_fake_supports_matrix():
    assert FakeWorkspace().supports(w.VOICE)
    ws = FakeWorkspace(supports=[w.DM])
    assert ws.supports(w.DM) and not ws.supports(w.BUTTONS) and not ws.supports("nope")
