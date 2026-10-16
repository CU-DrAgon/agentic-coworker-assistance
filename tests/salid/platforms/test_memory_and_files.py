"""Team memory (recording and who may read what) and files sent to Salid."""

import pytest
from conftest import FixedClockSettings
from platform_fakes import FakePort, gid_of, inc, make_assistant, rid_of, stored

from salid.guild_settings import GuildSettingsUpdate
from salid.memory_access import ChannelAccess
from salid.platforms import core, ids
from salid.platforms.core import FileRef, RoomInfo, RoomView, compute_scope
from salid.qa import FileActions, FileInput


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


async def run(a, port, **kw):
    msg = inc(**kw)
    await a.handle(msg, port)
    await a.drain()
    return msg


# ------------------------------------------------------------------ recording


async def test_humans_and_salids_replies_are_remembered(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    msg = await run(a, port, text="we ship friday", message="1.0")
    await run(a, port, mentions_bot=True, text="when do we ship?", message="2.0")
    rows = stored(a, msg)
    assert sorted((m.author_name, m.kind, m.text) for m in rows) == [
        ("Alice", "chat", "we ship friday"),
        ("Alice", "chat", "when do we ship?"),
        ("Salid", "salid", "ok"),
    ]
    assert all(m.guild_id == gid_of(msg) and m.channel_id == rid_of(msg) for m in rows)
    (mine,) = [m for m in rows if m.kind == "salid"]
    assert mine.author_id == ids.bot_user_id("slack", "T1")


async def test_files_are_remembered_as_descriptors(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    f = FileRef("receipt.png", "image/png", 10, lambda: b"x")
    msg = await run(a, port, text="look", files=[f])
    assert stored(a, msg)[0].text == "look [image: receipt.png]"


async def test_memory_switch_off_records_nothing(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    probe = inc()
    a.team.store.save_settings(gid_of(probe), GuildSettingsUpdate(features={"memory": False}), "t")
    msg = await run(a, port, text="private plans", mentions_bot=True)
    assert stored(a, msg) == [] and a.ask_fn.calls[0][1]["memory"] is None


async def test_edits_and_deletes_update_the_memory(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    msg = await run(a, port, text="budget is 50", message="1.0")
    await a.on_edit(inc(text="budget is 80", message="1.0"))
    await a.drain()
    assert stored(a, msg)[0].text == "budget is 80"
    await a.on_delete("slack", "T1", "C1", "1.0")
    await a.drain()
    assert stored(a, msg) == []


async def test_tenants_do_not_share_memory(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    one = await run(a, port, text="secret of workspace one", message="1.0")
    two = await run(a, port, text="other", message="2.0", tenant="T2")
    assert [m.text for m in stored(a, one)] == ["secret of workspace one"]
    assert [m.text for m in stored(a, two)] == ["other"]
    # the same channel id in two workspaces is two different rooms
    assert rid_of(one) != rid_of(two) and gid_of(one) != gid_of(two)


# ------------------------------------------------------------------ scope (pure)


def acc(room: int, name: str, *viewers: int) -> ChannelAccess:
    return ChannelAccess(room, name, frozenset(viewers))


def test_scope_needs_the_asker_in_the_room_and_the_audience_covered():
    asker, bob, eve = 1, 2, 3
    rooms = [
        acc(10, "public", asker, bob, eve),
        acc(11, "leads", asker, bob),  # eve cannot read it
        acc(12, "others", bob, eve),  # the asker cannot read it
    ]
    # answered in #public (asker, bob, eve): only rooms all three can read
    got = compute_scope(
        answering=10, answering_name="public", asker=asker,
        audience=frozenset({asker, bob, eve}), rooms=rooms,
    )  # fmt: skip
    assert got == {10: "public"}
    # answered in #leads (asker, bob): leads itself and public (both readers are in public)
    got = compute_scope(
        answering=11, answering_name="leads", asker=asker,
        audience=frozenset({asker, bob}), rooms=rooms,
    )  # fmt: skip
    assert got == {10: "public", 11: "leads"}


def test_scope_in_a_dm_is_the_askers_rooms():
    rooms = [acc(10, "public", 1, 2), acc(12, "others", 2, 3)]
    got = compute_scope(
        answering=99, answering_name="dm", asker=1, audience=frozenset({1}), rooms=rooms
    )
    assert got == {10: "public", 99: "dm"}


@pytest.mark.parametrize("unknown_members", [None])
def test_unknown_audience_fails_closed_to_the_room_itself(unknown_members):
    rooms = [acc(10, "public", 1, 2, 3)]
    got = compute_scope(
        answering=11, answering_name="big", asker=1, audience=unknown_members, rooms=rooms
    )
    assert got == {11: "big"}


# ------------------------------------------------------------------ scope (through the assistant)


async def test_slack_scope_passed_to_ask(s, tmp_path):
    view = RoomView(
        frozenset({"U1", "U2"}),
        [
            RoomInfo("C2", "leads", frozenset({"U1", "U2", "U3"})),  # superset: readable
            RoomInfo("C3", "secret", frozenset({"U1", "U9"})),  # U2 cannot read: no
            RoomInfo("C4", "nope", frozenset({"U2"})),  # the asker is not in it: no
        ],
    )
    a, port = make_assistant(s, tmp_path), FakePort(view=view)
    await run(a, port, mentions_bot=True)
    mem = a.ask_fn.calls[0][1]["memory"]
    names = sorted(mem.allowed.values())
    assert names == ["general", "leads"] and mem.guild_id == ids.tenant_id("slack", "T1")


async def test_rooms_in_off_mode_are_never_in_scope(s, tmp_path):
    view = RoomView(frozenset({"U1"}), [RoomInfo("C2", "muted", frozenset({"U1"}))])
    a, port = make_assistant(s, tmp_path), FakePort(view=view)
    gid, muted = gid_of(inc()), ids.room_id("slack", "T1", "C2")
    a.team.store.save_settings(gid, GuildSettingsUpdate(channel_modes={str(muted): "off"}), "t")
    await run(a, port, mentions_bot=True)
    assert sorted(a.ask_fn.calls[0][1]["memory"].allowed.values()) == ["general"]


async def test_line_group_memory_stays_in_that_group(s, tmp_path):
    from salid.platforms.line import LinePort

    class Api:
        pass

    port = LinePort(Api(), a_ps := make_assistant(s, tmp_path).ps)
    a = make_assistant(s, tmp_path)
    a.ps = a_ps
    group = await run(a, port, platform="line", tenant="Ubot", room="Cgroup1", user="Ualice",
                      mentions_bot=True, text="summary?", room_name="Team")  # fmt: skip
    mem = a.ask_fn.calls[0][1]["memory"]
    assert mem.allowed == {rid_of(group): "Team"}
    # a 1:1 chat reads only itself, even though the same person is in the group
    dm = await run(a, port, platform="line", tenant="Ubot", room="Ualice", user="Ualice",
                   is_direct=True, text="and me?", message="m2", room_name="")  # fmt: skip
    assert a.ask_fn.calls[1][1]["memory"].allowed == {rid_of(dm): "Ualice"}


async def test_any_scope_error_means_no_memory_tools(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    port.view_error = RuntimeError("members call failed")
    await run(a, port, mentions_bot=True)
    assert a.ask_fn.calls[0][1]["memory"] is None


async def test_memory_tools_read_only_their_tenant_and_scope(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    await run(a, port, text="alpha launch code 7", message="1.0")
    await run(a, port, text="beta budget 9", message="2.0", tenant="T2")
    await run(a, port, mentions_bot=True, message="3.0", text="what was the launch code?")
    mem = a.ask_fn.calls[0][1]["memory"]
    found = mem.chat_messages(start_date="2026-10-14", end_date="2026-10-14")
    texts = " ".join(m["text"] for m in found["messages"])
    assert "alpha launch code 7" in texts and "beta budget" not in texts


# ------------------------------------------------------------------ files


def fref(name, mime, size, data=b"bytes", calls=None):
    def fetch():
        if calls is not None:
            calls.append(name)
        return data

    return FileRef(name, mime, size, fetch)


async def test_files_reach_ask_and_receipts_bind_ids_in_code(s, tmp_path, monkeypatch):
    captured = []

    def fake_ingest(store, llm, settings, **kw):
        captured.append(kw)
        return "result"

    monkeypatch.setattr(core, "ingest_attachment", fake_ingest)
    a, port = make_assistant(s, tmp_path), FakePort()
    fetched = []
    msg = await run(
        a, port, mentions_bot=True, text="log this", message="42.5",
        files=[fref("bill.jpg", "image/jpeg", 100, calls=fetched)],
    )  # fmt: skip
    _q, kw = a.ask_fn.calls[0]
    (fi,) = kw["files"]
    assert isinstance(fi, FileInput) and fi.filename == "bill.jpg" and fi.data == b"bytes"
    assert fetched == ["bill.jpg"]
    actions: FileActions = kw["actions"]
    # the model's tool only chooses the attachment and a note: everything else is closed over
    assert actions.record_receipt(fi, "from the model") == "result"
    (call,) = captured
    assert call["guild_id"] == gid_of(msg)
    assert call["submitter_id"] == ids.user_id("slack", "U1")
    assert call["submitter_name"] == "Alice"
    assert call["channel_id"] == rid_of(msg)
    assert call["message_id"] == ids.message_id("slack", "T1", "C1", "42.5")
    assert call["note"] == "log this from the model" and call["data"] == b"bytes"
    assert actions.start_deep_task is None


async def test_receipts_switch_off_removes_the_record_tool(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    a.team.store.save_settings(
        gid_of(inc()), GuildSettingsUpdate(features={"receipts": False}), "t"
    )
    await run(a, port, mentions_bot=True, files=[fref("a.png", "image/png", 5)])
    assert a.ask_fn.calls[0][1]["actions"].record_receipt is None


async def test_receipt_lands_in_the_same_tenant_scoped_store(s, tmp_path, monkeypatch):
    from salid.extract import IngestResult
    from salid.models import Receipt

    def fake_extract(client, settings, data, mime, note):
        from salid.models import ReceiptExtraction

        return ReceiptExtraction(
            is_receipt=True, vendor="Makro", receipt_date="2026-10-14", total=120.0, confidence=0.9
        )

    monkeypatch.setattr("salid.extract.extract_receipt", fake_extract)
    a, port = make_assistant(s, tmp_path), FakePort()
    await run(a, port, mentions_bot=True, files=[fref("a.png", "image/png", 5, b"img")])
    fi = a.ask_fn.calls[0][1]["files"][0]
    res = a.ask_fn.calls[0][1]["actions"].record_receipt(fi, "")
    assert isinstance(res, IngestResult) and res.status == "created"
    gid = ids.tenant_id("slack", "T1")
    r: Receipt = res.receipt
    assert r.guild_id == gid and r.submitter_id == ids.user_id("slack", "U1")
    assert a.store.get_by_sha(gid, r.sha256) is not None
    assert a.store.get_by_sha(ids.tenant_id("slack", "T2"), r.sha256) is None


async def test_limits_skip_files_without_downloading_them(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    fetched = []
    files = [
        fref("big.pdf", "application/pdf", 16 * 1024 * 1024, calls=fetched),
        fref("movie.mp4", "video/mp4", 100, calls=fetched),
        fref("ok.png", "image/png", 100, calls=fetched),
    ]
    await run(a, port, mentions_bot=True, files=files)
    inputs = a.ask_fn.calls[0][1]["files"]
    assert fetched == ["ok.png"]
    assert [bool(f.skip) for f in inputs] == [True, True, False]
    assert "over 15 MB" in inputs[0].skip and "unsupported" in inputs[1].skip


async def test_download_failure_and_oversize_download_are_skipped(s, tmp_path, caplog):
    def broken():
        raise OSError("token xoxb-LEAK in url")

    a, port = make_assistant(s, tmp_path), FakePort()
    files = [
        FileRef("a.png", "image/png", 0, broken),
        FileRef("b.png", "image/png", 0, lambda: b"x" * (15 * 1024 * 1024 + 1)),
    ]
    caplog.set_level("DEBUG")
    await run(a, port, mentions_bot=True, files=files)
    skips = [f.skip for f in a.ask_fn.calls[0][1]["files"]]
    assert skips == ["could not be downloaded", "over the size limit"]
    assert "xoxb-LEAK" not in caplog.text


async def test_actual_downloaded_bytes_enforce_the_total_limit(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    files = [
        FileRef("a.png", "image/png", 0, lambda: b"a" * (11 * 1024 * 1024)),
        FileRef("b.png", "image/png", 0, lambda: b"b" * (10 * 1024 * 1024)),
    ]
    await run(a, port, mentions_bot=True, files=files)
    inputs = a.ask_fn.calls[0][1]["files"]
    assert inputs[0].data and not inputs[0].skip
    assert not inputs[1].data and inputs[1].skip == "total size over 20 MB"
