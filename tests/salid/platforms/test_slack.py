"""Slack: event translation, replies in threads, scope inputs, import. A fake Web API client."""

import pytest
from conftest import FixedClockSettings
from platform_fakes import NOW, Clock, gid_of, inc, make_assistant, rid_of, stored

from salid.platforms import ids, slack, slack_import
from salid.platforms.slack import SlackAdapter, SlackPort

BOT, TEAM = "UBOT", "T1"


class FakeSlack:
    """Just enough of AsyncWebClient. Responses are plain dicts (the SDK's support .get)."""

    def __init__(self):
        self.users = {
            "U1": {"id": "U1", "name": "alice", "profile": {"display_name": "Alice"}},
            "U2": {"id": "U2", "real_name": "Bob Lee", "profile": {}, "is_admin": True},
            "U3": {"id": "U3", "name": "eve", "profile": {}},
        }
        self.channels = {
            "C1": {"id": "C1", "name": "general", "members": ["U1", "U2", BOT]},
            "C2": {"id": "C2", "name": "leads", "members": ["U1", "U2", "U3", BOT]},
            "C3": {"id": "C3", "name": "secret", "members": ["U1", "U9", BOT]},
        }
        self.posts: list[dict] = []
        self.updates: list[dict] = []
        self.deleted: list[dict] = []
        self.calls: list[str] = []
        self.fail_update = False
        self.history: dict[str, list[dict]] = {}
        self.replies: dict[str, list[dict]] = {}
        self.member_pages = 1

    async def users_info(self, user):
        self.calls.append("users_info")
        if user not in self.users:
            raise RuntimeError("user_not_found")
        return {"user": self.users[user]}

    async def conversations_info(self, channel):
        return {"channel": self.channels[channel]}

    async def conversations_members(self, channel, limit=200, cursor=None):
        self.calls.append("members:" + channel)
        members = self.channels[channel]["members"]
        n = self.member_pages
        size = max(len(members) // n, 1)
        start = int(cursor or 0)
        page = members[start : start + size]
        nxt = start + size
        return {
            "members": page,
            "response_metadata": {"next_cursor": str(nxt) if nxt < len(members) else ""},
        }

    async def users_conversations(
        self, user=None, types=None, exclude_archived=None, limit=200, cursor=None
    ):
        who = user or BOT
        rows = [
            {"id": c["id"], "name": c["name"]}
            for c in self.channels.values()
            if who in c["members"]
        ]
        return {"channels": rows, "response_metadata": {"next_cursor": ""}}

    async def chat_postMessage(self, **kw):
        self.posts.append(kw)
        return {"ts": f"9{len(self.posts)}.000"}

    async def chat_update(self, **kw):
        if self.fail_update:
            raise RuntimeError("message_not_found")
        self.updates.append(kw)
        return {"ok": True}

    async def chat_delete(self, **kw):
        self.deleted.append(kw)
        return {"ok": True}

    async def conversations_history(self, channel, limit=200, oldest=None, cursor=None):
        rows = self.history.get(channel, [])
        if oldest:
            rows = [m for m in rows if float(m["ts"]) > float(oldest)]
        rows = sorted(rows, key=lambda m: -float(m["ts"]))  # Slack: newest first
        start = int(cursor or 0)
        page = rows[start : start + limit]
        nxt = start + limit
        return {
            "messages": page,
            "response_metadata": {"next_cursor": str(nxt) if nxt < len(rows) else ""},
        }

    async def conversations_replies(self, channel, ts, limit=100, oldest=None):
        return {"messages": self.replies.get(ts, [])}


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


@pytest.fixture
def fx(s, tmp_path):
    client = FakeSlack()
    a = make_assistant(s, tmp_path)
    port = SlackPort(client, TEAM, BOT, clock=Clock())
    downloads = []
    adapter = SlackAdapter(a, port, lambda url: downloads.append(url) or b"filebytes")
    adapter.downloads = downloads
    return client, a, port, adapter


def ev(**kw):
    base = {"type": "message", "channel": "C1", "channel_type": "channel", "user": "U1",
            "text": "hi", "ts": "1791954000.000100"}  # fmt: skip
    base.update(kw)
    return {k: v for k, v in base.items() if v is not None}


# ------------------------------------------------------------------ events -> Incoming


async def test_mention_in_a_channel_is_answered_in_a_thread_under_the_message(fx):
    client, a, port, adapter = fx
    await adapter.on_message(ev(text=f"<@{BOT}> what is the total?"), TEAM)
    await a.drain()
    assert a.ask_fn.calls[0][0] == "what is the total?"
    (post,) = client.posts
    assert post["channel"] == "C1" and post["thread_ts"] == "1791954000.000100"
    assert post["reply_broadcast"] is False and post["text"] == "ok"


async def test_reply_inside_an_existing_thread_stays_in_that_thread(fx):
    client, a, port, adapter = fx
    await adapter.on_message(
        ev(text=f"<@{BOT}> more", thread_ts="1791953000.000001", ts="1791954000.000200"), TEAM
    )
    assert client.posts[0]["thread_ts"] == "1791953000.000001"


async def test_dm_is_answered_without_a_mention_and_not_in_a_thread(fx):
    client, a, port, adapter = fx
    await adapter.on_message(ev(channel="D1", channel_type="im", text="hello there"), TEAM)
    assert a.ask_fn.calls and client.posts[0]["thread_ts"] is None


async def test_thread_started_by_salid_counts_as_addressed(fx):
    client, a, port, adapter = fx
    await adapter.on_message(
        ev(text="and the month?", thread_ts="1.0", parent_user_id=BOT, ts="2.0"), TEAM
    )
    assert len(a.ask_fn.calls) == 1


async def test_plain_channel_chatter_is_only_remembered(fx):
    client, a, port, adapter = fx
    await adapter.on_message(ev(text="lunch?"), TEAM)
    await a.drain()
    assert not a.ask_fn.calls and not client.posts
    assert [m.text for m in stored(a, inc())] == ["lunch?"]


@pytest.mark.parametrize(
    "event",
    [
        ev(bot_id="B1"),  # any bot
        ev(user=BOT),  # Salid itself
        ev(subtype="channel_join"),
        ev(subtype="bot_message", user=None),
        ev(user=None),
        ev(subtype="message_changed", message={"user": BOT, "ts": "5.0", "text": "x"}),
        ev(subtype="message_changed", message={"bot_id": "B1", "user": "U1", "ts": "5.0"}),
    ],
)
async def test_bots_self_and_noise_are_ignored(fx, event):
    client, a, port, adapter = fx
    await adapter.on_message(event, TEAM)
    await a.drain()
    assert not a.ask_fn.calls and not client.posts and stored(a, inc()) == []


async def test_mentions_of_other_people_become_names_and_mark_the_message(fx):
    client, a, port, adapter = fx
    await adapter.on_message(ev(text=f"<@{BOT}> tell <@U2> the total"), TEAM)
    assert a.ask_fn.calls[0][0] == "tell @Bob Lee the total"
    a2_msg = ev(text="<@U2> can you check", ts="3.0")
    await adapter.on_message(a2_msg, TEAM)
    assert len(a.ask_fn.calls) == 1  # talking to Bob, not to Salid


async def test_app_mention_and_message_event_for_one_message_answer_once(fx):
    client, a, port, adapter = fx
    e = ev(text=f"<@{BOT}> hi", ts="7.0")
    await adapter.on_message(e, TEAM)
    await adapter.on_message(e, TEAM, mention=True)
    assert len(a.ask_fn.calls) == 1 and len(client.posts) == 1


async def test_edit_updates_memory_and_deleting_removes_it(fx):
    client, a, port, adapter = fx
    await adapter.on_message(ev(text="budget 50", ts="1791954000.000900"), TEAM)
    await adapter.on_message(
        ev(
            subtype="message_changed",
            message={"user": "U1", "ts": "1791954000.000900", "text": "budget 80"},
        ),
        TEAM,
    )
    await a.drain()
    assert [m.text for m in stored(a, inc())] == ["budget 80"]
    assert not a.ask_fn.calls  # an edit is never answered
    await adapter.on_message(ev(subtype="message_deleted", deleted_ts="1791954000.000900"), TEAM)
    await a.drain()
    assert stored(a, inc()) == []


async def test_files_become_file_refs_and_download_with_the_token_inside(fx):
    client, a, port, adapter = fx
    await adapter.on_message(
        ev(
            text=f"<@{BOT}> read this",
            subtype="file_share",
            files=[
                {"name": "bill.png", "mimetype": "image/png", "size": 5,
                 "url_private_download": "https://files.slack.com/x/bill.png"},
                {"name": "gone.png", "mode": "tombstone", "url_private": "https://files.slack.com/y"},
            ],
        ),
        TEAM,
    )  # fmt: skip
    (fi,) = a.ask_fn.calls[0][1]["files"]
    assert fi.filename == "bill.png" and fi.data == b"filebytes"
    assert adapter.downloads == ["https://files.slack.com/x/bill.png"]


def test_download_refuses_foreign_hosts_so_the_token_never_leaves_slack():
    sent = []
    with pytest.raises(ValueError):
        slack.download("xoxb-secret", "https://evil.example/file", get=lambda u, h: sent.append(h))
    assert sent == []
    got = slack.download(
        "xoxb-secret", "https://files.slack.com/f", get=lambda u, h: h["Authorization"].encode()
    )
    assert got == b"Bearer xoxb-secret"


async def test_room_name_user_names_and_admin_lookup(fx):
    client, a, port, adapter = fx
    await adapter.on_message(ev(text=f"<@{BOT}> hi", user="U2"), TEAM)
    msg_exec = a.ask_fn.calls[0][1]["is_exec"]
    assert msg_exec is True  # U2 is a workspace admin (users.info is_admin)
    await adapter.on_message(ev(text=f"<@{BOT}> hi", user="U1", ts="2.0"), TEAM)
    assert a.ask_fn.calls[1][1]["is_exec"] is False
    assert a.names[(gid_of(inc()), rid_of(inc()))] == "general"


# ------------------------------------------------------------------ replies


async def test_long_answers_are_chunked_and_markdown_becomes_mrkdwn(fx):
    client, a, port, adapter = fx
    a.ask_fn.text = "# Report\n**Total** is 5\n" + "\n\n".join(["word " * 120] * 8)
    await adapter.on_message(ev(text=f"<@{BOT}> report"), TEAM)
    assert len(client.posts) > 1
    assert all(len(p["text"]) <= slack.CHUNK for p in client.posts)
    assert client.posts[0]["text"].startswith("*Report*\n*Total* is 5")
    assert all(p["thread_ts"] == "1791954000.000100" for p in client.posts)


async def test_ack_is_edited_into_the_first_chunk(fx):
    client, a, port, adapter = fx
    msg = inc(room="C1", message="5.0")
    await port.deliver(msg, ["one", "two"], ("C1", "88.0"))
    assert client.updates == [{"channel": "C1", "ts": "88.0", "text": "one", "mrkdwn": True}]
    assert [p["text"] for p in client.posts] == ["two"]


async def test_ack_that_cannot_be_edited_is_removed_and_the_answer_posted(fx):
    client, a, port, adapter = fx
    client.fail_update = True
    sent = await port.deliver(inc(message="5.0"), ["answer"], ("C1", "88.0"))
    assert client.deleted == [{"channel": "C1", "ts": "88.0"}]
    assert [p["text"] for p in client.posts] == ["answer"] and len(sent) == 1


async def test_ack_text_is_posted_in_the_thread(fx):
    client, a, port, adapter = fx
    ack = await port.ack(inc(message="5.0"), "one moment")
    assert ack == ("C1", "91.000") and client.posts[0]["thread_ts"] == "5.0"


# ------------------------------------------------------------------ audience for memory scope


async def test_room_view_removes_the_bot_and_intersects_the_askers_rooms(fx):
    client, a, port, adapter = fx
    view = await port.room_view(inc(room="C1", user="U1"))
    assert view.audience == frozenset({"U1", "U2"})
    # U1 is in C1, C2, C3 and so is the bot; C1 is the answering room
    assert {r.key: r.members for r in view.rooms} == {
        "C2": frozenset({"U1", "U2", "U3"}),
        "C3": frozenset({"U1", "U9"}),
    }


async def test_room_view_in_a_dm_is_the_user_alone(fx):
    client, a, port, adapter = fx
    view = await port.room_view(inc(room="D1", is_direct=True, user="U1"))
    assert view.audience == frozenset({"U1"})


async def test_unreadable_or_huge_membership_means_unknown_audience(fx, monkeypatch):
    client, a, port, adapter = fx

    async def boom(*a, **k):
        raise RuntimeError("missing_scope")

    monkeypatch.setattr(client, "conversations_members", boom)
    assert (await port.room_view(inc())).audience is None
    monkeypatch.undo()
    monkeypatch.setattr(slack, "MAX_MEMBERS", 1)
    port._members.clear()
    assert (await port.room_view(inc())).audience is None


async def test_membership_is_paged_and_cached_but_the_answering_room_is_read_live(fx):
    client, a, port, adapter = fx
    client.member_pages = 2
    first = await port.members("C2")
    assert first == frozenset({"U1", "U2", "U3"})
    n = client.calls.count("members:C2")
    await port.members("C2")
    assert client.calls.count("members:C2") == n  # cached
    await port.members("C2", fresh=True)
    assert client.calls.count("members:C2") > n


async def test_slack_scope_end_to_end_through_the_assistant(fx):
    client, a, port, adapter = fx
    await adapter.on_message(ev(text=f"<@{BOT}> what did leads say?"), TEAM)
    mem = a.ask_fn.calls[0][1]["memory"]
    # answering in #general (U1, U2): #leads has both, #secret lacks U2
    assert sorted(mem.allowed.values()) == ["general", "leads"]


async def test_history_is_read_from_slack_for_a_thread_after_a_restart(fx):
    client, a, port, adapter = fx
    client.replies["1791953000.000001"] = [
        {"ts": "1791953000.000001", "user": "U2", "text": "the vendor is Makro"},
        {"ts": "1791953100.000001", "user": BOT, "text": "noted", "bot_id": "B0"},
    ]
    await adapter.on_message(
        ev(text=f"<@{BOT}> which vendor?", thread_ts="1791953000.000001", ts="1791954000.000200"),
        TEAM,
    )
    hist = a.ask_fn.calls[0][1]["history"]
    assert "Bob Lee: the vendor is Makro" in hist and "สลิด: noted" in hist


# ------------------------------------------------------------------ import


def msgs(n, start=1791946000, user="U1"):
    return [{"ts": f"{start + i}.000100", "user": user, "text": f"message {i}"} for i in range(n)]


async def test_import_reads_history_once_and_is_idempotent(fx):
    client, a, port, adapter = fx
    from datetime import UTC, datetime

    a._now = lambda: datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    rows = msgs(5) + [{"ts": "1791946100.000100", "user": "U2", "text": "hi", "bot_id": "B1"},
                      {"ts": "1791946200.000100", "subtype": "channel_join", "user": "U2", "text": "joined"}]  # fmt: skip
    client.history["C1"] = rows
    imp = slack_import.SlackImporter(a, port, page_sleep=0)
    n = await imp.run()
    assert n == 5
    imported = a.memory_store.messages(
        ids.tenant_id("slack", TEAM), [ids.room_id("slack", TEAM, "C1")], "2025-01-01", "2026-12-31",
        limit=100, newest_first=False,
    )  # fmt: skip
    assert [m.text for m in imported] == [f"message {i}" for i in range(5)]
    assert {m.author_name for m in imported} == {"Alice"}
    await imp.run()  # a second run adds nothing
    again = a.memory_store.messages(
        ids.tenant_id("slack", TEAM), [ids.room_id("slack", TEAM, "C1")], "2025-01-01", "2026-12-31",
        limit=100, newest_first=False,
    )  # fmt: skip
    assert len(again) == 5


async def test_import_includes_thread_replies_and_skips_off_channels(fx):
    client, a, port, adapter = fx
    from datetime import UTC, datetime

    from salid.guild_settings import GuildSettingsUpdate

    a._now = lambda: datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    root = {"ts": "1791946000.000100", "user": "U1", "text": "root", "reply_count": 1,
            "thread_ts": "1791946000.000100"}  # fmt: skip
    client.history["C1"] = [root]
    client.replies["1791946000.000100"] = [
        root,
        {
            "ts": "1791946050.000100",
            "user": "U2",
            "text": "a reply",
            "thread_ts": "1791946000.000100",
        },
    ]
    client.history["C2"] = msgs(2)
    muted = ids.room_id("slack", TEAM, "C2")
    a.team.store.save_settings(
        ids.tenant_id("slack", TEAM), GuildSettingsUpdate(channel_modes={str(muted): "off"}), "t"
    )
    imp = slack_import.SlackImporter(a, port, page_sleep=0)
    assert await imp.run() == 2
    texts = {
        m.text
        for m in a.memory_store.messages(
            ids.tenant_id("slack", TEAM),
            [ids.room_id("slack", TEAM, "C1"), muted],
            "2025-01-01",
            "2026-12-31",
            limit=100,
        )  # fmt: skip
    }
    assert texts == {"root", "a reply"}


async def test_a_window_too_large_is_skipped_not_imported_with_a_hole(fx, monkeypatch):
    client, a, port, adapter = fx
    from datetime import UTC, datetime

    a._now = lambda: datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(slack_import, "MAX_WINDOW", 3)
    client.history["C1"] = msgs(10)
    imp = slack_import.SlackImporter(a, port, page_sleep=0)
    assert await imp.run() == 0
    assert (
        a.memory_store.messages(
            ids.tenant_id("slack", TEAM),
            [ids.room_id("slack", TEAM, "C1")],
            "2025-01-01",
            "2026-12-31",
        )
        == []
    )


async def test_import_respects_the_memory_switch(fx):
    from salid.guild_settings import GuildSettingsUpdate

    client, a, port, adapter = fx
    a.team.store.save_settings(
        ids.tenant_id("slack", TEAM), GuildSettingsUpdate(features={"memory": False}), "t"
    )
    client.history["C1"] = msgs(3)
    assert await slack_import.SlackImporter(a, port, page_sleep=0).run() == 0
    assert NOW  # (clock constant used by sibling tests)
