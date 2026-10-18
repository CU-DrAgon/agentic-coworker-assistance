"""Slack: event translation, replies in threads, scope inputs, import. A fake Web API client."""

import asyncio
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from conftest import FixedClockSettings
from platform_fakes import NOW, Clock, FakeAsk, gid_of, inc, make_assistant, rid_of, stored

from salid.guild_settings import GuildSettingsUpdate
from salid.platforms import ids, slack, slack_import
from salid.platforms.slack import SlackAdapter, SlackPort

BOT, TEAM = "UBOT", "T1"


class FakeSlack:
    """Just enough of AsyncWebClient. Responses are plain dicts (the SDK's support .get)."""

    def __init__(self):
        self.users = {
            "U1": {
                "id": "U1",
                "team_id": TEAM,
                "name": "alice",
                "profile": {"display_name": "Alice"},
            },
            "U2": {
                "id": "U2",
                "team_id": TEAM,
                "real_name": "Bob Lee",
                "profile": {},
                "is_admin": True,
            },
            "U3": {"id": "U3", "team_id": TEAM, "name": "eve", "profile": {}},
            # Slack Connect: an admin of ANOTHER workspace, a member of a shared channel
            "UX": {
                "id": "UX",
                "team_id": "T-EXT",
                "name": "ext",
                "profile": {},
                "is_admin": True,
                "is_owner": True,
            },
            "UB": {
                "id": "UB",
                "team_id": TEAM,
                "name": "bot",
                "profile": {},
                "is_admin": True,
                "is_bot": True,
            },
            "UD": {
                "id": "UD",
                "team_id": TEAM,
                "name": "gone",
                "profile": {},
                "is_admin": True,
                "deleted": True,
            },
            "UN": {"id": "UN", "name": "noteam", "profile": {}, "is_admin": True},
        }
        self.channels = {
            "C1": {"id": "C1", "name": "general", "is_private": True, "members": ["U1", "U2", BOT]},
            "C2": {
                "id": "C2",
                "name": "leads",
                "is_private": True,
                "members": ["U1", "U2", "U3", BOT],
            },
            "C3": {"id": "C3", "name": "secret", "is_private": True, "members": ["U1", "U9", BOT]},
        }
        self.posts: list[dict] = []
        self.updates: list[dict] = []
        self.deleted: list[dict] = []
        self.calls: list[str] = []
        self.omit_share_fields: set[str] = set()  # flags Slack leaves out of conversations.info
        self.fail_update = False
        self.history: dict[str, list[dict]] = {}
        self.replies: dict[str, list[dict]] = {}
        self.reply_page = 1000  # replies per conversations.replies page
        self.broken_threads: set[str] = set()
        self.member_pages = 1
        self.errors: list[Exception] = []  # raised by the next history/replies calls, in order
        self.max_page: int | None = None  # the restricted tier clamps `limit` (15)
        self.history_calls: list[dict] = []
        self.reply_calls: list[dict] = []
        self.files: dict[str, dict] = {}
        self.file_calls: list[str] = []
        self.canvases: dict[str, str] = {}  # canvas id -> markdown
        self.canvas_errors: set[str] = set()
        self.api_calls: list[tuple[str, dict]] = []

    async def api_call(self, method, json=None, **kw):
        self.api_calls.append((method, json))
        assert method == "canvases.getContent"
        cid = json["canvas_id"]
        if cid in self.canvas_errors or cid not in self.canvases:
            raise RuntimeError("canvas_not_found")
        return {"ok": True, "content": self.canvases[cid]}

    async def users_info(self, user):
        self.calls.append("users_info")
        if user not in self.users:
            raise RuntimeError("user_not_found")
        return {"user": self.users[user]}

    async def conversations_info(self, channel):
        self.calls.append("info:" + channel)
        info = {k: False for k in ("is_shared", "is_ext_shared", "is_org_shared")}
        info.update(self.channels[channel])  # Slack states the share flags explicitly
        for k in self.omit_share_fields:
            info.pop(k, None)
        return {"channel": info}

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

    async def files_info(self, file):
        self.file_calls.append(file)
        if file not in self.files:
            raise RuntimeError("file_not_found")
        return {"file": self.files[file]}

    async def conversations_history(
        self, channel, limit=200, oldest=None, latest=None, inclusive=False, cursor=None
    ):
        self.history_calls.append({"channel": channel, "limit": limit, "oldest": oldest})
        if self.errors:
            raise self.errors.pop(0)
        if self.max_page:
            limit = min(limit, self.max_page)
        rows = self.history.get(channel, [])
        ge = (lambda a, b: a >= b) if inclusive else (lambda a, b: a > b)
        le = (lambda a, b: a <= b) if inclusive else (lambda a, b: a < b)
        if oldest:
            rows = [m for m in rows if ge(float(m["ts"]), float(oldest))]
        if latest:
            rows = [m for m in rows if le(float(m["ts"]), float(latest))]
        rows = sorted(rows, key=lambda m: -float(m["ts"]))  # Slack: newest first
        start = int(cursor or 0)
        page = rows[start : start + limit]
        nxt = start + limit
        return {
            "messages": page,
            "response_metadata": {"next_cursor": str(nxt) if nxt < len(rows) else ""},
        }

    async def conversations_replies(self, channel, ts, limit=100, oldest=None, cursor=None):
        self.reply_calls.append({"channel": channel, "ts": ts, "limit": limit})
        if self.errors:
            raise self.errors.pop(0)
        if ts in self.broken_threads:
            raise RuntimeError("ratelimited")
        rows = self.replies.get(ts, [])
        start = int(cursor or 0)
        page = rows[start : start + self.reply_page]
        nxt = start + self.reply_page
        return {
            "messages": page,
            "response_metadata": {"next_cursor": str(nxt) if nxt < len(rows) else ""},
        }


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


async def test_only_admins_of_this_workspace_are_executives(fx):
    client, a, port, adapter = fx
    assert await port.is_admin(inc(user="U2")) is True
    for uid in ("UX", "UB", "UD", "UN", "U1"):  # other workspace, bot, deleted, no team_id, member
        assert await port.is_admin(inc(user=uid)) is False, uid


async def test_slack_connect_admin_in_a_shared_channel_is_not_an_executive_in_q_and_a(fx):
    client, a, port, adapter = fx
    client.channels["C1"]["is_ext_shared"] = True
    client.channels["C1"]["members"].append("UX")
    await adapter.on_message(ev(text=f"<@{BOT}> show every receipt", user="UX"), TEAM)
    kw = a.ask_fn.calls[0][1]
    assert kw["is_exec"] is False  # own receipts only
    assert kw["user_id"] == ids.user_id("slack", "UX")


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


async def test_public_answering_room_never_receives_other_room_memory(fx):
    client, a, port, adapter = fx
    client.channels["C1"]["is_private"] = False
    await adapter.on_message(ev(text=f"<@{BOT}> what did leads say?"), TEAM)
    mem = a.ask_fn.calls[0][1]["memory"]
    # conversations.members lists only joined users, but every workspace member can read a public
    # channel. Its true audience is therefore unknown, so no other room may enter the answer.
    assert list(mem.allowed.values()) == ["general"]


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


@pytest.mark.parametrize("flag", ["is_shared", "is_ext_shared", "is_org_shared"])
async def test_a_shared_room_limits_memory_to_that_room(fx, flag):
    client, a, port, adapter = fx
    client.channels["C1"][flag] = True  # still private, still listable, but not ours alone
    view = await port.room_view(inc(room="C1", user="U1"))
    assert view.audience is None and view.rooms == []
    await adapter.on_message(ev(text=f"<@{BOT}> what did leads say?"), TEAM)
    assert list(a.ask_fn.calls[0][1]["memory"].allowed.values()) == ["general"]


@pytest.mark.parametrize("omitted", ["is_shared", "is_ext_shared", "is_org_shared"])
async def test_omitted_share_flags_mean_unknown_audience(fx, omitted):
    client, a, port, adapter = fx
    client.omit_share_fields = {omitted}  # not explicitly non-shared: fail closed
    view = await port.room_view(inc(room="C1", user="U1"))
    assert view.audience is None and view.rooms == []


async def test_the_shared_flag_is_cached_and_refreshed_after_a_room_event(fx):
    client, a, port, adapter = fx
    assert (await port.room_view(inc(room="C1", user="U1"))).audience is not None
    n = client.calls.count("info:C1")
    await port.room_view(inc(room="C1", user="U1"))
    assert client.calls.count("info:C1") == n  # cached with the other room caches
    client.channels["C1"]["is_ext_shared"] = True
    await adapter.on_event({"type": "channel_rename", "channel": "C1"}, TEAM)
    assert (await port.room_view(inc(room="C1", user="U1"))).audience is None


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


def msgs(n, start=1791946000, user="U1", step=1):
    return [
        {"ts": f"{start + i * step}.000100", "user": user, "text": f"message {i}"} for i in range(n)
    ]


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


async def test_long_threads_are_read_page_by_page(fx):
    client, a, port, adapter = fx
    from datetime import UTC, datetime

    a._now = lambda: datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    root_ts = "1791946000.000100"
    root = {"ts": root_ts, "user": "U1", "text": "root", "reply_count": 5, "thread_ts": root_ts}
    replies = [
        {"ts": f"17919460{10 + i}.000100", "user": "U2", "text": f"r{i}", "thread_ts": root_ts}
        for i in range(5)
    ]
    client.history["C1"] = [root]
    client.replies[root_ts] = [root, *replies]
    client.reply_page = 2  # 3 pages
    assert await slack_import.SlackImporter(a, port, page_sleep=0).run() == 6


async def test_a_thread_that_cannot_be_read_skips_the_channel_for_this_run(fx):
    client, a, port, adapter = fx
    from datetime import UTC, datetime

    a._now = lambda: datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    root_ts = "1791946000.000100"
    client.history["C1"] = [
        {"ts": root_ts, "user": "U1", "text": "root", "reply_count": 1, "thread_ts": root_ts}
    ]
    client.broken_threads.add(root_ts)
    imp = slack_import.SlackImporter(a, port, page_sleep=0)
    assert await imp.run() == 0  # nothing marked complete with a missing thread
    client.broken_threads.clear()
    client.replies[root_ts] = [client.history["C1"][0]]
    assert await imp.run() == 1  # retried next run


async def test_a_cell_too_large_is_split_and_still_imported_whole(fx, monkeypatch):
    client, a, port, adapter = fx
    from datetime import UTC, datetime

    a._now = lambda: datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(slack_import, "MAX_CELL", 3)
    # 10 messages spread 2 h apart: the 7-day cell is too big, halves until each fits
    client.history["C1"] = msgs(10, start=1791900000, step=3600)
    imp = slack_import.SlackImporter(a, port, page_sleep=0)
    assert await imp.run() == 10
    assert await slack_import.SlackImporter(a, port, page_sleep=0).run() == 0  # idempotent


async def test_import_respects_the_memory_switch(fx):
    from salid.guild_settings import GuildSettingsUpdate

    client, a, port, adapter = fx
    a.team.store.save_settings(
        ids.tenant_id("slack", TEAM), GuildSettingsUpdate(features={"memory": False}), "t"
    )
    client.history["C1"] = msgs(3)
    assert await slack_import.SlackImporter(a, port, page_sleep=0).run() == 0
    assert NOW  # (clock constant used by sibling tests)


# ------------------------------------------------------------------ rate limits and uninstall


class HttpError(Exception):
    """What the SDK raises for an HTTP error: `.response` with a status and headers."""

    def __init__(self, status: int = 429, retry: float | None = None):
        super().__init__(f"http {status}")
        headers = {"Retry-After": str(retry)} if retry is not None else {}
        self.response = SimpleNamespace(status_code=status, headers=headers)


class FakeTime:
    """An injected clock + sleep: sleeping advances the clock and is recorded, never real."""

    def __init__(self):
        self.clock = Clock()
        self.sleeps: list[float] = []

    def __call__(self):
        return self.clock()

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.clock.advance(seconds)


def timed_port(client, ft, **kw):
    return SlackPort(client, TEAM, BOT, clock=ft, sleep=ft.sleep, **kw)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "app_uninstalled"},
        {"type": "tokens_revoked", "tokens": {"oauth": ["U1"], "bot": [BOT]}},
    ],
)
async def test_uninstall_or_bot_token_revocation_stops_answers_and_erases_memory(fx, event):
    client, a, port, adapter = fx
    deleted = []
    real = a.memory.delete_guild
    a.memory.delete_guild = lambda gid: deleted.append(gid) or real(gid)
    await adapter.on_message(ev(text="budget is 80"), TEAM)
    await a.drain()
    assert [m.text for m in stored(a, inc())] == ["budget is 80"]

    await adapter.on_event(event, TEAM)

    assert port.revoked and deleted == [gid_of(inc())]
    assert stored(a, inc()) == []
    await adapter.on_message(ev(text=f"<@{BOT}> still there?", ts="2.0"), TEAM)
    await a.drain()
    assert not a.ask_fn.calls and not client.posts
    assert await port.history(inc(), 5) == []  # no Slack read either


async def test_uninstall_while_an_answer_is_in_flight_posts_and_remembers_nothing(fx):
    client, a, port, adapter = fx
    release = threading.Event()
    a.ask_fn = FakeAsk("late answer", release=release)
    task = asyncio.create_task(adapter.on_message(ev(text=f"<@{BOT}> hi"), TEAM))
    while not a.ask_fn.calls:  # the model call is running in its worker thread
        await asyncio.sleep(0)
    await adapter.on_event({"type": "app_uninstalled"}, TEAM)
    assert port.revoked
    release.set()
    await task
    await a.drain()
    assert client.posts == [] and client.updates == []
    assert await port.ack(inc(), "one moment") is None and client.posts == []
    assert stored(a, inc()) == []
    assert not any(line.mine for line in a.buffer.lines((gid_of(inc()), rid_of(inc()), None)))


async def test_tokens_revoked_for_user_tokens_only_does_nothing(fx):
    client, a, port, adapter = fx
    deleted = []
    a.memory.delete_guild = lambda gid: deleted.append(gid)
    for tokens in ({"oauth": ["U1"]}, {}, None):
        await adapter.on_event({"type": "tokens_revoked", "tokens": tokens}, TEAM)
    await adapter.on_event({"type": "app_uninstalled"}, "T-OTHER")  # not this workspace
    assert not port.revoked and deleted == []
    await adapter.on_message(ev(text=f"<@{BOT}> hi"), TEAM)
    assert len(a.ask_fn.calls) == 1


# ------------------------------------------------------------------ Slack Connect files


async def test_slack_connect_file_stub_is_resolved_with_files_info_then_downloaded(fx):
    client, a, port, adapter = fx
    client.files["F1"] = {
        "id": "F1", "name": "bill.png", "mimetype": "image/png", "size": 5,
        "url_private_download": "https://files.slack.com/x/bill.png",
    }  # fmt: skip
    await adapter.on_message(
        ev(
            text=f"<@{BOT}> read this",
            subtype="file_share",
            files=[
                {"id": "F1", "file_access": "check_file_info"},
                {"id": "F404", "file_access": "check_file_info"},  # cannot be resolved: skipped
            ],
        ),
        TEAM,
    )
    assert client.file_calls == ["F1", "F404"]
    (fi,) = a.ask_fn.calls[0][1]["files"]
    assert fi.filename == "bill.png" and fi.data == b"filebytes"
    assert adapter.downloads == ["https://files.slack.com/x/bill.png"]


async def test_a_file_that_already_has_its_url_is_not_looked_up(fx):
    client, a, port, adapter = fx
    await adapter.on_message(
        ev(
            text=f"<@{BOT}> read this",
            subtype="file_share",
            files=[{"id": "F1", "name": "a.pdf", "url_private": "https://files.slack.com/a"}],
        ),
        TEAM,
    )
    assert client.file_calls == []


# ------------------------------------------------------------------ room cache refresh


async def test_room_name_cache_refreshes_after_a_rename_event(fx):
    _, a, _, _ = fx
    for kind, channel in (("channel_rename", "C1"), ("group_rename", "C2")):
        client = FakeSlack()
        port = SlackPort(client, TEAM, BOT, clock=Clock())
        adapter = SlackAdapter(a, port, lambda url: b"")
        old = client.channels[channel]["name"]
        assert await port.room_name(channel) == old
        client.channels[channel]["name"] = "renamed"
        assert await port.room_name(channel) == old  # cached
        await adapter.on_event({"type": kind, "channel": {"id": channel, "name": "renamed"}}, TEAM)
        assert await port.room_name(channel) == "renamed"


async def test_room_name_and_privacy_caches_expire_after_the_ttl(fx):
    _, a, _, _ = fx
    client, clock = FakeSlack(), Clock()
    port = SlackPort(client, TEAM, BOT, clock=clock)
    msg = inc()
    assert await port.room_name("C1") == "general"
    assert (await port.room_view(msg)).audience == frozenset({"U1", "U2"})  # private: members
    client.channels["C1"].update(name="ops", is_private=False)
    assert await port.room_name("C1") == "general"
    assert (await port.room_view(msg)).audience == frozenset({"U1", "U2"})
    clock.advance(slack.ROOM_TTL_S + 1)
    assert await port.room_name("C1") == "ops"
    assert (await port.room_view(msg)).audience is None  # now public: audience unknown


async def test_a_rename_event_drops_the_privacy_cache_too(fx):
    _, a, _, _ = fx
    client = FakeSlack()
    port = SlackPort(client, TEAM, BOT, clock=Clock())
    adapter = SlackAdapter(a, port, lambda url: b"")
    msg = inc()
    assert (await port.room_view(msg)).audience == frozenset({"U1", "U2"})
    client.channels["C1"].update(name="ops", is_private=False)
    await adapter.on_event({"type": "channel_rename", "channel": {"id": "C1"}}, TEAM)
    assert (await port.room_view(msg)).audience is None


# ------------------------------------------------------------------ paced replies


def record_post_times(client, ft):
    times = []
    original = client.chat_postMessage

    async def post(**kw):
        times.append(ft())
        return await original(**kw)

    client.chat_postMessage = post
    return times


async def test_two_chunks_in_one_channel_are_posted_at_least_1_1_seconds_apart():
    client, ft = FakeSlack(), FakeTime()
    times = record_post_times(client, ft)
    port = timed_port(client, ft)
    await port.deliver(inc(), ["first", "second"], None)
    assert [p["text"] for p in client.posts] == ["first", "second"]
    assert times[1] - times[0] >= slack.POST_GAP_S
    assert ft.sleeps == [pytest.approx(slack.POST_GAP_S)]


async def test_other_channels_are_not_delayed_by_a_busy_one():
    client, ft = FakeSlack(), FakeTime()
    port = timed_port(client, ft)
    await port.deliver(inc(room="C1"), ["a"], None)
    await port.deliver(inc(room="C2"), ["b"], None)
    assert ft.sleeps == []


async def test_a_429_with_retry_after_waits_and_the_next_chunk_is_still_posted():
    client, ft = FakeSlack(), FakeTime()
    times = record_post_times(client, ft)
    port = timed_port(client, ft)
    real = client.chat_postMessage
    failures = [HttpError(429, 7)]

    async def flaky(**kw):
        if failures:
            raise failures.pop()
        return await real(**kw)

    client.chat_postMessage = flaky
    await port.deliver(inc(), ["first", "second"], None)
    assert [p["text"] for p in client.posts] == ["first", "second"]
    assert ft.sleeps[0] == 7.0  # the Retry-After, not a guess
    assert times[1] - times[0] >= slack.POST_GAP_S
    assert len(ft.sleeps) == 2


async def test_a_failing_chunk_does_not_stop_the_later_ones_and_the_failure_is_raised_after():
    client, ft = FakeSlack(), FakeTime()
    port = timed_port(client, ft)
    real = client.chat_postMessage

    async def post(**kw):
        if kw["text"] == "bad":
            raise RuntimeError("msg_too_long")
        return await real(**kw)

    client.chat_postMessage = post
    with pytest.raises(RuntimeError, match="msg_too_long"):
        await port.deliver(inc(), ["one", "bad", "three"], None)
    assert [p["text"] for p in client.posts] == ["one", "three"]


async def test_a_429_that_never_clears_gives_up_after_the_tries_and_raises():
    client, ft = FakeSlack(), FakeTime()
    port = timed_port(client, ft)

    async def post(**kw):
        raise HttpError(429, 2)

    client.chat_postMessage = post
    with pytest.raises(HttpError):
        await port.deliver(inc(), ["only"], None)
    assert ft.sleeps.count(2.0) == slack.POST_TRIES - 1


# ------------------------------------------------------------------ live history is bounded


class SlowHistory:
    def __init__(self):
        self.calls = 0

    async def conversations_history(self, **kw):
        self.calls += 1
        await asyncio.Event().wait()  # never answers


class LimitedHistory:
    def __init__(self, retry=120):
        self.calls, self.retry = 0, retry

    async def conversations_history(self, **kw):
        self.calls += 1
        raise HttpError(429, self.retry)


async def test_a_slow_live_history_call_returns_within_the_timeout_and_falls_back():
    ft = FakeTime()
    slow = SlowHistory()
    port = timed_port(FakeSlack(), ft, history_client=slow, history_timeout=0.01)
    assert await port.history(inc(), 10) == []
    assert slow.calls == 1 and ft.sleeps == []


async def test_a_429_on_live_history_does_not_sleep_and_blocks_further_reads_for_retry_after():
    ft = FakeTime()
    limited = LimitedHistory(retry=120)
    port = timed_port(FakeSlack(), ft, history_client=limited)
    assert await port.history(inc(), 10) == []
    assert limited.calls == 1 and ft.sleeps == []  # the Retry-After is not slept
    ft.clock.advance(60)
    assert await port.history(inc(), 10) == []
    assert limited.calls == 1  # still inside the 120 s window: no call at all
    ft.clock.advance(61)
    await port.history(inc(), 10)
    assert limited.calls == 2 and ft.sleeps == []


async def test_an_answer_is_not_held_up_by_a_rate_limited_history_read(fx):
    _, a, _, _ = fx
    ft, client = FakeTime(), FakeSlack()
    port = timed_port(client, ft, history_client=LimitedHistory(retry=300))
    adapter = SlackAdapter(a, port, lambda url: b"")
    await adapter.on_message(ev(text=f"<@{BOT}> total?"), TEAM)
    await a.drain()
    assert len(a.ask_fn.calls) == 1 and client.posts[0]["text"] == "ok"
    assert ft.sleeps == []


# ------------------------------------------------------------------ import under rate limits


def far_future(a):
    a._now = lambda: datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


async def test_import_waits_for_retry_after_and_then_imports_everything(fx):
    client, a, port, adapter = fx
    a._now = lambda: datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    client.history["C1"] = msgs(5)
    client.errors = [HttpError(429, 7)]
    ft = FakeTime()
    imp = slack_import.SlackImporter(a, port, sleep=ft.sleep, page_sleep=0, clock=ft)
    assert await imp.run() == 5
    assert 7.0 in ft.sleeps  # the wait Slack asked for
    assert not imp.capped and not imp.throttle.restricted


async def test_exact_15_message_pages_mark_the_restricted_tier_and_slow_the_import(fx):
    client, a, port, adapter = fx
    a._now = lambda: datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    client.history["C1"] = msgs(40)
    client.max_page = 15  # Slack clamps every page to 15, whatever was asked
    ft = FakeTime()
    imp = slack_import.SlackImporter(a, port, sleep=ft.sleep, page_sleep=0, clock=ft)
    assert await imp.run() == 40
    assert imp.throttle.restricted
    limits = [c["limit"] for c in client.history_calls]
    first_small = limits.index(15)  # earlier (empty) weeks cannot reveal the clamp
    assert set(limits[:first_small]) == {200} and set(limits[first_small:]) == {15}
    assert ft.sleeps and all(s >= 60 for s in ft.sleeps)  # one request a minute from then on


async def test_a_short_page_is_not_mistaken_for_the_restricted_tier(fx):
    client, a, port, adapter = fx
    a._now = lambda: datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    client.history["C1"] = msgs(15)  # exactly 15 rows, but no next page: not clamped
    ft = FakeTime()
    imp = slack_import.SlackImporter(a, port, sleep=ft.sleep, page_sleep=0, clock=ft)
    assert await imp.run() == 15
    assert not imp.throttle.restricted


async def test_consecutive_429s_stop_the_run_keeping_progress_and_the_next_run_resumes(
    fx, monkeypatch
):
    client, a, port, adapter = fx
    far_future(a)
    monkeypatch.setattr(slack_import, "MAX_429", 3)
    # two weekly cells with 3 messages each; the importer stores a page of 3 at a time
    client.history["C1"] = msgs(3, start=1791946000) + msgs(3, start=1792030000)
    ft = FakeTime()
    imp = slack_import.SlackImporter(a, port, sleep=ft.sleep, page_sleep=0, clock=ft)
    imp.importer.page_size = 3

    real = client.conversations_history
    limited = {"on": True}
    second_cell = 1792022400  # the week after the first three messages

    async def history(**kw):
        if limited["on"] and float(kw["oldest"]) >= second_cell:  # 429 forever from there on
            raise HttpError(429, 5)
        return await real(**kw)

    client.conversations_history = history

    def stored_texts():
        return sorted(
            m.text
            for m in a.memory_store.messages(
                ids.tenant_id("slack", TEAM),
                [ids.room_id("slack", TEAM, "C1")],
                "2025-01-01",
                "2026-12-31",
                limit=100,
            )
        )

    assert await imp.run() == 3
    assert imp.capped  # the loop comes back soon
    assert len(stored_texts()) == 3  # the first cell is kept
    waits = [s for s in ft.sleeps if s == 5.0]
    assert len(waits) == 3  # MAX_429 waits, then it gave up instead of waiting forever

    limited["on"] = False
    assert await imp.run() == 3  # resumes after the stored cursor
    assert not imp.capped
    assert len(stored_texts()) == 6


# ------------------------------------------------------------------ huddle AI notes (canvas)


def canvas(fx, fid="F9", text="# Notes\n- ship friday", **info):
    client = fx[0]
    client.files[fid] = {
        "id": fid, "filetype": "quip", "pretty_type": "Canvas",
        "created": int(NOW.timestamp()), **info,
    }  # fmt: skip
    client.canvases[fid] = text
    return fid


def huddle_notes(a, room="C1"):
    return [m for m in stored(a, inc(), room) if m.kind == "huddle"]


def shared(fid="F9", channel="C1"):
    return {"type": "file_shared", "file_id": fid, "channel_id": channel, "user_id": "U1"}


async def test_a_shared_canvas_is_remembered_once_as_huddle_notes_and_never_answered(fx):
    client, a, port, adapter = fx
    canvas(fx)
    await adapter.on_event(shared(), TEAM)
    await a.drain()
    (note,) = huddle_notes(a)
    assert note.text == "[Huddle notes] # Notes\n- ship friday"
    assert client.api_calls == [("canvases.getContent", {"canvas_id": "F9"})]
    assert not a.ask_fn.calls and not client.posts
    assert [m.kind for m in stored(a, inc())] == ["huddle"]


async def test_resharing_or_editing_a_canvas_replaces_the_record(fx):
    client, a, port, adapter = fx
    canvas(fx)
    await adapter.on_event(shared(), TEAM)
    client.canvases["F9"] = "# Notes\n- ship monday"
    await adapter.on_event(shared(), TEAM)
    await a.drain()
    (note,) = huddle_notes(a)
    assert note.text.endswith("ship monday")


async def test_a_file_that_is_not_a_canvas_is_ignored(fx):
    client, a, port, adapter = fx
    client.files["F2"] = {"id": "F2", "filetype": "pdf", "pretty_type": "PDF", "created": 1}
    await adapter.on_event(shared("F2"), TEAM)
    await a.drain()
    assert client.file_calls == ["F2"] and client.api_calls == []
    assert stored(a, inc()) == []


async def test_a_canvas_in_a_room_with_mode_off_is_not_fetched_or_remembered(fx):
    client, a, port, adapter = fx
    canvas(fx)
    rid = ids.room_id("slack", TEAM, "C1")
    a.team.store.save_settings(
        ids.tenant_id("slack", TEAM), GuildSettingsUpdate(channel_modes={str(rid): "off"}), "t"
    )
    await adapter.on_event(shared(), TEAM)
    await a.drain()
    assert client.api_calls == [] and stored(a, inc()) == []


async def test_workspace_with_memory_off_or_the_env_flag_off_remembers_nothing(fx):
    client, a, port, adapter = fx
    canvas(fx)
    a.team.store.save_settings(
        ids.tenant_id("slack", TEAM), GuildSettingsUpdate(features={"memory": False}), "t"
    )
    await adapter.on_event(shared(), TEAM)
    a.team.store.save_settings(
        ids.tenant_id("slack", TEAM), GuildSettingsUpdate(features={"memory": True}), "t"
    )
    a.ps.slack_huddle_notes = False
    await adapter.on_event(shared(), TEAM)
    await a.drain()
    assert client.api_calls == [] and stored(a, inc()) == []


async def test_unreadable_or_empty_canvas_records_nothing(fx):
    client, a, port, adapter = fx
    canvas(fx)
    client.canvas_errors.add("F9")  # canvas_not_found: the bot cannot view it
    await adapter.on_event(shared(), TEAM)
    canvas(fx, "F8", text="  \n ")
    await adapter.on_event(shared("F8"), TEAM)
    await a.drain()
    assert stored(a, inc()) == []


@pytest.mark.parametrize("flag", ["is_ext_shared", "is_shared", "is_org_shared"])
async def test_slack_connect_and_unknown_rooms_fail_closed(fx, flag):
    client, a, port, adapter = fx
    canvas(fx)
    client.channels["C1"][flag] = True
    await adapter.on_event(shared(), TEAM)
    client.channels["C1"][flag] = False
    client.omit_share_fields.add("is_org_shared")  # Slack left a flag out: unknown
    port.forget_room("C1")
    await adapter.on_event(shared(), TEAM)
    await a.drain()
    assert client.api_calls == [] and stored(a, inc()) == []


async def test_a_one_to_one_dm_canvas_is_never_remembered(fx):
    client, a, port, adapter = fx
    canvas(fx)
    await adapter.on_event(shared(channel="D1"), TEAM)
    await a.drain()
    assert client.api_calls == []


async def test_huddle_thread_message_is_metadata_only_and_remembers_its_canvas(fx):
    client, a, port, adapter = fx
    canvas(fx)
    event = ev(
        subtype="huddle_thread", user="USLACKBOT", text="", ts="1791954000.000500",
        room={"id": "R1", "call_family": "huddle", "has_ended": True,
              "thread_root_ts": "1791954000.000500", "attached_file_ids": ["F9"]},
    )  # fmt: skip
    await adapter.on_message(event, TEAM)
    await adapter.on_message({**event, "text": f"<@{BOT}> summarise"}, TEAM, mention=True)
    await a.drain()
    assert not a.ask_fn.calls and not client.posts
    assert [m.kind for m in stored(a, inc())] == ["huddle"]  # no chat line for the huddle message


async def test_huddle_thread_without_a_canvas_records_and_answers_nothing(fx):
    client, a, port, adapter = fx
    await adapter.on_message(
        ev(subtype="huddle_thread", user="USLACKBOT", text=f"<@{BOT}> hi",
           room={"id": "R1", "has_ended": False}),
        TEAM,
    )  # fmt: skip
    await a.drain()
    assert not a.ask_fn.calls and not client.posts and stored(a, inc()) == []


async def test_a_bot_reply_in_a_known_huddle_thread_that_carries_a_canvas_is_remembered(fx):
    client, a, port, adapter = fx
    canvas(fx)
    root = "1791954000.000500"
    await adapter.on_message(
        ev(subtype="huddle_thread", user="USLACKBOT", text="", ts=root, room={"id": "R1"}), TEAM
    )
    reply = ev(
        ts="1791954999.000100", thread_ts=root, user="USLACKBOT", bot_id="B1", text="notes",
        files=[{"id": "F9"}],
    )  # fmt: skip
    await adapter.on_message(reply, TEAM)
    await a.drain()
    assert len(huddle_notes(a)) == 1 and not a.ask_fn.calls
    # the same files in an unrelated thread are not huddle notes
    await adapter.on_message({**reply, "ts": "5.0", "thread_ts": "4.0"}, TEAM)
    assert len(client.api_calls) == 1


async def test_user_huddle_changed_is_logged_without_re_verifying_the_account(fx, caplog):
    client, a, port, adapter = fx
    caplog.set_level("INFO", logger="salid.platforms.slack")
    before = list(client.calls)
    await adapter.on_event(
        {"type": "user_huddle_changed",
         "user": {"id": "U1", "profile": {"huddle_state": "in_a_huddle",
                                           "huddle_state_call_id": "R1"}}},
        TEAM,
    )  # fmt: skip
    assert client.calls == before
    assert "huddle_state=in_a_huddle" in caplog.text and "call_id=R1" in caplog.text


async def test_diagnostics_log_ids_only_never_text(fx, caplog):
    client, a, port, adapter = fx
    caplog.set_level("INFO", logger="salid.platforms.slack")
    canvas(fx, text="TOP SECRET TRANSCRIPT")
    await adapter.on_event(shared(), TEAM)
    assert "file_shared file=F9 channel=C1" in caplog.text
    assert "canvas=True" in caplog.text and "TOP SECRET" not in caplog.text


def test_lifecycle_listener_receives_file_shared_and_huddle_events():
    assert slack.LIFECYCLE_EVENTS.match("file_shared")
    assert slack.LIFECYCLE_EVENTS.match("user_huddle_changed")
