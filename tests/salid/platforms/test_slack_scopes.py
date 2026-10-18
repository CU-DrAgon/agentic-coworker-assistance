"""Slack features of the core scopes: files:write, im:write, reactions:read/write, commands,
usergroups:read. A fake Web API client, no network."""

import logging

from conftest import FixedClockSettings
from platform_fakes import Clock, inc, make_assistant
from test_slack import BOT, TEAM, FakeSlack

from salid.platforms import slack
from salid.platforms.config import PlatformSettings
from salid.platforms.slack import SlackAdapter, SlackPort
from salid.qa import Answer


class ScopeSlack(FakeSlack):
    def __init__(self):
        super().__init__()
        self.reactions: list[tuple[str, dict]] = []
        self.uploads: list[dict] = []
        self.opened: list[str] = []
        self.groups = {"S1": {"handle": "design-team", "users": ["U1", "U3", "U9"]}}
        self.group_calls: list[str] = []
        self.fail_reactions = False
        self.fail_upload = False

    async def reactions_add(self, **kw):
        if self.fail_reactions:
            raise RuntimeError("missing_scope")
        self.reactions.append(("add", kw))
        return {"ok": True}

    async def reactions_remove(self, **kw):
        self.reactions.append(("remove", kw))
        return {"ok": True}

    async def files_upload_v2(self, **kw):
        if self.fail_upload:
            raise RuntimeError("boom")
        self.uploads.append(kw)
        return {"ok": True}

    async def conversations_open(self, users):
        self.opened.append(users)
        return {"channel": {"id": "D" + users}}

    async def usergroups_list(self, **kw):
        self.group_calls.append("list")
        return {"usergroups": [{"id": k, "handle": v["handle"]} for k, v in self.groups.items()]}

    async def usergroups_users_list(self, usergroup):
        self.group_calls.append("users:" + usergroup)
        if usergroup not in self.groups:
            raise RuntimeError("no_such_subteam")
        return {"users": self.groups[usergroup]["users"]}


def build(tmp_path, ask=None, **ps_kw):
    s = FixedClockSettings(exec_user_ids=set())
    client = ScopeSlack()
    a = make_assistant(s, tmp_path, ask=ask, **ps_kw)
    port = SlackPort(client, TEAM, BOT, clock=Clock())
    adapter = SlackAdapter(a, port, lambda url: b"x")
    return client, a, port, adapter


def ev(**kw):
    base = {"type": "message", "channel": "C1", "channel_type": "channel", "user": "U1",
            "text": f"<@{BOT}> hi", "ts": "1791954000.000100"}  # fmt: skip
    base.update(kw)
    return base


class AskWith:
    def __init__(self, **answer):
        self.answer, self.calls = answer, []

    def __call__(self, settings, store, question, **kw):
        self.calls.append((question, kw))
        return Answer("done", 0, 0.0, [], **self.answer)


# ------------------------------------------------------------------ files:write


async def test_generated_files_are_uploaded_into_the_answer_thread(tmp_path):
    ask = AskWith(attachments=[("report.pdf", b"%PDF-1"), ("deck.pptx", b"PK")])
    client, a, port, adapter = build(tmp_path, ask=ask)
    await adapter.on_message(ev(), TEAM)
    await a.drain()
    (up,) = client.uploads
    assert up["channel"] == "C1" and up["thread_ts"] == "1791954000.000100"
    assert [f["filename"] for f in up["file_uploads"]] == ["report.pdf", "deck.pptx"]
    assert up["file_uploads"][0]["file"] == b"%PDF-1"


async def test_upload_skips_empty_and_oversize_files_and_never_breaks_the_answer(tmp_path):
    big = b"x" * (slack.UPLOAD_MAX_BYTES + 1)
    client, a, port, adapter = build(
        tmp_path, ask=AskWith(attachments=[("e.pdf", b""), ("b.bin", big)])
    )
    await adapter.on_message(ev(), TEAM)
    assert client.uploads == [] and client.posts[0]["text"] == "done"

    client, a, port, adapter = build(tmp_path, ask=AskWith(attachments=[("r.pdf", b"%PDF")]))
    client.fail_upload = True
    await adapter.on_message(ev(), TEAM)
    assert client.posts[0]["text"] == "done"  # the text answer was still delivered


async def test_a_dm_upload_is_not_threaded(tmp_path):
    client, a, port, adapter = build(tmp_path, ask=AskWith(attachments=[("r.pdf", b"%PDF")]))
    await adapter.on_message(ev(channel="D1", channel_type="im", text="report please"), TEAM)
    assert "thread_ts" not in client.uploads[0]


async def test_a_port_without_the_hooks_just_ignores_files(tmp_path):
    from platform_fakes import FakePort

    s = FixedClockSettings(exec_user_ids=set())
    a = make_assistant(
        s, tmp_path, ask=AskWith(attachments=[("r.pdf", b"%PDF")], receipts=[(1, "✅")])
    )
    port = FakePort()
    await a.handle(inc(is_direct=True, text="hi"), port)
    assert port.texts == ["done"]


# ------------------------------------------------------------------ im:write


async def test_open_dm_is_cached_and_send_dm_posts_formatted_text(tmp_path):
    client, a, port, adapter = build(tmp_path)
    ts = await port.send_dm("U1", "**Morning brief**")
    assert ts == "91.000" and await port.open_dm("U1") == "DU1"
    await port.send_dm("U1", "second")
    assert client.opened == ["U1"]  # conversations.open once
    assert client.posts[0]["channel"] == "DU1" and "*Morning brief*" in client.posts[0]["text"]


async def test_send_dm_returns_none_when_the_dm_cannot_be_opened(tmp_path):
    client, a, port, adapter = build(tmp_path)

    async def boom(users):
        raise RuntimeError("missing_scope")

    client.conversations_open = boom
    assert await port.send_dm("U1", "hi") is None and client.posts == []
    port.revoked = True
    assert await port.open_dm("U1") is None


# ------------------------------------------------------------------ reactions:write


async def test_eyes_is_added_while_working_and_removed_with_the_answer(tmp_path):
    client, a, port, adapter = build(tmp_path)
    await adapter.on_message(ev(), TEAM)
    kinds = [(k, kw["name"], kw["channel"], kw["timestamp"]) for k, kw in client.reactions]
    ts = "1791954000.000100"
    assert kinds == [("add", "eyes", "C1", ts), ("remove", "eyes", "C1", ts)]


async def test_recorded_and_flagged_receipts_get_their_emoji(tmp_path):
    ask = AskWith(receipts=[(1, "✅"), (2, "⚠️"), (3, None)])
    client, a, port, adapter = build(tmp_path, ask=ask)
    await adapter.on_message(ev(), TEAM)
    added = [kw["name"] for k, kw in client.reactions if k == "add"]
    assert added == ["eyes", "white_check_mark", "warning"]


async def test_reaction_failures_never_break_the_answer(tmp_path):
    client, a, port, adapter = build(tmp_path)
    client.fail_reactions = True
    await adapter.on_message(ev(), TEAM)
    assert client.posts[0]["text"] == "ok" and client.reactions == []


async def test_nothing_is_removed_that_salid_did_not_add(tmp_path):
    client, a, port, adapter = build(tmp_path)
    await port.deliver(inc(message="5.0"), ["x"], None)
    assert client.reactions == []


# ------------------------------------------------------------------ reactions:read


async def test_reaction_added_logs_ids_only_and_ignores_the_bot_itself(tmp_path, caplog):
    client, a, port, adapter = build(tmp_path)
    caplog.set_level(logging.INFO, logger="salid.platforms.slack")
    item = {"type": "message", "channel": "C1", "ts": "5.0"}
    await adapter.on_event(
        {"type": "reaction_added", "user": "U1", "reaction": "x", "item": item, "item_user": BOT}
    )
    assert "slack reaction_added user=U1 name=x channel=C1 ts=5.0" in caplog.text
    assert not client.posts


async def test_a_check_mark_in_a_dm_confirms_an_open_change_notice(tmp_path):
    client, a, port, adapter = build(tmp_path)
    seen = []

    class Changes:
        async def on_message(self, msg):
            seen.append((msg.user, msg.text, msg.is_direct))
            return "thanks"

    a.changes = Changes()
    item = {"type": "message", "channel": "D1", "ts": "5.0"}
    base = {"type": "reaction_added", "reaction": "white_check_mark", "item": item}
    await adapter.on_event({**base, "user": "U1", "item_user": BOT})
    assert seen == [("U1", "OK", True)] and client.posts[0]["text"] == "thanks"
    # not in a DM, someone else's message, the bot's own reaction, or another emoji: ignored
    await adapter.on_event(
        {**base, "user": "U1", "item_user": BOT, "item": {**item, "channel": "C1"}}
    )
    await adapter.on_event({**base, "user": "U1", "item_user": "U2"})
    await adapter.on_event({**base, "user": BOT, "item_user": BOT})
    await adapter.on_event({**base, "user": "U1", "item_user": BOT, "reaction": "x"})
    assert len(seen) == 1


# ------------------------------------------------------------------ commands


class Ack:
    def __init__(self):
        self.calls = []

    async def __call__(self, body=None):
        self.calls.append(body)


def cmd(**kw):
    base = {"command": "/salid", "text": "what did we spend?", "channel_id": "C1",
            "user_id": "U1", "trigger_id": "trig.1"}  # fmt: skip
    base.update(kw)
    return base


async def test_slash_question_acks_ephemerally_then_answers_in_the_channel(tmp_path):
    client, a, port, adapter = build(tmp_path)
    ack = Ack()
    await adapter.on_command(cmd(), TEAM, ack)
    await a.drain()
    assert ack.calls[0]["response_type"] == "ephemeral" and "moment" in ack.calls[0]["text"]
    assert a.ask_fn.calls[0][0] == "what did we spend?"
    (post,) = client.posts
    assert post["channel"] == "C1" and post["thread_ts"] is None  # top level, no ts to thread under
    assert post["text"].startswith("> what did we spend?") and post["text"].endswith("ok")
    assert client.reactions == []  # a slash command has no message to react to


async def test_slash_help_and_empty_text_show_the_help_without_asking(tmp_path):
    client, a, port, adapter = build(tmp_path)
    for text in ("help", "", "  HELP "):
        ack = Ack()
        await adapter.on_command(cmd(text=text), TEAM, ack)
        assert ack.calls == [{"response_type": "ephemeral", "text": slack.SLASH_HELP}]
    assert not a.ask_fn.calls and not client.posts


async def test_slash_respects_a_channel_that_is_switched_off(tmp_path):
    client, a, port, adapter = build(tmp_path)
    a._mode = lambda gid, rid, name, direct: "off"
    ack = Ack()
    await adapter.on_command(cmd(), TEAM, ack)
    assert ack.calls[0]["text"] == slack.SLASH_OFF and not a.ask_fn.calls and not client.posts


async def test_slash_in_a_dm_answers_there_and_a_revoked_tenant_is_silent(tmp_path):
    client, a, port, adapter = build(tmp_path)
    await adapter.on_command(cmd(channel_id="D1"), TEAM, Ack())
    assert client.posts[0]["channel"] == "D1"
    port.revoked = True
    client.posts.clear()
    await adapter.on_command(cmd(trigger_id="trig.2"), TEAM, Ack())
    assert not client.posts


async def test_a_failing_slash_handler_still_acks_and_never_raises(tmp_path):
    client, a, port, adapter = build(tmp_path)

    async def boom(*args):
        raise RuntimeError("secret")

    a.handle = boom
    ack = Ack()
    await adapter.on_command(cmd(), TEAM, ack)
    assert ack.calls  # Slack got its answer within 3 seconds


# ------------------------------------------------------------------ usergroups:read


async def test_a_group_mention_becomes_handle_and_head_count(tmp_path):
    client, a, port, adapter = build(tmp_path)
    text = f"<@{BOT}> ask <!subteam^S1|@design-team> about the logo"
    await adapter.on_message(ev(text=text), TEAM)
    assert a.ask_fn.calls[0][0] == "ask @design-team (3 people) about the logo"
    await adapter.on_message(ev(text=text, ts="2.0"), TEAM)
    assert client.group_calls == ["users:S1"]  # cached for 10 minutes


async def test_a_group_without_a_handle_in_the_text_is_looked_up_and_survives_a_missing_scope(
    tmp_path,
):
    client, a, port, adapter = build(tmp_path)
    assert await port.group_label("S1") == "@design-team (3 people)"
    client2, a2, port2, adapter2 = build(tmp_path)
    assert await port2.group_label("S404", "@ghost") == "@ghost"  # users.list failed: no count


async def test_members_of_an_exec_group_are_executives_but_guests_and_foreigners_are_not(tmp_path):
    client, a, port, adapter = build(tmp_path)
    client.groups["S1"]["users"] = ["U1", "UX", "U3"]
    client.users["U3"]["is_restricted"] = True
    assert not await port.is_admin_user("U1")  # no exec groups configured
    port.exec_usergroups = frozenset({"@Design-Team"})  # a handle, any case
    assert await port.is_admin_user("U1")
    assert not await port.is_admin_user("UX")  # another workspace
    assert not await port.is_admin_user("U3")  # guest
    port.exec_usergroups = frozenset({"S1"})  # an id
    assert await port.is_admin_user("U1")
    port.exec_usergroups = frozenset({"nope"})
    assert not await port.is_admin_user("U1")


async def test_exec_group_lookup_failure_fails_closed(tmp_path):
    client, a, port, adapter = build(tmp_path)
    port.exec_usergroups = frozenset({"S9"})
    assert not await port.is_admin_user("U1")


def test_exec_usergroups_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("SALID_SLACK_EXEC_USERGROUPS", "leadership, S0123ABC")
    assert PlatformSettings(_env_file=None).slack_exec_usergroups == {"leadership", "S0123ABC"}
