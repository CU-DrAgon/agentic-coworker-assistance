"""Change outbox on the chat platforms: a notice is pushed 1:1 with "reply OK", resent, acked by a
reply in the 1:1 (never in a group, never by someone else), apply targets are acked as nothing to
apply, and a platform nobody serves keeps its targets. Offline: fake pushers, SQLite."""

from datetime import UTC, datetime, timedelta

import pytest
from conftest import FixedClockSettings
from platform_fakes import FakePort, inc, make_assistant

from salid.change_outbox import ChangeOutbox, diff_members
from salid.coo_store import SQLiteDocStore
from salid.i18n import t
from salid.orgs import Member, OrgStore, Workspace, workspace_id
from salid.platforms import change_outbox_platform as cop

T0 = datetime(2026, 10, 18, 9, 0, tzinfo=UTC)


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    def advance(self, minutes):
        self.t += timedelta(minutes=minutes)


class World:
    pass


@pytest.fixture
def wd(tmp_path):
    w = World()
    w.docs = SQLiteDocStore(":memory:")
    w.orgs = OrgStore(w.docs)
    w.clock = Clock()
    w.orgs.create(
        "Cafe",
        "boss@example.com",
        org_id="org_a",
        members=[
            Member(email="boss@example.com", role="owner", identities={"line": "U-boss"}),
            Member(email="mew@example.com", identities={"line": "U-mew", "slack": "S-mew"}),
        ],
    )
    for platform, tenant in (("line", "Ubot"), ("slack", "T1")):
        w.orgs.attach_workspace(
            "org_a",
            Workspace(id=workspace_id(platform, tenant), platform=platform, external_id=tenant),
        )
    w.outbox = ChangeOutbox(w.docs, w.clock, max_attempts=3)
    w.a = make_assistant(FixedClockSettings(exec_user_ids=set()), tmp_path)
    w.changes = cop.PlatformChanges(w.outbox, w.orgs, w.a.team)
    w.a.changes = w.changes
    w.sent = []

    def pusher(platform):
        async def push(user, text):
            w.sent.append((platform, user, text))
            return f"ref{len(w.sent)}"

        return push

    w.pusher = pusher
    w.port = FakePort(platform="line")
    return w


def make_change(w, **fields):
    before = w.orgs.get("org_a")
    w.orgs.update_members(
        "org_a",
        lambda ms: [setattr(m, k, v) for m in ms if m.email == "mew@example.com"
                    for k, v in fields.items()] or True,
    )  # fmt: skip
    after = w.orgs.get("org_a")
    (c,) = w.outbox.record(diff_members(before, after, by="boss@example.com"), after)
    return c


def states(w, c):
    return {
        (x.workspace_id, x.kind): x.status for x in w.outbox.get(c.id).targets
    }  # fmt: skip


async def test_a_notice_is_pushed_with_the_reply_hint_and_apply_is_nothing_to_apply(wd):
    wd.changes.add_pusher("line", "Ubot", wd.pusher("line"))
    c = make_change(wd, title="CTO")
    await wd.changes.tick()
    ((platform, user, text),) = wd.sent
    assert (platform, user) == ("line", "U-mew")
    assert "CTO" in text and t("chg_hint", wd.changes._lang("line", "Ubot")) in text
    assert states(wd, c) == {
        ("line-Ubot", "apply"): "acked",
        ("line-Ubot", "notify"): "sent",
        ("slack-T1", "apply"): "pending",  # Slack is not served by this process
        ("slack-T1", "notify"): "pending",
    }
    assert wd.outbox.get(c.id).targets[0].acked_how == "nothing_to_apply"


async def test_the_notice_is_resent_every_fifteen_minutes_with_a_count(wd):
    wd.changes.add_pusher("line", "Ubot", wd.pusher("line"))
    make_change(wd, title="CTO")
    await wd.changes.tick()
    wd.clock.advance(14)
    await wd.changes.tick()
    assert len(wd.sent) == 1
    wd.clock.advance(1)
    await wd.changes.tick()
    assert (
        len(wd.sent) == 2 and "1" in wd.sent[1][2] and wd.sent[1][2].startswith(wd.sent[0][2][:20])
    )


async def test_a_reply_in_the_1to1_acks_and_stops_the_resends(wd):
    wd.changes.add_pusher("line", "Ubot", wd.pusher("line"))
    c = make_change(wd, title="CTO")
    await wd.changes.tick()
    await wd.a.handle(inc(platform="line", tenant="Ubot", room="U-mew", user="U-mew",
                          text="รับทราบค่ะ", message="m1", is_direct=True), wd.port)  # fmt: skip
    await wd.a.drain()
    assert states(wd, c)[("line-Ubot", "notify")] == "acked"
    assert wd.port.delivered, "she thanks them"
    wd.clock.advance(60)
    await wd.changes.tick()
    assert len([m for m in wd.sent if "Reply OK" in m[2]]) == 1


async def test_ok_from_someone_else_or_in_a_group_acks_nothing(wd):
    wd.changes.add_pusher("line", "Ubot", wd.pusher("line"))
    c = make_change(wd, title="CTO")
    await wd.changes.tick()
    await wd.a.handle(inc(platform="line", tenant="Ubot", room="U-boss", user="U-boss",
                          text="ok", message="m2", is_direct=True), wd.port)  # fmt: skip
    await wd.a.handle(inc(platform="line", tenant="Ubot", room="Group1", user="U-mew",
                          text="ok", message="m3", is_direct=False), wd.port)  # fmt: skip
    await wd.a.drain()
    assert states(wd, c)[("line-Ubot", "notify")] == "sent"


async def test_a_push_that_fails_is_counted_and_retried(wd):
    async def broken(user, text):
        raise RuntimeError("push refused")

    wd.changes.add_pusher("line", "Ubot", broken)
    c = make_change(wd, title="CTO")
    await wd.changes.tick()
    note = next(
        x for x in wd.outbox.get(c.id).targets if x.kind == "notify" and x.platform == "line"
    )
    assert (note.status, note.attempts, note.last_error) == ("pending", 1, "RuntimeError")


async def test_a_platform_nobody_serves_fails_at_the_deadline_and_served_admins_are_told(wd):
    wd.changes.add_pusher("line", "Ubot", wd.pusher("line"))
    c = make_change(wd, title="CTO")
    for _ in range(5):
        wd.clock.advance(15)
        await wd.changes.tick()
    after = wd.outbox.get(c.id)
    assert {x.status for x in after.targets if x.platform == "slack"} == {"failed"}
    alerts = [s for s in wd.sent if s[1] == "U-boss"]
    assert len(alerts) == 1 and "mew@example.com" in alerts[0][2]
    assert "line" in after.alerted_on
    wd.clock.advance(30)
    await wd.changes.tick()
    assert len([s for s in wd.sent if s[1] == "U-boss"]) == 1


async def test_without_a_pusher_the_tick_does_nothing(wd):
    c = make_change(wd, title="CTO")
    await wd.changes.tick()
    assert wd.sent == [] and {x.status for x in wd.outbox.get(c.id).targets} == {"pending"}


async def test_the_assistant_without_changes_is_unchanged(wd):
    wd.a.changes = None
    await wd.a.handle(inc(platform="line", tenant="Ubot", room="U-mew", user="U-mew",
                          text="ok", message="m9", is_direct=True), wd.port)  # fmt: skip
    await wd.a.drain()


async def test_each_pusher_sends_one_message_to_the_person():
    calls = []

    class Port:
        client = None

        async def _call(self, fn, **kw):
            calls.append((fn.__name__, kw))
            return {"channel": {"id": "D1"}, "ts": "1.2"}

    class Client:
        async def conversations_open(self, **kw): ...
        async def chat_postMessage(self, **kw): ...

    port = Port()
    port.client = Client()
    assert await cop.slack_pusher(port)("U1", "hi") == "1.2"
    assert calls == [
        ("conversations_open", {"users": "U1"}),
        ("chat_postMessage", {"channel": "D1", "text": "hi"}),
    ]

    class Line:
        def push(self, to, texts):
            calls.append(("line", to, texts))
            return ["m1"]

    assert await cop.line_pusher(Line())("U2", "hi") == "m1"

    class Tg:
        async def send_message(self, params):
            calls.append(("tg", params))
            return {"message_id": 7}

    assert await cop.telegram_pusher(Tg())("42", "hi") == "7"

    class Wa:
        async def send_text(self, to, body, reply_to):
            calls.append(("wa", to, body, reply_to))
            return "wamid"

    assert await cop.whatsapp_pusher(Wa())("66", "hi") == "wamid"
    assert ("tg", {"chat_id": 42, "text": "hi"}) in calls and ("wa", "66", "hi", None) in calls
