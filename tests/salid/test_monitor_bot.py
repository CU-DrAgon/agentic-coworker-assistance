# ruff: noqa: F811, F401
"""Workflow monitor on the bot tick and in the platform bridges: the emitter is flushed once per
tick, the heartbeat (with queue sizes) is written every 30 s, the delivery queues (web -> Discord
actions, the change outbox) emit an event on every state change, finished actions are pruned
after 30 days, and nothing at all is emitted while `SALID_MONITOR_ENABLED` is off. Offline:
SQLite DocStore / fake Firestore, fake Discord objects."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from conftest import GUILD
from fake_firestore import FakeFirestore
from test_change_outbox import drig, tick, web_edit  # noqa: F401  (fixture + helpers)
from test_coordinator import world  # noqa: F401  (fixture: members, as `drig` needs)
from test_modes import bot  # noqa: F401  (fixture)
from test_positions_bot import link, rig  # noqa: F401  (fixture)

from salid import monitor
from salid.actions import KEEP_DAYS, ActionQueue
from salid.bot import SecretaryBot
from salid.change_outbox import ChangeOutbox, diff_members, record_org_edit
from salid.config import Settings
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.monitor import HEARTBEAT, HEARTBEAT_EVERY_S, Audience, Emitter, read_log
from salid.orgs import Member, Org, OrgStore, Workspace, workspace_memory_id
from salid.platforms import runtime
from salid.platforms.config import PlatformSettings
from salid.platforms.runtime import PlatformMonitor, monitor_tick, platform_monitor

NOW = datetime(2026, 10, 18, 9, 0, tzinfo=UTC)
QUEUES = {"actions", "outbox_pending", "memory"}


def ev(i: int = 1) -> dict:
    """A producer event as `monitor.event_for` builds it."""
    aud = Audience(people=["discord:1"]).model_dump()
    return {
        "kind": "action",
        "id": f"a{i}",
        "status": "queued",
        "attention": False,
        "audience": aud,
    }


class Spy:
    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, *a, **k):
        self.n += 1
        return self.fn(*a, **k)


def spy(monkeypatch, owner, name: str) -> Spy:
    s = Spy(getattr(owner, name))
    monkeypatch.setattr(owner, name, s)
    return s


class Clock:
    def __init__(self):
        self.t = 500.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


async def ready():  # stands in for discord.Client.wait_until_ready
    return None


async def until(cond, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


# ------------------------------------------------------------------ bot
def test_flag_defaults_off(bot):
    assert Settings(_env_file=None).monitor_enabled is False
    assert bot.monitor.enabled is False
    assert bot._monitor_queues == {"actions": 0, "outbox_pending": 0, "memory": 0}


async def test_heartbeat_every_30_s_with_queues(bot, monkeypatch):
    bot.monitor.enabled = True
    await bot._monitor_due()
    hb = bot.docs.get(HEARTBEAT, 0, "bot")
    assert hb["process"] == "bot" and set(hb["queues"]) == QUEUES
    writes = spy(monkeypatch, monitor, "write_heartbeat")
    await bot._monitor_due()
    assert writes.n == 0
    bot.clock.advance(HEARTBEAT_EVERY_S - 1)
    await bot._monitor_due()
    assert writes.n == 0
    bot.clock.advance(1)
    await bot._monitor_due()
    assert writes.n == 1


async def test_heartbeat_carries_noted_queues_and_memory_backlog(bot):
    bot.monitor.enabled = True
    bot.memory.scheduler.note(GUILD, 10, "2026-10-14")
    bot.memory.scheduler.note(GUILD, 11, "2026-10-14")
    bot._monitor_note_queue("outbox_pending", 4)
    await bot._monitor_due()
    hb = bot.docs.get(HEARTBEAT, 0, "bot")
    assert hb["queues"] == {"actions": 0, "outbox_pending": 4, "memory": 2}


async def test_disabled_writes_nothing(bot, monkeypatch):
    flushes = spy(monkeypatch, bot.monitor, "flush")
    bot.monitor.add_for_guild(GUILD, ev())  # dropped: the emitter is off
    await bot._monitor_due()
    assert bot.docs.get(HEARTBEAT, 0, "bot") is None
    assert flushes.n == 0
    assert read_log(bot.docs, f"guild:{GUILD}")["seq"] == 0


async def test_flush_resolves_guild_to_org_or_unlinked_key(bot):
    bot.monitor.enabled = True
    bot.monitor.add_for_guild(GUILD, ev())
    await bot._monitor_due()
    assert read_log(bot.docs, f"guild:{GUILD}")["seq"] == 1
    link(bot)
    bot._monitor_init()  # fresh resolver cache
    bot.monitor.enabled = True
    bot.monitor.add_for_guild(GUILD, ev(2))
    await bot._monitor_due()
    assert read_log(bot.docs, "o1")["seq"] == 1
    assert read_log(bot.docs, f"guild:{GUILD}")["seq"] == 1  # unchanged


async def test_flushes_every_tick_between_heartbeats(bot):
    bot.monitor.enabled = True
    await bot._monitor_due()
    bot.monitor.add("o1", ev(1))
    await bot._monitor_due()  # no heartbeat due, the buffer is still written
    bot.monitor.add("o1", ev(2))
    await bot._monitor_due()
    assert [e["id"] for e in read_log(bot.docs, "o1")["events"]] == ["a1", "a2"]


async def test_store_work_runs_off_the_event_loop(bot, monkeypatch):
    seen: list[int] = []

    def on_thread(fn):
        return lambda *a, **k: seen.append(threading.get_ident()) or fn(*a, **k)

    monkeypatch.setattr(monitor, "write_heartbeat", on_thread(monitor.write_heartbeat))
    bot.monitor.enabled = True
    monkeypatch.setattr(bot.monitor, "flush", on_thread(bot.monitor.flush))
    sched = bot.memory.scheduler
    monkeypatch.setattr(sched, "backlog", on_thread(sched.backlog))
    await bot._monitor_due()
    docs = SQLiteDocStore(":memory:")
    em = Emitter(docs, enabled=True)
    monkeypatch.setattr(em, "flush", on_thread(em.flush))
    await monitor_tick(PlatformMonitor(em, docs, on_thread(dict)))
    # flush + queue sizes (they take the scheduler lock) + heartbeat, for the bot and platforms
    assert len(seen) == 6 and threading.get_ident() not in seen


async def test_monitor_due_never_raises(bot, monkeypatch, caplog):
    bot.monitor.enabled = True

    def boom():
        raise RuntimeError("secret text")

    monkeypatch.setattr(bot.monitor, "flush", boom)
    await bot._monitor_due()  # logged, not raised
    assert "RuntimeError" in caplog.text and "secret text" not in caplog.text


async def test_tick_notes_actions_and_leaves_the_monitor_to_its_loop(bot, monkeypatch):
    bot.monitor.enabled = True
    dues = []

    async def three():
        return 3

    async def due():
        dues.append(1)

    monkeypatch.setattr(bot, "_actions_due", three)
    monkeypatch.setattr(bot, "_monitor_due", due)
    await bot.tick()
    assert dues == [] and bot._monitor_queues["actions"] == 3


async def test_monitor_loop_beats_while_a_tick_job_hangs_or_fails(bot, monkeypatch):
    bot.monitor.enabled = True
    monkeypatch.setattr(bot, "wait_until_ready", ready)
    never = asyncio.Event()

    async def hang():
        await never.wait()

    async def boom():
        raise RuntimeError("tick job failed")

    monkeypatch.setattr(bot, "_signoff_due", hang)  # the first tick job never finishes
    tick = asyncio.create_task(bot.tick())
    loop = asyncio.create_task(bot._monitor_loop(0.01))
    try:
        await until(lambda: bot.docs.get(HEARTBEAT, 0, "bot") is not None)
        writes = spy(monkeypatch, monitor, "write_heartbeat")
        bot.monitor.add("o1", ev(1))
        bot.clock.advance(HEARTBEAT_EVERY_S + 1)  # over 30 s later, the tick is still stuck
        await until(lambda: writes.n == 1 and read_log(bot.docs, "o1")["seq"] == 1)
        assert not tick.done()
        monkeypatch.setattr(bot, "_signoff_due", boom)
        with pytest.raises(RuntimeError):
            await bot.tick()
        bot.monitor.add("o1", ev(2))
        bot.clock.advance(HEARTBEAT_EVERY_S)
        await until(lambda: writes.n == 2 and read_log(bot.docs, "o1")["seq"] == 2)
    finally:
        tick.cancel()
        loop.cancel()
        await asyncio.gather(tick, loop, return_exceptions=True)


async def test_monitor_start_once_only_when_on_and_stop_cancels(bot, monkeypatch):
    monkeypatch.setattr(bot, "wait_until_ready", ready)
    bot._monitor_start(0.01)
    assert bot._monitor_task is None  # off: no loop at all
    bot.monitor.enabled = True
    bot._monitor_start(0.01)
    task = bot._monitor_task
    bot._monitor_start(0.01)
    assert bot._monitor_task is task  # one loop only
    await until(lambda: bot.docs.get(HEARTBEAT, 0, "bot") is not None)
    bot._monitor_stop()
    assert bot._monitor_task is None
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_monitor_loop_ends_when_the_bot_closes(bot, monkeypatch):
    bot.monitor.enabled = True
    monkeypatch.setattr(bot, "wait_until_ready", ready)
    closed: list[bool] = []
    monkeypatch.setattr(bot, "is_closed", lambda: bool(closed))
    dues = spy(monkeypatch, bot, "_monitor_due")
    loop = asyncio.create_task(bot._monitor_loop(0.01))
    await until(lambda: dues.n >= 2)
    closed.append(True)
    await asyncio.wait_for(loop, 1)  # returns by itself, no error
    assert not loop.cancelled() and loop.exception() is None


async def test_change_outbox_run_notes_pending_targets(bot, monkeypatch):
    def target(platform, ws):
        return SimpleNamespace(platform=platform, workspace_id=ws, kind="notify", person="1")

    due = [
        (SimpleNamespace(org_id="o1"), target("discord", f"discord-{GUILD}")),
        (SimpleNamespace(org_id="o1"), target("discord", f"discord-{GUILD}")),
        (SimpleNamespace(org_id="o1"), target("discord", "discord-999")),  # not this bot's
        (SimpleNamespace(org_id="o1"), target("slack", "slack-T1")),
    ]
    bot.change_outbox = SimpleNamespace(prune=lambda org_ids: 0, due=lambda org_ids: due)

    async def nothing(*a, **k):
        return None

    monkeypatch.setattr(bot, "_change_notify", nothing)
    monkeypatch.setattr(bot, "_change_alerts", nothing)
    await bot._change_outbox_run()
    assert bot._monitor_queues["outbox_pending"] == 2


# ------------------------------------------------------------------ delivery producers
def action_guild(sent: list) -> SimpleNamespace:
    """A Discord server with a #general the bot can post in (as in test_center_api)."""

    class Msg:
        id, jump_url = 42, "https://discord.com/channels/1/2/42"

    async def send(text):
        sent.append(text)
        return Msg()

    ch = SimpleNamespace(id=1001, name="general", send=send, guild=None)
    return SimpleNamespace(
        id=GUILD,
        text_channels=[ch],
        voice_channels=[],
        get_channel=lambda i: ch if i == 1001 else None,
    )


def seq_order(events: list[dict]) -> list[dict]:
    return sorted(events, key=lambda e: e["seq"])


async def test_action_lifecycle_emits_queued_then_done(bot, monkeypatch):
    bot.monitor.enabled = True
    link(bot)
    sent: list = []
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [action_guild(sent)]))
    a = bot.actions.enqueue(
        GUILD, "post_message", {"channel": "general", "text": "Hello team"}, by="boss@x.com"
    )
    assert await bot._actions_due() == 1 and sent == ["Hello team"]
    await bot._monitor_due()
    events = seq_order(read_log(bot.docs, "o1")["events"])
    assert [(e["kind"], e["id"], e["status"]) for e in events] == [
        ("action", a["id"], "queued"),
        ("action", a["id"], "done"),
    ]
    assert events[0]["audience"] == {
        "admin_only": True,
        "exec_only": False,
        "dm": False,
        "people": ["boss@x.com"],
        "channel_ids": [],
    }
    assert "Hello team" not in str(events)  # never the action's args


async def test_outbox_updates_emit_change_events(drig):
    drig.bot.monitor.enabled = True
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)  # the apply target is acked, the notice is sent
    await drig.bot._monitor_due()
    events = seq_order(read_log(drig.bot.docs, "o1")["events"])
    (c,) = drig.bot.change_outbox.list_changes("o1")
    assert len(events) >= 3 and {(e["kind"], e["id"]) for e in events} == {("change", c.id)}
    assert events[0]["status"] == "queued" and events[-1]["status"] == "waiting"
    assert all(e["audience"]["admin_only"] and not e["audience"]["people"] for e in events)
    assert "CTO" not in str(events) and "peer" not in str(events)  # no values, no names


async def test_outbox_emits_nothing_while_the_monitor_is_off(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    drig.bot.monitor.enabled = True  # turned on afterwards: nothing was buffered meanwhile
    await drig.bot._monitor_due()
    assert read_log(drig.bot.docs, "o1")["seq"] == 0


def test_prune_keeps_30_days_and_pending(monkeypatch):
    docs = SQLiteDocStore(":memory:")
    bkk = ZoneInfo("Asia/Bangkok")

    def queued(days: float, *, end: str = "done", zone=UTC) -> str:
        q = ActionQueue(docs, clock=lambda: (NOW - timedelta(days=days)).astimezone(zone))
        a = q.enqueue(GUILD, "announce", {"text": "x"}, by="boss@x.com")
        if end == "done":
            q.finish(GUILD, a["id"], result={})
        elif end == "failed":
            q.finish(GUILD, a["id"], error="missing permission")
        return a["id"]

    old = queued(KEEP_DAYS + 1)
    old_failed = queued(KEEP_DAYS + 1, end="failed", zone=bkk)  # the bot writes Bangkok time
    recent = queued(KEEP_DAYS - 1, zone=bkk)
    edge = queued(KEEP_DAYS - 1 / 24)  # 23 h short of the cut-off
    waiting = queued(KEEP_DAYS + 1, end="")  # still pending: `expire` finishes it, never prune
    other = ActionQueue(docs, clock=lambda: NOW - timedelta(days=40))
    elsewhere = other.enqueue(2000, "announce", {"text": "x"}, by="boss@x.com")["id"]
    other.finish(2000, elsewhere, result={})

    q = ActionQueue(docs, clock=lambda: NOW)
    assert q.prune(GUILD) == 2
    assert q.get(GUILD, old) is None and q.get(GUILD, old_failed) is None
    assert all(q.get(GUILD, a) for a in (recent, edge, waiting))
    assert q.get(2000, elsewhere) is not None  # another guild is not touched

    reads = [spy(monkeypatch, docs, "find"), spy(monkeypatch, docs, "find_items")]
    assert q.prune(GUILD) == 0 and sum(r.n for r in reads) == 0  # within the hour: no read
    assert q.prune(GUILD, force=True) == 0 and sum(r.n for r in reads) == 1
    assert q.prune(2000) == 1 and q.get(2000, elsewhere) is None  # the hour is per guild


async def test_actions_due_prunes_each_guild_hourly(bot, monkeypatch):
    sent: list = []
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [action_guild(sent)]))
    pruned: list[int] = []
    real = bot.actions.prune
    monkeypatch.setattr(bot.actions, "prune", lambda gid, **k: pruned.append(gid) or real(gid))
    old = ActionQueue(bot.docs, clock=lambda: bot.s.now() - timedelta(days=KEEP_DAYS + 1))
    a = old.enqueue(GUILD, "announce", {"text": "x"}, by="boss@x.com")
    old.finish(GUILD, a["id"], result={})
    assert await bot._actions_due() == 0
    assert pruned == [GUILD] and bot.actions.get(GUILD, a["id"]) is None


async def test_a_failing_log_never_breaks_enqueue_or_the_tick(bot, monkeypatch, caplog):
    bot.monitor.enabled = True
    link(bot)
    sent: list = []
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [action_guild(sent)]))

    def broken(fn):
        raise RuntimeError("secret text")

    monkeypatch.setattr(bot.docs, "transact", broken)  # every change-log write fails
    failed = monitor.EMIT_STATS["failed"]
    a = bot.actions.enqueue(
        GUILD, "post_message", {"channel": "general", "text": "hi"}, by="boss@x.com"
    )
    assert a["status"] == "pending" and bot.actions.get(GUILD, a["id"]) is not None
    assert await bot._actions_due() == 1
    assert bot.actions.get(GUILD, a["id"])["status"] == "done" and sent == ["hi"]
    await bot._monitor_due()  # logged and counted, not raised
    assert monitor.EMIT_STATS["failed"] > failed
    assert read_log(bot.docs, "o1")["seq"] == 0
    assert "RuntimeError" in caplog.text and "secret text" not in caplog.text


def test_a_raising_hook_is_logged_and_swallowed(caplog):
    docs = SQLiteDocStore(":memory:")

    def boom(_):
        raise RuntimeError("secret text")

    q = ActionQueue(docs, on_change=boom)
    a = q.enqueue(GUILD, "announce", {"text": "x"}, by="boss@x.com")
    q.finish(GUILD, a["id"], result={})
    assert q.get(GUILD, a["id"])["status"] == "done"
    org, before, after = outbox_org()
    outbox = ChangeOutbox(docs, on_change=boom)
    (c,) = outbox.record(diff_members(before, after), org)
    assert outbox.mark_attempt(c.id, "slack-T1", "", org_id="o1")
    assert outbox.get(c.id, "o1").targets[0].status == "sent"
    assert "RuntimeError" in caplog.text and "secret text" not in caplog.text


def outbox_org() -> tuple[Org, Org, Org]:
    """An org with one Slack workspace, and a role edit of one member: (org, before, after)."""
    ws = [Workspace(id="slack-T1", platform="slack", external_id="T1")]
    before = Org(id="o1", name="Cafe", members=[Member(email="a@x.com")], workspaces=ws)
    after = Org(
        id="o1", name="Cafe", members=[Member(email="a@x.com", role="admin")], workspaces=ws
    )
    return after, before, after


@pytest.mark.parametrize("kind", ["sqlite", "firestore"])
def test_outbox_hook_gets_what_was_stored_or_committed(kind):
    docs = SQLiteDocStore(":memory:") if kind == "sqlite" else FirestoreDocStore(FakeFirestore())
    seen: list = []

    def hook(changes):  # reads the store: the hook runs after the transaction committed
        seen.append([(c.id, outbox.get(c.id, "o1").targets[0].status) for c in changes])

    outbox = ChangeOutbox(docs, lambda: NOW, on_change=hook)
    org, before, after = outbox_org()
    (c,) = outbox.record(diff_members(before, after), org)
    assert seen == [[(c.id, "pending")]]
    assert outbox.record(diff_members(before, after), org) == []  # a duplicate: no call
    assert outbox.mark_attempt(c.id, "slack-T1", "", org_id="o1")
    assert seen[-1] == [(c.id, "sent")]
    assert not outbox.mark_attempt(c.id, "slack-T9", "", org_id="o1")  # no such target
    assert outbox.ack(c.id, "slack-T1", "", "applied")  # the org is found from the change id
    assert seen[-1] == [(c.id, "acked")] and len(seen) == 3


def test_from_settings_and_record_org_edit_pass_the_hook():
    docs = SQLiteDocStore(":memory:")
    seen: list = []
    org, before, after = outbox_org()
    s = Settings(_env_file=None)
    assert ChangeOutbox.from_settings(docs, s, on_change=seen.append).on_change == seen.append
    assert record_org_edit(docs, before, after, "boss@x.com", on_change=seen.append) == 1
    owner = after.model_copy(update={"members": [Member(email="a@x.com", role="owner")]})
    assert record_org_edit(docs, before, owner, "x", settings=s, on_change=seen.append) == 1
    assert [len(x) for x in seen] == [1, 1]


def test_memory_backlog_counts_pending_entries(bot):
    sched = bot.memory.scheduler
    assert sched.backlog() == 0
    sched.note(GUILD, 10, "2026-10-14")
    sched.note(GUILD, 10, "2026-10-14")  # same summary
    sched.mark_dirty(GUILD, 11, "2026-10-14")
    assert sched.backlog() == 2


def test_headless_web_bot_never_buffers_events(monkeypatch, tmp_path):
    """The web service's headless bot (`salid.headless`) never connects, so its `setup_hook`
    never starts the monitor loop: even with `SALID_MONITOR_ENABLED` set its emitter stays off,
    else producer events would pile up in a buffer nothing flushes."""
    from salid import headless, llm

    monkeypatch.setenv("SALID_MONITOR_ENABLED", "true")
    monkeypatch.setenv("SALID_STORE", "sqlite")
    monkeypatch.setenv("SALID_DB_PATH", str(tmp_path / "ledger.db"))
    monkeypatch.setenv("SALID_DOCS_LOCAL_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(llm, "make_client", lambda s: None)
    headless._headless.cache_clear()
    try:
        bot = headless.get_headless_bot()
        assert bot.s.monitor_enabled is True and bot.headless is True
        assert bot.monitor.enabled is False
        bot._monitor_action(
            {"id": "a1", "guild_id": GUILD, "kind": "announce", "status": "pending"}
        )
        bot._monitor_job(
            {"job_id": "j1", "guild_id": GUILD, "kind": "research", "state": "running"}
        )
        bot.monitor.add("o1", ev())
        assert bot.monitor._buf == []  # nothing buffered
        assert bot.monitor.flush() == 0
    finally:
        headless._headless.cache_clear()  # later tests build their own headless bot


# ------------------------------------------------------------------ platform bridges
async def test_platform_monitor_tick_writes_platforms_heartbeat_and_flushes(monkeypatch):
    docs = SQLiteDocStore(":memory:")
    em = Emitter(docs, enabled=True, now=lambda: NOW)
    clock = Clock()
    m = PlatformMonitor(em, docs, lambda: {"memory": 4}, clock=clock)
    em.add("o1", ev(1))
    await monitor_tick(m)
    assert read_log(docs, "o1")["seq"] == 1
    hb = docs.get(HEARTBEAT, 0, "platforms")
    assert hb["process"] == "platforms" and hb["tick_at"] == NOW.isoformat()
    assert hb["queues"] == {"actions": 0, "outbox_pending": 0, "memory": 4}
    writes = spy(monkeypatch, monitor, "write_heartbeat")
    em.add("o1", ev(2))
    await monitor_tick(m)
    assert writes.n == 0 and read_log(docs, "o1")["seq"] == 2  # flushed, no heartbeat yet
    clock.advance(HEARTBEAT_EVERY_S)
    await monitor_tick(m)
    assert writes.n == 1


async def test_platform_monitor_tick_never_raises(caplog):
    docs = SQLiteDocStore(":memory:")

    def broken():
        raise RuntimeError("secret text")

    m = PlatformMonitor(Emitter(docs, enabled=True), docs, broken)
    await monitor_tick(m)  # the queues callback failed: logged, not raised
    assert "RuntimeError" in caplog.text and "secret text" not in caplog.text
    assert docs.get(HEARTBEAT, 0, "platforms") is None


def fake_assistant(docs, backlog: int = 0):
    return SimpleNamespace(
        identity=SimpleNamespace(orgs=SimpleNamespace(docs=docs)),
        memory=SimpleNamespace(scheduler=SimpleNamespace(backlog=lambda: backlog)),
    )


def test_platform_monitor_only_with_flag_and_org_store():
    docs = SQLiteDocStore(":memory:")
    on = Settings(_env_file=None, monitor_enabled=True)
    assert platform_monitor(Settings(_env_file=None), fake_assistant(docs)) is None
    assert platform_monitor(on, SimpleNamespace(identity=None)) is None
    m = platform_monitor(on, fake_assistant(docs, backlog=7))
    assert m.docs is docs and m.emitter.docs is docs and m.emitter.enabled
    assert m.queues() == {"actions": 0, "outbox_pending": 0, "memory": 7}
    store = OrgStore(docs)
    store.create("Cafe", "boss@x.com", org_id="o1")
    store.attach_workspace("o1", Workspace(id="slack-T1", platform="slack", external_id="T1"))
    linked = platform_monitor(on, fake_assistant(docs))
    assert linked.emitter.resolve(workspace_memory_id("slack", "T1")) == "o1"  # the org's log
    assert linked.emitter.resolve(5) == "guild:5"


async def test_run_starts_the_monitor_loop(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    started = []
    sentinel = object()

    async def fake_loop(m):
        started.append(m)

    monkeypatch.setattr(
        runtime, "build_assistant", lambda s, ps: type("A", (), {"identity": None})()
    )
    monkeypatch.setattr(runtime, "memory_loop", lambda a: asyncio.sleep(0))
    monkeypatch.setattr(runtime, "platform_monitor", lambda s, a: sentinel)
    monkeypatch.setattr(runtime, "monitor_loop", fake_loop)
    await runtime.run(object(), PlatformSettings(_env_file=None))
    assert started == [sentinel]
    started.clear()
    monkeypatch.setattr(runtime, "platform_monitor", lambda s, a: None)
    await runtime.run(object(), PlatformSettings(_env_file=None))
    assert started == []


def test_platform_change_hook_buffers_events_in_the_platform_emitter():
    assert runtime.change_events(None) is None  # the monitor is off: no hook at all
    docs = SQLiteDocStore(":memory:")
    em = Emitter(docs, enabled=True, now=lambda: NOW)
    outbox = ChangeOutbox(
        docs, lambda: NOW, on_change=runtime.change_events(PlatformMonitor(em, docs, dict))
    )
    org, before, after = outbox_org()
    (c,) = outbox.record(diff_members(before, after), org)
    assert outbox.mark_attempt(c.id, "slack-T1", "", org_id="o1")
    assert read_log(docs, "o1")["seq"] == 0  # buffered until the monitor tick flushes
    em.flush()
    events = seq_order(read_log(docs, "o1")["events"])
    assert [(e["kind"], e["id"], e["status"]) for e in events] == [
        ("change", c.id, "queued"),
        ("change", c.id, "waiting"),
    ]


async def test_run_gives_the_platform_outbox_the_monitor_hook(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    docs = SQLiteDocStore(":memory:")
    em = Emitter(docs, enabled=True)
    a = SimpleNamespace(identity=SimpleNamespace(orgs=OrgStore(docs)), team=None, changes=None)

    async def idle(*_):
        return None

    monkeypatch.setattr(runtime, "build_assistant", lambda s, ps: a)
    monkeypatch.setattr(runtime, "memory_loop", idle)
    monkeypatch.setattr(runtime, "monitor_loop", idle)
    monkeypatch.setattr(runtime.change_outbox_platform, "changes_loop", idle)
    monkeypatch.setattr(runtime, "platform_monitor", lambda s, a: PlatformMonitor(em, docs, dict))
    await runtime.run(Settings(_env_file=None), PlatformSettings(_env_file=None))
    org, before, after = outbox_org()
    a.changes.outbox.record(diff_members(before, after), org)
    em.flush()
    assert [e["kind"] for e in read_log(docs, "o1")["events"]] == ["change"]
    monkeypatch.setattr(runtime, "platform_monitor", lambda s, a: None)
    await runtime.run(Settings(_env_file=None), PlatformSettings(_env_file=None))
    assert a.changes.outbox.on_change is None  # the monitor is off: no hook
