# ruff: noqa: F811, F401
"""Workflow monitor on the bot tick and in the platform bridges: the emitter is flushed once per
tick, the heartbeat (with queue sizes) is written every 30 s, and nothing at all happens while
`SALID_MONITOR_ENABLED` is off. Offline: SQLite DocStore, fake Discord objects."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

from conftest import GUILD
from test_modes import bot, world  # noqa: F401  (fixtures)
from test_positions_bot import link

from salid import monitor
from salid.config import Settings
from salid.coo_store import SQLiteDocStore
from salid.monitor import HEARTBEAT, HEARTBEAT_EVERY_S, Audience, Emitter, read_log
from salid.orgs import OrgStore, Workspace, workspace_memory_id
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
    await bot._monitor_due()
    docs = SQLiteDocStore(":memory:")
    em = Emitter(docs, enabled=True)
    monkeypatch.setattr(em, "flush", on_thread(em.flush))
    await monitor_tick(PlatformMonitor(em, docs, dict))
    assert len(seen) == 4 and threading.get_ident() not in seen  # flush + heartbeat, twice


async def test_monitor_due_never_raises(bot, monkeypatch, caplog):
    bot.monitor.enabled = True

    def boom():
        raise RuntimeError("secret text")

    monkeypatch.setattr(bot.monitor, "flush", boom)
    await bot._monitor_due()  # logged, not raised
    assert "RuntimeError" in caplog.text and "secret text" not in caplog.text


async def test_tick_records_actions_and_ends_with_the_monitor(bot, monkeypatch):
    bot.monitor.enabled = True

    async def three():
        return 3

    monkeypatch.setattr(bot, "_actions_due", three)
    await bot.tick()
    assert bot.docs.get(HEARTBEAT, 0, "bot")["queues"]["actions"] == 3


async def test_change_outbox_run_notes_pending_targets(bot, monkeypatch):
    def target(platform, ws):
        return SimpleNamespace(platform=platform, workspace_id=ws, kind="notify", person="1")

    due = [
        (SimpleNamespace(org_id="o1"), target("discord", f"discord-{GUILD}")),
        (SimpleNamespace(org_id="o1"), target("discord", f"discord-{GUILD}")),
        (SimpleNamespace(org_id="o1"), target("discord", "discord-999")),  # not this bot's
        (SimpleNamespace(org_id="o1"), target("slack", "slack-T1")),
    ]
    bot.change_outbox = SimpleNamespace(prune=lambda: 0, due=lambda: due)

    async def nothing(*a, **k):
        return None

    monkeypatch.setattr(bot, "_change_notify", nothing)
    monkeypatch.setattr(bot, "_change_alerts", nothing)
    await bot._change_outbox_run()
    assert bot._monitor_queues["outbox_pending"] == 2


def test_memory_backlog_counts_pending_entries(bot):
    sched = bot.memory.scheduler
    assert sched.backlog() == 0
    sched.note(GUILD, 10, "2026-10-14")
    sched.note(GUILD, 10, "2026-10-14")  # same summary
    sched.mark_dirty(GUILD, 11, "2026-10-14")
    assert sched.backlog() == 2


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
