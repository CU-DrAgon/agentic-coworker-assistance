"""Fakes for the chat-platform tests: no network, no model, no real clock."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from salid.guild_settings import BotGuildSettings, MemoryGuildStore
from salid.memory import MemoryScheduler, MemoryService
from salid.memory_store import SQLiteMemoryStore
from salid.platforms import ids
from salid.platforms.config import PlatformSettings
from salid.platforms.core import Assistant, HistLine, Incoming, RoomView
from salid.qa import Answer
from salid.store import SQLiteStore

NOW = datetime(2026, 10, 14, 5, 0, tzinfo=UTC)  # 12:00 in Bangkok (the settings fixture's clock)


def inc(**kw) -> Incoming:
    base = dict(
        platform="slack",
        tenant="T1",
        room="C1",
        user="U1",
        user_name="Alice",
        text="hello",
        message="100.001",
        at=NOW,
        room_name="general",
    )
    base.update(kw)
    return Incoming(**base)


class FakePort:
    platform = "slack"
    max_messages = 10

    def __init__(self, view: RoomView | None = None, admin: bool = False, platform="slack"):
        self.platform = platform
        self.view = view if view is not None else RoomView(None, [])
        self.admin = admin
        self.delivered: list[tuple] = []
        self.acks: list[str] = []
        self.dropped: list = []
        self.hist: list[HistLine] = []
        self.view_error: Exception | None = None

    async def room_view(self, msg):
        if self.view_error:
            raise self.view_error
        return self.view

    async def is_admin(self, msg):
        return self.admin

    def format(self, text):
        return [text]

    async def deliver(self, msg, texts, ack):
        self.delivered.append((msg, list(texts), ack))
        return [f"s{len(self.delivered)}.{i}" for i in range(len(texts))]

    async def ack(self, msg, text):
        self.acks.append(text)
        return ("ack", len(self.acks))

    async def drop_ack(self, ack):
        self.dropped.append(ack)

    async def history(self, msg, limit):
        return list(self.hist)

    @property
    def texts(self) -> list[str]:
        return [t for _m, texts, _a in self.delivered for t in texts]


class FakeAsk:
    """Stands in for qa.ask: records the call, returns a fixed answer (optionally blocking)."""

    def __init__(self, text="ok", release: threading.Event | None = None, fail=False):
        self.text, self.release, self.fail = text, release, fail
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, settings, store, question, **kw):
        self.calls.append((question, kw))
        if self.release is not None:
            self.release.wait(5)
        if self.fail:
            raise RuntimeError("secret model failure with user text")
        return Answer(self.text, 0, 0.0, [("spending_summary", {})])


async def _never(_seconds):
    """The slow-answer timer that never fires (the tests that want it pass `immediately`)."""
    await asyncio.Event().wait()


async def immediately(_seconds):
    await asyncio.sleep(0)


def make_assistant(
    settings, tmp_path, *, ps=None, ask=None, gate=None, sleep=None, clock=None, **ps_kw
):
    ps = ps or PlatformSettings(_env_file=None, **ps_kw)
    store = SQLiteStore(str(tmp_path / "ledger.db"))
    team = BotGuildSettings(MemoryGuildStore(collection="g"))
    mstore = SQLiteMemoryStore(str(tmp_path / "ledger.db"), settings.zone)
    scheduler = MemoryScheduler(mstore, settings, None, now=lambda: NOW)
    gate_calls: list = []

    def default_gate(llm, model, history, line, *, channel_id):
        gate_calls.append((history, line))
        return True

    a = Assistant(
        settings=settings,
        ps=ps,
        store=store,
        llm=None,
        team=team,
        memory=MemoryService(mstore, settings, scheduler),
        memory_store=mstore,
        ask_fn=ask or FakeAsk(),
        gate_fn=gate or default_gate,
        now=lambda: NOW,
        sleep=sleep or _never,
        choose=lambda xs: xs[0],
        **({"clock": clock} if clock else {}),
    )
    a.gate_calls = gate_calls
    return a


def gid_of(msg: Incoming) -> int:
    return ids.tenant_id(msg.platform, msg.tenant)


def rid_of(msg: Incoming) -> int:
    return ids.room_id(msg.platform, msg.tenant, msg.room)


def stored(a: Assistant, msg: Incoming, room: str | None = None) -> list:
    """Messages in the memory store for a room of the message's tenant."""
    rid = ids.room_id(msg.platform, msg.tenant, room or msg.room)
    day = (NOW + timedelta(hours=7)).date().isoformat()
    return a.memory_store.messages(gid_of(msg), [rid], day, day, limit=100, newest_first=False)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def ns(**kw):
    return SimpleNamespace(**kw)
