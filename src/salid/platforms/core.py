"""Platform-neutral Salid for chat platforms (Slack, LINE, Telegram, WhatsApp).

`Incoming` is one message as an adapter saw it; `Port` is what the adapter offers back (who is in
the room, how to send). `Assistant` does the rest, the same way the Discord bot does for a
channel message:

    dedupe -> tenant / room settings -> remember (memory log + recent buffer) -> decide whether
    Salid is addressed -> history + memory scope + files -> `qa.ask` in a worker thread -> reply

Scope (who sees what) is decided here in code, never by the model:
* ledger: the same `ask(guild_id, is_exec, user_id)` rule as Discord (non-executives see only
  their own receipts);
* memory: `compute_scope`. Memory of a room is readable only if the asker is in that room AND every
  reader of the answering room is too. Unknown audience, unknown membership or any error means
  "this room only" or nothing at all, never a guess.

Everything blocking runs in worker threads. Logs carry ids, counts and timings, never text.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
import traceback
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from types import SimpleNamespace
from typing import Protocol

from .. import gate
from ..extract import ingest_attachment
from ..fillers import RecentLines, pick_fallback
from ..gate import GateLine
from ..guild_settings import BotGuildSettings
from ..i18n import resolve_lang, t
from ..identity_links import (
    REPLY_ATTACHED,
    REPLY_INVALID,
    REPLY_LINKED,
    REPLY_ONE_CODE,
    REPLY_PRIVATE,
    REPLY_SLOW_DOWN,
    CodeError,
    IdentityLinks,
    code_candidates,
)
from ..memory import MemoryService, build_message, message_text
from ..memory_access import ChannelAccess, allowed_channel_ids
from ..memory_summary import render_summary
from ..message_control import is_thanks
from ..naming import mentions_name
from ..org_scope import OrgView
from ..org_stores import MultiGuildLedgerStore, MultiGuildMemoryStore
from ..qa import MAX_FILE_BYTES as MAX_FILE_BYTES
from ..qa import MAX_TOTAL_BYTES as MAX_TOTAL_BYTES
from ..qa import FileActions, FileInput, MemoryTools, ask, file_mime, select_files
from ..web_read import extract_urls
from . import ids
from .config import PlatformSettings

logger = logging.getLogger(__name__)

HISTORY_MESSAGES = 30
HISTORY_SECONDS = 3 * 3600
HISTORY_CHARS = 8000
HISTORY_LINE_CHARS = 500
LINK_ATTEMPTS = 5  # link-code attempts per platform user ...
LINK_ATTEMPTS_TENANT = 30  # ... and per tenant, in LINK_WINDOW_S (guessing protection)
LINK_WINDOW_S = 600.0
# re-verify a Slack user (still a full member, same email) this often; also the oldest verdict
# an org-wide widening may rest on
VERIFY_TTL_S = 300.0
FOLLOWUP_SECONDS = 120  # a bare "@Salid" picks up the same person's message from this recently
GATE_HISTORY = 10
GREETING_KEYS = ("greeting", "greeting_2", "greeting_3")
SEEN_KEEP = 5000
MEMORY_QUEUE_MAX = 20000
TRUNCATED = {
    "th": "\n… (ข้อความยาวเกินไป ตัดส่วนที่เหลือออกค่ะ)",
    "en": "\n… (the rest was cut: the answer was too long)",
}


def safe_exc(e: BaseException) -> str:
    """Type and the innermost frame (file:line), never the message: exception texts may quote
    user content or tokens."""
    tb = traceback.extract_tb(e.__traceback__)
    where = f"{tb[-1].filename.rsplit('/', 1)[-1]}:{tb[-1].lineno}" if tb else "?"
    return f"{type(e).__name__} at {where}"


# ------------------------------------------------------------------ data types


@dataclass
class FileRef:
    """A file attached to a message. `fetch` downloads it (blocking: called in a worker thread);
    the platform credentials live inside the closure."""

    name: str
    mime: str | None
    size: int
    fetch: Callable[[], bytes]


@dataclass(frozen=True)
class AccountState:
    """What a platform says about a person right now (see `SlackPort.account_state`)."""

    status: str  # member | revoked | unavailable
    email: str = ""


@dataclass
class Incoming:
    platform: str  # slack | line | telegram | whatsapp
    tenant: str  # Slack team_id / LINE bot user id / Telegram bot id / WhatsApp phone number id
    room: str  # Slack channel id / LINE groupId, roomId or userId (1:1) / Telegram chat id / WhatsApp wa_id
    user: str
    user_name: str
    text: str  # plain text, Salid's own @mention already removed
    message: str  # platform message id (Slack ts / LINE message id)
    at: datetime
    is_direct: bool = False  # a 1:1 chat with Salid
    mentions_bot: bool = False
    mentions_others: bool = False  # tags someone else (talking to them, not to Salid)
    thread: str | None = None  # Slack thread root ts
    replies_to_bot: bool = False  # Slack: the thread was started by Salid
    files: list[FileRef] = field(default_factory=list)
    room_name: str = ""
    reply_token: str = ""  # LINE
    raw_at: float = 0.0  # arrival time on the platform clock (LINE reply token age)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.platform, self.tenant, self.room, self.message)


@dataclass(frozen=True)
class HistLine:
    at: datetime
    who: str
    text: str
    mine: bool = False


@dataclass
class RoomInfo:
    key: str
    name: str
    members: frozenset[str]  # human member ids (platform ids)


@dataclass
class RoomView:
    """What the platform knows about the room an answer is posted to. `audience` None = the
    readers of the answering room are unknown: memory is then limited to that room alone."""

    audience: frozenset[str] | None
    rooms: list[RoomInfo] = field(default_factory=list)


class Port(Protocol):
    """The adapter's side: a few awaitable calls, faked in tests."""

    platform: str

    async def room_view(self, msg: Incoming) -> RoomView: ...

    async def is_admin(self, msg: Incoming) -> bool: ...

    def format(self, text: str) -> list[str]:
        """Model markdown -> messages that fit the platform."""
        ...

    async def deliver(self, msg: Incoming, texts: list[str], ack: object | None) -> list[str]:
        """Send `texts` as the answer; returns the sent message ids. `ack` is what `ack`
        returned (the first text replaces it where the platform can edit)."""
        ...

    async def ack(self, msg: Incoming, text: str) -> object | None:
        """A quick "one moment" (or the platform's typing indicator). May return None."""
        ...

    async def drop_ack(self, ack: object) -> None: ...

    async def history(self, msg: Incoming, limit: int) -> list[HistLine]:
        """Recent messages of the room / thread from the platform (oldest first); [] if it
        cannot be read."""
        ...


# ------------------------------------------------------------------ pure scope + history


def compute_scope(
    *,
    answering: int,
    answering_name: str,
    asker: int,
    audience: frozenset[int] | None,
    rooms: list[ChannelAccess],
) -> dict[int, str]:
    """Room id -> name the asker may get memory from (see `memory_access.allowed_channel_ids`).
    The answering room itself is always allowed. When its audience is unknown nothing else is."""
    if audience is None:
        return {answering: answering_name}
    access = [c for c in rooms if c.channel_id != answering]
    access.append(ChannelAccess(answering, answering_name, audience))
    allowed = allowed_channel_ids(access, asker, answering)
    return {c.channel_id: c.name for c in access if c.channel_id in allowed}


def render_history(lines: list[HistLine], now: datetime, zone) -> str:
    """Recent chat as `[HH:MM] who: text`, oldest first, bounded by age, count and size."""
    keep: list[str] = []
    for line in reversed(lines):
        if (now - line.at).total_seconds() > HISTORY_SECONDS:
            break
        if not line.text:
            continue
        at = line.at.astimezone(zone).strftime("%H:%M")
        who = "สลิด" if line.mine else line.who
        keep.append(f"[{at}] {who}: {line.text[:HISTORY_LINE_CHARS]}")
        if len(keep) >= HISTORY_MESSAGES:
            break
    keep.reverse()
    while keep and sum(len(x) + 1 for x in keep) > HISTORY_CHARS:
        keep.pop(0)
    return "\n".join(keep)


class RoomBuffer:
    """Recent lines per (tenant, room, thread), in memory only. A room is `primed` once the
    platform's own history was read (or there was nothing to read)."""

    def __init__(self, per_key: int = HISTORY_MESSAGES + 10, max_keys: int = 2000):
        self._per, self._max = per_key, max_keys
        self._d: OrderedDict[tuple, deque[tuple[str, HistLine]]] = OrderedDict()
        self.primed: set[tuple] = set()

    def _q(self, key: tuple) -> deque:
        q = self._d.get(key)
        if q is None:
            q = self._d[key] = deque(maxlen=self._per)
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            old, _ = self._d.popitem(last=False)
            self.primed.discard(old)
        return q

    def add(self, key: tuple, message_id: str, line: HistLine) -> None:
        q = self._q(key)
        if not any(mid == message_id for mid, _ in q):
            q.append((message_id, line))

    def edit(self, key: tuple, message_id: str, text: str) -> None:
        q = self._d.get(key)
        for i, (mid, line) in enumerate(q or ()):
            if mid == message_id:
                q[i] = (mid, HistLine(line.at, line.who, text, line.mine))

    def remove(self, room: tuple, message_id: str) -> None:
        """Forget a deleted message in every thread of a room (`room` = (tenant, room))."""
        for key, q in self._d.items():
            if key[:2] == room:
                kept = [item for item in q if item[0] != message_id]
                if len(kept) != len(q):
                    q.clear()
                    q.extend(kept)

    def lines(self, key: tuple, before: str | None = None) -> list[HistLine]:
        out: list[HistLine] = []
        for mid, line in self._d.get(key, ()):
            if before is not None and mid == before:
                break
            out.append(line)
        return out

    def prime(self, key: tuple, lines: list[HistLine]) -> None:
        """Merge platform history (older lines) with what the buffer already saw."""
        q = self._q(key)
        have = {(line.at, line.text) for _, line in q}
        merged = [(f"h{i}", x) for i, x in enumerate(lines) if (x.at, x.text) not in have]
        merged += list(q)
        merged.sort(key=lambda item: item[1].at)
        q.clear()
        q.extend(merged[-self._per :])
        self.primed.add(key)


# ------------------------------------------------------------------ the assistant


def tenant_label(platform: str, tenant: str) -> str:
    """The owner recorded in a tenant's settings document (its `guild_name`)."""
    return f"{platform.title()} {tenant}"


class Assistant:
    def __init__(
        self,
        *,
        settings,
        ps: PlatformSettings,
        store,
        llm,
        team: BotGuildSettings,
        memory: MemoryService,
        memory_store,
        embedder=None,
        archive=None,
        guilds_store=None,
        ask_fn=ask,
        gate_fn=gate.addressed_to_salid,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable] = asyncio.sleep,
        choose: Callable = random.choice,
        names: dict[tuple[int, int], str] | None = None,
        identity: IdentityLinks | None = None,
        org_sources=None,
    ) -> None:
        self.s, self.ps, self.store, self.llm, self.team = settings, ps, store, llm, team
        self.memory, self.memory_store, self.embedder = memory, memory_store, embedder
        self.archive, self.guilds_store = archive, guilds_store
        self.ask_fn, self.gate_fn = ask_fn, gate_fn
        self._clock, self._now, self._sleep, self._choose = clock, now, sleep, choose
        self.buffer = RoomBuffer()
        self.exec_ids = {
            "slack": ps.slack_exec_user_ids,
            "line": ps.line_exec_user_ids,
            "telegram": ps.telegram_exec_user_ids,
            "whatsapp": ps.whatsapp_exec_numbers,
        }
        self.sem = asyncio.Semaphore(max(ps.platform_max_concurrent, 1))
        # (tenant, room) -> name, for summaries
        self.names: dict[tuple[int, int], str] = names if names is not None else {}
        self.tenants: dict[int, tuple[str, str]] = {}  # tenant id -> (platform, tenant key)
        self._seen: OrderedDict[tuple, None] = OrderedDict()
        self._windows: dict[tuple[int, int, int], float] = {}  # (tenant, room, user) -> last reply
        self._threads: dict[tuple[int, int, str], float] = {}  # Salid answered in this thread
        self._answers: dict[tuple[int, int], deque[float]] = {}  # per room, loop protection
        self._registered: set[int] = set()
        self._recent_acks = RecentLines()
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
        self.on_room_seen: Callable[[Incoming], None] | None = None  # e.g. start a history import
        # verified identities / org workspaces (None: platforms run on their own, as before)
        self.identity = identity
        # platforms.org_sources.OrgSources: what a verified member may read across the org's
        # workspaces; None = every message keeps its own tenant's scope
        self.org_sources = org_sources
        self._link_tries: dict[tuple, deque[float]] = {}
        # (tenant, user) -> (checked at, still a verified member?) and the messages whose sender
        # could not be verified (identity_of answers None for them)
        self._verdicts: OrderedDict[tuple[str, str], tuple[float, bool]] = OrderedDict()
        self._denied: OrderedDict[tuple, None] = OrderedDict()

    # -------------------------------------------------------------- bookkeeping
    def _first_time(self, key: tuple) -> bool:
        if key in self._seen:
            return False
        self._seen[key] = None
        while len(self._seen) > SEEN_KEEP:
            self._seen.popitem(last=False)
        return True

    def spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def _enqueue(self, job) -> None:
        """One worker runs the blocking memory writes in order, off the event loop."""
        if self._queue is None:
            self._queue = asyncio.Queue(MEMORY_QUEUE_MAX)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_queue())
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.warning("platform memory queue full; dropped one record")

    async def _run_queue(self) -> None:
        queue = self._queue
        while True:
            job = await queue.get()
            try:
                await asyncio.to_thread(job)
            except Exception as e:
                logger.warning("platform memory write failed (%s)", safe_exc(e))
            finally:
                queue.task_done()

    async def drain(self) -> None:
        """Wait for queued memory writes and running answers (tests, shutdown)."""
        while self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)
        if self._queue is not None:
            await self._queue.join()

    def _owner_ok(self, msg: Incoming, gid: int) -> bool:
        """The tenant id is a 60-bit hash of the workspace / channel key. The settings document
        records which key owns it (`guild_name`); a different owner (a hash collision, or a Discord
        guild) is refused so two tenants never share a ledger or memory. Blocking."""
        expected = tenant_label(msg.platform, msg.tenant)
        known = self.tenants.get(gid)
        if known is not None:
            return known == (msg.platform, msg.tenant)
        if self.guilds_store is not None:
            try:
                doc = self.guilds_store.find(gid)
            except Exception as e:
                logger.warning("could not check tenant %s (%s)", gid, safe_exc(e))
                return False  # fail closed: unknown owner
            if doc is not None and doc.guild_name and doc.guild_name != expected:
                logger.error("tenant id %s already belongs to another tenant: refused", gid)
                return False
        self.tenants[gid] = (msg.platform, msg.tenant)
        return True

    def _register(self, msg: Incoming, gid: int) -> None:
        if gid in self._registered or self.guilds_store is None:
            return
        self._registered.add(gid)

        def run() -> None:
            try:
                self.guilds_store.upsert_guild(
                    gid,
                    name=tenant_label(msg.platform, msg.tenant),
                    icon_url="",
                    channels=[],
                    installed=True,
                )
            except Exception as e:
                logger.warning("could not register tenant %s (%s)", gid, safe_exc(e))

        self.spawn(asyncio.to_thread(run))

    # -------------------------------------------------------------- settings
    def _mode(self, gid: int, rid: int, name: str, direct: bool) -> str:
        if direct:
            return "chat"
        return self.team.channel_mode(gid, rid, name, set())

    def _language_setting(self, gid: int, rid: int) -> str:
        """th / en / auto. A tenant nobody configured speaks the language it is spoken to."""
        g = self.team.current(gid)
        if not g.updated_at and not g.channel_languages.get(str(rid)):
            return "auto"
        return self.team.channel_language(gid, rid)

    async def _is_exec(self, msg: Incoming, port: Port) -> bool:
        if msg.user in self.exec_ids.get(msg.platform, ()):
            return True
        try:
            return bool(await port.is_admin(msg))
        except Exception as e:
            logger.warning("admin lookup failed (%s)", safe_exc(e))
            return False

    # -------------------------------------------------------------- entry points
    async def handle(self, msg: Incoming, port: Port) -> None:
        """One incoming message. Never raises."""
        try:
            await self._handle(msg, port)
        except Exception as e:
            logger.error("platform handler failed (%s)", safe_exc(e))

    async def _handle(self, msg: Incoming, port: Port) -> None:
        if not self._first_time(msg.key):
            return
        if self.identity is not None and (codes := code_candidates(msg.text)):
            # a link / attach code, even inside a sentence: never buffered, remembered, logged,
            # gated or shown to the model (and never used in a group, where everyone reads it)
            await self._link_command(msg, port, codes)
            return
        gid = ids.tenant_id(msg.platform, msg.tenant)
        rid = ids.room_id(msg.platform, msg.tenant, msg.room)
        if not await asyncio.to_thread(self._owner_ok, msg, gid):
            return
        self.names[(gid, rid)] = msg.room_name or msg.room
        self._register(msg, gid)
        await asyncio.to_thread(self.team.current, gid)  # refresh the cache off the loop
        mode = self._mode(gid, rid, msg.room_name, msg.is_direct)
        if mode == "off":
            return
        bkey = self._bkey(msg, gid, rid)
        self.buffer.add(
            bkey,
            msg.message,
            HistLine(msg.at, msg.user_name or msg.user, msg.text or _files_line(msg)),
        )
        self._record(msg, gid, rid)
        if self.on_room_seen is not None:
            self.on_room_seen(msg)
        if not await self._addressed(msg, port, gid, rid, mode):
            return
        await self._verify_account(msg, port)
        await self._answer(msg, port, gid, rid)

    async def on_edit(self, msg: Incoming) -> None:
        """A message was edited: keep the memory text and the recent buffer current."""
        if self.identity is not None and code_candidates(msg.text):
            return  # an edit that adds a link code is as private as the original
        gid = ids.tenant_id(msg.platform, msg.tenant)
        rid = ids.room_id(msg.platform, msg.tenant, msg.room)
        self.buffer.edit(self._bkey(msg, gid, rid), msg.message, msg.text)
        mid = ids.message_id(msg.platform, msg.tenant, msg.room, msg.message)
        text = message_text(msg.text, _descriptors(msg.files))
        if text:
            self._enqueue(partial(self.memory.edit, gid, mid, text, self._now()))

    async def on_delete(self, platform: str, tenant: str, room: str, message: str) -> None:
        gid = ids.tenant_id(platform, tenant)
        mid = ids.message_id(platform, tenant, room, message)
        # a deleted / unsent message must not reach the next model call through the live buffer
        self.buffer.remove((gid, ids.room_id(platform, tenant, room)), message)
        self._enqueue(partial(self.memory.delete, gid, [mid]))

    # -------------------------------------------------------------- verified identity
    def identity_of(self, msg: Incoming) -> tuple[str, str] | None:
        """(org id, member email) when the sender is a VERIFIED member of the org that owns this
        tenant; None for a guest of the room (also when no org store is wired). Blocking."""
        if self.identity is None or msg.key in self._denied:
            return None
        try:
            return self.identity.identity_of(msg.platform, msg.tenant, msg.user)
        except Exception as e:
            logger.warning("identity lookup failed (%s)", safe_exc(e))
            return None

    def tenant_revoked(self, platform: str, tenant: str) -> None:
        """The platform workspace was uninstalled: forget the member links of that platform in
        its org (unless the org keeps another workspace there). Blocking, never raises."""
        if self.identity is None:
            return
        try:
            orgs = self.identity.orgs
            found = orgs.org_of_workspace(platform, tenant)
            if found:
                orgs.drop_platform_identities(found[0].id, platform, keep_workspace=found[1].id)
        except Exception as e:
            logger.warning("could not drop identities of %s tenant (%s)", platform, safe_exc(e))

    async def _org_view(self, msg: Incoming) -> OrgView | None:
        """What this sender may use org-wide: only in a 1:1 (its one reader is the asker) with a
        VERIFIED member of the org that owns the tenant. Anything else, and any error: None, so
        the message keeps today's single-tenant scope."""
        if self.org_sources is None or self.identity is None or not msg.is_direct:
            return None
        try:
            who = await asyncio.to_thread(self.identity_of, msg)
            if who is None:
                return None
            return await self.org_sources.view(
                who[0],
                who[1],
                asker=msg.user,
                is_direct=msg.is_direct,
                audience=frozenset({msg.user}) if msg.is_direct else None,
                asking=(msg.platform, msg.tenant),
            )
        except Exception as e:
            logger.warning("could not resolve the org scope (%s)", safe_exc(e))
            return None

    def _tries_ok(self, key: tuple, limit: int) -> bool:
        q = self._link_tries.setdefault(key, deque())
        now = self._clock()
        while q and now - q[0] > LINK_WINDOW_S:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        if len(self._link_tries) > 5000:  # stale keys of people who never came back
            self._link_tries = {k: v for k, v in self._link_tries.items() if v}
        return True

    async def _link_command(self, msg: Incoming, port: Port, codes: list[str]) -> None:
        """A message with a link / attach code: in a private chat with exactly one code, redeem
        it without the model and answer with a fixed line; with several codes, a fixed
        instruction. In a group the code is NOT used (everyone there can read it)."""
        if not msg.is_direct or len(codes) != 1:
            gid = ids.tenant_id(msg.platform, msg.tenant)
            if self._rate_ok(gid, ids.room_id(msg.platform, msg.tenant, msg.room)):
                await self._say(msg, port, REPLY_PRIVATE if not msg.is_direct else REPLY_ONE_CODE)
            return
        code = codes[0]
        if not (
            self._tries_ok(("u", msg.platform, msg.tenant, msg.user), LINK_ATTEMPTS)
            and self._tries_ok(("t", msg.platform, msg.tenant), LINK_ATTEMPTS_TENANT)
        ):
            await self._say(msg, port, REPLY_SLOW_DOWN)
            return
        try:
            done = await asyncio.to_thread(
                self.identity.redeem, code, msg.platform, msg.user, msg.tenant
            )
        except CodeError as e:
            logger.info("link refused platform=%s reason=%s", msg.platform, e.reason)
            await self._say(msg, port, REPLY_INVALID)
            return
        except Exception as e:
            logger.error("link failed platform=%s (%s)", msg.platform, safe_exc(e))
            await self._say(msg, port, REPLY_INVALID)
            return
        logger.info("link ok platform=%s kind=%s", msg.platform, done.kind)
        await self._say(msg, port, REPLY_ATTACHED if done.kind == "attach" else REPLY_LINKED)

    async def _verify_account(self, msg: Incoming, port: Port) -> None:
        """Slack: keep the verified identity of the sender in step with the workspace.

        The Slack profile email of a full member of THIS team that equals a member email of the
        org owning the workspace links the account (no code); a linked account that is now
        deleted, a guest, from another team or has another email loses its link. Re-checked at
        most every `VERIFY_TTL_S` (and at once on a `user_change` event). When Slack cannot be
        asked (error, missing scope) this request is treated as unverified, nothing is unlinked
        and the next message asks again."""
        key = (msg.tenant, msg.user)
        now = self._clock()
        hit = self._verdicts.get(key)
        if hit is not None and now - hit[0] < VERIFY_TTL_S:
            ok = hit[1]
        else:
            ok = await self._check_account(msg.platform, msg.tenant, msg.user, port)
            if ok is not None:
                self._verdicts[key] = (now, ok)
                self._verdicts.move_to_end(key)
                while len(self._verdicts) > SEEN_KEEP:
                    self._verdicts.popitem(last=False)
        if ok is None or ok is False:
            self._denied[msg.key] = None
            while len(self._denied) > SEEN_KEEP:
                self._denied.popitem(last=False)

    async def recheck_account(self, platform: str, tenant: str, user: str, port: Port) -> None:
        """The platform says the person changed: verify again now. Never raises."""
        self._verdicts.pop((tenant, user), None)
        try:
            ok = await self._check_account(platform, tenant, user, port)
        except Exception as e:  # pragma: no cover - _check_account already guards
            logger.warning("could not re-verify the account (%s)", safe_exc(e))
            return
        if ok is not None:
            self._verdicts[(tenant, user)] = (self._clock(), ok)

    async def _check_account(
        self, platform: str, tenant: str, user: str, port: Port
    ) -> bool | None:
        """True: a verified-or-verifiable member (or nothing to verify). False: revoked, the
        link was removed. None: Slack could not tell, fail closed without touching any link."""
        state_of = getattr(port, "account_state", None)
        if self.identity is None or state_of is None or platform != "slack":
            return True
        try:
            if not await asyncio.to_thread(self.identity.orgs.org_of_workspace, platform, tenant):
                return True
            state = await state_of(user)
            if state.status == "unavailable":
                return None
            await asyncio.to_thread(
                self.identity.reconcile,
                platform,
                tenant,
                user,
                state.status == "member",
                state.email,
            )
            return state.status == "member"
        except Exception as e:
            logger.warning("could not verify the account (%s)", safe_exc(e))
            return None

    # -------------------------------------------------------------- addressing
    @staticmethod
    def _bkey(msg: Incoming, gid: int, rid: int) -> tuple:
        return (gid, rid, msg.thread)

    async def _addressed(self, msg: Incoming, port: Port, gid: int, rid: int, mode: str) -> bool:
        """Is this message for Salid? DM / 1:1 always; chat mode always; a tag, the name, a reply
        in a thread Salid started (or answered in) in call/join mode; listen mode only a tag; a
        follow-up of the person Salid just answered when the gate model agrees. Else: record only."""
        if msg.is_direct or mode == "chat":
            return True
        if mode == "listen":
            return msg.mentions_bot
        if msg.mentions_bot or (msg.text and mentions_name(msg.text)):
            return True
        if msg.thread and (msg.replies_to_bot or self._thread_open(gid, rid, msg.thread)):
            return True
        if msg.mentions_others:
            return False
        if not (msg.text or msg.files) or (not msg.files and is_thanks(msg.text)):
            return False
        opened = self._windows.get((gid, rid, ids.user_id(msg.platform, msg.user)))
        if opened is None or self._clock() - opened >= self.ps.platform_window_seconds:
            return False
        lines = self.buffer.lines(self._bkey(msg, gid, rid), before=msg.message)[-GATE_HISTORY:]
        history = [
            GateLine(str(i), ln.who if not ln.mine else "สลิด", ln.text)
            for i, ln in enumerate(lines)
        ]
        line = GateLine(msg.message, msg.user_name or msg.user, msg.text or _files_line(msg))
        return bool(
            await asyncio.to_thread(
                self.gate_fn, self.llm, self.s.gate_model, history, line, channel_id=rid
            )
        )

    def _thread_open(self, gid: int, rid: int, thread: str) -> bool:
        at = self._threads.get((gid, rid, thread))
        return at is not None and self._clock() - at < self.ps.platform_thread_seconds

    def _rate_ok(self, gid: int, rid: int) -> bool:
        q = self._answers.setdefault((gid, rid), deque())
        now = self._clock()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.ps.platform_answers_per_minute:
            return False
        q.append(now)
        return True

    # -------------------------------------------------------------- memory
    def _record(
        self,
        msg: Incoming,
        gid: int,
        rid: int,
        *,
        text: str | None = None,
        mine: bool = False,
        key: str | None = None,
    ) -> None:
        if not self.team.enabled(gid, "memory"):
            return
        body = text if text is not None else message_text(msg.text, _descriptors(msg.files))
        if not body:
            return
        thread = (
            ids.room_id(msg.platform, msg.tenant, f"thread:{msg.thread}") if msg.thread else None
        )
        rec = build_message(
            self.s,
            guild_id=gid,
            channel_id=rid,
            thread_id=thread,
            message_id=ids.message_id(msg.platform, msg.tenant, msg.room, key or msg.message),
            author_id=ids.bot_user_id(msg.platform, msg.tenant)
            if mine
            else ids.user_id(msg.platform, msg.user),
            author_name="Salid" if mine else (msg.user_name or msg.user),
            created_at=self._now() if mine else msg.at,
            text=body,
            kind="salid" if mine else "chat",
        )
        self._enqueue(partial(self.memory.record, rec))

    async def _memory_tools(
        self, msg: Incoming, port: Port, gid: int, rid: int, wide: OrgView | None = None
    ) -> MemoryTools | None:
        """Memory tools scoped to what the asker AND everyone reading the answering room may
        see. Any failure -> no memory tools at all."""
        if not self.team.enabled(gid, "memory"):
            return None
        try:
            view = await port.room_view(msg)
            asker = ids.user_id(msg.platform, msg.user)
            rooms = []
            for r in view.rooms:
                r_id = ids.room_id(msg.platform, msg.tenant, r.key)
                if self.team.channel_mode(gid, r_id, r.name, set()) == "off":
                    continue
                rooms.append(
                    ChannelAccess(
                        r_id, r.name, frozenset(ids.user_id(msg.platform, m) for m in r.members)
                    )
                )
            audience = (
                None
                if view.audience is None
                else frozenset(ids.user_id(msg.platform, m) for m in view.audience)
            )
            allowed = compute_scope(
                answering=rid,
                answering_name=msg.room_name or msg.room,
                asker=asker,
                audience=audience,
                rooms=rooms,
            )
        except Exception as e:
            logger.warning("could not compute memory scope (%s)", safe_exc(e))
            return None
        store, guilds = self.memory_store, [gid]
        if wide is not None:
            try:
                store, allowed, guilds = self._widen_memory(wide, gid, allowed)
            except Exception as e:  # fail closed: the asking tenant's scope, unchanged
                logger.warning("could not widen the memory scope (%s)", safe_exc(e))
        return MemoryTools(
            store,
            gid,
            allowed,
            self.s,
            min(self.team.memory_retention_days(g) for g in guilds),
            opener=None,
            visibility=None,
            embedder=self.embedder,
            retention_of=lambda: min(
                self.team.fresh_memory_retention_days(g, fail_closed=True) for g in guilds
            ),
        )

    def _widen_memory(self, wide: OrgView, gid: int, allowed: dict[int, str]):
        """The asker's own tenant scope plus every channel of the org's workspaces they may read
        (a workspace with memory switched off or a channel in mode off adds nothing).
        Returns (store, channel scope, guilds involved)."""
        routes: dict[int, int] = {c: gid for c in allowed}
        scope = dict(allowed)
        guilds = [gid]
        for src in wide.sources:
            if not self.team.enabled(src.guild_id, "memory"):
                continue
            added = False
            for cid, label in src.channels.items():
                if cid in scope:
                    continue  # the tenant's own scope already names it
                if (
                    cid not in src.direct
                    and self.team.channel_mode(src.guild_id, cid, label, set()) == "off"
                ):
                    continue
                scope[cid] = label
                routes[cid] = src.guild_id
                added = True
            if added and src.guild_id not in guilds:
                guilds.append(src.guild_id)
        if len(guilds) == 1:
            return self.memory_store, scope, guilds
        return MultiGuildMemoryStore(self.memory_store, routes, gid, self.embedder), scope, guilds

    async def _today_summary(self, gid: int, rid: int) -> str:
        if not self.team.enabled(gid, "memory"):
            return ""
        try:
            day = datetime.now(self.s.zone).date().isoformat()
            got = await asyncio.to_thread(self.memory_store.get_summary, gid, rid, day)
            if got is None or got.dirty or not got.summary:
                return ""
            return render_summary(got.summary, max_chars=1500)
        except Exception as e:
            logger.warning("could not read today's summary (%s)", safe_exc(e))
            return ""

    # -------------------------------------------------------------- history
    async def _history(self, msg: Incoming, port: Port, gid: int, rid: int) -> str:
        key = self._bkey(msg, gid, rid)
        if key not in self.buffer.primed:
            try:
                got = await port.history(msg, HISTORY_MESSAGES)
            except Exception as e:
                logger.warning("could not read platform history (%s)", safe_exc(e))
                got = []
            self.buffer.prime(key, [g for g in got if g.at < msg.at])
        lines = self.buffer.lines(key, before=msg.message)
        parts = []
        summary = await self._today_summary(gid, rid)
        if summary:
            parts.append(f"Summary of this room today (precedes the chat below):\n{summary}")
        chat = render_history(lines, msg.at, self.s.zone)
        if chat:
            parts.append(chat)
        return "\n\n".join(parts)

    def _previous_text(self, msg: Incoming, gid: int, rid: int) -> str:
        """The author's own message just before a bare @mention ("Hi" then "@Salid")."""
        for line in reversed(self.buffer.lines(self._bkey(msg, gid, rid), before=msg.message)):
            if line.who == (msg.user_name or msg.user) and not line.mine:
                if (msg.at - line.at).total_seconds() <= FOLLOWUP_SECONDS and line.text:
                    return line.text
                return ""
        return ""

    # -------------------------------------------------------------- files
    async def _file_inputs(self, msg: Incoming) -> list[FileInput]:
        inputs = [
            FileInput(i, f.name, file_mime(f.name, f.mime), size=int(f.size or 0))
            for i, f in enumerate(msg.files)
        ]
        usable, _skipped = select_files(inputs)
        actual_total = 0
        for fi in usable:
            try:
                data = await asyncio.to_thread(msg.files[fi.index].fetch)
            except Exception as e:
                logger.warning("attachment download failed (%s)", safe_exc(e))
                fi.skip = "could not be downloaded"
                continue
            if len(data) > MAX_FILE_BYTES:
                fi.skip = "over the size limit"
                continue
            if actual_total + len(data) > MAX_TOTAL_BYTES:
                fi.skip = f"total size over {MAX_TOTAL_BYTES // 1024 // 1024} MB"
                continue
            fi.data, fi.size = data, len(data)
            actual_total += len(data)
        return inputs

    def _file_actions(self, msg: Incoming, gid: int, rid: int, lang: str) -> FileActions:
        """The side effects the agent may trigger for this message. Tenant, submitter, room and
        message are closed over here; the model only chooses which attachment."""
        uid = ids.user_id(msg.platform, msg.user)
        mid = ids.message_id(msg.platform, msg.tenant, msg.room, msg.message)

        def record(f: FileInput, note: str):
            return ingest_attachment(
                self.store,
                self.llm,
                self.s,
                guild_id=gid,
                data=f.data,
                filename=f.filename,
                content_type=f.mime,
                submitter_id=uid,
                submitter_name=msg.user_name or msg.user,
                channel_id=rid,
                message_id=mid,
                note=" ".join(x for x in ((msg.text or "").strip(), note.strip()) if x),
                archive=self.archive,
                lang=lang,
            )

        return FileActions(record_receipt=record if self.team.enabled(gid, "receipts") else None)

    # -------------------------------------------------------------- the answer
    def _web_kw(self, gid: int) -> dict:
        if self.llm is not None and self.team.enabled(gid, "web"):
            return {"web_client": self.llm}
        return {}

    async def _say(self, msg: Incoming, port: Port, text: str) -> list[str]:
        sent = await port.deliver(msg, port.format(text), None)
        return sent

    async def _answer(self, msg: Incoming, port: Port, gid: int, rid: int) -> None:
        t0 = self._clock()
        uid = ids.user_id(msg.platform, msg.user)
        setting = self._language_setting(gid, rid)
        q = (msg.text or "").strip()
        if not q and not msg.files:
            q = self._previous_text(msg, gid, rid)
        if not q and not msg.files:
            lang = resolve_lang(setting, "")
            await self._reply(msg, port, gid, rid, t(self._choose(GREETING_KEYS), lang), cap=lang)
            return
        lang = resolve_lang(setting, q)
        if not self.team.enabled(gid, "qa"):
            if msg.is_direct or msg.mentions_bot:
                await self._say(msg, port, t("qa_off", lang))
            return
        if not self._rate_ok(gid, rid):
            logger.warning("answer rate limit tenant=%s room=%s", gid, rid)
            return
        is_exec = await self._is_exec(msg, port)
        wide = await self._org_view(msg)
        store = self.store
        if wide is not None:
            # an org owner / admin counts as an executive for the ledger across the org
            is_exec = is_exec or wide.is_admin
            pairs = wide.ledger_sources(is_exec)
            if not any(g == gid for g, _ in pairs):
                pairs.append((gid, None if is_exec else uid))
            store = MultiGuildLedgerStore(self.store, pairs)
        history = await self._history(msg, port, gid, rid)
        memory = await self._memory_tools(msg, port, gid, rid, wide)
        kwargs: dict = {"language": setting, "memory": memory, **self._web_kw(gid)}
        if msg.files:
            files = await self._file_inputs(msg)
            kwargs.update(files=files, actions=self._file_actions(msg, gid, rid, lang))
            lang = resolve_lang(setting, q)
        t_ctx = self._clock()
        slow = _Slow(self, msg, port, q, lang, rid)
        slow.start()
        try:
            async with self.sem:
                ans = await asyncio.to_thread(
                    self.ask_fn,
                    self.s,
                    store,
                    q,
                    guild_id=gid,
                    is_exec=is_exec,
                    user_id=uid,
                    history=history,
                    **kwargs,
                )
        except Exception as e:
            ack = await slow.finish()
            logger.error("ask failed tenant=%s room=%s (%s)", gid, rid, safe_exc(e))
            await self._reply(msg, port, gid, rid, t("qa_failed", lang), ack=ack, remember=False)
            return
        ack = await slow.finish()
        logger.info(
            "answer platform=%s tenant=%s room=%s exec=%s ms_total=%d ms_context=%d tools=%s",
            msg.platform, gid, rid, is_exec,
            (self._clock() - t0) * 1000, (t_ctx - t0) * 1000,
            ",".join(name for name, _ in ans.tool_calls),
        )  # fmt: skip
        await self._reply(msg, port, gid, rid, ans.text, ack=ack, cap=lang)

    async def _reply(
        self,
        msg: Incoming,
        port: Port,
        gid: int,
        rid: int,
        text: str,
        *,
        ack: object | None = None,
        remember: bool = True,
        cap: str = "th",
    ) -> None:
        if getattr(port, "revoked", False):
            return  # the tenant was removed while this answer was in flight: nothing to say or keep
        parts = port.format(text)
        limit = getattr(port, "max_messages", 10)
        if len(parts) > limit:
            parts = [*parts[: limit - 1], parts[limit - 1] + t_trunc(cap)]
        try:
            sent = await port.deliver(msg, parts, ack)
        except Exception as e:
            logger.error("deliver failed tenant=%s room=%s (%s)", gid, rid, safe_exc(e))
            return
        if not remember or getattr(port, "revoked", False):
            return
        now = self._clock()
        self._windows[(gid, rid, ids.user_id(msg.platform, msg.user))] = now
        self._windows = {k: v for k, v in self._windows.items() if now - v < 3600}
        thread = msg.thread or (
            msg.message if msg.platform == "slack" and not msg.is_direct else None
        )
        if thread:
            self._threads[(gid, rid, thread)] = now
        bkey = (gid, rid, msg.thread)
        for i, (chunk, key) in enumerate(zip(parts, [*sent, *[""] * len(parts)], strict=False)):
            key = key or f"salid:{msg.message}:{i}"
            self.buffer.add(bkey, key, HistLine(self._now(), "Salid", chunk, mine=True))
            self._record(msg, gid, rid, text=chunk, mine=True, key=key)


def t_trunc(lang: str) -> str:
    return TRUNCATED["en" if lang == "en" else "th"]


class _Slow:
    """The "one moment" of a slow answer: posted only if no answer is back after
    `platform_slow_after_seconds`, and handed to `deliver` (which replaces it)."""

    def __init__(self, a: Assistant, msg: Incoming, port: Port, q: str, lang: str, rid: int):
        self.a, self.msg, self.port, self.q, self.lang, self.rid = a, msg, port, q, lang, rid
        self.ack: object | None = None
        self._task: asyncio.Task | None = None
        self._posting = False

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def _kind(self) -> str:
        if self.msg.files:
            return (
                "image"
                if any((f.mime or "").startswith("image/") for f in self.msg.files)
                else "doc"
            )
        return "link" if extract_urls(self.q) else "slow"

    async def _run(self) -> None:
        await self.a._sleep(self.a.ps.platform_slow_after_seconds)
        self._posting = True
        key = (self.msg.platform, self.rid)
        line = pick_fallback(
            self._kind(),
            self.lang,
            self.a._recent_acks.get(key),
            str(getattr(self.a.s, "persona_particle", "ค่ะ") or "ค่ะ"),
            self.a._choose,
        )
        try:
            self.ack = await self.port.ack(self.msg, line)
            self.a._recent_acks.add(key, line)
        except Exception as e:
            logger.warning("ack failed (%s)", safe_exc(e))

    async def finish(self) -> object | None:
        task = self._task
        if task is None:
            return None
        if not self._posting and not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(task, return_exceptions=True)
        return self.ack


# ------------------------------------------------------------------ small helpers


def _descriptors(files: list[FileRef]) -> list:
    return [SimpleNamespace(filename=f.name, content_type=f.mime or "") for f in files]


def _files_line(msg: Incoming) -> str:
    return " ".join(f"[{f.name}]" for f in msg.files)
