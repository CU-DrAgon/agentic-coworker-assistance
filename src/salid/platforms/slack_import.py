"""Optional Slack history import for the team memory.

Reuses `salid.memory_import.MemoryImporter` (resumable per-channel state, idempotent writes by
message id, retention window, per-run cap, throttled pages, past-day summaries). Slack's history
call returns the newest messages first while the importer wants oldest-first pages, so
`SlackHistory` reads the channel in time *cells* (a week at a time, oldest cell first) and serves
the importer's pages from them. Every page the importer stores is persisted in the channel state
before the next cell is read, so a restart, a 429 or a crash never loses stored work: the next run
resumes at the cursor.

Slack throttles `conversations.history`/`replies` per workspace and method. `Throttle` paces both,
honours `Retry-After` (backing off instead of giving up) and detects the restricted tier that new
unlisted commercial apps get since 2025-05-29 (1 request/minute, 15 messages per page): it then
imports slowly (and says so once in the log) while answers are not affected, because the live
history read uses its own client and gives up after a few seconds.

Limit: a thread whose parent is older than the resume cursor does not get its later replies from a
restarted import (the live events record them as they happen). Only conversations Salid is a
member of are read (that is all Slack allows), 1:1 chats with people are not. Counts only in the
logs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ..i18n import resolve_lang
from ..memory import build_message, message_text
from ..memory_import import MemoryImporter, Target
from . import ids, textfmt
from .core import Assistant, FileRef, _descriptors, safe_exc
from .slack import ACCEPTED_SUBTYPES, PLATFORM, SlackPort, _ts, retry_after

logger = logging.getLogger(__name__)

MAX_CELL = 20_000  # messages of one cell; a bigger cell is split in halves
MAX_THREADS = 200  # threads per cell whose replies are read; more splits the cell
PAGE = 200
RESTRICTED_PAGE = 15  # the page size of the restricted tier
RESTRICTED_INTERVAL_S = 61.0  # ... and its rate: one request per minute
LONG_RETRY_S = 30.0  # a Retry-After this long means the restricted tier
MAX_429 = 30  # consecutive 429s before the run gives up (it resumes on the next run)
MAX_WAIT_S = 300.0
CELL = timedelta(days=7)
MIN_CELL = timedelta(hours=1)


class Throttled(Exception):
    """Slack kept answering 429; the run stops here and resumes later."""


class _Split(Exception):
    """A cell holds too much: read it again in smaller cells."""


class Throttle:
    """Paces `conversations.history` and `conversations.replies` (separate buckets, shared by all
    channels of the workspace) and turns a 429 into a wait for its Retry-After."""

    def __init__(self, sleep=asyncio.sleep, clock=time.monotonic, interval: float = 1.0) -> None:
        self._sleep, self._clock, self.base = sleep, clock, interval
        self.interval = interval
        self.restricted = False
        self._last: dict[str, float] = {}
        self._long_429 = 0

    @property
    def page(self) -> int:
        return RESTRICTED_PAGE if self.restricted else PAGE

    def mark_restricted(self, why: str) -> None:
        if not self.restricted:
            self.restricted = True
            self.interval = max(self.base, RESTRICTED_INTERVAL_S)
            logger.warning(
                "slack history is rate limited to 1 request/minute and 15 messages per page "
                "(%s): the import continues slowly, answers are not affected",
                why,
            )

    def saw_page(self, requested: int, got: int, more: bool) -> None:
        # the restricted tier clamps every page to exactly 15 even when more rows exist
        if more and requested > RESTRICTED_PAGE and got == RESTRICTED_PAGE:
            self.mark_restricted("short pages")

    async def call(self, bucket: str, fn, **kw):
        waits = 0
        while True:
            last = self._last.get(bucket)
            if last is not None:
                gap = self.interval - (self._clock() - last)
                if gap > 0:
                    await self._sleep(gap)
            try:
                resp = await fn(**kw)
            except Exception as e:
                self._last[bucket] = self._clock()
                wait = retry_after(e)
                if wait is None:
                    raise
                waits += 1
                if wait >= LONG_RETRY_S:
                    self._long_429 += 1
                    if self._long_429 >= 2:
                        self.mark_restricted("repeated 429")
                if waits > MAX_429:
                    raise Throttled() from e
                logger.info("slack import: 429, waiting %.0fs", wait)
                await self._sleep(min(max(wait, 1.0), MAX_WAIT_S))
                continue
            self._last[bucket] = self._clock()
            return resp


class SlackHistory:
    """One channel's history in the shape `MemoryImporter` expects (`history(after=, before=,
    limit=, oldest_first=)`, an async iterator of objects with `created_at`)."""

    def __init__(self, port: SlackPort, channel: str, throttle: Throttle, now, on_error=None):
        self.port, self.channel, self.throttle, self._now = port, channel, throttle, now
        self._on_error = on_error
        self._width = CELL
        self._cells: dict[int, list[SimpleNamespace]] = {}
        self._pending: dict[str, SimpleNamespace] = {}  # replies read ahead of their cell

    # ------------------------------------------------------------ cells
    def _bounds(self, k: int) -> tuple[datetime, datetime]:
        w = self._width.total_seconds()
        return datetime.fromtimestamp(k * w, UTC), datetime.fromtimestamp((k + 1) * w, UTC)

    def _index(self, t: datetime) -> int:
        return int(t.timestamp() // self._width.total_seconds())

    async def _row(self, m: dict) -> SimpleNamespace | None:
        ts = m.get("ts")
        if not ts or m.get("subtype") not in ACCEPTED_SUBTYPES or m.get("bot_id"):
            return None
        if not m.get("user"):
            return None
        return SimpleNamespace(
            created_at=_ts(ts),
            ts=ts,
            user=m["user"],
            name=await self.port.user_name(m["user"]),
            text=textfmt.slack_to_plain(m.get("text") or "", self.port.bot_user_id),
            thread_ts=m.get("thread_ts") if m.get("thread_ts") != ts else None,
            files=[
                FileRef(f.get("name") or "file", f.get("mimetype"), 0, lambda: b"")
                for f in (m.get("files") or [])
            ],
        )

    async def _read_cell(self, k: int, after: datetime) -> list[SimpleNamespace]:
        lo, hi = self._bounds(k)
        client = self.port.import_client
        raw: list[dict] = []
        cursor = ""
        while True:
            limit = self.throttle.page
            resp = await self.throttle.call(
                "history",
                client.conversations_history,
                channel=self.channel,
                oldest=f"{lo.timestamp():.6f}",
                latest=f"{hi.timestamp():.6f}",
                inclusive=True,
                limit=limit,
                **({"cursor": cursor} if cursor else {}),
            )
            got = resp.get("messages") or []
            cursor = ((resp.get("response_metadata") or {}).get("next_cursor")) or ""
            self.throttle.saw_page(limit, len(got), bool(cursor))
            raw.extend(got)
            if len(raw) > MAX_CELL and self._width > MIN_CELL:
                raise _Split()
            if not cursor:
                break
        raw = [m for m in raw if lo <= _ts(m.get("ts")) < hi]
        threads = [
            m["ts"]
            for m in raw
            if (m.get("reply_count") or 0) > 0 and m.get("ts") and _ts(m["ts"]) > after
        ]
        if len(threads) > MAX_THREADS and self._width > MIN_CELL:
            raise _Split()
        rows: dict[str, SimpleNamespace] = {}
        for m in raw:
            r = await self._row(m)
            if r is not None:
                rows[r.ts] = r
        for ts in threads:  # every page of every thread, or the cell is not imported
            tcursor = ""
            while True:
                limit = self.throttle.page
                resp = await self.throttle.call(
                    "replies",
                    client.conversations_replies,
                    channel=self.channel,
                    ts=ts,
                    limit=limit,
                    **({"cursor": tcursor} if tcursor else {}),
                )
                tcursor = ((resp.get("response_metadata") or {}).get("next_cursor")) or ""
                self.throttle.saw_page(limit, len(resp.get("messages") or []), bool(tcursor))
                for m in resp.get("messages") or []:
                    if m.get("ts") == ts:
                        continue
                    r = await self._row(m)
                    if r is None:
                        continue
                    if r.created_at < hi:
                        rows[r.ts] = r
                    else:
                        self._pending[r.ts] = r  # belongs to a later cell of this run
                if not tcursor:
                    break
        for ts, r in list(self._pending.items()):
            if lo <= r.created_at < hi:
                rows[ts] = r
                del self._pending[ts]
        return sorted(rows.values(), key=lambda r: r.created_at)

    async def _cell(self, t: datetime, after: datetime) -> tuple[list[SimpleNamespace], datetime]:
        """The rows of the cell holding `t` and where that cell ends."""
        while True:
            k = self._index(t)
            if k in self._cells:
                return self._cells[k], self._bounds(k)[1]
            for old in [c for c in self._cells if c < k]:
                del self._cells[old]
            try:
                self._cells[k] = await self._read_cell(k, after)
            except _Split:
                # smaller cells (the grid is re-aligned, so what was read is dropped)
                self._width = max(self._width / 2, MIN_CELL)
                self._cells.clear()
                continue
            return self._cells[k], self._bounds(k)[1]

    # ------------------------------------------------------------ the importer's view
    async def history(self, *, limit, after, oldest_first=True, before=None):
        end = before or self._now()
        out: list[SimpleNamespace] = []
        t = after
        try:
            while len(out) < limit and t < end:
                rows, t = await self._cell(t, after)
                out.extend(
                    r
                    for r in rows
                    if r.created_at > after and (before is None or r.created_at < before)
                )
        except Exception:
            if self._on_error is not None:
                self._on_error(self.channel)
            raise
        for r in out[:limit]:
            yield r


class SlackImporter:
    def __init__(
        self,
        assistant: Assistant,
        port: SlackPort,
        *,
        sleep=asyncio.sleep,
        page_sleep: float = 1.3,
        clock=time.monotonic,
    ) -> None:
        self.a, self.port = assistant, port
        self._sleep, self._page_sleep = sleep, page_sleep
        self.throttle = Throttle(sleep, clock, page_sleep)
        self.capped = False
        self._failed: set[str] = set()
        self._built = 0
        self.importer = MemoryImporter(
            assistant.memory_store,
            assistant.s,
            assistant.memory,
            assistant.llm,
            build_record=self._record,
            retention_of=assistant.team.memory_retention_days,
            lang_of=lambda gid: resolve_lang(assistant.team.current(gid).default_language),
            channel_name_of=lambda gid, cid: assistant.names.get((gid, cid), str(cid)),
            now=assistant.s.now,
            page_sleep=0.0,  # the Throttle paces the Slack calls
        )

    def _record(self, m, tg: Target):
        """The memory record of an imported Slack message (None: nothing to remember)."""
        text = message_text(m.text, _descriptors(m.files))
        if not text:
            return None
        team = self.port.team_id
        channel = tg.channel.channel
        self._built += 1
        return build_message(
            self.a.s,
            guild_id=ids.tenant_id(PLATFORM, team),
            channel_id=tg.cid,
            thread_id=ids.room_id(PLATFORM, team, f"thread:{m.thread_ts}") if m.thread_ts else None,
            message_id=ids.message_id(PLATFORM, team, channel, m.ts),
            author_id=ids.user_id(PLATFORM, m.user),
            author_name=m.name,
            created_at=m.created_at,
            text=text,
        )

    async def run(self) -> int:
        """Import every conversation Salid is in. Returns the messages imported. A run that was
        throttled or failed sets `capped`, so the loop comes back soon and resumes."""
        team = self.port.team_id
        gid = ids.tenant_id(PLATFORM, team)
        self.capped = False
        if self.port.revoked or not self.a.team.enabled(gid, "memory"):
            return 0
        try:
            rooms = await self.port.bot_rooms()
        except Exception as e:
            logger.warning("could not list Slack conversations (%s)", safe_exc(e))
            self.capped = True
            return 0
        targets = []
        for channel, (name, _id) in rooms.items():
            rid = ids.room_id(PLATFORM, team, channel)
            self.a.names[(gid, rid)] = name
            if self.a.team.channel_mode(gid, rid, name, set()) == "off":
                continue
            history = SlackHistory(
                self.port, channel, self.throttle, self.a._now, on_error=self._failed.add
            )
            targets.append(Target(channel=history, cid=rid))
        # a channel that failed last time goes last, so it cannot starve the others
        targets.sort(key=lambda t: t.channel.channel in self._failed)
        if not targets:
            return 0
        before_n = self._built
        try:
            res = await self.importer.run_guild(gid, targets)
        except Throttled:
            logger.warning("slack import paused: Slack keeps rate limiting, resuming later")
            self.capped = True
            return self._built - before_n
        except Exception as e:
            logger.warning("slack import stopped (%s), resuming later", safe_exc(e))
            self.capped = True
            return self._built - before_n
        self._failed.clear()
        self.capped = res.capped
        return res.messages
