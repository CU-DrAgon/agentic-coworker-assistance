"""Salid's organisational memory: recording (layer 1) and summary scheduling (layer 2).

Layer 1 is the raw log: every human message (and Salid's own replies) in channels whose mode is not
`off`, with no LLM involved. Layer 2 turns each channel-day into a short summary with the cheap
gate model, in batches, never per message (see `MemoryScheduler`). Layer 3, answering, is in
`salid.qa` (tools) and `salid.memory_access` (who may see what).

Everything here is blocking and meant to run in a worker thread (`asyncio.to_thread`); the bot
wraps it so recording can never slow or break replying. Logs carry counts and ids, never message
contents.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from . import memory_summary
from .config import Settings
from .memory_store import DaySummary, MemoryMessage, MemoryStore, iso

logger = logging.getLogger(__name__)
UTC = UTC

MAX_TEXT_CHARS = 4000
MAX_UPDATES_PER_TICK = 3
MAX_FETCH = 600  # messages read per channel-day update
RETRY_AFTER = timedelta(minutes=5)  # after a failed summary call
DISCORD_EPOCH_MS = 1420070400000


def describe_attachment(att) -> str:
    """ "[image: receipt.jpg]" for an attachment object with `filename` and `content_type`."""
    ctype = str(getattr(att, "content_type", "") or "")
    kind = (
        ctype.split("/", 1)[0] if ctype.split("/", 1)[0] in ("image", "video", "audio") else "file"
    )
    return f"[{kind}: {getattr(att, 'filename', '') or 'attachment'}]"


def message_text(
    content: str | None, attachments: Iterable = (), notes: dict[int, str] | None = None
) -> str:
    """Content plus one descriptor per attachment; `notes` (attachment index -> text, e.g. an
    image description) replaces that attachment's plain descriptor."""
    notes = notes or {}
    parts = [
        (content or "").strip(),
        *(notes.get(i) or describe_attachment(a) for i, a in enumerate(attachments)),
    ]
    return " ".join(p for p in parts if p)[:MAX_TEXT_CHARS]


def synthetic_id(at: datetime, n: int = 0) -> int:
    """A Discord-snowflake-like id for records that are not Discord messages (voice segments)."""
    ms = int(at.timestamp() * 1000) - DISCORD_EPOCH_MS
    return (ms << 22) | (1 << 21) | (n & 0xFFFFF)  # bit 21 marks "not a real message"


def build_message(
    settings: Settings,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    author_id: int,
    author_name: str,
    created_at: datetime,
    text: str,
    jump_url: str = "",
    kind: str = "chat",
    thread_id: int | None = None,
) -> MemoryMessage:
    created = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
    created = created.astimezone(UTC)
    return MemoryMessage(
        guild_id=guild_id,
        channel_id=channel_id,
        thread_id=thread_id,
        message_id=message_id,
        author_id=author_id,
        author_name=author_name,
        created_at=created,
        local_date=created.astimezone(settings.zone).date().isoformat(),
        text=text,
        jump_url=jump_url,
        kind=kind,
    )


@dataclass
class _Entry:
    count: int = 0  # new messages since the last update
    last_at: datetime | None = None  # when the latest one arrived (scheduler clock)
    dirty: bool = False
    retry_at: datetime | None = None


class MemoryScheduler:
    """Decides when a channel-day summary is updated and does it, at most `max_updates` per tick.

    Today's summary is living: it is updated when the channel has `memory_update_every` new
    messages, or any new messages and `memory_quiet_minutes` of quiet. After the local day ends
    the summary is brought up to date once more and marked final; it is never rewritten after
    that, except once when a message of that day was deleted (dirty flag).
    """

    def __init__(
        self,
        store: MemoryStore,
        settings: Settings,
        llm,
        *,
        now: Callable[[], datetime] | None = None,
        lang_of: Callable[[int], str] = lambda gid: "th",
        retention_of: Callable[[int], int] | None = None,
        channel_name_of: Callable[[int, int], str] = lambda gid, cid: str(cid),
        enabled: Callable[[int], bool] = lambda gid: True,
        max_updates: int = MAX_UPDATES_PER_TICK,
        on_summary: Callable[[int, int, str, dict, list[MemoryMessage]], None] | None = None,
    ) -> None:
        self.store, self.s, self.llm = store, settings, llm
        # called after a summary was stored: (guild, channel, day, summary dict, new messages)
        self.on_summary = on_summary
        self.now = now or settings.now
        self.lang_of, self.channel_name_of, self.enabled = lang_of, channel_name_of, enabled
        self.max_updates = max_updates
        self.retention_of = retention_of or (lambda gid: settings.memory_retention_days)
        self._entries: dict[tuple[int, int, str], _Entry] = {}
        self._seeded: set[int] = set()
        self._known_guilds: set[int] = set()  # guilds seen by tick(): their own retention applies
        self._lock = threading.Lock()
        self._purged_on: str | None = None

    # ------------------------------------------------------------ events from the log
    def note(self, guild_id: int, channel_id: int, local_date: str) -> None:
        with self._lock:
            e = self._entries.setdefault((guild_id, channel_id, local_date), _Entry())
            e.count += 1
            e.last_at = self.now()

    def mark_dirty(self, guild_id: int, channel_id: int, local_date: str) -> None:
        with self._lock:
            self._entries.setdefault((guild_id, channel_id, local_date), _Entry()).dirty = True

    # ------------------------------------------------------------ tick
    def tick(self, guild_ids: Iterable[int]) -> int:
        """Seed new guilds, run due updates, purge once a day. Returns the updates performed."""
        now = self.now()
        guild_ids = list(guild_ids)
        self._known_guilds |= set(guild_ids)
        for gid in guild_ids:
            if gid not in self._seeded and self.enabled(gid):
                self._seed(gid, now)
        done = 0
        for key in self._due(now):
            if done >= self.max_updates:
                break
            if self._update(key):
                done += 1
        self.maybe_purge(now)
        return done

    def _seed(self, guild_id: int, now: datetime) -> None:
        """After a restart: pick up unfinished work from the store (open summaries, and today's
        and yesterday's channels that have messages after their summary)."""
        self._seeded.add(guild_id)
        today = now.date()
        with self._lock:
            for s in self.store.open_summaries(guild_id):
                e = self._entries.setdefault((guild_id, s.channel_id, s.local_date), _Entry())
                e.dirty = e.dirty or s.dirty
        for day in (today.isoformat(), (today - timedelta(days=1)).isoformat()):
            for cid in self.store.day_channels(guild_id, day):
                s = self.store.get_summary(guild_id, cid, day)
                if s is not None and s.final and not s.dirty:
                    continue
                n = len(
                    self.store.messages_after(
                        guild_id, cid, day, s.cursor if s else None, MAX_FETCH
                    )
                )
                if n:
                    with self._lock:
                        e = self._entries.setdefault((guild_id, cid, day), _Entry())
                        e.count = max(e.count, n)
                        e.last_at = e.last_at or now
        logger.info("memory scheduler seeded guild %s", guild_id)

    def _due(self, now: datetime) -> list[tuple[int, int, str]]:
        today = now.date().isoformat()
        quiet = timedelta(minutes=self.s.memory_quiet_minutes)
        ranked = []
        with self._lock:
            for key, e in self._entries.items():
                gid, _cid, day = key
                if not self.enabled(gid) or (e.retry_at and now < e.retry_at):
                    continue
                if e.dirty:
                    rank = 0
                elif day < today:
                    rank = 1
                elif e.count >= self.s.memory_update_every or (
                    e.count > 0 and e.last_at is not None and now - e.last_at >= quiet
                ):
                    rank = 2
                else:
                    continue
                ranked.append((rank, e.last_at or now, key))
        ranked.sort(key=lambda r: (r[0], r[1]))
        return [k for _, _, k in ranked]

    # ------------------------------------------------------------ one update
    def _update(self, key: tuple[int, int, str]) -> bool:
        gid, cid, day = key
        now = self.now()
        summary = self.store.get_summary(gid, cid, day)
        with self._lock:
            e = self._entries[key]
            snapshot, e.count = e.count, 0
            was_dirty, e.dirty = e.dirty, False
        dirty = was_dirty or bool(summary and summary.dirty)
        past = day < now.date().isoformat()
        if summary is not None and summary.final and not dirty:
            self._drop(key)
            return False
        try:
            after = None if dirty or summary is None else summary.cursor
            msgs = self.store.messages_after(gid, cid, day, after, MAX_FETCH)
            # a DM note its author marked private is never summarised
            shown = [m for m in msgs if m.visibility != "private"]
            if msgs or dirty:
                prev = None if dirty or summary is None else summary.summary
                new = (
                    memory_summary.update_summary(
                        self.llm,
                        self.s.gate_model,
                        prev,
                        shown,
                        channel=self.channel_name_of(gid, cid),
                        day=day,
                        lang=self.lang_of(gid),
                        zone=self.s.zone,
                    )
                    if shown
                    else (prev or memory_summary.SummaryDoc().model_dump())
                )
                calls = bool(shown)
            else:
                new, calls = (summary.summary if summary else {}), False
        except Exception:
            logger.warning(
                "memory summary failed for %s/%s %s; will retry", gid, cid, day, exc_info=True
            )
            with self._lock:
                e.count += snapshot
                e.dirty = e.dirty or was_dirty
                e.retry_at = self.now() + RETRY_AFTER
            return False
        more = len(msgs) >= MAX_FETCH  # a big backlog: finish it on the next tick
        if summary is None and not msgs:
            self._drop(key)
            return False
        base = 0 if dirty or summary is None else summary.message_count
        last = msgs[-1] if msgs else None
        fresh = self.store.get_summary(gid, cid, day)
        out = DaySummary(
            guild_id=gid,
            channel_id=cid,
            local_date=day,
            summary=new,
            last_message_at=iso(last.created_at)
            if last
            else (summary.last_message_at if summary and not dirty else None),
            last_message_id=last.message_id
            if last
            else (summary.last_message_id if summary and not dirty else 0),
            message_count=base + len(msgs),
            updated_at=iso(now.astimezone(UTC)),
            final=past and not more,
            dirty=bool(fresh and fresh.dirty and not dirty),
        )
        self.store.put_summary(out)
        if self.on_summary is not None and msgs:
            try:
                self.on_summary(gid, cid, day, new, [m for m in msgs if m.visibility != "private"])
            except Exception:
                logger.warning("summary hook failed for %s/%s %s", gid, cid, day, exc_info=True)
        logger.info(
            "memory update guild=%s channel=%s day=%s new=%d llm=%s final=%s",
            gid, cid, day, len(msgs), calls, out.final,
        )  # fmt: skip
        with self._lock:
            if out.final:
                self._entries.pop(key, None)
            elif more or out.dirty:
                e.count += 1 if more else 0
                e.dirty = out.dirty
        return calls

    def _drop(self, key) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def maybe_purge(self, now: datetime) -> tuple[int, int] | None:
        """Once per local day: delete what is past each guild's CURRENT retention (so a changed
        setting applies to data already stored)."""
        day = now.date().isoformat()
        if self._purged_on == day:
            return None
        self._purged_on = day
        try:
            utc = now.astimezone(UTC)
            cutoffs = {
                gid: utc - timedelta(days=self.retention_of(gid)) for gid in self._known_guilds
            }
            default = utc - timedelta(days=self.s.memory_retention_days)
            m, s = self.store.purge(cutoffs, default)
        except Exception:
            logger.warning("memory purge failed", exc_info=True)
            self._purged_on = None
            return None
        logger.info("memory purge: %d messages, %d summaries", m, s)
        return m, s


class MemoryService:
    """What the bot calls: record / edit / delete (layer 1) plus the scheduler."""

    def __init__(self, store: MemoryStore, settings: Settings, scheduler: MemoryScheduler):
        self.store, self.s, self.scheduler = store, settings, scheduler

    def record(self, msg: MemoryMessage) -> None:
        self.store.add_message(msg)
        self.scheduler.note(msg.guild_id, msg.channel_id, msg.local_date)

    def import_records(self, records: list[MemoryMessage]) -> None:
        """History import: idempotent writes (by message id). Only today's and yesterday's
        messages feed the living-summary scheduler; older days are summarised once by the
        importer's own capped queue."""
        edge = (self.scheduler.now().date() - timedelta(days=1)).isoformat()
        for m in records:
            self.store.add_message(m)
            if m.local_date >= edge:
                self.scheduler.note(m.guild_id, m.channel_id, m.local_date)

    def edit(self, guild_id: int, message_id: int, text: str, edited_at: datetime | None) -> bool:
        return self.store.update_message_text(guild_id, message_id, text, edited_at)

    def delete(self, guild_id: int, message_ids: Collection[int]) -> int:
        affected = self.store.delete_messages(guild_id, message_ids)
        for cid, day in affected:
            self.scheduler.mark_dirty(guild_id, cid, day)
        return len(affected)

    def delete_guild(self, guild_id: int) -> int:
        """Remove everything remembered for a guild (not called automatically: the bot keeps all
        other guild data when it is removed, so memory is kept too until retention expires it)."""
        return self.store.delete_guild(guild_id)
