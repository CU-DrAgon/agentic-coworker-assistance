"""History import for Salid's memory: everything that still exists in Discord within a guild's
retention window, including messages from before memory was switched on.

Per readable, non-off channel or thread a `ChannelState` records which time range the log covers
(`imported_from` .. `last_message_at`), so the import is resumable and idempotent (records are
written by message id):

* forward phase: messages after `last_message_at` up to now. This is also the downtime catch-up
  after a restart, and the whole import of a new channel (starting at now - retention).
* gap phase: when retention was lengthened, only the newly covered older range
  `[now - retention, imported_from)` is imported.

Throttled on purpose: one channel at a time, pages of 100, a pause between pages, at most
`memory_import_guild_cap` messages per guild per run (the next run resumes). Past channel-days
with >= 3 messages get one final summary, generated through a daily capped queue, newest days
first. Logs carry counts only.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from . import memory_summary
from .config import Settings
from .memory import MAX_FETCH, MemoryService
from .memory_store import ChannelState, DaySummary, MemoryMessage, MemoryStore, iso

logger = logging.getLogger(__name__)

PAGE_SIZE = 100
MIN_DAY_MESSAGES = 3  # channel-days with fewer messages are left to their raw messages


@dataclass
class Target:
    """A channel or thread to import. `cid` is the id used for access (a thread's parent)."""

    channel: object
    cid: int
    thread_id: int | None = None

    @property
    def state_id(self) -> int:
        return self.thread_id or self.cid


@dataclass
class RunResult:
    messages: int = 0
    channels_done: int = 0
    channels_total: int = 0
    capped: bool = False  # the guild cap stopped the run: schedule another one


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class MemoryImporter:
    def __init__(
        self,
        store: MemoryStore,
        settings: Settings,
        service: MemoryService,
        llm,
        *,
        build_record: Callable[[object, Target], MemoryMessage | None],
        retention_of: Callable[[int], int],
        lang_of: Callable[[int], str] = lambda gid: "th",
        channel_name_of: Callable[[int, int], str] = lambda gid, cid: str(cid),
        now: Callable[[], datetime] | None = None,
        page_sleep: float = 0.5,
        page_size: int = PAGE_SIZE,
    ) -> None:
        self.store, self.s, self.service, self.llm = store, settings, service, llm
        self.build_record, self.retention_of = build_record, retention_of
        self.lang_of, self.channel_name_of = lang_of, channel_name_of
        self.now = now or settings.now
        self.page_sleep, self.page_size = page_sleep, page_size

    # ------------------------------------------------------------ guild budget accounting
    def guild_state(self, gid: int):
        st = self.store.get_guild_state(gid)
        today = self.now().date().isoformat()
        if st.day != today:
            st.day, st.images_done, st.summaries_done = today, 0, 0
        return st

    # ------------------------------------------------------------ the run
    async def run_guild(self, gid: int, targets: list[Target]) -> RunResult:
        now = self.now().astimezone(UTC)
        target_from = now - timedelta(days=self.retention_of(gid))
        gstate = self.guild_state(gid)
        gstate.channels_total = len(targets)
        res = RunResult(channels_total=len(targets))
        budget = self.s.memory_import_guild_cap
        for tg in targets:
            if budget <= 0:
                res.capped = True
                break
            st = await asyncio.to_thread(self.store.get_channel_state, gid, tg.state_id)
            if st is None:
                st = ChannelState(
                    guild_id=gid,
                    channel_id=tg.state_id,
                    parent_id=tg.cid if tg.thread_id else None,
                    imported_from=iso(target_from),
                    last_message_at=iso(target_from),
                )
            else:
                # data older than the window is purged anyway: never claim to cover it
                st.imported_from = max(st.imported_from, iso(target_from))
                st.last_message_at = max(st.last_message_at, iso(target_from))
            used, forward_done = await self._forward(gid, tg, st, budget)
            budget -= used
            res.messages += used
            gap_done = True
            if forward_done and budget > 0 and st.imported_from > iso(target_from):
                used, gap_done = await self._gap(gid, tg, st, target_from, budget)
                budget -= used
                res.messages += used
            elif st.imported_from > iso(target_from):
                gap_done = False
            st.complete = forward_done and gap_done
            st.updated_at = iso(now)
            await asyncio.to_thread(self.store.put_channel_state, st)
            if not st.complete:
                res.capped = True
        res.channels_done = sum(
            1 for s in await asyncio.to_thread(self.store.list_channel_states, gid) if s.complete
        )
        await asyncio.to_thread(self.store.put_guild_state, gstate)
        await self.summarise_past(gid)
        logger.info(
            "memory import guild=%s messages=%d channels=%d/%d capped=%s",
            gid, res.messages, res.channels_done, res.channels_total, res.capped,
        )  # fmt: skip
        return res

    async def _pages(self, tg: Target, after: datetime, before: datetime | None):
        kw = {"limit": self.page_size, "after": after, "oldest_first": True}
        if before is not None:
            kw["before"] = before
        return [m async for m in tg.channel.history(**kw)]

    async def _forward(self, gid, tg, st, budget) -> tuple[int, bool]:
        """Messages after `last_message_at`. Returns (imported, reached the end)."""
        used, touched = 0, set()
        while True:
            if used >= budget:
                return used, False
            msgs = await self._pages(tg, _dt(st.last_message_at), None)
            if not msgs:
                await self._queue_days(st, touched, None)
                await asyncio.to_thread(self.store.put_channel_state, st)
                return used, True
            n = await self._store_page(gid, tg, msgs, touched)
            used += n
            st.last_message_at = iso(msgs[-1].created_at)
            st.messages += n
            await self._queue_days(st, touched, msgs[-1].created_at)
            await asyncio.to_thread(self.store.put_channel_state, st)
            if len(msgs) < self.page_size:
                await self._queue_days(st, touched, None)
                await asyncio.to_thread(self.store.put_channel_state, st)
                return used, True
            await asyncio.sleep(self.page_sleep)

    async def _gap(self, gid, tg, st, target_from, budget) -> tuple[int, bool]:
        """Messages in `[target_from, imported_from)` after a retention increase."""
        used, touched = 0, set()
        before = _dt(st.imported_from)
        while True:
            if used >= budget:
                return used, False
            after = _dt(st.gap_cursor) if st.gap_cursor else target_from
            msgs = await self._pages(tg, after, before)
            finished = len(msgs) < self.page_size
            if msgs:
                n = await self._store_page(gid, tg, msgs, touched)
                used += n
                st.messages += n
                st.gap_cursor = iso(msgs[-1].created_at)
                await self._queue_days(st, touched, None if finished else msgs[-1].created_at)
            if finished:
                await self._queue_days(st, touched, None)
                st.imported_from, st.gap_cursor = iso(target_from), None
                await asyncio.to_thread(self.store.put_channel_state, st)
                return used, True
            await asyncio.to_thread(self.store.put_channel_state, st)
            await asyncio.sleep(self.page_sleep)

    async def _store_page(self, gid, tg, msgs, touched: set) -> int:
        records = [r for m in msgs if (r := self.build_record(m, tg)) is not None]
        await asyncio.to_thread(self.service.import_records, records)
        touched.update(r.local_date for r in records)
        return len(records)

    async def _queue_days(self, st: ChannelState, touched: set, cursor: datetime | None) -> None:
        """Past days that are complete (the cursor moved past them, or the phase ended) and have
        >= 3 messages wait for their one final summary."""
        today = self.now().date()
        edge = (today - timedelta(days=1)).isoformat()  # today and yesterday: the scheduler's
        cursor_day = cursor.astimezone(self.s.zone).date().isoformat() if cursor else "9999-12-31"
        cid = st.parent_id or st.channel_id
        for day in sorted(touched):
            if day >= cursor_day or day >= edge or day in st.summary_days:
                continue
            touched.discard(day)
            few = await asyncio.to_thread(
                self.store.messages_after, st.guild_id, cid, day, None, MIN_DAY_MESSAGES
            )
            if len(few) >= MIN_DAY_MESSAGES:
                st.summary_days.append(day)

    # ------------------------------------------------------------ past-day summaries
    async def summarise_past(self, gid: int) -> int:
        """Generate the one final summary of queued past channel-days: newest days first, at
        most `memory_import_summaries_per_day` per guild per day. Returns how many were made."""
        states = await asyncio.to_thread(self.store.list_channel_states, gid)
        gstate = self.guild_state(gid)
        pending = sorted(
            {(day, s.parent_id or s.channel_id) for s in states for day in s.summary_days},
            reverse=True,
        )
        made = 0
        for day, cid in pending:
            if gstate.summaries_done >= self.s.memory_import_summaries_per_day:
                break
            done = await asyncio.to_thread(self._summarise_day, gid, cid, day)
            if done:
                gstate.summaries_done += 1
                made += 1
            if done is not None:  # None = leave it queued (a failure to retry later)
                for s in states:
                    if day in s.summary_days and (s.parent_id or s.channel_id) == cid:
                        s.summary_days.remove(day)
                        await asyncio.to_thread(self.store.put_channel_state, s)
        gstate.updated_at = iso(self.now().astimezone(UTC))
        await asyncio.to_thread(self.store.put_guild_state, gstate)
        return made

    def _summarise_day(self, gid: int, cid: int, day: str) -> bool | None:
        """True = summary generated, False = nothing to do, None = failed (retry later)."""
        existing = self.store.get_summary(gid, cid, day)
        if existing is not None and existing.final and not existing.dirty:
            return False
        msgs = self.store.messages_after(gid, cid, day, None, MAX_FETCH)
        if len(msgs) < MIN_DAY_MESSAGES:
            return False
        try:
            new = memory_summary.update_summary(
                self.llm,
                self.s.gate_model,
                None,
                msgs,
                channel=self.channel_name_of(gid, cid),
                day=day,
                lang=self.lang_of(gid),
                zone=self.s.zone,
            )
        except Exception:
            logger.warning("past-day summary failed for %s/%s %s", gid, cid, day, exc_info=True)
            return None
        last = msgs[-1]
        self.store.put_summary(
            DaySummary(
                guild_id=gid,
                channel_id=cid,
                local_date=day,
                summary=new,
                last_message_at=iso(last.created_at),
                last_message_id=last.message_id,
                message_count=len(msgs),
                updated_at=iso(self.now().astimezone(UTC)),
                final=True,
            )
        )
        return True


def import_progress(store: MemoryStore, guild_id: int, images_cap: int, today: str) -> dict:
    """What `/ความจำ` and the hub show: channels imported, messages, historical images today."""
    states = store.list_channel_states(guild_id)
    g = store.get_guild_state(guild_id)
    return {
        "channels_done": sum(1 for s in states if s.complete),
        "channels_total": max(g.channels_total, len(states)),
        "messages": sum(s.messages for s in states),
        "images_done": g.images_done if g.day == today else 0,
        "images_cap": images_cap,
    }
