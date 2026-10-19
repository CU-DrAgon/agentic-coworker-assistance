"""The bot's side of the workflow monitor (`salid.monitor`).

`MonitorMixin` owns `self.monitor`, the `Emitter` that producers add change-log events to
(`add` with an org id, `add_for_guild` with a guild id). Its own loop (`_monitor_loop`, every
bot `TICK_SECONDS`, started next to the tick loop) runs `_monitor_due`: it flushes the buffer (one
`emit_many` per log key) and, every `HEARTBEAT_EVERY_S`, writes the `bot` heartbeat with the
queue sizes noted by the tick (`actions` handled, `outbox_pending` change targets) and the memory
scheduler's backlog. The loop does not wait for the tick, so a slow or failing tick job never
delays the heartbeat or the flush (spec 3.4: the heartbeat tells a stuck job from a stopped bot).
The store work runs off the event loop. Flag `SALID_MONITOR_ENABLED` (`Settings.monitor_enabled`):
off = no loop, nothing buffered, flushed or written.

Producer hooks: `_monitor_action` (the action queue's `on_change`: one event per new or finished
web -> Discord action, logged under the guild's org), `_monitor_changes` (the change outbox's
`on_change`: one event per stored or updated change, under its org), `_monitor_video` (every
video-ad job record `videoad_bot` writes, the restart sweep's included, under the guild's org)
and `_monitor_job` (the deep-task / research `JobBook`'s `on_change`: every record it writes,
the restart sweep's included, under the guild's org). They only buffer (no store work on the
caller's thread beyond building the content-free event) and never raise.
"""

from __future__ import annotations

import asyncio
import logging

from . import monitor
from .change_outbox import Change
from .monitor import HEARTBEAT_EVERY_S, Emitter, event_for, org_key_resolver
from .monitor_sources import adapt_action, adapt_change, adapt_job, adapt_video

logger = logging.getLogger(__name__)


class MonitorMixin:
    """Needs `self.docs`, `self.s`, `self._clock`, `self.memory` and the `discord.Client`
    `wait_until_ready` / `is_closed`. Call `_monitor_init()` once `self.docs` exists,
    `_monitor_start(TICK_SECONDS)` where the tick loop starts and `_monitor_stop()` in `close`."""

    def _monitor_init(self) -> None:
        self.monitor = Emitter(
            self.docs, enabled=self.s.monitor_enabled, resolve=org_key_resolver(self.docs)
        )
        self._monitor_beat = float("-inf")
        self._monitor_queues = {"actions": 0, "outbox_pending": 0, "memory": 0}
        self._monitor_task: asyncio.Task | None = None

    def _monitor_note_queue(self, name: str, n: int) -> None:
        """Remember a queue size for the next heartbeat."""
        self._monitor_queues[name] = int(n)

    def _monitor_action(self, rec: dict) -> None:
        """`ActionQueue.on_change`: buffer the action's event for its guild's log. Never raises."""
        if not self.monitor.enabled:
            return
        try:
            self.monitor.add_for_guild(int(rec["guild_id"]), event_for(adapt_action(rec)))
        except Exception as e:  # the type only (the text may quote the record)
            logger.warning("monitor action event failed: %s", type(e).__name__)

    def _monitor_video(self, guild_id: int, job_id: str, rec: dict) -> None:
        """A video-ad job record was just written (`videoad_bot`): buffer its event for the
        guild's log. Never raises."""
        if not self.monitor.enabled:
            return
        try:
            self.monitor.add_for_guild(int(guild_id), event_for(adapt_video(job_id, rec)))
        except Exception as e:  # the type only (the text may quote the record)
            logger.warning("monitor video event failed: %s", type(e).__name__)

    def _monitor_job(self, rec: dict) -> None:
        """`JobBook.on_change`: a deep-task or research record was just written; buffer its
        event for the guild's log. Never raises."""
        if not self.monitor.enabled:
            return
        try:
            self.monitor.add_for_guild(int(rec["guild_id"]), event_for(adapt_job(rec)))
        except Exception as e:  # the type only (the text may quote the record)
            logger.warning("monitor job event failed: %s", type(e).__name__)

    def _monitor_changes(self, changes: list[Change]) -> None:
        """`ChangeOutbox.on_change`: buffer one event per change for its org's log. Never
        raises."""
        if not self.monitor.enabled:
            return
        try:
            now = self.s.now()
            for c in changes:
                self.monitor.add(c.org_id, event_for(adapt_change(c, now=now)))
        except Exception as e:  # the type only (the text may quote the change)
            logger.warning("monitor change event failed: %s", type(e).__name__)

    def _monitor_start(self, every_s: float) -> None:
        """Start the monitor loop; nothing while the monitor is off or the loop already runs."""
        if self.monitor.enabled and self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop(every_s))

    def _monitor_stop(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None

    async def _monitor_loop(self, every_s: float) -> None:
        """`_monitor_due` every `every_s` until the bot closes or the task is cancelled."""
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                await self._monitor_due()
            except Exception as e:  # `_monitor_due` never raises; keep the loop alive regardless
                logger.warning("monitor loop failed: %s", type(e).__name__)
            await asyncio.sleep(every_s)

    async def _monitor_due(self) -> None:
        """Flush the emitter and write the heartbeat when it is due. Never raises; does nothing
        while the monitor is off."""
        if not self.monitor.enabled:
            return
        try:
            await asyncio.to_thread(self.monitor.flush)
            now = self._clock()
            if now - self._monitor_beat < HEARTBEAT_EVERY_S:
                return
            self._monitor_beat = now  # a failed write is tried again in 30 s, not every 5 s
            queues = dict(self._monitor_queues)

            def beat() -> bool:  # the backlog takes the scheduler lock: off the event loop too
                queues["memory"] = self.memory.scheduler.backlog()
                return monitor.write_heartbeat(self.docs, "bot", queues=queues, now=self.s.now())

            await asyncio.to_thread(beat)
        except Exception as e:  # the type only (the text may quote stored data)
            logger.warning("monitor tick failed: %s", type(e).__name__)
