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
"""

from __future__ import annotations

import asyncio
import logging

from . import monitor
from .monitor import HEARTBEAT_EVERY_S, Emitter, org_key_resolver

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
