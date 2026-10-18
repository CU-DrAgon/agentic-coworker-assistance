"""The bot's side of the workflow monitor (`salid.monitor`).

`MonitorMixin` owns `self.monitor`, the `Emitter` that producers add change-log events to
(`add` with an org id, `add_for_guild` with a guild id), and runs last in every bot tick
(`_monitor_due`): it flushes the buffer (one `emit_many` per log key) and, every
`HEARTBEAT_EVERY_S`, writes the `bot` heartbeat with the queue sizes noted during the tick
(`actions` handled, `outbox_pending` change targets, the memory scheduler's backlog). The store
work runs off the event loop. Flag `SALID_MONITOR_ENABLED` (`Settings.monitor_enabled`): off =
nothing is buffered, flushed or written.
"""

from __future__ import annotations

import asyncio
import logging

from . import monitor
from .monitor import HEARTBEAT_EVERY_S, Emitter, org_key_resolver

logger = logging.getLogger(__name__)


class MonitorMixin:
    """Needs `self.docs`, `self.s`, `self._clock` and `self.memory`. Call `_monitor_init()` once
    `self.docs` exists; `tick` ends with `_monitor_due()`."""

    def _monitor_init(self) -> None:
        self.monitor = Emitter(
            self.docs, enabled=self.s.monitor_enabled, resolve=org_key_resolver(self.docs)
        )
        self._monitor_beat = float("-inf")
        self._monitor_queues = {"actions": 0, "outbox_pending": 0, "memory": 0}

    def _monitor_note_queue(self, name: str, n: int) -> None:
        """Remember a queue size for the next heartbeat."""
        self._monitor_queues[name] = int(n)

    async def _monitor_due(self) -> None:
        """Flush the emitter (every tick) and write the heartbeat when it is due. Never raises;
        does nothing while the monitor is off."""
        if not self.monitor.enabled:
            return
        try:
            await asyncio.to_thread(self.monitor.flush)
            now = self._clock()
            if now - self._monitor_beat < HEARTBEAT_EVERY_S:
                return
            self._monitor_beat = now  # a failed write is tried again in 30 s, not every tick
            self._monitor_queues["memory"] = self.memory.scheduler.backlog()
            await asyncio.to_thread(
                monitor.write_heartbeat,
                self.docs,
                "bot",
                queues=dict(self._monitor_queues),
                now=self.s.now(),
            )
        except Exception as e:  # never break the tick; the type only (the text may quote data)
            logger.warning("monitor tick failed: %s", type(e).__name__)
