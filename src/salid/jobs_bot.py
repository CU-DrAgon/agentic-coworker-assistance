"""The bot's side of the deep-task and deep-research job records (`salid.jobs`).

`JobsMixin` gives `SecretaryBot` `self.jobs`, a `JobBook` whose every record write buffers a
workflow-monitor event (`MonitorMixin._monitor_job`; nothing while the monitor is off):
- `_job_start` / `_job_end`: `_run_deep` and `_run_research` record their job when it starts
  (index entry, then the `running` record) and when it ends (in `finally`: done, failed with a
  reason code, or cancelled by the user), all store work off the event loop. The job waits at
  most `START_WAIT_S` for its start record, then goes on without it (a user's cancel never waits
  for it); when the start record is still being written as the job ends, the end record is
  written after it by a background task, so the end record always comes last. A
  `CancelledError` without the user's cancel flag is a shutdown: nothing is written, and the
  record stays `running` in the index for the next start's sweep. Recording never stops, delays
  past the bound or fails the job;
- `dm`: a job asked for in a DM or in a member's private Salid channel (a failed lookup counts
  as private) is private to its requester in the monitor (the video ads' rule,
  `VideoAdMixin._video_private`);
- `_sweep_jobs`: on the first `on_ready`, jobs a restart cut off become `restarted`
  (`JobBook.sweep`, in a worker thread; `_job_live` keeps this process's running jobs and those
  whose records are still being written).

Never the task text or the research question. Jobs live in this process (one bot process).
"""

from __future__ import annotations

import asyncio
import logging

from .jobs import JobBook

logger = logging.getLogger(__name__)

START_WAIT_S = 5.0  # the longest a job waits for its start record before it goes on


def _settled(write: asyncio.Future) -> None:
    """Done callback of a start write: its outcome is always retrieved (never "exception was
    never retrieved"); a failure is logged by its type only."""
    if write.cancelled():
        return
    exc = write.exception()
    if exc is not None:
        logger.warning("job record not started: %s", type(exc).__name__)


class JobsMixin:
    """Needs `self.docs`, `self.s`, `_deep_jobs`, `_research_jobs` (`SearchMixin`),
    `_video_private` (`VideoAdMixin`) and `_monitor_job` (`MonitorMixin`). Call `_jobs_init()`
    once `self.docs` exists."""

    def _jobs_init(self) -> None:
        self.jobs = JobBook(self.docs, self.s.now, on_change=self._monitor_job)
        self._jobs_swept = False
        # job id -> its start write, until its end record is written (or it is left)
        self._job_writes: dict[str, asyncio.Future] = {}
        self._job_tails: set[asyncio.Task] = set()  # end records waiting for their start

    @staticmethod
    def _job_cancel_state(handle, exc: BaseException) -> str | None:
        """How a stopped job ends: `cancelled` when the user cancelled it (the job's own
        cancel error, or a `CancelledError` with the handle's `cancelled` flag set), else None:
        a shutdown, left for the next start's sweep."""
        if not isinstance(exc, asyncio.CancelledError):
            return "cancelled"
        return "cancelled" if getattr(handle, "cancelled", False) else None

    async def _job_start(
        self, kind: str, job_id: str, message, guild_id: int, *, requester_id: int | None = None
    ) -> None:
        """Record that job `job_id` of `kind` started in `guild_id`, asked for by
        `requester_id` (default: the message's author) in the message's channel. Waits at most
        `START_WAIT_S` for the write. Nothing without a job id; never raises."""
        if not job_id:
            return
        try:
            author = getattr(message, "author", None)
            rid = int((getattr(author, "id", 0) if requester_id is None else requester_id) or 0)
            cid = int(getattr(getattr(message, "channel", None), "id", 0) or 0)
            gid = int(guild_id)
        except Exception as e:  # the type only
            logger.warning("job record not started: %s", type(e).__name__)
            return

        def start() -> bool:  # the private-channel lookup may read the store too
            dm = self._video_private(message)
            return self.jobs.start(
                kind, job_id=job_id, guild_id=gid, channel_id=cid, requester_id=rid, dm=dm
            )

        write = asyncio.ensure_future(asyncio.to_thread(start))
        write.add_done_callback(_settled)
        self._job_writes[job_id] = write
        # `wait` never cancels the write (on a timeout or when the job is cancelled meanwhile)
        done, _pending = await asyncio.wait({write}, timeout=START_WAIT_S)
        if not done:
            logger.warning("job record start is slow: the job goes on without it")

    async def _job_end(
        self, guild_id: int, job_id: str, state: str | None, reason: str = ""
    ) -> None:
        """Record how job `job_id` ended; `state` None (a shutdown) writes nothing and leaves
        the record for the next start's sweep. A start record still being written is waited
        for by a background task that then writes this one, so the job does not wait. Nothing
        without a job id; never raises."""
        if not job_id:
            return
        if state is None:
            self._job_writes.pop(job_id, None)
            return
        write = self._job_writes.get(job_id)
        if write is not None and not write.done():
            tail = asyncio.ensure_future(
                self._job_finish(guild_id, job_id, state, reason, after=write)
            )
            self._job_tails.add(tail)
            tail.add_done_callback(self._job_tails.discard)
            return
        await self._job_finish(guild_id, job_id, state, reason)

    async def _job_finish(
        self, guild_id: int, job_id: str, state: str, reason: str, after=None
    ) -> None:
        """Write the end record (after the start write `after`, when given). Never raises."""
        try:
            if after is not None:
                await asyncio.wait({after})  # the start record lands first
            await asyncio.to_thread(self.jobs.finish, int(guild_id), job_id, state, reason)
        except Exception as e:
            logger.warning("job record not finished: %s", type(e).__name__)
        finally:
            self._job_writes.pop(job_id, None)

    def _job_live(self, job_id: str) -> bool:
        """This process runs the deep task or research `job_id` (an entry whose task is not
        done), or its records are still being written. Called from the sweep's worker thread:
        it only reads, from copies."""
        if not job_id:
            return False
        if job_id in self._job_writes:
            return True
        running = [*list(self._deep_jobs.values()), *list(self._research_jobs.values())]
        return any(getattr(j, "job_id", "") == job_id and not j.task.done() for j in running)

    async def _sweep_jobs(self) -> None:
        """Once per process (the first on_ready; reconnects fire it again): jobs a restart cut
        off become `restarted` and leave the index, in a worker thread. Never raises."""
        if self._jobs_swept:
            return
        self._jobs_swept = True
        try:
            swept = await asyncio.to_thread(self.jobs.sweep, self._job_live)
        except Exception as e:
            logger.warning("job sweep failed: %s", type(e).__name__)
            return
        if swept:
            logger.info("jobs a restart cut off: %d", len(swept))
