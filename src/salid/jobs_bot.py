"""The bot's side of the deep-task and deep-research job records (`salid.jobs`).

`JobsMixin` gives `SecretaryBot` `self.jobs`, a `JobBook` whose every record write buffers a
workflow-monitor event (`MonitorMixin._monitor_job`; nothing while the monitor is off):
- `_job_start` / `_job_end`: `_run_deep` and `_run_research` record their job when it starts
  (index entry, then the `running` record) and when it ends (in `finally`: done, failed with a
  reason code, or cancelled), all store work off the event loop. A cancel that lands while the
  start record is being written waits for it, so the end record always comes last. Recording
  never stops or fails the job;
- `dm`: a job asked for in a DM or in the requester's private Salid channel is private to its
  requester in the monitor (the video ads' rule, `VideoAdMixin._video_private`);
- `_sweep_jobs`: on the first `on_ready`, jobs a restart cut off become `restarted`
  (`JobBook.sweep`, in a worker thread; `_job_live` keeps this process's running jobs).

Never the task text or the research question. Jobs live in this process (one bot process).
"""

from __future__ import annotations

import asyncio
import logging

from .jobs import JobBook

logger = logging.getLogger(__name__)


class JobsMixin:
    """Needs `self.docs`, `self.s`, `_deep_jobs`, `_research_jobs` (`SearchMixin`),
    `_video_private` (`VideoAdMixin`) and `_monitor_job` (`MonitorMixin`). Call `_jobs_init()`
    once `self.docs` exists."""

    def _jobs_init(self) -> None:
        self.jobs = JobBook(self.docs, self.s.now, on_change=self._monitor_job)
        self._jobs_swept = False

    async def _job_start(
        self, kind: str, job_id: str, message, guild_id: int, *, requester_id: int | None = None
    ) -> None:
        """Record that job `job_id` of `kind` started in `guild_id`, asked for by
        `requester_id` (default: the message's author) in the message's channel. Nothing
        without a job id; never raises."""
        if not job_id:
            return
        try:
            author = getattr(message, "author", None)
            rid = int((getattr(author, "id", 0) if requester_id is None else requester_id) or 0)
            cid = int(getattr(getattr(message, "channel", None), "id", 0) or 0)
            gid = int(guild_id)

            def start() -> bool:  # the private-channel lookup may read the store too
                dm = self._video_private(message)
                return self.jobs.start(
                    kind, job_id=job_id, guild_id=gid, channel_id=cid, requester_id=rid, dm=dm
                )

            write = asyncio.ensure_future(asyncio.to_thread(start))
            try:
                await asyncio.shield(write)
            except asyncio.CancelledError:
                await asyncio.wait({write})  # the start record lands before the end record
                raise
        except Exception as e:  # the type only (the text may quote stored data)
            logger.warning("job record not started: %s", type(e).__name__)

    async def _job_end(self, guild_id: int, job_id: str, state: str, reason: str = "") -> None:
        """Record how job `job_id` ended. Nothing without a job id; never raises."""
        if not job_id:
            return
        try:
            await asyncio.to_thread(self.jobs.finish, int(guild_id), job_id, state, reason)
        except Exception as e:
            logger.warning("job record not finished: %s", type(e).__name__)

    def _job_live(self, job_id: str) -> bool:
        """This process runs the deep task or research `job_id` (an entry whose task is not
        done). Called from the sweep's worker thread: it only reads, from copies."""
        if not job_id:
            return False
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
