"""Deep task and deep research job records: what runs, for whom, and how it ended.

No ``discord`` import. ``JobBook`` keeps one record per job in DocStore kind ``salid_jobs``
(guild = the Discord server, key = job id) and one active-jobs index in the same kind (guild 0,
key ``_active``, ``{job id: guild id}``), copying the video-ad jobs (``videoad_bot``): a job is
indexed before its first record is written and leaves the index after its final one, so after a
restart the index lists exactly the jobs that may have been cut off; ``sweep`` marks those
``restarted``. Records are never deleted.

A record is ``{job_id, kind, guild_id, channel_id, requester_id, dm, state, created_at,
updated_at, finished_at, reason}``: ids, times and states only (Bangkok ISO times from the bot's
clock). It never holds the task text, the research question, the report or an error message:
``reason`` is one of ``REASONS`` (codes of ``monitor.ERROR_TEXT``). Every record written is handed
to ``on_change`` (the bot's workflow-monitor hook). Nothing here raises: a store error is logged
(the error type only, never its text, which may quote stored data) and the call answers False,
``{}`` or ``[]``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .coo_store import DocStore

logger = logging.getLogger(__name__)

KIND = "salid_jobs"  # DocStore kind: per Discord server, key = job id
INDEX_GUILD, INDEX_KEY = 0, "_active"  # in KIND: {job id: guild id} of every job not yet ended
KINDS = ("deep_task", "research")
RUNNING = "running"
FINAL = frozenset({"done", "failed", "cancelled", "restarted"})
# the only reasons a record keeps (anything else is stored as "error")
REASONS = frozenset({"", "failed", "error", "restarted"})
FIELDS = (
    "job_id",
    "kind",
    "guild_id",
    "channel_id",
    "requester_id",
    "dm",
    "state",
    "created_at",
    "updated_at",
    "finished_at",
    "reason",
)


def new_id() -> str:
    """A fresh job id: 12 hex characters."""
    return uuid.uuid4().hex[:12]


def _failed(what: str, exc: BaseException) -> None:
    logger.warning("job %s failed: %s", what, type(exc).__name__)


class JobBook:
    """The job records and the active-jobs index on a DocStore. Blocking: the bot calls it from
    a worker thread. `now` is the bot's clock; `on_change(record)` runs after every record write
    (a raising hook is logged and ignored)."""

    def __init__(
        self,
        docs: DocStore,
        now: Callable[[], datetime],
        on_change: Callable[[dict], None] | None = None,
    ):
        self.docs, self.now, self.on_change = docs, now, on_change

    # ------------------------------------------------------------- lifecycle
    def start(
        self,
        kind: str,
        *,
        job_id: str,
        guild_id: int,
        channel_id: int,
        requester_id: int,
        dm: bool = False,
    ) -> bool:
        """Index the job (one transaction), then write its `running` record. False (nothing
        more written) for an unknown kind, an empty id or a store error; an index entry whose
        record was not written is dropped by the next sweep."""
        if kind not in KINDS or not job_id:
            return False
        try:
            gid = int(guild_id)
            self._set_index(job_id, gid)
        except Exception as exc:
            _failed("index", exc)
            return False
        at = self._stamp()
        rec = {
            "job_id": job_id,
            "kind": kind,
            "guild_id": gid,
            "channel_id": int(channel_id),
            "requester_id": int(requester_id),
            "dm": bool(dm),
            "state": RUNNING,
            "created_at": at,
            "updated_at": at,
            "finished_at": "",
            "reason": "",
        }
        return self._put(rec)

    def finish(self, guild_id: int, job_id: str, state: str, reason: str = "") -> bool:
        """Write the job's final record (`state` in FINAL; `reason` outside REASONS is stored as
        "error"), then take it out of the index. False when nothing was written: a state that
        is not final, a store error (the job stays indexed, so a restart marks it restarted
        instead of leaving it running) or a missing record (the entry just leaves the index)."""
        if state not in FINAL or not job_id:
            return False
        try:
            gid = int(guild_id)
            rec = self.docs.get(KIND, gid, job_id)
        except Exception as exc:
            _failed("read", exc)
            return False
        if not rec:
            self._drop(job_id)
            return False
        at = self._stamp()
        code = reason if reason in REASONS else "error"
        done = {
            **_known(rec),
            "job_id": job_id,
            "guild_id": gid,
            "state": state,
            "reason": code,
            "updated_at": at,
            "finished_at": at,
        }
        if not self._put(done):
            return False
        return self._drop(job_id)

    def active(self) -> dict[str, int]:
        """The active-jobs index: {job id: guild id} ({} on a store error)."""
        try:
            return self._read_index()
        except Exception as exc:
            _failed("index read", exc)
            return {}

    def sweep(self, is_live: Callable[[str], bool]) -> list[dict]:
        """Reading only the index and the records it lists: every listed job that is not
        `is_live` (run by this process) and has not ended becomes `restarted` (reason
        "restarted") and leaves the index; an ended or missing record just leaves it. Returns
        the records marked ([] on a store error)."""
        try:
            index = self._read_index()
        except Exception as exc:
            _failed("sweep index read", exc)
            return []
        out: list[dict] = []
        for job_id, gid in index.items():
            try:
                rec = self._sweep_one(job_id, gid, is_live)
            except Exception as exc:
                _failed("sweep", exc)
                continue
            if rec is not None:
                out.append(rec)
        return out

    # ------------------------------------------------------------- helpers
    def _sweep_one(self, job_id: str, gid: int, is_live: Callable[[str], bool]) -> dict | None:
        if is_live(job_id):  # runs here: it leaves the index itself when it ends
            return None
        rec = self.docs.get(KIND, gid, job_id)
        if not rec or rec.get("state") in FINAL:
            self._drop(job_id)
            return None
        at = self._stamp()
        cut = {
            **_known(rec),
            "job_id": job_id,
            "guild_id": gid,
            "state": "restarted",
            "reason": "restarted",
            "updated_at": at,
            "finished_at": at,
        }
        self.docs.put(KIND, gid, job_id, cut)
        self._changed(cut)
        self._drop(job_id)
        return cut

    def _stamp(self) -> str:
        return self.now().isoformat(timespec="seconds")

    def _put(self, rec: dict) -> bool:
        try:
            self.docs.put(KIND, rec["guild_id"], rec["job_id"], rec)
        except Exception as exc:
            _failed("record", exc)
            return False
        self._changed(rec)
        return True

    def _changed(self, rec: dict) -> None:
        if self.on_change is None:
            return
        try:
            self.on_change(dict(rec))
        except Exception as exc:
            _failed("change hook", exc)

    def _read_index(self) -> dict[str, int]:
        return _entries(self.docs.get(KIND, INDEX_GUILD, INDEX_KEY))

    def _set_index(self, job_id: str, guild_id: int | None) -> None:
        """Add `job_id` (of `guild_id`) to the index, or remove it (None), in one transaction.
        Raises on a store error."""

        def change(tx) -> None:
            doc = _entries(tx.get(KIND, INDEX_GUILD, INDEX_KEY))
            if guild_id is None:
                if job_id not in doc:
                    return
                del doc[job_id]
            else:
                doc[job_id] = guild_id
            tx.put(KIND, INDEX_GUILD, INDEX_KEY, doc)

        self.docs.transact(change)

    def _drop(self, job_id: str) -> bool:
        """Take an ended job out of the index; False (logged) on a store error."""
        try:
            self._set_index(job_id, None)
        except Exception as exc:
            _failed("unindex", exc)
            return False
        return True


def _entries(doc: dict[str, Any] | None) -> dict[str, int]:
    """The index document as {job id: guild id} (Firestore adds a `guild_id` field on write).
    An entry whose guild id is not a number is skipped (logged by error type), so one bad
    entry never stops the recording; the next index write drops it."""
    out: dict[str, int] = {}
    for key, value in dict(doc or {}).items():
        if key == "guild_id":
            continue
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError) as exc:
            _failed("index entry", exc)
    return out


def _known(rec: dict[str, Any]) -> dict[str, Any]:
    """Only the record's own fields: anything else a stored document carries is dropped."""
    return {k: rec.get(k) for k in FIELDS}
