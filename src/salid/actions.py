"""Act on Discord from anywhere: a queue of actions the bot carries out.

The web service holds no Discord token (by design), yet the command center and Salid on the web
must be able to CHANGE things in the company's Discord: post a message, create a scheduled event,
make a channel, DM someone. So they enqueue an action here (DocStore kind `salid_actions`, keyed
by guild) and the bot, which has the token and the permissions, picks it up on its tick (every
few seconds), performs it with the Discord API and writes the result back. Anyone who queued an
action can poll it. Actions are small dicts so they survive any process restart.

    q = ActionQueue(docs)
    a = q.enqueue(guild_id, "post_message", {"channel": "general", "text": "hi"}, by="boss@x")
    # "announce" {text}: no channel given, the bot picks the server's ask channel / general
    ...bot...: for a in q.pending(guild_id): ...; q.finish(guild_id, a["id"], result={...})

`on_change` (optional) is told every new or finished record, so the workflow monitor can log the
state change (`salid.monitor`); a failing hook is logged and never breaks the queue. Finished
records are deleted `KEEP_DAYS` after they were queued (`prune`, at most hourly per guild).
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from .coo_store import DocStore

logger = logging.getLogger(__name__)

KIND = "salid_actions"
KINDS = (
    "post_message",
    "announce",
    "create_event",
    "create_channel",
    "send_dm",
    "edit_event",
    "delete_event",
)
MAX_PENDING = 50  # per guild: a runaway caller cannot flood the bot
MAX_AGE_S = 3600  # pending actions older than this are abandoned with an error
KEEP_DAYS = 30  # finished (done / failed) actions are deleted this long after they were queued
PRUNE_EVERY_S = 3600.0  # `prune` reads a guild's actions at most this often


def _now() -> datetime:
    return datetime.now(UTC)


def _created(rec: dict) -> datetime | None:
    """When the action was queued, as an aware datetime (naive = UTC); None when unreadable."""
    try:
        at = datetime.fromisoformat(str(rec.get("created_at") or ""))
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=UTC)


class ActionQueue:
    def __init__(
        self,
        docs: DocStore,
        clock: Callable[[], datetime] | None = None,
        on_change: Callable[[dict], None] | None = None,
    ):
        self.docs, self._clock, self.on_change = docs, clock or _now, on_change
        self._pruned: dict[int, float] = {}  # guild id -> monotonic time of its last prune

    def _changed(self, rec: dict) -> None:
        """Tell `on_change` about a stored record. Never raises: the error type is logged (its
        text may quote the record)."""
        if self.on_change is None:
            return
        try:
            self.on_change(rec)
        except Exception as e:
            logger.warning("action queue: change hook failed (%s)", type(e).__name__)

    def enqueue(self, guild_id: int, kind: str, args: dict, *, by: str) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown action {kind!r}")
        if len(self.pending(guild_id)) >= MAX_PENDING:
            raise ValueError("too many pending actions; try again in a minute")
        aid = "A-" + secrets.token_hex(4)
        rec = {
            "id": aid,
            "guild_id": guild_id,
            "kind": kind,
            "args": dict(args),
            "by": by,
            "status": "pending",
            "created_at": self._clock().isoformat(timespec="seconds"),
            "done_at": "",
            "result": {},
            "error": "",
        }
        self.docs.put(KIND, guild_id, aid, rec)
        self._changed(rec)
        return rec

    def get(self, guild_id: int, aid: str) -> dict | None:
        return self.docs.get(KIND, guild_id, aid)

    def pending(self, guild_id: int) -> list[dict]:
        rows = self.docs.find(KIND, guild_id, lambda d: d.get("status") == "pending")
        return sorted(rows, key=lambda d: d.get("created_at", ""))

    def recent(self, guild_id: int, limit: int = 30) -> list[dict]:
        rows = self.docs.find(KIND, guild_id)
        return sorted(rows, key=lambda d: d.get("created_at", ""), reverse=True)[:limit]

    def finish(
        self, guild_id: int, aid: str, *, result: dict | None = None, error: str = ""
    ) -> None:
        rec = self.get(guild_id, aid)
        if not rec:
            return
        rec.update(
            status="failed" if error else "done",
            error=error[:500],
            result=result or {},
            done_at=self._clock().isoformat(timespec="seconds"),
        )
        self.docs.put(KIND, guild_id, aid, rec)
        self._changed(rec)

    def expire(self, guild_id: int) -> int:
        """Abandon pending actions nobody carried out within MAX_AGE_S (bot was down)."""
        now = self._clock()
        n = 0
        for a in self.pending(guild_id):
            try:
                age = (now - datetime.fromisoformat(a["created_at"])).total_seconds()
            except ValueError:
                age = MAX_AGE_S + 1
            if age > MAX_AGE_S:
                self.finish(guild_id, a["id"], error="expired before the bot could do it")
                n += 1
        return n

    def prune(self, guild_id: int, *, force: bool = False) -> int:
        """Delete this guild's finished actions (done or failed) queued more than `KEEP_DAYS`
        ago; pending ones are never deleted (`expire` finishes them first) and neither is a
        record whose queue time is unreadable. Reads the guild at most once per
        `PRUNE_EVERY_S` unless `force`. Returns how many were deleted."""
        mono = time.monotonic()
        last = self._pruned.get(guild_id)
        if not force and last is not None and mono - last < PRUNE_EVERY_S:
            return 0
        self._pruned[guild_id] = mono
        now = self._clock()
        cutoff = (now if now.tzinfo else now.replace(tzinfo=UTC)) - timedelta(days=KEEP_DAYS)
        n = 0
        for key, rec in self.docs.find_items(
            KIND, guild_id, lambda d: d.get("status") != "pending"
        ):
            at = _created(rec)
            if at is not None and at < cutoff:
                self.docs.delete(KIND, guild_id, key)
                n += 1
        return n
