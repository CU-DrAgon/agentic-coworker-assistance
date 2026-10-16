"""Optional Slack history import for the team memory.

Reuses `salid.memory_import.MemoryImporter` (resumable per-channel state, idempotent writes by
message id, retention window, per-run cap, throttled pages, past-day summaries). Slack's history
call returns newest messages first, so `SlackHistory` reads a channel's whole window once (cached
for the run) and then serves the importer's oldest-first pages from it. A window bigger than
`MAX_WINDOW` messages (or with more than `MAX_THREADS` threads, or a thread that could not be read
completely) is skipped for the run (logged) instead of being imported with a hole.

Only conversations Salid is a member of are read (that is all Slack allows), 1:1 chats with people
are not. Counts only in the logs.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ..i18n import resolve_lang
from ..memory import build_message, message_text
from ..memory_import import MemoryImporter, Target
from ..memory_store import iso
from . import ids, textfmt
from .core import Assistant, FileRef, _descriptors, safe_exc
from .slack import ACCEPTED_SUBTYPES, PLATFORM, SlackPort, _ts

logger = logging.getLogger(__name__)

MAX_WINDOW = 50_000  # messages of one channel read per run
MAX_THREADS = 200  # threads per channel whose replies are read per run
PAGE = 200


class WindowTooLarge(Exception):
    pass


class SlackHistory:
    """One channel's history in the shape `MemoryImporter` expects (`history(after=, before=,
    limit=, oldest_first=)`, an async iterator of objects with `created_at`)."""

    def __init__(self, port: SlackPort, channel: str, sleep=asyncio.sleep, page_sleep: float = 1.0):
        self.port, self.channel, self._sleep, self._page_sleep = port, channel, sleep, page_sleep
        self._rows: list[SimpleNamespace] = []
        self._from: datetime | None = None

    async def prefetch(self, oldest: datetime) -> int:
        """Read every message from `oldest` on (once; a later, younger request is served from it).
        Raises WindowTooLarge."""
        if self._from is not None and oldest >= self._from:
            return len(self._rows)
        client = self.port.client
        raw: list[dict] = []
        cursor = ""
        while True:
            resp = await client.conversations_history(
                channel=self.channel,
                oldest=f"{oldest.timestamp():.6f}",
                limit=PAGE,
                **({"cursor": cursor} if cursor else {}),
            )
            raw.extend(resp.get("messages") or [])
            if len(raw) > MAX_WINDOW:
                raise WindowTooLarge()
            cursor = ((resp.get("response_metadata") or {}).get("next_cursor")) or ""
            if not cursor:
                break
            await self._sleep(self._page_sleep)
        threads = [m["ts"] for m in raw if (m.get("reply_count") or 0) > 0 and m.get("ts")]
        if len(threads) > MAX_THREADS:
            raise WindowTooLarge()  # importing only some threads would mark the window complete
        for ts in threads:
            # Every page of every thread, or the channel is not imported this run (an unread
            # thread must never leave a permanent hole behind a "complete" window). A failed
            # read raises and is retried on the next run.
            cursor = ""
            while True:
                resp = await client.conversations_replies(
                    channel=self.channel,
                    ts=ts,
                    oldest=f"{oldest.timestamp():.6f}",
                    limit=PAGE,
                    **({"cursor": cursor} if cursor else {}),
                )
                raw.extend(m for m in resp.get("messages") or [] if m.get("ts") != ts)
                if len(raw) > MAX_WINDOW:
                    raise WindowTooLarge()
                await self._sleep(self._page_sleep)
                cursor = ((resp.get("response_metadata") or {}).get("next_cursor")) or ""
                if not cursor:
                    break
        rows, seen = [], set()
        for m in raw:
            ts = m.get("ts")
            if not ts or ts in seen or m.get("subtype") not in ACCEPTED_SUBTYPES or m.get("bot_id"):
                continue
            seen.add(ts)
            if not m.get("user"):
                continue
            rows.append(
                SimpleNamespace(
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
            )
        rows.sort(key=lambda r: r.created_at)
        self._rows, self._from = rows, oldest
        return len(rows)

    async def history(self, *, limit, after, oldest_first=True, before=None):
        rows = [
            r
            for r in self._rows
            if r.created_at > after and (before is None or r.created_at < before)
        ]
        for r in rows[:limit]:
            yield r


class SlackImporter:
    def __init__(
        self,
        assistant: Assistant,
        port: SlackPort,
        *,
        sleep=asyncio.sleep,
        page_sleep: float = 1.0,
    ) -> None:
        self.a, self.port = assistant, port
        self._sleep, self._page_sleep = sleep, page_sleep
        self.capped = False
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
            page_sleep=page_sleep,
        )

    def _record(self, m, tg: Target):
        """The memory record of an imported Slack message (None: nothing to remember)."""
        text = message_text(m.text, _descriptors(m.files))
        if not text:
            return None
        team = self.port.team_id
        channel = tg.channel.channel
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
        """Import every conversation Salid is in. Returns the messages imported."""
        team = self.port.team_id
        gid = ids.tenant_id(PLATFORM, team)
        if not self.a.team.enabled(gid, "memory"):
            return 0
        retention = self.a.team.memory_retention_days(gid)
        target_from = self.a._now() - timedelta(days=retention)
        try:
            rooms = await self.port.bot_rooms()
        except Exception as e:
            logger.warning("could not list Slack conversations (%s)", safe_exc(e))
            return 0
        targets = []
        for channel, (name, _id) in rooms.items():
            rid = ids.room_id(PLATFORM, team, channel)
            self.a.names[(gid, rid)] = name
            if self.a.team.channel_mode(gid, rid, name, set()) == "off":
                continue
            state = await asyncio.to_thread(self.importer.store.get_channel_state, gid, rid)
            oldest = target_from
            if state is not None and state.complete and state.imported_from <= iso(target_from):
                oldest = max(target_from, datetime.fromisoformat(state.last_message_at))
            history = SlackHistory(self.port, channel, self._sleep, self._page_sleep)
            try:
                n = await history.prefetch(oldest.astimezone(UTC))
            except WindowTooLarge:
                logger.warning(
                    "slack import skipped a channel: window over %d messages", MAX_WINDOW
                )
                continue
            except Exception as e:
                logger.warning("slack import could not read a channel (%s)", safe_exc(e))
                continue
            logger.info("slack import read channel=%s messages=%d", rid, n)
            targets.append(Target(channel=history, cid=rid))
        if not targets:
            return 0
        res = await self.importer.run_guild(gid, targets)
        self.capped = res.capped
        return res.messages
