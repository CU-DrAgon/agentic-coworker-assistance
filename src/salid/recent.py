"""Short-term context: the last messages of each channel, kept in memory from gateway events.

Salid reads the room's recent conversation before every answer (and before the follow-up gate).
Instead of a Discord history call each time, `RecentMessages` keeps the newest messages per channel
as they arrive (`on_message`, edits, deletes). A channel is trusted only after one history read has
primed it: until then (and after a gateway reconnect, when events may have been missed) callers
read Discord and prime the buffer with the result.

Bounded: `per_channel` messages per channel, `max_channels` channels (least recently used dropped),
channels quiet for longer than `max_age_s` dropped by `sweep`. In memory only, never stored.
Used from the event loop only.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Snapshot:
    """The fields of a Discord message that the context readers use (duck-types a message)."""

    id: int
    channel_id: int
    author: Any
    content: str
    created_at: datetime
    attachments: tuple = ()
    role_mentions: tuple = ()

    @classmethod
    def of(cls, message, channel_id: int) -> Snapshot:
        return cls(
            id=message.id,
            channel_id=channel_id,
            author=message.author,
            content=message.content or "",
            created_at=getattr(message, "created_at", None) or datetime.now(UTC),
            attachments=tuple(getattr(message, "attachments", None) or ()),
            role_mentions=tuple(getattr(message, "role_mentions", None) or ()),
        )


class RecentMessages:
    def __init__(
        self, per_channel: int = 40, max_channels: int = 1000, max_age_s: float = 3 * 3600
    ):
        self.per_channel, self.max_channels, self.max_age_s = per_channel, max_channels, max_age_s
        self._chans: OrderedDict[int, deque[Snapshot]] = OrderedDict()
        self._primed: set[int] = set()

    def _channel(self, channel_id: int) -> deque[Snapshot]:
        dq = self._chans.get(channel_id)
        if dq is None:
            dq = self._chans[channel_id] = deque(maxlen=self.per_channel)
        self._chans.move_to_end(channel_id)
        while len(self._chans) > self.max_channels:
            old, _ = self._chans.popitem(last=False)
            self._primed.discard(old)
        return dq

    def add(self, message) -> None:
        channel_id = getattr(getattr(message, "channel", None), "id", None)
        if channel_id is None:
            return
        dq = self._channel(channel_id)
        if any(s.id == message.id for s in dq):  # already there from a history read
            return
        dq.append(Snapshot.of(message, channel_id))

    def edit(self, channel_id: int, message_id: int, content: str) -> None:
        dq = self._chans.get(channel_id)
        for i, s in enumerate(dq or ()):
            if s.id == message_id:
                dq[i] = replace(s, content=content or "")

    def delete(self, channel_id: int, message_ids: Iterable[int]) -> None:
        dq = self._chans.get(channel_id)
        if dq is None:
            return
        gone = set(message_ids)
        kept = [s for s in dq if s.id not in gone]
        dq.clear()
        dq.extend(kept)

    def prime(self, channel_id: int, messages: Iterable) -> None:
        """Mark the channel complete with `messages` (a history read, any order). What the buffer
        already holds wins: it came from the gateway and may carry newer edits."""
        dq = self._channel(channel_id)
        merged = {s.id: s for s in dq}
        for m in messages:
            merged.setdefault(m.id, Snapshot.of(m, channel_id))
        dq.clear()
        dq.extend(sorted(merged.values(), key=lambda s: s.id)[-self.per_channel :])
        self._primed.add(channel_id)

    def newest_id(self, channel_id: int) -> int | None:
        dq = self._chans.get(channel_id)
        return dq[-1].id if dq else None

    def history(self, channel_id: int, limit: int, before_id: int | None = None) -> list | None:
        """Newest first, like Discord's history; None when the channel is not primed."""
        if channel_id not in self._primed:
            return None
        self._chans.move_to_end(channel_id)
        rows = [
            s for s in reversed(self._chans[channel_id]) if before_id is None or s.id < before_id
        ]
        return rows[:limit]

    def reset(self) -> None:
        """After a reconnect: events (edits, deletes) may have been missed, so everything is
        forgotten and every channel is read again."""
        self._chans.clear()
        self._primed.clear()

    def sweep(self, now: datetime) -> None:
        """Drop channels whose newest message is older than `max_age_s`."""
        for channel_id, dq in list(self._chans.items()):
            if not dq or (now - dq[-1].created_at).total_seconds() > self.max_age_s:
                self._chans.pop(channel_id, None)
                self._primed.discard(channel_id)
