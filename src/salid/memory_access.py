"""Who may read what from the memory: decided in code, never by the model.

`allowed_channel_ids` is a pure function over channels and the member ids who can read each one.
The Discord adapter (in the bot) only fills `ChannelAccess` from permissions.

Rules for a question asked by `asker_id`:
* the asker must be able to read the source channel now;
* when the answer is posted in a guild channel X, every (human) member who can read X must also be
  able to read the source channel S, so a private channel's content never reaches a wider
  audience. X itself is always allowed;
* in a DM (`answering_channel_id=None`) only the first rule applies.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelAccess:
    channel_id: int
    name: str
    viewers: frozenset[int] = field(default_factory=frozenset)  # humans who can read + see history


def allowed_channel_ids(
    channels: Iterable[ChannelAccess], asker_id: int, answering_channel_id: int | None = None
) -> set[int]:
    chans = list(channels)
    by_id = {c.channel_id: c for c in chans}
    audience: frozenset[int] | None = None
    if answering_channel_id is not None:
        answering = by_id.get(answering_channel_id)
        if answering is None:
            return set()  # audience unknown: share nothing rather than guess
        audience = answering.viewers
    allowed = set()
    for c in chans:
        if (
            c.channel_id == answering_channel_id
            or asker_id in c.viewers
            and (audience is None or audience <= c.viewers)
        ):
            allowed.add(c.channel_id)
    return allowed
