"""Who may read what from the memory: decided in code, never by the model.

`allowed_channel_ids` is a pure function over channels and the member ids who can read each one.
The Discord adapter (in the bot) only fills `ChannelAccess` from permissions.

Rules for a question asked by `asker_id`:
* the asker must be able to read the source channel now;
* when the answer is posted in a guild channel X, every (human) member who can read X must also be
  able to read the source channel S, so a private channel's content never reaches a wider
  audience. X itself is always allowed;
* in a DM (`answering_channel_id=None`) only the first rule applies.

Role hierarchy (on top of the channel rules): information ABOUT a person's work that the asker
cannot read in its channel is visible only when the asker strictly outranks that person (or is
that person). The same audience rule applies: in a guild channel X every member who can read X
must be that person or outrank them, so in practice hierarchy-granted information appears in DMs
and in manager-only channels. Rank is a plain int (see `member_rank`). A DM note ("dm" kind,
stored under the synthetic per-user scope `dm_scope(user_id)`) is visible to its author and, unless
the author marked it "private", to everyone who outranks the author. A disclosure grant makes
specific message ids visible to one grantee, in DMs only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

OWNER_RANK = 2_000_000  # the guild owner outranks everyone
ADMIN_RANK = 1_000_000  # Administrator counts as top (plus the role position as a tie-breaker)


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


def dm_scope(user_id: int) -> int:
    """Synthetic channel id under which a member's DM notes are stored (never a Discord id)."""
    return -int(user_id)


def is_dm_scope(channel_id: int) -> bool:
    return channel_id < 0


def member_rank(*, is_owner: bool, is_admin: bool, top_role_position: int) -> int:
    """Rank of a guild member from plain facts (the Discord adapter fills them in)."""
    if is_owner:
        return OWNER_RANK
    return (ADMIN_RANK if is_admin else 0) + max(int(top_role_position), 0)


def hierarchy_authors(
    asker_id: int,
    ranks: dict[int, int],
    audience: Iterable[int] | None = None,
) -> frozenset[int]:
    """Author ids whose work the asker may read beyond the channels they can read: themself and
    everyone they strictly outrank. With `audience` (the members who can read the answering
    channel; None = a DM or ephemeral reply) an author only counts when every audience member is
    that author or outranks them, so a reply never reaches someone who may not see it."""
    mine = ranks.get(asker_id)
    if mine is None:
        return frozenset()
    people = None if audience is None else set(audience) | {asker_id}
    out = set()
    for author, rank in ranks.items():
        if author != asker_id and mine <= rank:
            continue
        if people is not None and any(m != author and ranks.get(m, -1) <= rank for m in people):
            continue
        out.add(author)
    return frozenset(out)


def audience_authors(ranks: dict[int, int], audience: Iterable[int]) -> frozenset[int]:
    """Authors whose work every member of `audience` may read (each is that author or outranks
    them): what a channel's report may say about people. Used for the reports channel."""
    people = set(audience)
    return frozenset(
        a for a, r in ranks.items() if all(m == a or ranks.get(m, -1) > r for m in people)
    )


@dataclass(frozen=True)
class Scope:
    """Everything one request may read: built once per question by the bot."""

    asker_id: int
    channels: frozenset[int] = frozenset()  # readable channels (allowed_channel_ids)
    authors: frozenset[int] = frozenset()  # hierarchy_authors for the answering audience
    dm_authors: frozenset[int] = frozenset()  # same without the audience rule (a DM would do)
    unlocked: frozenset[int] = frozenset()  # message ids disclosed to the asker (DMs only)
    dm: bool = True  # the answer goes to the asker alone (a DM): grants apply


def message_visible(m, scope: Scope) -> bool:
    """Can the asker see this record (a `MemoryMessage`)? Pure."""
    if scope.dm and m.message_id in scope.unlocked:
        return True
    if is_dm_scope(
        m.channel_id
    ):  # a DM note: its author, or whoever outranks them (not if private)
        private = getattr(m, "visibility", "normal") == "private"
        if private and m.author_id != scope.asker_id:
            return False
        return m.author_id in scope.authors
    if m.channel_id in scope.channels:
        return True
    return m.kind != "salid" and m.author_id in scope.authors


def hidden_but_askable(m, scope: Scope) -> bool:
    """Not visible now, and worth offering a disclosure request for: a hit the asker may not see.
    Private DM notes are never hinted at."""
    if message_visible(m, scope):
        return False
    return not (is_dm_scope(m.channel_id) and getattr(m, "visibility", "normal") == "private")


@dataclass(frozen=True)
class Visibility:
    """What the memory tools need beyond the plain channel list: the scope, every channel that
    may hold hierarchy-visible or hidden records, and member names for the `person` filter."""

    scope: Scope
    read_channels: dict[int, str] = field(default_factory=dict)  # id -> name, hidden ones too
    people: dict[int, str] = field(default_factory=dict)  # member id -> display name
    dm_names: dict[int, str] = field(default_factory=dict)  # DM scopes whose summaries may be read
