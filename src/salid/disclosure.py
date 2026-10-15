"""Disclosure requests: how someone gets information they may not see, with the owner's consent.

When a memory tool withholds matching records (they belong to a higher-ranked person or sit in a
channel the asker cannot read), the agent may call `request_disclosure`. Salid then DMs the
owner (the author of the withheld records; if several, the highest-ranked) with who asks, the
question and a neutral description of what would be shared, never the content, and two buttons.
Approval stores a grant in `salid_disclosures`: exactly those message ids become visible to that one
grantee (in DMs), which is how Salid "remembers" it was disclosed. Declining stores the refusal.

This module is pure storage and rules; the Discord side (DMs, buttons) is in `salid.coo`.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Collection
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from .coo_store import DISCLOSURES, DocStore
from .i18n import t
from .memory_access import is_dm_scope
from .memory_store import MemoryMessage

MAX_HITS = 50  # message ids in one request


class Disclosure(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    guild_id: int
    owner_id: int  # whose information it is (the one who must agree)
    grantee_id: int  # who asked
    topic: str
    question: str = ""
    description: str = ""  # neutral text shown to the owner
    message_ids: list[int] = Field(default_factory=list)  # the source records
    status: str = "pending"  # pending | approved | declined
    lang: str = "th"
    requested_at: str = ""
    granted_at: str = ""
    expires_at: str = ""  # "" = no expiry


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def pick_owner(hits: Collection[MemoryMessage], ranks: dict[int, int]) -> int:
    """The person to ask: the highest-ranked author of the withheld records (most records, then
    the lowest id, break ties). 0 when there is no human author."""
    counts = Counter(m.author_id for m in hits if m.author_id and m.kind != "salid")
    if not counts:
        return 0
    return max(counts, key=lambda a: (ranks.get(a, -1), counts[a], -a))


def describe_hits(hits: Collection[MemoryMessage], names: dict[int, str], lang: str = "th") -> str:
    """Count, date span and place of the records; never their content."""
    days = sorted(m.local_date for m in hits)
    places = {
        t("disc_where_dm", lang)
        if is_dm_scope(m.channel_id)
        else f"#{names.get(m.channel_id, '?')}"
        for m in hits
    }
    span = days[0] if days[0] == days[-1] else f"{days[0]} - {days[-1]}"
    return t("disc_desc", lang, n=len(hits), span=span, where=", ".join(sorted(places)[:4]))


class DisclosureRepo:
    def __init__(self, docs: DocStore, now=lambda: datetime.now(UTC)):
        self.docs, self.now = docs, now

    def request(
        self,
        *,
        guild_id: int,
        owner_id: int,
        grantee_id: int,
        topic: str,
        question: str,
        hits: Collection[MemoryMessage],
        names: dict[int, str],
        lang: str = "th",
    ) -> tuple[Disclosure, bool]:
        """Create a pending request for the owner's records among `hits`. An identical pending
        request (same people and topic) is returned instead of a second one: (doc, created)."""
        mine = [m for m in hits if m.author_id == owner_id]
        ids = sorted({m.message_id for m in mine})[:MAX_HITS]
        for d in self.docs.find(DISCLOSURES, guild_id):
            if (
                d["status"] == "pending"
                and d["owner_id"] == owner_id
                and d["grantee_id"] == grantee_id
                and d["topic"].casefold() == topic.strip().casefold()
            ):
                return Disclosure(**d), False
        doc = Disclosure(
            guild_id=guild_id,
            owner_id=owner_id,
            grantee_id=grantee_id,
            topic=topic.strip()[:200],
            question=question.strip()[:400],
            description=describe_hits(mine, names, lang),
            message_ids=ids,
            lang=lang,
            requested_at=_iso(self.now()),
        )
        self.docs.put(DISCLOSURES, guild_id, doc.id, doc.model_dump())
        return doc, True

    def get(self, request_id: str) -> Disclosure | None:
        d = self.docs.find_any(DISCLOSURES, request_id)
        return Disclosure(**d) if d else None

    def _decide(
        self, request_id: str, status: str, expires_in: timedelta | None
    ) -> Disclosure | None:
        doc = self.get(request_id)
        if doc is None or doc.status != "pending":
            return None  # unknown, or already decided (a second click does nothing)
        doc.status = status
        if status == "approved":
            doc.granted_at = _iso(self.now())
            doc.expires_at = _iso(self.now() + expires_in) if expires_in else ""
        self.docs.put(DISCLOSURES, doc.guild_id, doc.id, doc.model_dump())
        return doc

    def approve(self, request_id: str, expires_in: timedelta | None = None) -> Disclosure | None:
        return self._decide(request_id, "approved", expires_in)

    def decline(self, request_id: str) -> Disclosure | None:
        return self._decide(request_id, "declined", None)

    def unlocked_ids(self, guild_id: int, grantee_id: int) -> frozenset[int]:
        """Message ids this person may see through grants (approved, not expired)."""
        now = _iso(self.now())
        out: set[int] = set()
        for d in self.docs.find(DISCLOSURES, guild_id):
            if (
                d["status"] == "approved"
                and d["grantee_id"] == grantee_id
                and (not d.get("expires_at") or d["expires_at"] > now)
            ):
                out.update(d["message_ids"])
        return frozenset(out)
