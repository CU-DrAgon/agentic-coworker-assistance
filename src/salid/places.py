"""Places Salid remembers: the company's head office and branches, and people's own places.

Who may know a place is decided in code, never by the model (pure functions over ranks and the
members who can read each channel; the Discord adapter lives in `places_bot.py`):

* Company place: its audience is the union of the audiences of every channel where a human stated
  it (`sources`), plus the role hierarchy (a member who outranks someone in that audience may know
  it too). A later mention in a wider channel appends a source and so widens it; it never narrows
  by itself. An executive can reset the sources to a chosen set of channels.
* Personal place (`owner_id` set): only the owner, unless the owner shares it (superiors, named
  people, everyone) or states it in a channel themselves (then that channel's audience may know).
  Nobody else can set, change or widen it.

An answer that reaches several people (a channel, a voice room) may only carry a place that every
one of them may know (`audience`).
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from urllib.parse import quote

from google.genai import types
from pydantic import BaseModel

from .llm import generate
from .memory_access import dm_scope
from .tasks import match_person

logger = logging.getLogger(__name__)

PLACES = "salid_places"
OFFERS = "salid_place_offers"  # "asked once" markers: a hash, never the address itself
COMPANY_KINDS = ("hq", "branch", "warehouse", "site", "client", "venue", "other")
PERSONAL_KINDS = ("home", "other")
SHARE_MODES = ("owner", "superiors", "people", "everyone", "channel")
MAX_LIST = 30
MAX_ADDRESS = 400
MAPS_SEARCH = "https://www.google.com/maps/search/?api=1&query="

Viewers = Callable[[int], frozenset[int]]  # channel id -> humans who can read it


# ---------------------------------------------------------------------------- the record
@dataclass
class Place:
    id: str
    guild_id: int
    name: str
    address: str
    kind: str = "other"
    aliases: list[str] = field(default_factory=list)
    owner_id: int | None = None  # personal place: the person; None = company place
    created_by: int = 0
    notes: str = ""
    lat: float | None = None
    lng: float | None = None
    maps_url: str = ""
    sources: list[dict] = field(default_factory=list)  # {channel_id, message_id, message_link,
    # disclosed_by, at}; or {members: [ids], ...} for a voice room; channel_id < 0 = a DM
    share_superiors: bool = False  # personal only
    share_people: list[int] = field(default_factory=list)  # personal only
    share_everyone: bool = False  # personal only
    created_at: str = ""

    @property
    def personal(self) -> bool:
        return self.owner_id is not None

    @classmethod
    def from_doc(cls, d: dict) -> Place:
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})

    def doc(self) -> dict:
        return asdict(self)


def norm(text: str) -> str:
    return re.sub(r"[\s,.\-_/()]+", "", (text or "").casefold())


def maps_link(p: Place) -> str:
    if p.maps_url:
        return p.maps_url
    if p.lat is not None and p.lng is not None:
        return MAPS_SEARCH + quote(f"{p.lat},{p.lng}", safe="")
    return MAPS_SEARCH + quote(p.address or p.name, safe="")


def matches(p: Place, query: str) -> int:
    """0 = no match, 2 = exact name/alias, 1 = partial (name, alias or address contains it)."""
    q = norm(query)
    if not q:
        return 0
    names = [norm(n) for n in (p.name, *p.aliases) if norm(n)]
    if q in names:
        return 2
    if any(q in n or (len(n) >= 3 and n in q) for n in names):
        return 1
    return 1 if len(q) >= 4 and q in norm(p.address) else 0


# ---------------------------------------------------------------------------- who may know
def source_viewers(src: dict, viewers_of: Viewers) -> frozenset[int]:
    if src.get("members"):
        return frozenset(int(x) for x in src["members"])
    cid = int(src.get("channel_id", 0))
    if cid < 0:  # a DM
        return frozenset({-cid})
    return frozenset(viewers_of(cid))


def place_audience(p: Place, ranks: dict[int, int], viewers_of: Viewers) -> frozenset[int]:
    """Every member who may know this place."""
    if p.personal:
        owner = int(p.owner_id)
        out = {owner, *p.share_people}
        if p.share_everyone:
            out |= set(ranks)
        if p.share_superiors and owner in ranks:
            out |= {u for u, r in ranks.items() if r > ranks[owner]}
        for src in p.sources:  # the owner stated it in that channel themself
            if src.get("disclosed_by") == owner:
                out |= source_viewers(src, viewers_of)
        return frozenset(out)
    base: set[int] = set()
    for src in p.sources:
        base |= source_viewers(src, viewers_of)
    if not base:
        base = {p.created_by}
    known = [ranks[u] for u in base if u in ranks]
    if known:  # whoever outranks someone who may know it may know it too
        floor = min(known)
        base |= {u for u, r in ranks.items() if r > floor}
    return frozenset(base)


def may_know(
    p: Place,
    user_id: int,
    ranks: dict[int, int],
    viewers_of: Viewers,
    audience: frozenset[int] | None = None,
) -> bool:
    """The asker may know it, and (when the answer reaches `audience`) so may every listener."""
    who = place_audience(p, ranks, viewers_of)
    if user_id not in who:
        return False
    return audience is None or set(audience) <= who


# ---------------------------------------------------------------------------- repository
class PlaceRepo:
    def __init__(self, docs, clock: Callable[[], datetime] | None = None):
        self.docs = docs
        self.clock = clock or (lambda: datetime.now(UTC))

    def list(self, guild_id: int) -> list[Place]:
        return sorted(
            (Place.from_doc(d) for d in self.docs.find(PLACES, guild_id)), key=lambda p: p.name
        )

    def get(self, guild_id: int, place_id: str) -> Place | None:
        d = self.docs.get(PLACES, guild_id, place_id)
        return Place.from_doc(d) if d else None

    def put(self, p: Place) -> Place:
        self.docs.put(PLACES, p.guild_id, p.id, p.doc())
        return p

    def delete(self, p: Place) -> None:
        self.docs.delete(PLACES, p.guild_id, p.id)

    def create(self, guild_id: int, **kw) -> Place:
        p = Place(
            id=uuid.uuid4().hex[:10],
            guild_id=guild_id,
            created_at=self.clock().isoformat(timespec="seconds"),
            **kw,
        )
        return self.put(p)

    def search(self, guild_id: int, query: str, owner_id: int | None | str = "any") -> list[Place]:
        scored = [(matches(p, query), p) for p in self.list(guild_id)]
        out = [(s, p) for s, p in scored if s and (owner_id == "any" or p.owner_id == owner_id)]
        return [p for _s, p in sorted(out, key=lambda x: -x[0])]

    def same(self, guild_id: int, name: str, address: str, owner_id: int | None) -> Place | None:
        """An existing place of the same owner with this name/alias or this address."""
        n, a = norm(name), norm(address)
        for p in self.list(guild_id):
            if p.owner_id != owner_id:
                continue
            if n and n in [norm(x) for x in (p.name, *p.aliases)]:
                return p
            if a and len(a) >= 8 and a == norm(p.address):
                return p
        return None

    def add_source(self, p: Place, src: dict) -> bool:
        """Append a source unless that channel (or the same room) is already one. True = widened
        or changed."""
        for s in p.sources:
            if src.get("members"):
                if sorted(s.get("members") or []) == sorted(src["members"]):
                    return False
            elif not s.get("members") and s.get("channel_id") == src.get("channel_id"):
                return False
        p.sources.append(src)
        self.put(p)
        return True

    # one-time "may I remember your home?" offers
    @staticmethod
    def _offer_key(owner_id: int, address: str) -> str:
        return hashlib.sha1(f"{owner_id}|{norm(address)}".encode()).hexdigest()[:16]

    def claim_offer(self, guild_id: int, owner_id: int, address: str) -> bool:
        """True the first time for this person and address (and records that it was asked)."""
        key = self._offer_key(owner_id, address)
        if self.docs.get(OFFERS, guild_id, key):
            return False
        self.docs.put(
            OFFERS,
            guild_id,
            key,
            {"owner_id": owner_id, "at": self.clock().isoformat(timespec="seconds")},
        )
        return True

    def mark_offered(self, guild_id: int, owner_id: int, address: str) -> None:
        self.claim_offer(guild_id, owner_id, address)


def make_source(
    channel_id: int | None,
    disclosed_by: int,
    at: datetime,
    *,
    message_id: int = 0,
    link: str = "",
    room: frozenset[int] | None = None,
) -> dict:
    src: dict = {
        "channel_id": dm_scope(disclosed_by) if channel_id is None else channel_id,
        "message_id": message_id,
        "message_link": link,
        "disclosed_by": disclosed_by,
        "at": at.isoformat(timespec="seconds"),
    }
    if room:
        src["members"] = sorted(room)
    return src


# ---------------------------------------------------------------------------- learning
ADDRESS_RE = re.compile(
    r"เลขที่|ถนน|ถ\.|ซอย|แขวง|เขต|ตำบล|อำเภอ|จังหวัด|จ\.\s?[ก-๙]|หมู่ที่|รหัสไปรษณีย์"
    r"|สาขา|สำนักงานใหญ่|สำนักงาน|ออฟฟิศ|โกดัง|คลังสินค้า|บ้านผม|บ้านฉัน|บ้านของผม|บ้านหนู"
    r"|maps\.app\.goo\.gl|goo\.gl/maps|google\.[a-z.]+/maps|maps\.google"
    r"|\b(?:road|rd\.|street|st\.|soi|district|province|tambon|amphoe|building|floor)\b"
    r"|\b(?:hq|head\s*office|headquarters|office|branch|warehouse|my\s+home|my\s+house)\b"
    r"|\b\d{5}\b",
    re.I,
)
MIN_LEARN_CHARS = 12
MAX_LEARN_CHARS = 1200


def looks_like_place(text: str) -> bool:
    """Free prefilter: only messages that may hold an address go to the model."""
    text = text or ""
    return len(text) >= MIN_LEARN_CHARS and bool(ADDRESS_RE.search(text))


class Candidate(BaseModel):
    name: str = ""
    address: str = ""
    kind: str = "other"
    scope: str = "other"  # company | own_home | other
    aliases: list[str] = []


class _Candidates(BaseModel):
    places: list[Candidate] = []


EXTRACT_PROMPT = """You read one chat message from a company's Discord (Thai or English).
List the places whose ADDRESS or Google Maps link is actually written in the message.
For each: name (short, as people call it, e.g. "สาขาบางนา", "Head office"), address (copy it as written, or the maps link),
kind (company places: hq, branch, warehouse, site, client, venue, other; personal: home, other),
aliases (other names used for it, Thai/English), and scope:
- "company": an office/branch/warehouse/site/client/venue of the company or a place used for company work;
- "own_home": the author's own home or personal address (the author says it is theirs);
- "other": anything else (somebody else's home, a restaurant, a made-up or incomplete address, a place only
  named without an address).
Do not invent anything. If no address is written, return an empty list.

Author: {author}
Message:
{text}
"""


def extract_candidates(client, model: str, text: str, author: str) -> list[Candidate]:
    """One cheap-model call. Fails quiet (no candidates)."""
    try:
        resp = generate(
            client,
            model,
            EXTRACT_PROMPT.format(author=author, text=text[:MAX_LEARN_CHARS]),
            types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=500,
                response_mime_type="application/json",
                response_schema=_Candidates,
            ),
        )
        return _Candidates.model_validate_json(resp.text).places[:3]
    except Exception:
        logger.warning("place extraction failed", exc_info=True)
        return []


@dataclass
class Learned:
    created: list[Place] = field(default_factory=list)
    widened: list[Place] = field(default_factory=list)
    offers: list[Candidate] = field(default_factory=list)  # own-home candidates to ask about


def learn(
    repo: PlaceRepo,
    cands: list[Candidate],
    *,
    guild_id: int,
    author_id: int,
    channel_id: int,
    message_id: int,
    link: str,
    at: datetime,
    addressed: bool = False,
) -> Learned:
    """Apply what the model found in one human message. Company places are stored or widened;
    an author's own home is never stored here (an offer is returned instead, once), but it widens
    an existing personal place of the same author. Anything else is ignored."""
    out = Learned()
    for c in cands:
        address = (c.address or "").strip()[:MAX_ADDRESS]
        name = (c.name or "").strip()[:80]
        if not address or not (name or c.scope == "own_home"):
            continue
        src = make_source(channel_id, author_id, at, message_id=message_id, link=link)
        if c.scope == "company":
            kind = c.kind if c.kind in COMPANY_KINDS else "other"
            hit = repo.same(guild_id, name, address, None)
            if hit is None:
                out.created.append(
                    repo.create(
                        guild_id,
                        name=name,
                        address=address,
                        kind=kind,
                        aliases=[a.strip() for a in c.aliases if a.strip()][:5],
                        created_by=author_id,
                        sources=[src],
                    )
                )
            elif repo.add_source(hit, src):
                out.widened.append(hit)
        elif c.scope == "own_home":
            mine = repo.same(guild_id, name or "บ้าน", address, author_id)
            if mine is not None:  # the owner says it again in a channel: that audience may know
                if repo.add_source(mine, src):
                    out.widened.append(mine)
            elif not addressed and repo.claim_offer(guild_id, author_id, address):
                out.offers.append(c)
    return out


# ---------------------------------------------------------------------------- agent tools
@dataclass
class PlacesCtx:
    """Everything the tools are bound to for one request (set by the bot, never by the model)."""

    guild_id: int
    asker_id: int
    is_exec: bool
    repo: PlaceRepo
    ranks: dict[int, int]
    viewers_of: Viewers
    audience: frozenset[int] | None  # who the answer reaches (None = the asker alone, a DM)
    channel_id: int | None  # where the asker is talking (None = a DM)
    people: dict[int, str]
    channels: dict[int, str]  # readable channels, id -> name
    confirm: Callable[[tuple], bool] | None = None  # two-step gate for company deletions
    room: frozenset[int] | None = None  # voice room
    message_id: int = 0
    message_link: str = ""
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    def can_see(self, p: Place) -> bool:
        return may_know(p, self.asker_id, self.ranks, self.viewers_of, self.audience)

    def source(self) -> dict:
        return make_source(
            self.channel_id,
            self.asker_id,
            self.now(),
            message_id=self.message_id,
            link=self.message_link,
            room=self.room,
        )


class PlaceTools:
    def __init__(self, ctx: PlacesCtx):
        self.c = ctx

    # ---- helpers
    def _visible(self, query: str) -> list[Place]:
        c = self.c
        return [p for p in c.repo.search(c.guild_id, query) if c.can_see(p)]

    def _own_or_company(self, query: str) -> Place | None:
        """Best visible match; the asker's own personal places win a tie."""
        hits = self._visible(query)
        hits.sort(key=lambda p: (p.owner_id != self.c.asker_id, -matches(p, query)))
        return hits[0] if hits else None

    def _who(self, p: Place) -> str | list[str]:
        """Who can see it, for the owner / creator / an executive only."""
        c = self.c
        if p.personal:
            parts = ["owner only"]
            if p.share_everyone:
                parts = ["everyone in the company"]
            else:
                if p.share_superiors:
                    parts.append("people ranked above the owner")
                if p.share_people:
                    parts.append(
                        "named: " + ", ".join(c.people.get(i, str(i)) for i in p.share_people)
                    )
            chans = [
                c.channels.get(s["channel_id"], "a channel")
                for s in p.sources
                if s.get("disclosed_by") == p.owner_id and s.get("channel_id", 0) > 0
            ]
            if chans:
                parts.append("readers of: " + ", ".join(sorted(set(chans))))
            return parts
        chans = []
        for s in p.sources:
            if s.get("members"):
                chans.append("a voice meeting")
            elif s.get("channel_id", 0) < 0:
                chans.append("a private chat")
            else:
                chans.append(c.channels.get(s["channel_id"], "a channel"))
        return ["readers of: " + ", ".join(sorted(set(chans))), "plus people ranked above them"]

    def _view(self, p: Place, detail: bool = True) -> dict:
        c = self.c
        out = {
            "name": p.name,
            "kind": p.kind,
            "address": p.address,
            "maps_link": maps_link(p),
            "personal": p.personal,
        }
        if p.personal and p.owner_id != c.asker_id:
            out["owner"] = c.people.get(int(p.owner_id), "")
        if detail:
            if p.aliases:
                out["aliases"] = p.aliases
            if p.notes:
                out["notes"] = p.notes
            if p.owner_id == c.asker_id or p.created_by == c.asker_id or c.is_exec:
                out["who_can_see"] = self._who(p)
        return out

    # ---- tools
    def remember_place(
        self,
        name: str,
        address: str,
        kind: str = "other",
        personal: bool = False,
        aliases: list[str] | None = None,
        notes: str = "",
    ) -> dict:
        """Remember a place. personal=False: a place of the company (head office, branch,
        warehouse, site, client, venue) told by someone, visible to whoever may read where it was
        said. personal=True: the ASKER's own place (e.g. their home), only when the asker asked for
        it or agreed to it; private to the asker until they share it. Never for someone else's
        personal place. kind: hq, branch, warehouse, site, client, venue, other (company) or home,
        other (personal)."""
        c = self.c
        name, address = (name or "").strip()[:80], (address or "").strip()[:MAX_ADDRESS]
        if not name or not address:
            return {"error": "name and address are required"}
        kinds = PERSONAL_KINDS if personal else COMPANY_KINDS
        kind = kind if kind in kinds else "other"
        al = [a.strip() for a in (aliases or []) if a and a.strip()][:5]
        owner = c.asker_id if personal else None
        hit = c.repo.same(c.guild_id, name, address, owner)
        if (
            hit is not None
            and not personal
            and not c.can_see(hit)
            and norm(hit.address) != norm(address)
        ):
            hit = None  # a name alone never touches or reveals a place the asker may not know
        if hit is None:
            p = c.repo.create(
                c.guild_id,
                name=name,
                address=address,
                kind=kind,
                aliases=al,
                owner_id=owner,
                created_by=c.asker_id,
                notes=(notes or "")[:300],
                sources=[] if personal else [c.source()],
            )
            if personal:
                c.repo.claim_offer(c.guild_id, c.asker_id, address)  # never ask about it later
            return {"status": "saved", "place": self._view(p)}
        # an existing place
        if norm(hit.address) != norm(address):
            if hit.owner_id == c.asker_id or hit.created_by == c.asker_id or c.is_exec:
                hit.address = address
            else:
                return {
                    "error": "not_allowed",
                    "note": "a place with this name exists with another address; only the person "
                    "who saved it or an executive can change it",
                }
        if al:
            hit.aliases = sorted({*hit.aliases, *al})
        if notes and (hit.owner_id == c.asker_id or not hit.personal):
            hit.notes = notes[:300]
        if not personal:
            c.repo.add_source(hit, c.source())  # said again here: this audience may know it too
        c.repo.put(hit)
        return {"status": "updated", "place": self._view(hit)}

    def find_place(self, query: str) -> dict:
        """Look up a place by name, nickname or part of its address. Returns the address, a Google
        Maps link and (to its owner / creator / an executive) who can see it. Only places the
        asker, and everyone who will hear the answer, may know."""
        hits = self._visible(query)
        if not hits:
            return {"status": "not_found", "note": "no place that you may know matches"}
        hits.sort(key=lambda p: (p.owner_id != self.c.asker_id, -matches(p, query)))
        return {"count": len(hits), "places": [self._view(p) for p in hits[:5]]}

    def list_places(self, kind: str = "") -> dict:
        """List the places the asker (and everyone who will hear the answer) may know."""
        rows = [
            p
            for p in self.c.repo.list(self.c.guild_id)
            if self.c.can_see(p) and (not kind or p.kind == kind)
        ]
        out: dict = {"count": len(rows), "places": [self._view(p, False) for p in rows[:MAX_LIST]]}
        if not rows:
            out["note"] = "no places that you may know"
        return out

    def share_place(self, place: str, audience: str, people: list[str] | None = None) -> dict:
        """Change who may know a place. audience: owner (only me), superiors (people ranked above
        the owner), people (the named `people`), everyone (the whole company), channel (whoever
        reads this channel). Personal places: only the owner. Company places: only an executive
        or the person who saved it, and only channel or everyone."""
        c = self.c
        if audience not in SHARE_MODES:
            return {"error": f"audience must be one of {', '.join(SHARE_MODES)}"}
        p = self._own_or_company(place)
        if p is None:
            return {"status": "not_found"}
        if p.personal:
            if p.owner_id != c.asker_id:
                return {"error": "not_allowed", "note": "only the owner can share a personal place"}
            if audience == "owner":
                p.share_superiors = p.share_everyone = False
                p.share_people = []
                p.sources = [s for s in p.sources if s.get("disclosed_by") != p.owner_id]
            elif audience == "superiors":
                p.share_superiors = True
            elif audience == "everyone":
                p.share_everyone = True
            elif audience == "people":
                ids, bad = [], []
                for n in people or []:
                    uid = match_person(n, c.people)
                    (ids if uid else bad).append(uid or n)
                if bad or not ids:
                    return {"error": "cannot tell who these people are", "unresolved": bad}
                p.share_people = sorted({*p.share_people, *ids})
            else:  # channel
                if c.channel_id is None or c.audience is None:
                    return {"error": "say this in the channel to share with"}
                c.repo.add_source(p, c.source())
            c.repo.put(p)
            return {"status": "updated", "place": self._view(p)}
        if not (c.is_exec or p.created_by == c.asker_id):
            return {"error": "not_allowed", "note": "only an executive or who saved it"}
        if audience == "channel":
            if c.channel_id is None:
                return {"error": "say this in the channel to share with"}
            c.repo.add_source(p, c.source())
        elif audience == "everyone":
            c.repo.add_source(p, make_source(None, c.asker_id, c.now(), room=frozenset(c.ranks)))
        else:
            return {"error": "company places are shared by channel: use channel or everyone"}
        c.repo.put(p)
        return {"status": "updated", "place": self._view(p)}

    def forget_place(self, place: str) -> dict:
        """Forget a place. A personal place: its owner, any time. A company place: the person who
        saved it or an executive, after confirmation in a later message."""
        c = self.c
        p = self._own_or_company(place)
        if p is None:
            return {"status": "not_found"}
        if p.personal:
            if p.owner_id != c.asker_id:
                return {"error": "not_allowed", "note": "only the owner can forget it"}
        else:
            if not (c.is_exec or p.created_by == c.asker_id):
                return {"error": "not_allowed", "note": "only an executive or who saved it"}
            if c.confirm is not None and not c.confirm(
                ("forget_place", c.guild_id, c.asker_id, p.id)
            ):
                return {
                    "status": "needs_confirmation",
                    "place": p.name,
                    "instruction": "Ask the user to confirm forgetting this company place, then "
                    "call forget_place again with the same name in a later message.",
                }
        c.repo.delete(p)
        return {"status": "forgotten", "place": p.name}

    def set_place_secret(self, place: str, channels: list[str]) -> dict:
        """Executives only: make a company place secret to the given channels, i.e. reset who may
        know it to the readers of exactly these channels (plus people ranked above them)."""
        c = self.c
        if not c.is_exec:
            return {"error": "not_allowed", "note": "executives only"}
        p = self._own_or_company(place)
        if p is None or p.personal:
            return {"status": "not_found"}
        ids, bad = [], []
        for ref in channels or []:
            m = re.search(r"<#(\d+)>", ref or "")
            cid = (
                int(m[1])
                if m
                else next(
                    (
                        i
                        for i, n in c.channels.items()
                        if n.casefold() == (ref or "").strip("# ").casefold()
                    ),
                    0,
                )
            )
            (ids if cid in c.channels else bad).append(cid or ref)
        if bad or not ids:
            return {"error": "unknown channels", "unresolved": bad}
        p.sources = [make_source(i, c.asker_id, c.now()) for i in ids]
        c.repo.put(p)
        return {"status": "updated", "place": self._view(p)}


def place_for_event(ctx: PlacesCtx, query: str, invitee_ids: list[int]) -> dict:
    """Scheduling: the place a meeting is at. `ok` + `public` (everyone in the guild may know it,
    so it may go into the Discord event itself) only when every invitee may know it; otherwise
    `restricted` with the names who may not (the bot then asks) or `not_found`."""
    hits = [
        p
        for p in ctx.repo.search(ctx.guild_id, query)
        if may_know(p, ctx.asker_id, ctx.ranks, ctx.viewers_of)
    ]
    if not hits:
        return {"status": "not_found"}
    hits.sort(key=lambda p: (p.owner_id != ctx.asker_id, -matches(p, query)))
    p = hits[0]
    who = place_audience(p, ctx.ranks, ctx.viewers_of)
    blocked = sorted({i for i in invitee_ids if i not in who} - {ctx.asker_id})
    if blocked:
        return {
            "status": "restricted",
            "place": p.name,
            "cannot_know": [ctx.people.get(i, str(i)) for i in blocked],
        }
    return {
        "status": "ok",
        "name": p.name,
        "address": p.address,
        "maps_link": maps_link(p),
        "public": set(ctx.ranks) <= who,
    }


def build_place_tools(ctx: PlacesCtx) -> list:
    t = PlaceTools(ctx)
    fns = [
        t.remember_place,
        t.find_place,
        t.list_places,
        t.share_place,
        t.forget_place,
        t.set_place_secret,
    ]
    for f in fns:  # lets qa tell these from the coordinator tools
        f.__func__.salid_group = "places"
    return fns


PLACES_PROMPT = (
    "\nสถานที่สำคัญที่สลิดจำไว้ (ที่ตั้งสำนักงานใหญ่ สาขา โกดัง ลูกค้า สถานที่ประชุม และสถานที่ส่วนตัวของแต่ละคน): "
    "ใช้ find_place เมื่อมีคนถามว่าที่ไหน/ที่อยู่/ขอแผนที่ (ตอบที่อยู่ + maps_link) และ list_places เมื่อถามว่าจำที่ไหนไว้บ้าง. "
    "ความลับตามผู้ที่เห็นได้ถูกตัดสินในโค้ด: ถ้าไม่พบ ให้บอกว่าไม่พบข้อมูลที่ผู้ถามเห็นได้ ห้ามเดา ห้ามบอกว่ามีอยู่ถ้าไม่พบ. "
    "ถ้ามีคนบอกที่อยู่ของที่ทำงาน/สาขาและบอกให้จำ (เช่น “จำไว้นะ สาขาบางนาอยู่ที่ …”) เรียก remember_place(personal=false). "
    "สถานที่ส่วนตัว (เช่น บ้านของใคร) เก็บได้เฉพาะเมื่อเจ้าตัวขอให้จำให้ตัวเอง หรือสลิดถามแล้วเจ้าตัวตอบตกลง "
    "จึงเรียก remember_place(personal=true) ห้ามเก็บบ้านของคนอื่นหรือเก็บเองโดยไม่ได้ขอ; ค่าเริ่มต้นเห็นเฉพาะเจ้าตัว. "
    "ถ้าเจ้าตัวพิมพ์ที่อยู่ลงห้องสาธารณะ ให้เตือนสั้นๆ ว่าข้อความนั้นคนในห้องเห็นอยู่แล้ว คราวหน้าบอกสลิดทาง DM จะเป็นส่วนตัวกว่า. "
    "เจ้าของสั่งแชร์ได้ด้วย share_place (“ให้หัวหน้าเห็นได้”=superiors, “ให้ A เห็นได้”=people, “ทุกคนในบริษัท”=everyone, "
    "“ให้ห้องนี้เห็น”=channel, “เห็นคนเดียว”=owner) และลบด้วย forget_place (“ลืมที่อยู่บ้านผม”). "
    "ผู้บริหารตั้ง “ที่อยู่นี้เป็นความลับ” ได้ด้วย set_place_secret (ระบุห้องที่ให้รู้). "
    "ผลลัพธ์ restricted หรือสถานที่ที่ผู้ฟังบางคนไม่มีสิทธิ์รู้ จะไม่ถูกส่งมา: อย่าบอกที่อยู่นั้นในห้องนี้ แนะนำให้ถามทาง DM. "
    "นัดประชุมที่สถานที่ที่จำไว้ ให้ใส่ชื่อสถานที่ใน place ของ schedule_meeting (ระบบเช็กสิทธิ์ผู้ถูกเชิญเอง)."
)
