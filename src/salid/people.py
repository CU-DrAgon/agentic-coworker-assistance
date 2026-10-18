"""Person dossiers: what a company knows about WHO DOES WHAT (pure logic, no Discord, no model).

Salid listens to the work, not the people: for each member the DocStore kind `salid_people` (key =
user id) keeps a `Dossier` with their role, responsibilities, how they work, systems they use,
recurring duties, topics they know (expertise) and who they deal with. Every learned item carries
`Evidence` (the day, the channel and the message it came from), so a reader is shown an item only
when they could read at least one of its source channels (`evidence_visible`); items stated by an
executive have no channel and are open. Facts learned in a DM scope are never stored. A dossier is
kept forever: when someone leaves the status becomes `left` and the generated handover stays.

This module is the model, the merge rules (normalised-text dedupe, caps, eviction of the oldest
low-confidence entry), the expert ranking behind `who_to_ask`, and the company-map builders
(`build_org`, `build_who_to_ask`, `build_key_contacts`). `people_listen.py` fills dossiers,
`people_tools.py` answers questions, `people_bot.py` is the Discord glue.
"""

from __future__ import annotations

import difflib
import re
import threading
import unicodedata
from collections import Counter
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .company_brain import KeyContact, OrgDept, OrgPosition, PersonRef, WhoTopic, _grams
from .coo_store import DocStore
from .memory_access import is_dm_scope

PEOPLE = "salid_people"
STATUSES = ("active", "left")
MAX_FACTS = 40
MAX_TOPICS = 60
MAX_CONTACTS = 80
MAX_EVIDENCE = 5
MAX_TEXT = 200
MAX_HANDOVER = 12000
ROLE_RANK = {"": 0, "heard": 1, "discord": 2, "org": 3, "stated": 4}
EVIDENCE_KINDS = ("chat", "meeting", "task", "stated")
BOT_NAMES = frozenset({"salid", "สลิด"})  # the assistant itself: never a person, never a contact
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "how",
        "who",
        "what",
        "with",
        "about",
        "our",
        "do",
        "we",
        "to",
        "of",
        "a",
        "an",
        "in",
        "on",
        "ask",
        "is",
        "it",
        "me",
        "my",
        "you",
        "หรือ",
        "และ",
        "ของ",
        "ที่",
        "ใน",
        "กับ",
        "ให้",
        "ไป",
        "มา",
    }
)


# ------------------------------------------------------------------ model
class Evidence(BaseModel):
    day: str = ""
    channel_id: int = 0  # 0 = stated by an executive / the system: open to every reader
    message_id: int = 0
    kind: str = "chat"  # chat | meeting | task | stated


class Fact(BaseModel):
    text: str
    since: str = ""
    last_seen: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.5


class Topic(BaseModel):
    topic: str
    score: float = 1.0
    last_seen: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class Contact(BaseModel):
    who: str  # a name as written (external party) or the member's display name
    user_id: int = 0  # > 0: a colleague
    about: str = ""
    count: int = 1
    last_seen: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class Dossier(BaseModel):
    user_id: int
    name: str = ""
    role_title: str = ""
    role_source: str = ""  # "" | heard | discord | org | stated (a higher source wins)
    department: str = ""
    dept_source: str = ""  # same scale, for the department
    status: str = "active"
    left_at: str = ""
    responsibilities: list[Fact] = Field(default_factory=list)
    how_they_work: list[Fact] = Field(default_factory=list)
    expertise: list[Topic] = Field(default_factory=list)
    contacts: list[Contact] = Field(default_factory=list)
    systems: list[Fact] = Field(default_factory=list)
    recurring: list[Fact] = Field(default_factory=list)
    handover: str = ""
    handover_at: str = ""
    successor_id: int = 0
    updated_at: str = ""


class HeardContact(BaseModel):
    who: str = ""
    about: str = ""
    user_id: int = 0


class Heard(BaseModel):
    """What was learned about one person from one batch (already name-resolved and cleaned)."""

    did: list[str] = Field(default_factory=list)
    how: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    recurring: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    contacts: list[HeardContact] = Field(default_factory=list)
    role_hint: str = ""


# ------------------------------------------------------------------ text helpers
def clean(text: str, limit: int = MAX_TEXT) -> str:
    return " ".join(str(text or "").split())[:limit]


def norm(text: str) -> str:
    """Letters, digits and marks of the text, lower-cased and single-spaced."""
    out: list[str] = []
    for ch in (text or "").casefold():
        if unicodedata.category(ch)[0] in "LNM":
            out.append(ch)
        elif out and out[-1] != " ":
            out.append(" ")
    return "".join(out).strip()


def is_bot_name(name: str, extra: Collection[str] = ()) -> bool:
    """Is `name` the assistant itself (Salid / สลิด, or one of `extra`, e.g. the bot's display name)?"""
    n = norm(name)
    return bool(n) and (n in BOT_NAMES or n in {norm(x) for x in extra})


def same(a: str, b: str) -> bool:
    """Two phrases that say the same thing (equal, one inside the other, or nearly equal)."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if re.findall(r"\d+", na) != re.findall(r"\d+", nb):
        return False  # "invoice 12" and "invoice 13" are different things
    if min(len(na), len(nb)) >= 8 and (na in nb or nb in na):
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.88


def _words(text: str) -> set[str]:
    return {g for g in _grams(text) if g not in _STOP}


def topic_match(query: str, text: str) -> float:
    """How well `text` answers the question `query`: 3 equal, 2 one inside the other, else half a
    point per shared word / Thai trigram (at most 1.5), 0 when nothing is shared."""
    nq, nt = norm(query), norm(text)
    if not nq or not nt:
        return 0.0
    if nq == nt:
        return 3.0
    if (len(nq) >= 2 and nq in nt) or (len(nt) >= 2 and nt in nq):
        return 2.0
    common = _words(query) & _words(text)
    return min(len(common), 3) * 0.5


# ------------------------------------------------------------------ evidence and visibility
def evidence_visible(evidence: list[Evidence], readable: Collection[int], see_all: bool) -> bool:
    """May this reader see an item? No evidence or a stated one: yes; else they must be able to
    read the channel of at least one evidence entry (`see_all`: an executive asking alone).
    A DM scope never counts."""
    if not evidence:
        return True
    return any(
        e.channel_id == 0 or (see_all and e.channel_id > 0) or e.channel_id in readable
        for e in evidence
        if not is_dm_scope(e.channel_id)
    )


def publicly_sourced(evidence: list[Evidence], public: Collection[int]) -> bool:
    """Is the item stated or sourced from a channel the whole server can read?"""
    return not evidence or any(e.channel_id == 0 or e.channel_id in public for e in evidence)


def _add_evidence(lst: list[Evidence], ev: Evidence) -> bool:
    """Add `ev` (newest last, MAX_EVIDENCE kept). False when it was already there."""
    key = (ev.day, ev.channel_id, ev.message_id)
    if any((e.day, e.channel_id, e.message_id) == key for e in lst):
        return False
    lst.append(ev)
    del lst[:-MAX_EVIDENCE]
    return True


# ------------------------------------------------------------------ merge
def _evict(items: list, cap: int, key: Callable) -> None:
    while len(items) > cap:
        items.remove(min(items, key=key))


def merge_fact(items: list[Fact], text: str, ev: Evidence, day: str, conf: float = 0.5) -> int:
    """Add or bump one fact; 1 when something changed. A repeat in the same channel-day (the
    day's summary is re-read after every update) changes nothing."""
    text = clean(text)
    if len(norm(text)) < 3:
        return 0
    for f in items:
        if same(f.text, text):
            fresh = _add_evidence(f.evidence, ev)
            if not fresh and f.last_seen == day:
                return 0
            f.last_seen = max(f.last_seen, day)
            f.confidence = round(min(1.0, f.confidence + (0.1 if fresh else 0.0)), 2)
            return 1
    items.append(Fact(text=text, since=day, last_seen=day, evidence=[ev], confidence=conf))
    _evict(items, MAX_FACTS, lambda f: (f.confidence, f.last_seen))
    return 1


def merge_topic(items: list[Topic], topic: str, ev: Evidence, day: str, weight: float = 1.0) -> int:
    topic = clean(topic, 80)
    if len(norm(topic)) < 2:
        return 0
    for t in items:
        if same(t.topic, topic):
            fresh = _add_evidence(t.evidence, ev)
            if not fresh and t.last_seen == day:
                return 0
            t.last_seen = max(t.last_seen, day)
            t.score = round(min(1000.0, t.score + weight), 2)
            return 1
    items.append(Topic(topic=topic, score=weight, last_seen=day, evidence=[ev]))
    _evict(items, MAX_TOPICS, lambda t: (t.score, t.last_seen))
    return 1


def merge_contact(items: list[Contact], c: HeardContact, ev: Evidence, day: str) -> int:
    who = clean(c.who, 80)
    if not who or is_bot_name(who):
        return 0  # talking to Salid is not a work contact
    about = clean(c.about, 120)
    for x in items:
        if (c.user_id and x.user_id == c.user_id) or (not c.user_id and same(x.who, who)):
            fresh = _add_evidence(x.evidence, ev)
            if not fresh and x.last_seen == day:
                return 0
            x.last_seen = max(x.last_seen, day)
            x.count += 1
            if about and not any(same(p, about) for p in x.about.split("; ") if p):
                x.about = clean(f"{x.about}; {about}" if x.about else about, 200)
            return 1
    items.append(
        Contact(who=who, user_id=c.user_id, about=about, count=1, last_seen=day, evidence=[ev])
    )
    _evict(items, MAX_CONTACTS, lambda x: (x.count, x.last_seen))
    return 1


def set_role(d: Dossier, title: str, source: str, department: str | None = None) -> bool:
    """Record a role and / or department. A lower source never overwrites a higher one."""
    changed = False
    rank = ROLE_RANK.get(source, 0)
    title = clean(title, 80)
    if (
        title
        and rank >= ROLE_RANK.get(d.role_source, 0)
        and (title != d.role_title or source != d.role_source)
    ):
        d.role_title, d.role_source, changed = title, source, True
    dept = clean(department or "", 60)
    if (
        dept
        and rank >= ROLE_RANK.get(d.dept_source, 0)
        and (dept != d.department or source != d.dept_source)
    ):
        d.department, d.dept_source, changed = dept, source, True
    return changed


def merge_extraction(
    d: Dossier,
    heard: Heard,
    day: str,
    channel_id: int,
    *,
    kind: str = "chat",
    message_id: int = 0,
    name: str = "",
) -> int:
    """Merge one batch's learning into the dossier. Returns how many items changed. A DM scope
    teaches nothing."""
    if is_dm_scope(channel_id):
        return 0
    ev = Evidence(day=day, channel_id=channel_id, message_id=message_id, kind=kind)
    n = 0
    for text in heard.did:
        n += merge_fact(d.responsibilities, text, ev.model_copy(), day)
    for text in heard.how:
        n += merge_fact(d.how_they_work, text, ev.model_copy(), day)
    for text in heard.tools:
        n += merge_fact(d.systems, text, ev.model_copy(), day)
    for text in heard.recurring:
        n += merge_fact(d.recurring, text, ev.model_copy(), day)
    for text in heard.topics:
        n += merge_topic(d.expertise, text, ev.model_copy(), day)
    for c in heard.contacts:
        n += merge_contact(d.contacts, c, ev.model_copy(), day)
    if heard.role_hint and set_role(d, heard.role_hint, "heard"):
        n += 1
    if name and name != d.name:
        d.name, n = name, n + 1
    return n


# ------------------------------------------------------------------ repository
class PeopleRepo:
    def __init__(self, docs: DocStore, clock: Callable[[], datetime] | None = None):
        self.docs = docs
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()  # listener thread vs the bot loop: read-modify-write
        # (user_id, name) -> True for the assistant itself / any bot (set by the Discord glue)
        self.bot_check: Callable[[int, str], bool] | None = None

    def is_bot(self, user_id: int, name: str = "") -> bool:
        if is_bot_name(name):
            return True
        try:
            return bool(self.bot_check and self.bot_check(user_id, name))
        except Exception:
            return False

    def now(self) -> str:
        return self._clock().isoformat(timespec="seconds")

    @staticmethod
    def _load(d: dict | None) -> Dossier | None:
        if not d:
            return None
        try:
            return Dossier(**{k: v for k, v in d.items() if k in Dossier.model_fields})
        except ValueError:
            return None

    def get(self, guild_id: int, user_id: int) -> Dossier | None:
        return self._load(self.docs.get(PEOPLE, guild_id, str(user_id)))

    def put(self, guild_id: int, d: Dossier) -> None:
        d.updated_at = self.now()
        self.docs.put(PEOPLE, guild_id, str(d.user_id), d.model_dump())

    def delete(self, guild_id: int, user_id: int) -> None:
        self.docs.delete(PEOPLE, guild_id, str(user_id))

    def list(self, guild_id: int, include_bots: bool = False) -> list[Dossier]:
        """Every dossier, active first. Bots (and the assistant itself) are left out unless asked."""
        out = [self._load(x) for x in self.docs.find(PEOPLE, guild_id)]
        return sorted(
            (d for d in out if d and (include_bots or not self.is_bot(d.user_id, d.name))),
            key=lambda d: (d.status != "active", d.name.casefold()),
        )

    def update(self, guild_id: int, user_id: int, name: str, fn: Callable[[Dossier], object]):
        """Read (or start) the dossier, apply `fn`, save it when `fn` returns truthy. Returns
        what `fn` returned. Bots and the assistant itself never get a dossier (returns None)."""
        if self.is_bot(user_id, name):
            return None
        with self._lock:
            d = self.get(guild_id, user_id) or Dossier(user_id=user_id, name=name)
            res = fn(d)
            if res:
                if name and not d.name:
                    d.name = name
                self.put(guild_id, d)
            return res

    def find(self, guild_id: int, who: str) -> list[Dossier]:
        """Dossiers a name, mention or id points at: an id / mention, else exact name, else every
        name that contains it (or is contained)."""
        raw = (who or "").strip()
        m = re.search(r"<@!?(\d+)>", raw)
        if m or raw.isdigit():
            d = self.get(guild_id, int(m[1] if m else raw))
            return [d] if d else []
        low = raw.casefold().lstrip("@")
        if not low:
            return []
        rows = self.list(guild_id)
        exact = [d for d in rows if d.name.casefold() == low]
        if exact:
            return exact
        return [
            d
            for d in rows
            if len(low) >= 2
            and (low in d.name.casefold() or (len(d.name) >= 2 and d.name.casefold() in low))
        ]


# ------------------------------------------------------------------ who to ask
@dataclass
class Expert:
    user_id: int
    name: str
    status: str
    score: float
    last_seen: str = ""
    why: list[dict] = field(default_factory=list)  # {"what", "text", "day", "channel_id"}
    successor_id: int = 0


def _best_ev(ev: list[Evidence], readable, see_all) -> Evidence | None:
    shown = [e for e in ev if evidence_visible([e], readable, see_all)]
    return max(shown, key=lambda e: e.day, default=None)


def _reason(
    query: str, what: str, text: str, ev: list[Evidence], weight: float, readable, see_all: bool
) -> dict | None:
    m = topic_match(query, text)
    if m <= 0 or not evidence_visible(ev, readable, see_all):
        return None
    best = _best_ev(ev, readable, see_all)
    return {
        "what": what,
        "text": text,
        "day": best.day if best else "",
        "channel_id": best.channel_id if best else 0,
        "_w": m * weight,
    }


def rank_experts(
    dossiers: Iterable[Dossier],
    query: str,
    *,
    readable: Collection[int] = (),
    see_all: bool = False,
    limit: int = 6,
) -> list[Expert]:
    """People who know about `query`, best first, each with the visible reasons. Left people are
    included (status "left"). Only items the reader may see count, so a hidden item neither
    ranks nor appears."""
    out: list[Expert] = []
    for d in dossiers:
        rows: list[dict] = []
        for t in d.expertise:
            rows.append(
                _reason(
                    query,
                    "knows",
                    t.topic,
                    t.evidence,
                    1.0 + min(t.score, 20) / 10,
                    readable,
                    see_all,
                )
            )
        for what, items in (
            ("does", d.responsibilities),
            ("recurring", d.recurring),
            ("uses", d.systems),
            ("works like", d.how_they_work),
        ):
            for f in items:
                rows.append(
                    _reason(
                        query,
                        what,
                        f.text,
                        f.evidence,
                        0.6 * (0.5 + f.confidence),
                        readable,
                        see_all,
                    )
                )
        for text in (d.role_title, d.department):
            if text:
                rows.append(_reason(query, "role", text, [], 0.8, readable, see_all))
        why = sorted((r for r in rows if r), key=lambda r: -r["_w"])
        score = sum(w["_w"] for w in why)
        if score <= 0:
            continue
        seen = max([d.updated_at[:10], *(w["day"] for w in why)])
        for w in why:
            w.pop("_w", None)
        out.append(
            Expert(d.user_id, d.name, d.status, round(score, 2), seen, why[:4], d.successor_id)
        )
    out.sort(key=lambda e: (e.status == "left", -e.score, e.last_seen))
    return out[:limit]


# ------------------------------------------------------------------ company map
@dataclass
class RosterEntry:
    user_id: int
    name: str
    department: str = ""  # from the org member / the dossier
    title: str = ""  # the dossier's role, else the top Discord role
    rank: int = 0  # hierarchy rank (higher = more senior)


def department_label(raw: str) -> str:
    """Org member departments are short codes (tech, business, finance, other)."""
    raw = clean(raw, 60)
    return raw[:1].upper() + raw[1:] if raw.islower() and raw.isascii() else raw


def build_org(roster: Iterable[RosterEntry]) -> list[OrgDept]:
    """Departments -> positions -> people. A department's head is its most senior member (ties:
    the first by name); unassigned people go into a department with an empty name."""
    depts: dict[str, dict[str, list[RosterEntry]]] = {}
    for r in roster:
        dept = department_label(r.department)
        depts.setdefault(dept, {}).setdefault(clean(r.title, 80), []).append(r)
    out: list[OrgDept] = []
    for dept, positions in depts.items():
        everyone = sorted(
            (r for rs in positions.values() for r in rs), key=lambda r: (-r.rank, r.name.casefold())
        )
        head = everyone[0] if everyone and everyone[0].rank > 0 else None
        out.append(
            OrgDept(
                name=dept,
                head=PersonRef(user_id=head.user_id, name=head.name) if head else None,
                positions=[
                    OrgPosition(
                        title=title,
                        people=[
                            PersonRef(user_id=r.user_id, name=r.name)
                            for r in sorted(rs, key=lambda r: (-r.rank, r.name.casefold()))
                        ],
                    )
                    for title, rs in sorted(positions.items(), key=lambda kv: (kv[0] == "", kv[0]))
                ],
            )
        )
    out.sort(key=lambda d: (d.name == "", d.name.casefold()))
    return out


def build_who_to_ask(
    dossiers: Iterable[Dossier], public: Collection[int], *, max_topics: int = 40
) -> list[WhoTopic]:
    """topic -> the best (active) people, from the topics whose evidence the whole server can
    read."""
    acc: dict[str, dict] = {}
    for d in dossiers:
        if d.status != "active":
            continue
        for t in d.expertise:
            if not publicly_sourced(t.evidence, public):
                continue
            key = norm(t.topic)
            if not key:
                continue
            slot = acc.setdefault(key, {"topic": t.topic, "people": {}})
            slot["people"][d.user_id] = (t.score, d.name)
    rows = []
    for slot in acc.values():
        ranked = sorted(slot["people"].items(), key=lambda kv: -kv[1][0])
        rows.append(
            (
                sum(s for s, _ in slot["people"].values()),
                WhoTopic(
                    topic=slot["topic"],
                    people=[PersonRef(user_id=u, name=n) for u, (_, n) in ranked[:3]],
                ),
            )
        )
    rows.sort(key=lambda r: -r[0])
    return [w for _, w in rows[:max_topics]]


def build_key_contacts(
    dossiers: Iterable[Dossier], public: Collection[int], *, limit: int = 20
) -> list[KeyContact]:
    """External parties (contacts that are not members) people deal with, most frequent first."""
    acc: dict[str, KeyContact] = {}
    for d in dossiers:
        if d.status != "active":
            continue
        for c in d.contacts:
            if c.user_id or is_bot_name(c.who) or not publicly_sourced(c.evidence, public):
                continue
            key = norm(c.who)
            if not key:
                continue
            k = acc.get(key)
            if k is None:
                acc[key] = KeyContact(
                    name=c.who, context=c.about, count=c.count, last_seen=c.last_seen
                )
            else:
                k.count += c.count
                k.last_seen = max(k.last_seen, c.last_seen)
                if c.about and not k.context:
                    k.context = c.about
    return sorted(acc.values(), key=lambda k: (-k.count, k.name.casefold()))[:limit]


def map_from_dossiers(
    dossiers: Iterable[Dossier], is_private: Callable[[int], bool] = lambda cid: False
) -> tuple[list[OrgDept], list[WhoTopic], list[KeyContact]]:
    """The company map from the dossiers alone (no Discord): org from role titles / departments,
    who-to-ask and key contacts treating every evidence channel as readable except DM scopes and
    the ones `is_private` marks. For `scripts/salid_people_backfill.py --rebuild-map`."""
    ds = [d for d in dossiers if d.status == "active"]
    seen: set[int] = set()
    for d in ds:
        for items in (d.responsibilities, d.how_they_work, d.systems, d.recurring, d.expertise):
            seen.update(e.channel_id for i in items for e in i.evidence)
        seen.update(e.channel_id for c in d.contacts for e in c.evidence)
    public = {c for c in seen if c > 0 and not is_dm_scope(c) and not is_private(c)}
    roster = [RosterEntry(d.user_id, d.name, d.department, d.role_title) for d in ds]
    return build_org(roster), build_who_to_ask(ds, public), build_key_contacts(ds, public)


# ------------------------------------------------------------------ who works with whom
@dataclass
class GraphNode:
    key: str  # "u<id>" for a member, "x:<name>" for an outside party
    name: str
    external: bool = False
    status: str = "active"
    department: str = ""
    neighbours: set[str] = field(default_factory=set)
    in_weight: int = 0  # how often others name this node
    out_weight: int = 0
    topics: Counter = field(default_factory=Counter)  # what they are contacted about

    @property
    def degree(self) -> int:
        return len(self.neighbours)


@dataclass
class GraphEdge:
    a: str
    b: str  # the pair, a < b
    weight: int = 0
    topics: Counter = field(default_factory=Counter)


@dataclass
class ContactGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[tuple[str, str], GraphEdge] = field(default_factory=dict)
    # directed: (from, to) -> (weight, topics): who names whom
    directed: dict[tuple[str, str], int] = field(default_factory=dict)

    def hubs(self, limit: int = 5) -> list[GraphNode]:
        """Most named by others (weighted in-degree)."""
        rows = [n for n in self.nodes.values() if n.in_weight > 0]
        rows.sort(key=lambda n: (-n.in_weight, -n.degree, n.name.casefold()))
        return rows[:limit]

    def bridges(self, limit: int = 5) -> list[tuple[GraphNode, list[str]]]:
        """People whose contacts span the most departments (at least two)."""
        out: list[tuple[GraphNode, list[str]]] = []
        for n in self.nodes.values():
            if n.external:
                continue
            depts = {
                self.nodes[k].department
                for k in n.neighbours
                if k in self.nodes and not self.nodes[k].external
            } - {""}
            if len(depts) >= 2:
                out.append((n, sorted(depts)))
        out.sort(key=lambda r: (-len(r[1]), -r[0].degree, r[0].name.casefold()))
        return out[:limit]

    def pairs(self, limit: int = 10) -> list[GraphEdge]:
        rows = sorted(
            self.edges.values(),
            key=lambda e: (
                -e.weight,
                self.nodes[e.a].name.casefold(),
                self.nodes[e.b].name.casefold(),
            ),
        )
        return rows[:limit]

    def around(self, key: str) -> list[tuple[GraphNode, GraphEdge]]:
        """Everybody `key` is connected to, strongest first."""
        out = []
        for k in self.nodes[key].neighbours:
            e = self.edges[(min(key, k), max(key, k))]
            out.append((self.nodes[k], e))
        out.sort(key=lambda r: (-r[1].weight, r[0].name.casefold()))
        return out

    def also_talk_to(self, key: str, limit: int = 5) -> list[tuple[GraphNode, list[str]]]:
        """Other members who deal with the same parties `key` deals with (shared neighbours)."""
        mine = self.nodes[key].neighbours
        out = []
        for n in self.nodes.values():
            if n.key == key or n.external:
                continue
            shared = sorted(self.nodes[k].name for k in (n.neighbours & mine) if k != n.key)
            if shared:
                out.append((n, shared))
        out.sort(key=lambda r: (-len(r[1]), r[0].name.casefold()))
        return out[:limit]


def _about_parts(about: str) -> list[str]:
    return [p.strip() for p in (about or "").split(";") if p.strip()]


def contact_graph(
    dossiers: Iterable[Dossier], readable: Collection[int], see_all: bool
) -> ContactGraph:
    """Who deals with whom, from the contacts in the dossiers. Nodes are members plus outside
    parties; an edge's weight is how many times it was seen, its topics the `about` strings. Only
    contacts the reader may see (`evidence_visible`: a DM never counts); the assistant itself is
    never a node."""
    g = ContactGraph()
    ds = [d for d in dossiers if not is_bot_name(d.name)]
    dept_of = {d.user_id: department_label(d.department) for d in ds}
    names = {d.user_id: d.name for d in ds}

    def node(uid: int, name: str, status: str = "active") -> GraphNode:
        key = f"u{uid}" if uid else f"x:{norm(name)}"
        n = g.nodes.get(key)
        if n is None:
            n = g.nodes[key] = GraphNode(
                key, names.get(uid) or name, external=not uid,
                department=dept_of.get(uid, ""), status=status,
            )  # fmt: skip
        return n

    for d in ds:
        for c in d.contacts:
            if is_bot_name(c.who) or c.user_id == d.user_id or not norm(c.who):
                continue
            if not evidence_visible(c.evidence, readable, see_all):
                continue
            a = node(d.user_id, d.name, d.status)
            a.status = d.status
            b = node(c.user_id, c.who)
            w = max(1, c.count)
            a.neighbours.add(b.key)
            b.neighbours.add(a.key)
            a.out_weight += w
            b.in_weight += w
            g.directed[(a.key, b.key)] = g.directed.get((a.key, b.key), 0) + w
            e = g.edges.setdefault(
                (min(a.key, b.key), max(a.key, b.key)),
                GraphEdge(min(a.key, b.key), max(a.key, b.key)),
            )
            e.weight += w
            for part in _about_parts(c.about):
                e.topics[part] += 1
                b.topics[part] += 1
    return g


# ------------------------------------------------------------------ views
def view_items(
    items: list, readable: Collection[int], see_all: bool, limit: int = 12, text_of=None
) -> list[dict]:
    """Visible items as small dicts (newest first), the source of each as day + channel id."""
    shown = [i for i in items if evidence_visible(i.evidence, readable, see_all)]
    shown.sort(key=lambda i: i.last_seen, reverse=True)
    out = []
    for i in shown[:limit]:
        best = _best_ev(i.evidence, readable, see_all)
        row = {"text": text_of(i) if text_of else i.text, "last_seen": i.last_seen}
        if best is not None and best.channel_id:
            row["source"] = {"day": best.day, "channel_id": best.channel_id, "kind": best.kind}
        out.append(row)
    return out
