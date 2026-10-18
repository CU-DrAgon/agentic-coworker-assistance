"""Salid learns how the company works by listening (feeds the person dossiers, `people.py`).

`PeopleListener.on_summary` is chained after `ProjectListener.on_summary` (the memory scheduler's
hook, see `coo._on_summary_all`). For each batch of new messages of a channel it makes ONE
cheap-model call: for each person who spoke, what they did, HOW they work, who they contacted,
which tools they used, which topics they know, their recurring duties and a role hint. Everything
the model returns is checked in code: unknown people are dropped, a contact that is a member is
matched by id, unsafe or sensitive text is dropped, lists are capped, and each learned item is
stored with its evidence (day, channel, message). Nothing is learned from a DM scope or from a
member's own private Salid channel (`excluded`). Meeting channels additionally feed from the
summary itself: owners of action items become responsibilities and the people of each topic gain
that expertise. A finished task adds "delivered X"; a project adds "owns project X".

Everything here is blocking (the summary's worker thread). It never raises into the caller.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable

from google.genai import types
from pydantic import BaseModel, Field

from .company_brain import injection_reason
from .llm import generate
from .memory_access import is_dm_scope
from .memory_store import MemoryMessage
from .memory_summary import format_lines
from .people import (
    Evidence,
    Heard,
    HeardContact,
    PeopleRepo,
    clean,
    is_bot_name,
    merge_extraction,
    merge_fact,
    merge_topic,
)
from .tasks import MAX_LINES_FOR_CHECK, match_person

logger = logging.getLogger(__name__)

MAX_PEOPLE = 8
MAX_ITEMS = 6
ITEM_CHARS = 160
MAX_SPEAKERS_IN_PROMPT = 12
# phone / bank / id numbers, e-mail addresses and credential talk never become knowledge
_SENSITIVE = re.compile(
    r"\d[\d\s-]{8,}\d|[\w.+-]+@[\w-]+\.[\w.]+|password|passcode|api[\s_-]*key|secret|token"
    r"|รหัสผ่าน|พาสเวิร์ด|ความลับ|โทเค็น",
    re.I,
)


class _Contact(BaseModel):
    who: str = ""
    about: str = ""


class _Person(BaseModel):
    user: str = ""
    did: list[str] = Field(default_factory=list)
    how: list[str] = Field(default_factory=list)
    contacted: list[_Contact] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    recurring: list[str] = Field(default_factory=list)
    role_hint: str = ""


class _Batch(BaseModel):
    people: list[_Person] = Field(default_factory=list)


LISTEN_PROMPT = """You build a company's institutional memory: who does what, and how work gets done. \
Nobody is talking to you; you only read a team chat and note durable WORK knowledge about the people \
who spoke.
Day of the messages: {day}

People who spoke (use these names exactly): {speakers}

Messages (HH:MM name: text):
{messages}

Return JSON {{"people": [{{"user", "did", "how", "contacted", "tools", "topics", "recurring", \
"role_hint"}}]}}, one entry per person for whom the messages SHOW something, where:
- "did": what they handled or are responsible for (short phrases, e.g. "prepares the monthly payroll").
- "how": HOW they solve problems or work (methods, habits, steps they follow, e.g. "checks the bank \
statement before approving").
- "contacted": people or outside parties they deal with, [{{"who": name as written, "about": \
what for}}].
- "tools": systems, apps or accounts they use (names only, never passwords, keys or numbers).
- "topics": subjects they know well enough that others could ask them.
- "recurring": duties that repeat (weekly report, month-end close, daily stock check).
- "role_hint": their job title or role ONLY if stated or obvious, else "".
Rules: only what the messages clearly state or show; no guesses; skip small talk, jokes, moods, \
opinions and anything personal (health, family, pay, private life). Each item at most 12 words, in \
the language of the messages. At most {limit} items per list and {people} people. Most batches have \
little to note: an empty list is a normal answer."""


def _ok(text: str) -> str:
    """The cleaned item, or "" when it is empty, unsafe or sensitive."""
    s = clean(text, ITEM_CHARS)
    if len(s) < 3 or injection_reason(s) or _SENSITIVE.search(s):
        return ""
    return s


def _items(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        s = _ok(v)
        if s and s not in out:
            out.append(s)
    return out[:MAX_ITEMS]


class PeopleListener:
    def __init__(
        self,
        repo: PeopleRepo,
        settings,
        llm,
        *,
        people_of: Callable[[int], dict[int, str]] = lambda gid: {},
        kind_of: Callable[[int, int], str] = lambda gid, cid: "summary",
        enabled: Callable[[int], bool] = lambda gid: True,
        excluded: Callable[[int, int], bool] = lambda gid, cid: False,
    ):
        self.repo, self.s, self.llm = repo, settings, llm
        self.people_of, self.kind_of = people_of, kind_of
        self.enabled, self.excluded = enabled, excluded

    # ------------------------------------------------------------------ hook
    def on_summary(
        self,
        guild_id: int,
        channel_id: int,
        day: str,
        summary: dict,
        messages: list[MemoryMessage],
    ) -> int:
        """Never raises. Returns how many dossier items were added or refreshed."""
        try:
            if not self.enabled(guild_id) or is_dm_scope(channel_id):
                return 0
            if self.excluded(guild_id, channel_id):
                return 0
            return self.learn(guild_id, channel_id, day, summary, messages)
        except Exception:
            logger.warning("people listening failed", exc_info=True)
            return 0

    def learn(
        self, guild_id: int, channel_id: int, day: str, summary: dict, messages: list[MemoryMessage]
    ) -> int:
        lines = [
            m
            for m in messages
            if m.kind in ("chat", "voice") and m.text.strip() and not is_dm_scope(m.channel_id)
        ]
        lines = lines[-MAX_LINES_FOR_CHECK:]
        people = {
            u: n for u, n in self.people_of(guild_id).items() if not self.repo.is_bot(u, n)
        }  # the assistant (and any bot) is never a person
        kind = "meeting" if self.kind_of(guild_id, channel_id) == "meeting" else "chat"
        n = 0
        if lines:
            n += self._from_chat(guild_id, channel_id, day, kind, lines, people)
        if kind == "meeting" and summary:
            n += self._from_meeting(guild_id, channel_id, day, summary, people)
        if n:
            logger.info("people: %d items learned in guild=%s channel=%s", n, guild_id, channel_id)
        return n

    # ------------------------------------------------------------------ chat
    def _speakers(self, lines: list[MemoryMessage], people: dict[int, str]) -> dict[int, str]:
        spoke: dict[int, int] = {}
        for m in lines:
            if m.author_id in people:
                spoke[m.author_id] = spoke.get(m.author_id, 0) + 1
        top = sorted(spoke, key=lambda u: -spoke[u])[:MAX_SPEAKERS_IN_PROMPT]
        return {u: people[u] for u in top}

    def _from_chat(
        self,
        guild_id: int,
        channel_id: int,
        day: str,
        kind: str,
        lines: list[MemoryMessage],
        people: dict[int, str],
    ) -> int:
        speakers = self._speakers(lines, people)
        if not speakers:
            return 0
        heard = self._ask(day, speakers, lines)
        n = 0
        for p in heard[:MAX_PEOPLE]:
            uid = match_person(p.user, speakers)
            if not uid:
                continue
            h = self._clean(p, uid, people)
            if not (
                h.did or h.how or h.tools or h.topics or h.recurring or h.contacts or h.role_hint
            ):
                continue
            mine = [m for m in lines if m.author_id == uid]
            anchor = mine[-1].message_id if mine else 0
            name = speakers[uid]
            n += (
                self.repo.update(
                    guild_id,
                    uid,
                    name,
                    lambda d, h=h, a=anchor, name=name: merge_extraction(
                        d, h, day, channel_id, kind=kind, message_id=a, name=name
                    ),
                )
                or 0
            )
        return n

    def _ask(self, day: str, speakers: dict[int, str], lines: list[MemoryMessage]) -> list[_Person]:
        prompt = LISTEN_PROMPT.format(
            day=day,
            speakers=", ".join(speakers.values()),
            messages=format_lines(lines, self.s.zone),
            limit=MAX_ITEMS,
            people=MAX_PEOPLE,
        )
        try:
            resp = generate(
                self.llm,
                self.s.gate_model,
                prompt,
                types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=2000,
                    response_mime_type="application/json",
                    response_schema=_Batch,
                ),
            )
            return _Batch.model_validate_json(resp.text).people
        except Exception:
            logger.warning("people listening call failed", exc_info=True)
            return []

    def _clean(self, p: _Person, uid: int, people: dict[int, str]) -> Heard:
        contacts: list[HeardContact] = []
        for c in p.contacted[:MAX_ITEMS]:
            who = clean(c.who, 80)
            about = _ok(c.about) if c.about else ""
            if not who or injection_reason(who) or _SENSITIVE.search(who):
                continue
            if is_bot_name(who):
                continue  # talking to Salid is not a work contact
            other = match_person(who, people)
            if other == uid:
                continue
            if other:
                contacts.append(HeardContact(who=people[other], user_id=other, about=about))
            else:
                contacts.append(HeardContact(who=who, about=about))
        role = _ok(p.role_hint)
        return Heard(
            did=_items(p.did),
            how=_items(p.how),
            tools=_items(p.tools),
            recurring=_items(p.recurring),
            topics=_items(p.topics),
            contacts=contacts,
            role_hint=role[:80],
        )

    # ------------------------------------------------------------------ meetings
    def _from_meeting(
        self, guild_id: int, channel_id: int, day: str, summary: dict, people: dict[int, str]
    ) -> int:
        """No model: owners of action items -> responsibilities, the people of a topic ->
        that expertise. Names are matched against the members."""
        ev = Evidence(day=day, channel_id=channel_id, kind="meeting")
        n = 0
        for a in summary.get("action_items") or []:
            what = _ok(a.get("what", ""))
            uid = match_person(str(a.get("who", "")), people)
            if what and uid:
                n += (
                    self.repo.update(
                        guild_id,
                        uid,
                        people[uid],
                        lambda d, what=what: merge_fact(
                            d.responsibilities, what, ev.model_copy(), day
                        ),
                    )
                    or 0
                )
        for tp in summary.get("topics") or []:
            title = _ok(tp.get("title", ""))
            if not title:
                continue
            for who in tp.get("people") or []:
                uid = match_person(str(who), people)
                if uid:
                    n += (
                        self.repo.update(
                            guild_id,
                            uid,
                            people[uid],
                            lambda d, title=title: merge_topic(
                                d.expertise, title, ev.model_copy(), day
                            ),
                        )
                        or 0
                    )
        return n

    # ------------------------------------------------------------------ tasks, projects
    def on_task_done(self, task) -> None:
        """A task was finished: its assignee delivered it. Never raises."""
        try:
            gid, uid = task.guild_id, task.assignee_id
            if not uid or not self.enabled(gid) or task.confirm == "declined":
                return  # "not a task" answers are marked done + declined: nothing was delivered
            title = _ok(task.title)
            if not title:
                return
            day = (task.done_at or task.updated_at or "")[:10]
            cid = (
                task.source_channel_id
                if task.source_channel_id and not is_dm_scope(task.source_channel_id)
                else 0
            )
            if cid and self.excluded(gid, cid):
                cid = 0
            ev = Evidence(day=day, channel_id=cid, kind="task")
            name = self.people_of(gid).get(uid) or task.assignee_name
            self.repo.update(
                gid,
                uid,
                name,
                lambda d: merge_fact(d.responsibilities, f"Delivered: {title}", ev, day, 0.6),
            )
        except Exception:
            logger.warning("people: task feed failed", exc_info=True)

    def refresh_projects(self, guild_id: int, projects: Iterable) -> int:
        """Owners of live projects: "Owns project X" (stated evidence: project ownership is
        recorded by the team, not heard in a private room)."""
        n = 0
        for p in projects:
            try:
                if p.status in ("completed", "cancelled") or getattr(p, "deleted", False):
                    continue
                uid = int(p.owner.discord_user_id or 0)
                name = _ok(p.name)
                if not uid or not name:
                    continue
                day = (p.updated_at or "")[:10]
                ev = Evidence(day=day, channel_id=0, kind="task")
                n += (
                    self.repo.update(
                        guild_id,
                        uid,
                        p.owner.name,
                        lambda d, name=name, day=day, ev=ev: merge_fact(
                            d.responsibilities, f"Owns project: {name}", ev.model_copy(), day, 0.7
                        ),
                    )
                    or 0
                )
            except Exception:
                logger.warning("people: project feed failed", exc_info=True)
        return n


# ------------------------------------------------------------------ backfill
BACKFILL_BATCH = 40


def backfill(
    listener: PeopleListener,
    store,
    guild_id: int,
    today,
    days: int,
    *,
    apply: bool = False,
) -> dict:
    """Run the learning pass over the stored days of a guild (`scripts/salid_people_backfill.py`).
    Dry run (`apply=False`) only counts. Returns counts, never message text."""
    from datetime import timedelta

    from .memory_store import date_range

    counts = {
        "days": 0,
        "channel_days": 0,
        "messages": 0,
        "skipped_dm": 0,
        "skipped_private": 0,
        "batches": 0,
        "items_learned": 0,
    }
    voice: set[int] = set()
    keep = listener.kind_of
    listener.kind_of = lambda gid, cid: "meeting" if cid in voice else "summary"
    try:
        start = (today - timedelta(days=days)).isoformat()
        for day in date_range(start, today.isoformat()):
            channels = store.day_channels(guild_id, day)
            if channels:
                counts["days"] += 1
            for cid in channels:
                if is_dm_scope(cid):
                    counts["skipped_dm"] += 1
                    continue
                if listener.excluded(guild_id, cid):
                    counts["skipped_private"] += 1
                    continue
                msgs = [
                    m
                    for m in store.messages_after(guild_id, cid, day, None, 5000)
                    if m.kind in ("chat", "voice") and m.text.strip()
                ]
                if not msgs:
                    continue
                counts["channel_days"] += 1
                counts["messages"] += len(msgs)
                if any(m.kind == "voice" for m in msgs):
                    voice.add(cid)
                for i in range(0, len(msgs), BACKFILL_BATCH):
                    counts["batches"] += 1
                    if not apply:
                        continue
                    got = store.get_summary(guild_id, cid, day)
                    summary = (got.summary if got and i == 0 else {}) or {}
                    counts["items_learned"] += listener.on_summary(
                        guild_id, cid, day, summary, msgs[i : i + BACKFILL_BATCH]
                    )
    finally:
        listener.kind_of = keep
    return counts
