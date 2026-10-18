"""People on the company's platforms: everyone Salid has SEEN, not only those with an account.

The command center's Members list only knows people who signed in. A company also wants to
know who is *there*: everyone who talks on the connected Discord server (later Slack, LINE).
This module builds that directory from what the bot already stores, with no new bot code:

* the team memory (`salid_messages`: author id + name, channel, day),
* the task ledger (`salid_tasks`: assignee id + name),
* the bot's own profiles (`salid_profiles`: nickname).

Every person gets a "floating" profile (`Person`, DocStore kind `salid_people`, guild = the
workspace memory id, key `discord:<user id>`): what an admin writes about them (nickname,
title, department, what they do) and, separately, what Salid OBSERVED from the conversations
(`Observed`: their apparent role, what they handle, who they work with, how they seem; the
cheap model, one call per person, only on request or when stale). A floating profile is linked
to a Salid account (`email`) as soon as a member verifies that Discord account, or by an admin.

Nothing here imports discord: ids, names and text only (the parity rule).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Collection
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from .coo_store import DocStore
from .i18n import lang_name

logger = logging.getLogger(__name__)

KIND = "salid_people_directory"  # NOT salid_people: that is the work-dossier store (people.py)
SCAN_DAYS = 90  # how far back a discovery scan looks
OBSERVE_LINES = 80  # the person's own recent messages the model reads
OBSERVE_STALE_H = 24  # an observation older than this is refreshed on request
BOT_NAMES = {"salid", "สลิด"}
PROFILE_LIMITS = {"nickname": 1000, "title": 1000, "department": 1000, "about": 50000}


class Seen(BaseModel):
    messages: int = 0
    first_seen: str = ""  # YYYY-MM-DD
    last_seen: str = ""
    channels: list[str] = Field(default_factory=list)  # names, most active first (<= 6)
    talks_with: list[str] = Field(default_factory=list)  # names, most shared days first (<= 6)
    tasks_open: int = 0
    tasks_done: int = 0


class Observed(BaseModel):
    """What Salid noticed from the conversations (execs only; derived from private rooms too)."""

    role: str = ""  # their apparent role, e.g. "runs sales and customer quotes"
    responsibilities: str = ""
    works_with: list[str] = Field(default_factory=list)
    how_they_are: str = ""  # tone / workload / mood, respectful, factual
    at: str = ""  # when this was written (ISO)
    messages: int = 0  # how many of their messages were read


class Person(BaseModel):
    id: str  # "<platform>:<external id>"
    platform: str = "discord"
    external_id: str
    name: str = ""  # the name seen on the platform
    nickname: str = ""  # from the bot's profile or an admin
    email: str = ""  # the Salid account this profile is linked to ("" = no account yet)
    title: str = ""
    department: str = ""
    about: str = ""
    hidden: bool = False  # an admin hid them (a bot, a guest...)
    seen: Seen = Field(default_factory=Seen)
    observed: Observed | None = None
    updated_at: str = ""

    def display_name(self) -> str:
        return self.nickname or self.name or self.external_id


def _stamp(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).isoformat(timespec="seconds")


class PeopleBook:
    def __init__(self, docs: DocStore):
        self.docs = docs

    # ---- records
    def list(self, guild_id: int, *, include_hidden: bool = True) -> list[Person]:
        out = [Person.model_validate(d) for d in self.docs.find(KIND, guild_id)]
        if not include_hidden:
            out = [p for p in out if not p.hidden]
        out.sort(key=lambda p: (p.seen.last_seen, p.seen.messages), reverse=True)
        return out

    def get(self, guild_id: int, pid: str) -> Person | None:
        d = self.docs.get(KIND, guild_id, pid)
        return Person.model_validate(d) if d else None

    def put(self, guild_id: int, p: Person, now: datetime | None = None) -> Person:
        p.updated_at = _stamp(now)
        self.docs.put(KIND, guild_id, p.id, p.model_dump())
        return p

    def edit(self, guild_id: int, pid: str, **fields: Any) -> Person:
        """What an admin writes: nickname / title / department / about / hidden / email."""
        p = self.get(guild_id, pid)
        if p is None:
            raise KeyError(pid)
        for k, v in fields.items():
            if v is None:
                continue
            if k in PROFILE_LIMITS:
                text = str(v).strip() if k == "about" else " ".join(str(v).split())
                setattr(p, k, text[: PROFILE_LIMITS[k]])
            elif k == "hidden":
                p.hidden = bool(v)
            elif k == "email":
                p.email = str(v).strip().lower()[:200]
            else:
                raise ValueError(f"unknown field {k!r}")
        return self.put(guild_id, p)

    # ---- discovery
    def discover(
        self,
        guild_id: int,
        memory: Any,
        channels: dict[int, str],
        *,
        today: date,
        days: int = SCAN_DAYS,
        tasks: Collection[dict] = (),
        nicknames: dict[str, str] | None = None,
        emails: dict[str, str] | None = None,
    ) -> list[Person]:
        """Rebuild `seen` for everyone who wrote in `channels` in the last `days`, keep what
        admins wrote and what Salid observed, link accounts (`emails`: discord id -> email).
        Bots and Salid herself are skipped."""
        start = (today - timedelta(days=days)).isoformat()
        rows = memory.messages(
            guild_id, list(channels), start, today.isoformat(), limit=5000, newest_first=True
        )
        per: dict[str, dict[str, Any]] = {}
        day_people: dict[tuple[int, str], set[str]] = defaultdict(set)
        for m in rows:
            if not m.author_id or m.kind in ("salid",) or m.author_name.casefold() in BOT_NAMES:
                continue
            uid = str(m.author_id)
            d = per.setdefault(
                uid, {"name": "", "n": 0, "first": "", "last": "", "chans": Counter()}
            )
            d["n"] += 1
            d["name"] = d["name"] or m.author_name  # newest first: the latest name wins
            d["first"] = min(d["first"] or m.local_date, m.local_date)
            d["last"] = max(d["last"], m.local_date)
            d["chans"][channels.get(m.channel_id, str(m.channel_id))] += 1
            day_people[(m.channel_id, m.local_date)].add(uid)
        together: dict[str, Counter] = defaultdict(Counter)
        for people in day_people.values():
            for a in people:
                for b in people:
                    if a != b:
                        together[a][b] += 1
        open_n, done_n, task_names = Counter(), Counter(), {}
        for t in tasks:
            uid = str(t.get("assignee_id") or "")
            if uid in ("", "0"):
                continue
            (open_n if t.get("status") == "open" else done_n)[uid] += 1
            task_names.setdefault(uid, t.get("assignee_name") or "")
        existing = {p.id: p for p in self.list(guild_id)}
        now = datetime.now(UTC)
        out: list[Person] = []
        for uid in set(per) | set(task_names):
            pid = f"discord:{uid}"
            p = existing.get(pid) or Person(id=pid, external_id=uid)
            d = per.get(uid)
            p.name = (d["name"] if d else "") or p.name or task_names.get(uid, "")
            if nicknames and nicknames.get(uid) and not p.nickname:
                p.nickname = nicknames[uid][: PROFILE_LIMITS["nickname"]]
            if emails and emails.get(uid):
                p.email = emails[uid]
            p.seen = Seen(
                messages=d["n"] if d else 0,
                first_seen=d["first"] if d else "",
                last_seen=d["last"] if d else "",
                channels=[c for c, _ in (d["chans"].most_common(6) if d else [])],
                talks_with=[
                    (
                        per.get(o, {}).get("name")
                        or existing.get(f"discord:{o}", Person(id="", external_id=o)).display_name()
                    )
                    for o, _ in together[uid].most_common(6)
                ],
                tasks_open=open_n[uid],
                tasks_done=done_n[uid],
            )
            out.append(self.put(guild_id, p, now))
        out.sort(key=lambda p: (p.seen.last_seen, p.seen.messages), reverse=True)
        return out

    # ---- what Salid observed
    def own_lines(
        self, guild_id: int, memory: Any, channels: dict[int, str], uid: str, *, today: date
    ) -> list[str]:
        start = (today - timedelta(days=SCAN_DAYS)).isoformat()
        rows = memory.messages(
            guild_id, list(channels), start, today.isoformat(), limit=5000, newest_first=True
        )
        mine = [m for m in rows if str(m.author_id) == uid][:OBSERVE_LINES]
        mine.reverse()
        return [
            f"[{m.local_date} #{channels.get(m.channel_id, m.channel_id)}] {m.text.strip()[:300]}"
            for m in mine
            if m.text.strip()
        ]

    def observe(
        self,
        guild_id: int,
        pid: str,
        *,
        lines: list[str],
        observer: Any,
        lang: str = "en",
        now: datetime | None = None,
    ) -> Person:
        """One cheap-model call: what this person seems to do, from their own messages."""
        p = self.get(guild_id, pid)
        if p is None:
            raise KeyError(pid)
        if not lines:
            p.observed = Observed(at=_stamp(now), messages=0)
            return self.put(guild_id, p, now)
        result = observer(build_observe_prompt(p, lines, lang))
        obs = Observed.model_validate({**result, "at": _stamp(now), "messages": len(lines)})
        obs.works_with = [str(x)[:60] for x in obs.works_with][:8]
        for k in ("role", "responsibilities", "how_they_are"):
            setattr(obs, k, " ".join(str(getattr(obs, k) or "").split())[:400])
        p.observed = obs
        return self.put(guild_id, p, now)


class ObservedOut(BaseModel):
    role: str = ""
    responsibilities: str = ""
    works_with: list[str] = Field(default_factory=list)
    how_they_are: str = ""


def build_observe_prompt(p: Person, lines: list[str], lang: str) -> str:
    who = p.display_name()
    extra = ", ".join(x for x in (p.title, p.department) if x)
    return (
        f"You are Salid, the company's assistant. Below are recent messages written by {who}"
        + (f" ({extra})" if extra else "")
        + " on the company's chat (channel names after #). Talks with: "
        + (", ".join(p.seen.talks_with) or "unknown")
        + ".\n\nFrom ONLY these messages, describe for the management, in "
        + lang_name(lang)
        + ":\n"
        "- role: what this person apparently does in the company (one sentence);\n"
        "- responsibilities: the things they handle or are asked about (one or two sentences);\n"
        "- works_with: names of the people they coordinate with most (from the messages);\n"
        "- how_they_are: how they come across lately: workload, tone, blockers, mood "
        "(one or two factual sentences, respectful, no speculation about private life, "
        "no diagnosis).\n"
        'Say "not enough to tell" when the messages do not show it. Return JSON only.\n\n'
        + "\n".join(lines)
    )


def make_observer(client: Any, model: str):
    """(prompt) -> dict with the ObservedOut fields, through Gemini (JSON mode)."""
    from google.genai import types

    from .llm import generate

    def run(prompt: str) -> dict:
        resp = generate(
            client,
            model,
            prompt,
            types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1500,
                response_mime_type="application/json",
                response_schema=ObservedOut,
            ),
        )
        return ObservedOut.model_validate(json.loads(resp.text or "{}")).model_dump()

    return run
