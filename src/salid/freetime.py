"""Find a time everyone is free (like Lark Calendar's find-time).

Pure logic, no Discord. Three parts:

* **Busy sources** (`BusySource`): anything that can say when a person is busy, never why.
  Shipped: Salid's own meetings (`MeetingsSource`), personal availability (`AvailabilitySource`,
  rules people tell Salid), Discord scheduled events (`CallbackSource`, the bot supplies the
  fetch) and two no-op defaults that later features plug into: `LeaveSource` (an approvals module
  exposing `approved_leave(guild_id, user_id, start, end)`) and `GoogleCalendarSource` (free/busy;
  needs credentials, so it reports nothing today). A `Busy` carries only a time range and a
  coarse kind, never a title, so another person's private events cannot leak.
* **Availability** (`AvailabilityRepo`): per (guild, user) weekly work hours, weekly busy blocks
  and one-off busy blocks, in the DocStore collection `salid_availability`. The agent edits it
  only through tools bound in code to the speaker.
* **The search** (`find_slots`) and the agent tools (`freetime_tools`).

Working hours default to 09:00-18:00 Mon-Fri in the guild zone (Asia/Bangkok); lunch 12:00-13:00
is skipped unless allowed.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from .coo_store import AVAILABILITY, DocStore
from .scheduling import fmt_when, parse_local, parse_range

logger = logging.getLogger(__name__)

DEFAULT_START = 9 * 60
DEFAULT_END = 18 * 60
LUNCH = (12 * 60, 13 * 60)
STEP_MIN = 30
MAX_PEOPLE = 25
MAX_SPAN_DAYS = 31
MAX_RULES = 20
MAX_NOTE = 80
SLOTS = 3
PARTS = {"morning": (9 * 60, 13 * 60), "afternoon": (13 * 60, 18 * 60), "day": (0, 24 * 60)}
_PART_ALIASES = {
    "เช้า": "morning",
    "ครึ่งเช้า": "morning",
    "am": "morning",
    "บ่าย": "afternoon",
    "ครึ่งบ่าย": "afternoon",
    "pm": "afternoon",
    "เต็มวัน": "day",
    "ทั้งวัน": "day",
    "all day": "day",
    "full": "day",
    "whole day": "day",
}
_DAY_NAMES = {
    0: ("mon", "monday", "จันทร์", "วันจันทร์"),
    1: ("tue", "tues", "tuesday", "อังคาร", "วันอังคาร"),
    2: ("wed", "wednesday", "พุธ", "วันพุธ"),
    3: ("thu", "thur", "thurs", "thursday", "พฤหัส", "พฤหัสบดี", "วันพฤหัสบดี", "วันพฤหัส"),
    4: ("fri", "friday", "ศุกร์", "วันศุกร์"),
    5: ("sat", "saturday", "เสาร์", "วันเสาร์"),
    6: ("sun", "sunday", "อาทิตย์", "วันอาทิตย์"),
}
_DAY_LOOKUP = {n: i for i, names in _DAY_NAMES.items() for n in names}
_GROUPS = {
    "weekdays": [0, 1, 2, 3, 4],
    "weekday": [0, 1, 2, 3, 4],
    "จันทร์-ศุกร์": [0, 1, 2, 3, 4],
    "จันทร์ถึงศุกร์": [0, 1, 2, 3, 4],
    "ทุกวันทำงาน": [0, 1, 2, 3, 4],
    "weekend": [5, 6],
    "weekends": [5, 6],
    "เสาร์อาทิตย์": [5, 6],
    "everyday": [0, 1, 2, 3, 4, 5, 6],
    "every day": [0, 1, 2, 3, 4, 5, 6],
    "daily": [0, 1, 2, 3, 4, 5, 6],
    "ทุกวัน": [0, 1, 2, 3, 4, 5, 6],
}
_ME = {"me", "myself", "i", "ผม", "ฉัน", "ดิฉัน", "เรา", "หนู", "ตัวเอง", "ฉันเอง", "ผมเอง"}
_MENTION = re.compile(r"^<@!?(\d{1,25})>$")
_CTRL = re.compile(r"[\x00-\x1f\x7f`]+")


# ------------------------------------------------------------------ busy sources
@dataclass(frozen=True)
class Busy:
    """A busy range. `kind` is coarse (meeting, event, availability, leave, calendar): no title."""

    start: datetime
    end: datetime
    kind: str = "busy"


class BusySource(Protocol):
    """Says when people are busy in [start, end). Must never raise for a missing person; return
    only time ranges. Add a source by implementing `busy` and passing it to `freetime_tools`."""

    def busy(
        self, guild_id: int, user_ids: Sequence[int], start: datetime, end: datetime
    ) -> dict[int, list[Busy]]: ...


def _overlaps(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    return a0 < b1 and b0 < a1


class MeetingsSource:
    """Salid's own scheduled meetings (and Discord events it tracks): `salid_meetings`."""

    def __init__(self, store) -> None:
        self.store = store

    def busy(self, guild_id, user_ids, start, end):
        want = set(user_ids)
        out: dict[int, list[Busy]] = {}
        lo = (start - timedelta(hours=12)).astimezone(UTC).isoformat()
        hi = end.astimezone(UTC).isoformat()
        for m in self.store.list_meetings(guild_id, lo, hi, statuses=("scheduled", "started")):
            try:
                s = datetime.fromisoformat(m["start_utc"])
                e = datetime.fromisoformat(m["end_utc"])
            except (KeyError, ValueError):
                continue
            if not _overlaps(s, e, start, end):
                continue
            who = set(m.get("attendee_ids") or [])
            if m.get("organizer_id") is not None:
                who.add(m["organizer_id"])
            for uid in who & want:
                out.setdefault(uid, []).append(Busy(s, e, "meeting"))
        return out


class CallbackSource:
    """Wraps a function the bot supplies (it fetches Discord scheduled events: who organises or
    is interested). `fn(guild_id, user_ids, start, end) -> {user_id: [Busy]}`."""

    def __init__(self, fn: Callable[..., dict[int, list[Busy]]]) -> None:
        self.fn = fn

    def busy(self, guild_id, user_ids, start, end):
        return self.fn(guild_id, user_ids, start, end)


class LeaveSource:
    """Approved leave from the approvals module. `approved_leave(guild_id, user_id, start, end)`
    returns (start, end) datetimes; with no module plugged in this reports nothing."""

    def __init__(self, approved_leave: Callable[..., Sequence[tuple[datetime, datetime]]] | None):
        self.approved_leave = approved_leave

    def busy(self, guild_id, user_ids, start, end):
        if self.approved_leave is None:
            return {}
        out: dict[int, list[Busy]] = {}
        for uid in user_ids:
            for s, e in self.approved_leave(guild_id, uid, start, end) or ():
                out.setdefault(uid, []).append(Busy(s, e, "leave"))
        return out


class GoogleCalendarSource:
    """Google Calendar free/busy. No credentials exist, so this is a no-op until `fetch`
    (`fn(guild_id, user_id, start, end) -> [(start, end)]`) is provided."""

    def __init__(self, fetch: Callable[..., Sequence[tuple[datetime, datetime]]] | None = None):
        self.fetch = fetch

    def busy(self, guild_id, user_ids, start, end):
        if self.fetch is None:
            return {}
        out: dict[int, list[Busy]] = {}
        for uid in user_ids:
            for s, e in self.fetch(guild_id, uid, start, end) or ():
                out.setdefault(uid, []).append(Busy(s, e, "calendar"))
        return out


# ------------------------------------------------------------------ availability
def clean_note(text) -> str:
    s = " ".join(_CTRL.sub(" ", str(text or "")).split())
    return s if len(s) <= MAX_NOTE else s[: MAX_NOTE - 1].rstrip() + "…"


def parse_minutes(text, *, allow_24: bool = False) -> int | None:
    """ "10", "10:30", "10.30", 10 -> minutes since midnight; None when unreadable."""
    s = str(text if text is not None else "").strip().lower().replace(".", ":")
    s = re.sub(r"\s*(น\.?|am|pm)$", "", s) if not s.endswith(("am", "pm")) else s
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?", s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    if mi > 59 or h > 24 or (h == 24 and (mi or not allow_24)):
        return None
    return h * 60 + mi


def parse_weekdays(values) -> list[int] | None:
    """Weekday names (Thai/English) or group words -> sorted 0=Mon..6=Sun; None when unreadable."""
    if values is None:
        return None
    items = [values] if isinstance(values, (str, int)) else list(values)
    out: set[int] = set()
    for v in items:
        if isinstance(v, int) and 0 <= v <= 6:
            out.add(v)
            continue
        key = str(v).strip().lower()
        if key in _GROUPS:
            out.update(_GROUPS[key])
        elif key in _DAY_LOOKUP:
            out.add(_DAY_LOOKUP[key])
        else:
            return None
    return sorted(out) or None


def _hm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class AvailabilityRepo:
    """One doc per (guild, user): {work, weekly: [...], once: [...]} in `salid_availability`."""

    def __init__(self, docs: DocStore) -> None:
        self.docs = docs

    def get(self, guild_id: int, user_id: int) -> dict:
        doc = self.docs.get(AVAILABILITY, guild_id, str(user_id)) or {}
        return {
            "work": doc.get("work"),
            "weekly": list(doc.get("weekly") or []),
            "once": list(doc.get("once") or []),
        }

    def save(self, guild_id: int, user_id: int, doc: dict) -> None:
        if not doc["work"] and not doc["weekly"] and not doc["once"]:
            self.docs.delete(AVAILABILITY, guild_id, str(user_id))
            return
        self.docs.put(AVAILABILITY, guild_id, str(user_id), doc)

    def clear(self, guild_id: int, user_id: int) -> None:
        self.docs.delete(AVAILABILITY, guild_id, str(user_id))

    def work_days(self, guild_id: int, user_id: int) -> set[int]:
        w = self.get(guild_id, user_id)["work"]
        return set(w["weekdays"]) if w else {0, 1, 2, 3, 4}


class AvailabilitySource:
    """Turns a person's own rules into busy ranges: outside their work hours / on non-work days,
    weekly busy blocks and one-off blocks."""

    def __init__(self, repo: AvailabilityRepo, zone) -> None:
        self.repo, self.zone = repo, zone

    def busy(self, guild_id, user_ids, start, end):
        out: dict[int, list[Busy]] = {}
        z = self.zone
        d0 = start.astimezone(z).replace(hour=0, minute=0, second=0, microsecond=0)
        days = []
        d = d0
        while d < end.astimezone(z) and len(days) <= MAX_SPAN_DAYS + 1:
            days.append(d)
            d = (d + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

        def at(day: datetime, minutes: int) -> datetime:
            return day.replace(hour=0, minute=0) + timedelta(minutes=minutes)

        for uid in user_ids:
            doc = self.repo.get(guild_id, uid)
            blocks: list[Busy] = []
            w = doc["work"]
            for day in days:
                if w:
                    if day.weekday() not in w["weekdays"]:
                        blocks.append(Busy(at(day, 0), at(day, 24 * 60), "availability"))
                    else:
                        blocks.append(Busy(at(day, 0), at(day, w["start"]), "availability"))
                        blocks.append(Busy(at(day, w["end"]), at(day, 24 * 60), "availability"))
                for r in doc["weekly"]:
                    if day.weekday() in r["weekdays"]:
                        blocks.append(Busy(at(day, r["start"]), at(day, r["end"]), "availability"))
            for o in doc["once"]:
                try:
                    blocks.append(
                        Busy(
                            datetime.fromisoformat(o["start"]),
                            datetime.fromisoformat(o["end"]),
                            "availability",
                        )
                    )
                except (KeyError, ValueError):
                    continue
            blocks = [
                b for b in blocks if b.end > b.start and _overlaps(b.start, b.end, start, end)
            ]
            if blocks:
                out[uid] = blocks
        return out


# ------------------------------------------------------------------ the search
@dataclass
class Person:
    id: int
    name: str


@dataclass
class Slot:
    start: datetime
    end: datetime
    free: list[int] = field(default_factory=list)
    busy: list[int] = field(default_factory=list)


def find_slots(
    *,
    people: Sequence[Person],
    busy: dict[int, list[Busy]],
    work_days: dict[int, set[int]],
    start: datetime,
    end: datetime,
    now: datetime,
    duration_min: int,
    earliest: int = DEFAULT_START,
    latest: int = DEFAULT_END,
    allow_lunch: bool = False,
    limit: int = SLOTS,
) -> list[Slot]:
    """The best `limit` slots in [start, end) (aware datetimes in the guild zone). Slots where
    everybody is free win; when there are none, the slots with the most people free. Picks are
    spread over different days first and never overlap."""
    if duration_min <= 0 or latest - earliest < duration_min:
        return []
    ids = [p.id for p in people]
    cands: list[Slot] = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        wd = day.weekday()
        searchable = wd < 5 or all(wd in work_days.get(i, {0, 1, 2, 3, 4}) for i in ids)
        if searchable:
            for m in range(earliest, latest - duration_min + 1, STEP_MIN):
                if not allow_lunch and m < LUNCH[1] and m + duration_min > LUNCH[0]:
                    continue
                s = day + timedelta(minutes=m)
                e = s + timedelta(minutes=duration_min)
                if s <= now or s < start or e > end:
                    continue
                free = [
                    i
                    for i in ids
                    if not any(_overlaps(s, e, b.start, b.end) for b in busy.get(i, ()))
                ]
                if free:
                    cands.append(Slot(s, e, free, [i for i in ids if i not in free]))
        day = (day + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    if not cands:
        return []
    best = max(len(c.free) for c in cands)
    cands = [c for c in cands if len(c.free) == best]
    picked: list[Slot] = []
    used_days: set = set()
    for pass_days in (True, False):
        for c in cands:
            if len(picked) >= limit:
                break
            if pass_days and c.start.date() in used_days:
                continue
            if any(_overlaps(c.start, c.end, p.start, p.end) for p in picked):
                continue
            picked.append(c)
            used_days.add(c.start.date())
    return sorted(picked, key=lambda s: s.start)


def parse_within(text: str | None, now: datetime) -> tuple[datetime, datetime]:
    """Search range [start, end): "" = the next 7 days, this week (to Sunday), next week, today,
    tomorrow, a date, or a date range (YYYY-MM-DD..YYYY-MM-DD). ValueError when unreadable."""
    s = (text or "").strip().lower()
    d0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    monday = d0 - timedelta(days=d0.weekday())
    if s in ("this week", "สัปดาห์นี้", "อาทิตย์นี้", "สัปดาห์นี้", "week"):
        return now, monday + timedelta(days=7)
    if s in ("next week", "สัปดาห์หน้า", "อาทิตย์หน้า", "สัปดาห์ถัดไป", "อาทิตย์ถัดไป"):
        return monday + timedelta(days=7), monday + timedelta(days=14)
    if s in ("", "upcoming", "next", "7d"):
        return now, d0 + timedelta(days=7)
    if s in ("month", "this month", "เดือนนี้", "30d"):
        return now, d0 + timedelta(days=MAX_SPAN_DAYS)
    a, b = parse_range(s, now)
    return max(a, now), min(b, d0 + timedelta(days=MAX_SPAN_DAYS + 1))


# ------------------------------------------------------------------ tools
@dataclass
class FreetimeContext:
    """Everything the tools need, bound in code to one guild and one asker."""

    guild_id: int
    asker: Person
    zone: Any
    tz_name: str
    lang: str
    repo: AvailabilityRepo
    sources: list[BusySource]
    now: Callable[[], datetime]
    match_people: Callable[[str], list[Person]]  # display-name search in this guild
    person_by_id: Callable[[int], Person | None] = lambda _i: None


class FreetimeTools:
    def __init__(self, c: FreetimeContext) -> None:
        self.c = c

    # ---- people
    def _resolve(self, names: Sequence[str] | None):
        c = self.c
        people: dict[int, Person] = {c.asker.id: c.asker}
        unresolved: list[str] = []
        ambiguous: dict[str, list[str]] = {}
        for raw in names or []:
            name = str(raw).strip()
            if not name or name.casefold() in _ME:
                continue
            mention = _MENTION.match(name)
            if mention:
                p = c.person_by_id(int(mention.group(1)))
                found = [p] if p else []
            else:
                found = c.match_people(name)
            if not found:
                unresolved.append(name)
            elif len(found) > 1:
                ambiguous[name] = [p.name for p in found[:10]]
            else:
                people[found[0].id] = found[0]
        return list(people.values()), unresolved, ambiguous

    def _busy(self, people: Sequence[Person], start: datetime, end: datetime):
        c = self.c
        ids = [p.id for p in people]
        merged: dict[int, list[Busy]] = {}
        for src in c.sources:
            try:
                got = src.busy(c.guild_id, ids, start, end) or {}
            except Exception:  # one broken source must not block the answer
                logger.exception("busy source %s failed", type(src).__name__)
                continue
            for uid, items in got.items():
                if uid in ids:
                    merged.setdefault(uid, []).extend(items)
        return merged

    # ---- find_free_time
    def find_free_time(
        self,
        people: list[str] | None = None,
        duration_minutes: int = 60,
        within: str = "",
        earliest_hour: str | None = None,
        latest_hour: str | None = None,
        allow_lunch: bool = False,
    ) -> dict:
        c = self.c
        now = c.now()
        try:
            minutes = int(duration_minutes or 60)
        except (TypeError, ValueError):
            minutes = 60
        minutes = max(15, min(minutes, 8 * 60))
        try:
            start, end = parse_within(within, now)
        except ValueError:
            return {
                "error": "within must be like 'this week', 'tomorrow' or YYYY-MM-DD..YYYY-MM-DD"
            }
        lo = DEFAULT_START if earliest_hour in (None, "") else parse_minutes(earliest_hour)
        hi = DEFAULT_END if latest_hour in (None, "") else parse_minutes(latest_hour, allow_24=True)
        if lo is None or hi is None or hi <= lo:
            return {"error": "earliest_hour / latest_hour must be hours like 9 or 17:30"}
        members, unresolved, ambiguous = self._resolve(people)
        if unresolved or ambiguous:
            return {
                "status": "needs_info",
                "unresolved": unresolved,
                "ambiguous": ambiguous,
                "instruction": "Ask the user exactly who these people are (Discord names or mentions). Do not guess.",
            }
        if len(members) > MAX_PEOPLE:
            return {"error": f"too many people (max {MAX_PEOPLE})"}
        busy = self._busy(members, start - timedelta(days=1), end)
        slots = find_slots(
            people=members,
            busy=busy,
            work_days={p.id: self.c.repo.work_days(c.guild_id, p.id) for p in members},
            start=start,
            end=end,
            now=now,
            duration_min=minutes,
            earliest=lo,
            latest=hi,
            allow_lunch=bool(allow_lunch),
        )
        names = {p.id: p.name for p in members}
        out = {
            "status": "ok" if slots else "no_slot",
            "timezone": c.tz_name,
            "duration_minutes": minutes,
            "attendees": [p.name for p in members],
            "searched": f"{fmt_when(start, c.lang, c.zone)} - {fmt_when(end - timedelta(seconds=1), c.lang, c.zone)}",
            "slots": [
                {
                    "n": i,
                    "start": s.start.isoformat(timespec="minutes"),
                    "end": s.end.isoformat(timespec="minutes"),
                    "label": f"{fmt_when(s.start, c.lang, c.zone)}-{s.end:%H:%M}",
                    "everyone_free": not s.busy,
                    "free": [names[i] for i in s.free],
                    "busy": [names[i] for i in s.busy],
                }
                for i, s in enumerate(slots, 1)
            ],
        }
        if slots:
            out["next_step"] = (
                "Show the numbered options (weekday, date, time, who is free or busy; busy/free "
                "only, never why). If the user picks one ('เอาอันแรก'), call schedule_meeting "
                "with that slot's start, the duration, and these attendees; it asks for confirmation."
            )
        else:
            out["hint"] = "No common time in that range; offer a wider range or a shorter meeting."
        return out

    # ---- availability (the asker's own only)
    def set_availability(
        self,
        kind: str,
        weekdays: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        date: str | None = None,
        end_date: str | None = None,
        part: str | None = None,
        note: str = "",
    ) -> dict:
        c = self.c
        kind = str(kind or "").strip().lower()
        doc = c.repo.get(c.guild_id, c.asker.id)
        part_key = _PART_ALIASES.get(
            str(part or "").strip().lower(), str(part or "").strip().lower()
        )
        if part_key and part_key not in PARTS:
            return {"error": "part must be morning, afternoon or day"}
        a = parse_minutes(start_time) if start_time not in (None, "") else None
        b = parse_minutes(end_time, allow_24=True) if end_time not in (None, "") else None
        if (start_time not in (None, "") and a is None) or (
            end_time not in (None, "") and b is None
        ):
            return {"error": "times must be like 10:00"}
        if kind == "work_hours":
            if a is None or b is None or b <= a:
                return {"error": "work_hours needs start_time and end_time, e.g. 10:00 and 19:00"}
            days = parse_weekdays(weekdays) if weekdays else [0, 1, 2, 3, 4]
            if days is None:
                return {"error": "weekdays must be names like Mon, Tue or พุธ"}
            doc["work"] = {"start": a, "end": b, "weekdays": days}
        elif kind == "busy_weekly":
            days = parse_weekdays(weekdays)
            if days is None:
                return {"error": "busy_weekly needs weekdays like ['wed'] or ['พุธ']"}
            if a is None or b is None:
                a, b = PARTS.get(part_key or "day", PARTS["day"]) if not (a and b) else (a, b)
            if b <= a:
                return {"error": "end_time must be after start_time"}
            if len(doc["weekly"]) >= MAX_RULES:
                return {"error": "too many rules; clear some with my_availability"}
            doc["weekly"].append(
                {
                    "id": uuid.uuid4().hex[:6],
                    "weekdays": days,
                    "start": a,
                    "end": b,
                    "note": clean_note(note),
                }
            )
        elif kind == "busy_once":
            err = self._add_once(doc, date, end_date, a, b, part_key, note)
            if err:
                return {"error": err}
        else:
            return {"error": "kind must be work_hours, busy_weekly or busy_once"}
        c.repo.save(c.guild_id, c.asker.id, doc)
        return {"status": "saved", "availability": self._view(doc)}

    def _add_once(self, doc, date, end_date, a, b, part_key, note) -> str | None:
        c = self.c
        from datetime import date as _date

        try:
            d0 = _date.fromisoformat(str(date).strip())
            d1 = _date.fromisoformat(str(end_date).strip()) if end_date else d0
        except ValueError:
            return "date must be YYYY-MM-DD (convert 'tomorrow' etc. using today's date)"
        if d1 < d0 or (d1 - d0).days >= MAX_SPAN_DAYS:
            return f"end_date must be on or after date, within {MAX_SPAN_DAYS} days"
        if a is None or b is None:
            a, b = PARTS.get(part_key or "day", PARTS["day"])
        if b <= a:
            return "end_time must be after start_time"
        today = c.now().date()
        if d1 < today:
            return "that date is in the past"
        day, once = d0, doc["once"]
        once[:] = [
            o for o in once if datetime.fromisoformat(o["end"]).date() >= today
        ]  # drop finished blocks
        while day <= d1:
            s = datetime(day.year, day.month, day.day, tzinfo=c.zone) + timedelta(minutes=a)
            e = datetime(day.year, day.month, day.day, tzinfo=c.zone) + timedelta(minutes=b)
            once.append(
                {
                    "id": uuid.uuid4().hex[:6],
                    "start": s.isoformat(),
                    "end": e.isoformat(),
                    "note": clean_note(note),
                }
            )
            day += timedelta(days=1)
        if len(once) > MAX_RULES * 3:
            return "too many blocks; clear some with my_availability"
        return None

    def _view(self, doc: dict) -> dict:
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        w = doc["work"]
        return {
            "work_hours": (
                f"{_hm(w['start'])}-{_hm(w['end'])} " + "/".join(names[d] for d in w["weekdays"])
                if w
                else "default 09:00-18:00 Mon-Fri"
            ),
            "busy_weekly": [
                {
                    "id": r["id"],
                    "days": "/".join(names[d] for d in r["weekdays"]),
                    "time": f"{_hm(r['start'])}-{_hm(r['end'])}",
                    "note": r.get("note", ""),
                }
                for r in doc["weekly"]
            ],
            "busy_once": [
                {
                    "id": o["id"],
                    "from": o["start"][:16].replace("T", " "),
                    "to": o["end"][:16].replace("T", " "),
                    "note": o.get("note", ""),
                }
                for o in sorted(doc["once"], key=lambda o: o["start"])
            ],
        }

    def my_availability(self, action: str = "show", rule_id: str | None = None) -> dict:
        c = self.c
        doc = c.repo.get(c.guild_id, c.asker.id)
        action = str(action or "show").strip().lower()
        if action == "show":
            return {"availability": self._view(doc)}
        if action != "clear":
            return {"error": "action must be show or clear"}
        if not rule_id:
            c.repo.clear(c.guild_id, c.asker.id)
            return {
                "status": "cleared",
                "availability": self._view(c.repo.get(c.guild_id, c.asker.id)),
            }
        rid = str(rule_id).strip()
        if rid == "work_hours":
            doc["work"] = None
        else:
            before = len(doc["weekly"]) + len(doc["once"])
            doc["weekly"] = [r for r in doc["weekly"] if r["id"] != rid]
            doc["once"] = [o for o in doc["once"] if o["id"] != rid]
            if len(doc["weekly"]) + len(doc["once"]) == before:
                return {"error": "rule_id not found; use my_availability(show)"}
        c.repo.save(c.guild_id, c.asker.id, doc)
        return {"status": "cleared", "availability": self._view(doc)}


FREETIME_TOOL_NAMES = ("find_free_time", "set_availability", "my_availability")


def freetime_tools(c: FreetimeContext, calls: list | None = None) -> list:
    """The three agent tools (plain functions; docstrings are the tool descriptions)."""
    t = FreetimeTools(c)
    log = calls if calls is not None else []

    def safe(name: str, args: dict, fn):
        log.append((name, {k: v for k, v in args.items() if v is not None}))
        try:
            return fn()
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

    def find_free_time(
        people: list[str] | None = None,
        duration_minutes: int = 60,
        within: str = "",
        earliest_hour: str | None = None,
        latest_hour: str | None = None,
        allow_lunch: bool = False,
    ) -> dict:
        """Find the best 3 times when the asker and the given people are all free (Asia/Bangkok,
        working hours 09:00-18:00 Mon-Fri, lunch 12-13 skipped). Use it when someone asks when
        people are free, or wants a meeting without giving a time. It reads meetings, Discord
        events and what people told Salid about their availability, and returns busy/free only.

        Args:
            people: the other people (display names or @mentions); the asker is always included.
            duration_minutes: meeting length, default 60.
            within: "this week", "next week", "today", "tomorrow", a date YYYY-MM-DD or a range
                YYYY-MM-DD..YYYY-MM-DD; empty = the next 7 days.
            earliest_hour: earliest start, like "9" or "13:30", optional.
            latest_hour: latest end, like "18" or "20:00", optional.
            allow_lunch: true only if the user accepts meeting over 12-13.
        """
        a = dict(
            people=people, duration_minutes=duration_minutes, within=within,
            earliest_hour=earliest_hour, latest_hour=latest_hour, allow_lunch=allow_lunch,
        )  # fmt: skip
        return safe("find_free_time", a, lambda: t.find_free_time(**a))

    def set_availability(
        kind: str,
        weekdays: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        date: str | None = None,
        end_date: str | None = None,
        part: str | None = None,
        note: str = "",
    ) -> dict:
        """Remember when the ASKER (only themself) is not available, so find_free_time avoids it.
        "ผมไม่ว่างทุกวันพุธบ่าย" = busy_weekly, weekdays ["wed"], part "afternoon".
        "I work 10-19" = work_hours, start_time 10:00, end_time 19:00.
        "พรุ่งนี้ลาครึ่งวันเช้า" = busy_once, date = tomorrow's YYYY-MM-DD, part "morning".

        Args:
            kind: work_hours (usual hours), busy_weekly (repeats every week) or busy_once (one date or range).
            weekdays: for work_hours / busy_weekly, e.g. ["mon","tue"] or ["พุธ"]; work_hours defaults to Mon-Fri.
            start_time: HH:MM, optional if part is given.
            end_time: HH:MM, optional if part is given.
            date: busy_once: YYYY-MM-DD (the model converts relative dates).
            end_date: busy_once: last day YYYY-MM-DD for a range, optional.
            part: morning (09-13), afternoon (13-18) or day (whole day), optional.
            note: short reason (only shown to the asker), optional.
        """
        a = dict(
            kind=kind, weekdays=weekdays, start_time=start_time, end_time=end_time, date=date,
            end_date=end_date, part=part, note=note,
        )  # fmt: skip
        return safe("set_availability", a, lambda: t.set_availability(**a))

    def my_availability(action: str = "show", rule_id: str | None = None) -> dict:
        """Show or clear the asker's own availability rules and busy blocks.

        Args:
            action: show (default) or clear.
            rule_id: with clear: one rule/block id from show, or "work_hours"; omit to clear everything.
        """
        a = dict(action=action, rule_id=rule_id)
        return safe("my_availability", a, lambda: t.my_availability(**a))

    return [find_free_time, set_availability, my_availability]


FREETIME_PROMPT = (
    "\nหาเวลาว่าง (เครื่องมือ find_free_time, set_availability, my_availability): "
    "เมื่อผู้ใช้ถามว่าใครว่างเมื่อไร หรืออยากนัดประชุมโดยไม่ระบุเวลา (เช่น “หาเวลาว่างประชุมกับ A B สัก 1 ชม. อาทิตย์นี้”, "
    "“when are we all free tomorrow?”) ให้เรียก find_free_time ก่อน แล้วเสนอ 1-3 ตัวเลือกเป็นข้อๆ พร้อมวันในสัปดาห์ วันที่ เวลา และใครว่าง/ไม่ว่าง "
    "(บอกแค่ว่าง/ไม่ว่าง ห้ามเดาหรือบอกเหตุผล). ถ้าได้ needs_info ให้ถามว่าหมายถึงใคร ห้ามเดาชื่อ. "
    "เมื่อผู้ใช้เลือก (“เอาอันแรก”, “ตัวที่ 2”) ให้เรียก schedule_meeting ด้วยเวลาและผู้เข้าร่วมของตัวเลือกนั้น (ยังต้องขอยืนยันตามปกติ). "
    "ถ้าผู้ใช้บอกว่าตัวเองไม่ว่าง/เวลาทำงาน/ลา (เช่น “ผมไม่ว่างทุกวันพุธบ่าย”, “I work 10-19”, “พรุ่งนี้ลาครึ่งวันเช้า”) "
    "ให้เรียก set_availability (แปลงวันที่สัมพัทธ์เป็น YYYY-MM-DD) ตั้งได้เฉพาะของผู้ถามเอง; ดูหรือล้างด้วย my_availability."
)

__all__ = [
    "AvailabilityRepo",
    "AvailabilitySource",
    "Busy",
    "BusySource",
    "CallbackSource",
    "FREETIME_PROMPT",
    "FREETIME_TOOL_NAMES",
    "FreetimeContext",
    "FreetimeTools",
    "GoogleCalendarSource",
    "LeaveSource",
    "MeetingsSource",
    "Person",
    "Slot",
    "find_slots",
    "freetime_tools",
    "parse_local",
    "parse_within",
]
