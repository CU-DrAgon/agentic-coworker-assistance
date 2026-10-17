"""Personal reminders, one-off and recurring (pure logic, no Discord).

Agent tools (`reminder_tools`, appended by `SecretaryBot._agent_kwargs` via `reminders_bot` so chat
and voice share them): `set_reminder`, `list_reminders`, `cancel_reminder`. The model converts
natural language into the small formats accepted here (`YYYY-MM-DD HH:MM`, `HH:MM`, `+30m`);
everything else (limits, who may remind whom, quiet hours, next-slot maths) is decided in code.

Storage: `ReminderBook` on the `DocStore` (kind `salid_reminders`, key = `R-0012`), so SQLite and
Firestore both work. Delivery and the tick live in `reminders_bot.py`.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from .dm import SENSITIVE_DM_RE
from .i18n import t

KIND = "salid_reminders"
MAX_MESSAGE = 300
MIN_INTERVAL_S = 15 * 60
MAX_INTERVAL_S = 365 * 86400
MAX_ACTIVE = 20  # per target person per guild
PAST_GRACE = timedelta(minutes=1)
MAX_AHEAD = timedelta(days=365)
LATE_GRACE = timedelta(minutes=30)  # a ONCE reminder later than this carries a "late" note
QUIET_START, QUIET_END = 22, 8  # local hours: no interval fires from 22:00 until 08:00
PRUNE_AFTER = timedelta(days=30)

_ID = re.compile(r"^R-(\d+)$")
_REL = re.compile(r"^\+\s*(\d+)\s*([mhd])$")
_FULL = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})[ t]+(\d{1,2})[:.](\d{2})$")
_CLOCK = re.compile(r"^(\d{1,2})[:.](\d{2})$")
_EVERY = re.compile(r"^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$", re.I)
_UNIT_S = {"m": 60, "h": 3600, "d": 86400}
_EN_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_TH_DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
_DAY_WORDS = {
    **{n: i for i, n in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])},
    **{n: i for i, n in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday"])},
    "saturday": 5,
    "sunday": 6,
    "tues": 1,
    "thur": 3,
    "thurs": 3,
    "จ": 0,
    "อ": 1,
    "พ": 2,
    "พฤ": 3,
    "ศ": 4,
    "ส": 5,
    "อา": 6,
    **{n: i for i, n in enumerate(_TH_DAYS)},
    **{f"วัน{n}": i for i, n in enumerate(_TH_DAYS)},
}
_GROUPS = {
    "daily": range(7),
    "everyday": range(7),
    "ทุกวัน": range(7),
    "weekdays": range(5),
    "weekday": range(5),
    "workdays": range(5),
    "วันทำงาน": range(5),
    "weekend": (5, 6),
    "weekends": (5, 6),
    "เสาร์อาทิตย์": (5, 6),
}
_ME = {"", "me", "myself", "self", "ฉัน", "ผม", "ตัวเอง", "เรา", "ดิฉัน", "หนู"}
_BROADCAST = re.compile(r"@(everyone|here)\b|<@&\d+>", re.I)


class ReminderError(Exception):
    """A refusal with an i18n reason (`rem_err_<code>`); never crashes the tool."""

    def __init__(self, code: str, **fmt):
        super().__init__(code)
        self.code, self.fmt = code, fmt


# ----------------------------------------------------------------------------- parsing
def parse_at(text: str, now: datetime) -> datetime:
    """`YYYY-MM-DD HH:MM` | `HH:MM` (today if still ahead, else tomorrow) | `+30m` `+2h` `+1d`,
    read in now's zone. Raises ReminderError(bad_time)."""
    s = (text or "").strip().lower().removesuffix("น.").removesuffix("น").strip()
    if m := _REL.match(s):
        n = int(m[1])
        if n <= 0 or n * _UNIT_S[m[2]] > MAX_AHEAD.total_seconds() * 2:
            raise ReminderError("bad_time")
        return (now + timedelta(seconds=n * _UNIT_S[m[2]])).replace(microsecond=0)
    try:
        if m := _FULL.match(s):
            y, mo, d, h, mi = (int(x) for x in m.groups())
            return datetime(y, mo, d, h, mi, tzinfo=now.tzinfo)
        if m := _CLOCK.match(s):
            h, mi = int(m[1]), int(m[2])
            cand = now.replace(hour=h, minute=mi, second=0, microsecond=0)
            return cand if cand > now else cand + timedelta(days=1)
    except ValueError:
        pass
    raise ReminderError("bad_time")


def parse_every(text: str) -> int:
    """`30m` `2h` `1d` -> seconds (min 15 minutes)."""
    m = _EVERY.match((text or "").strip())
    if not m:
        raise ReminderError("bad_every")
    secs = int(m[1]) * _UNIT_S[m[2][0].lower()]
    if secs <= 0 or secs > MAX_INTERVAL_S:
        raise ReminderError("bad_every")
    if secs < MIN_INTERVAL_S:
        raise ReminderError("min_interval")
    return secs


def parse_weekdays(text: str) -> list[int]:
    """`daily` | `weekdays` | `mon,tue` | `mon-fri` | `จ,อ,พ,พฤ,ศ,ส,อา` -> sorted 0..6 (Mon=0)."""
    out: set[int] = set()
    for tok in re.split(r"[,\s/]+", (text or "").strip().lower()):
        if not tok:
            continue
        if tok in _GROUPS:
            out.update(_GROUPS[tok])
        elif tok in _DAY_WORDS:
            out.add(_DAY_WORDS[tok])
        elif "-" in tok and all(p in _DAY_WORDS for p in tok.split("-", 1)):
            a, b = (_DAY_WORDS[p] for p in tok.split("-", 1))
            out.update(range(a, b + 1) if a <= b else [*range(a, 7), *range(0, b + 1)])
        else:
            raise ReminderError("bad_weekdays")
    if not out:
        raise ReminderError("bad_weekdays")
    return sorted(out)


def parse_until(text: str) -> date:
    try:
        return date.fromisoformat((text or "").strip())
    except ValueError:
        raise ReminderError("bad_until") from None


def clean_message(text: str) -> str:
    s = _BROADCAST.sub(lambda m: m[0].replace("@", "@​"), (text or "").strip())
    s = " ".join(s.split())
    if not s:
        raise ReminderError("empty")
    if len(s) > MAX_MESSAGE:
        raise ReminderError("too_long", n=MAX_MESSAGE)
    return s


def norm_id(text: str) -> str | None:
    """`R-0012` | `r-12` | `12` -> `R-0012`; None when it is not an id."""
    s = (text or "").strip().upper()
    if s.isdigit():
        s = f"R-{s}"
    m = _ID.match(s)
    return f"R-{int(m[1]):04d}" if m else None


# ----------------------------------------------------------------------------- schedule maths
def _utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def _from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def push_out_of_quiet(dt: datetime) -> datetime:
    """A local time inside 22:00-08:00 moves to the next 08:00."""
    if dt.hour >= QUIET_START:
        return (dt + timedelta(days=1)).replace(hour=QUIET_END, minute=0, second=0, microsecond=0)
    if dt.hour < QUIET_END:
        return dt.replace(hour=QUIET_END, minute=0, second=0, microsecond=0)
    return dt


def weekly_slot(weekdays: list[int], hhmm: str, after: datetime) -> datetime:
    """The first `hhmm` on one of `weekdays` strictly after `after` (local, aware)."""
    h, m = (int(x) for x in hhmm.split(":"))
    for i in range(0, 9):
        d = after.date() + timedelta(days=i)
        if d.weekday() in weekdays:
            cand = datetime.combine(d, time(h, m), tzinfo=after.tzinfo)
            if cand > after:
                return cand
    raise ValueError("no slot")  # unreachable for a non-empty weekday list


def interval_slot(base: datetime, every_s: int, now: datetime, quiet: bool) -> datetime:
    """base + k * every for the smallest k >= 1 that is after `now` (the cadence is kept; missed
    slots are skipped, never replayed), pushed out of the quiet window when asked."""
    k = max(1, int((now - base).total_seconds() // every_s) + 1)
    slot = base + timedelta(seconds=k * every_s)
    while slot <= now:
        slot += timedelta(seconds=every_s)
    return push_out_of_quiet(slot.astimezone(now.tzinfo)) if quiet else slot


def next_slot(rec: dict, now: datetime) -> datetime | None:
    """The next fire strictly after `now` (aware, in now's zone) for a recurring reminder; None for
    ONCE or when `until` passed."""
    kind = rec["kind"]
    if kind == "once":
        return None
    if kind == "weekly":
        slot = weekly_slot(rec["weekdays"], rec["at_hhmm"], now)
    else:
        base = _from_iso(rec["next_at"]).astimezone(now.tzinfo)
        slot = interval_slot(base, rec["every_s"], now, bool(rec.get("quiet")))
    until = rec.get("until")
    if until and slot.date() > date.fromisoformat(until):
        return None
    return slot


def advance(rec: dict, now: datetime) -> dict:
    """The reminder after a fire (or a skipped stale slot): next slot, or inactive."""
    out = dict(rec)
    slot = next_slot(rec, now)
    if slot is None:
        out.update(active=False, done_at=_utc(now))
    else:
        out["next_at"] = _utc(slot)
    return out


# ----------------------------------------------------------------------------- display
def _days_text(rec: dict, lang: str) -> str:
    wd = rec["weekdays"]
    if len(wd) == 7:
        return t("rem_every_day", lang)
    if wd == [0, 1, 2, 3, 4]:
        return t("rem_every_weekday", lang)
    names = _EN_DAYS if lang == "en" else _TH_DAYS
    return t("rem_every_on", lang, days=", ".join(names[i] for i in wd))


def _every_text(secs: int, lang: str) -> str:
    for unit, key in ((86400, "rem_unit_d"), (3600, "rem_unit_h"), (60, "rem_unit_m")):
        if secs % unit == 0:
            return t("rem_every_n", lang, n=secs // unit, unit=t(key, lang))
    return t("rem_every_n", lang, n=secs // 60, unit=t("rem_unit_m", lang))


def describe(rec: dict, lang: str) -> str:
    """ "ทุก 2 ชั่วโมง" / "every Mon, Wed 08:00" / "ครั้งเดียว"."""
    if rec["kind"] == "interval":
        return _every_text(rec["every_s"], lang)
    if rec["kind"] == "weekly":
        return f"{_days_text(rec, lang)} {rec['at_hhmm']}"
    return t("rem_once", lang)


def fmt_next(dt: datetime, now: datetime, lang: str) -> str:
    """ "วันนี้ 14:00" / "พรุ่งนี้ 09:00" / "จันทร์ 2026-10-19 09:00"."""
    dt = dt.astimezone(now.tzinfo)
    delta = (dt.date() - now.date()).days
    if delta == 0:
        return t("rem_today", lang, hm=f"{dt:%H:%M}")
    if delta == 1:
        return t("rem_tomorrow", lang, hm=f"{dt:%H:%M}")
    day = (_EN_DAYS if lang == "en" else _TH_DAYS)[dt.weekday()]
    return f"{day} {dt:%Y-%m-%d %H:%M}"


def fire_text(
    rec: dict,
    lang: str,
    now: datetime,
    *,
    nxt: datetime | None = None,
    who: str = "",
    mention: str = "",
    late_for: datetime | None = None,
) -> str:
    """The delivered message: ⏰ line (+ who asked, + a note when it is late for `late_for`), then
    for a reminder that repeats (`nxt` given) a second line with the schedule, the next time and
    how to cancel."""
    head = t("rem_fire_from" if who else "rem_fire", lang, message=rec["message"], who=who)
    if mention:
        head = f"{mention} {head}"
    if late_for is not None:
        head += "\n" + t("rem_late", lang, when=fmt_next(late_for, now, lang))
    if nxt is not None:
        head += "\n" + t(
            "rem_footer",
            lang,
            every=describe(rec, lang),
            next=fmt_next(nxt, now, lang),
            id=rec["id"],
        )
    return head


# ----------------------------------------------------------------------------- storage
class ReminderBook:
    """Reminders on the DocStore. Reads go through `find` (guild scale, filtered in Python)."""

    def __init__(self, docs):
        self.docs = docs
        self._lock = threading.Lock()

    def create(self, guild_id: int, fields: dict) -> dict:
        with self._lock:
            ids = [int(m[1]) for d in self.docs.find(KIND, guild_id) if (m := _ID.match(d["id"]))]
            rec = {**fields, "id": f"R-{max(ids, default=0) + 1:04d}", "guild_id": guild_id}
            self.docs.put(KIND, guild_id, rec["id"], rec)
        return rec

    def get(self, guild_id: int, rid: str) -> dict | None:
        return self.docs.get(KIND, guild_id, rid)

    def save(self, rec: dict) -> None:
        self.docs.put(KIND, rec["guild_id"], rec["id"], rec)

    def active(self, guild_id: int, where: Callable[[dict], bool] | None = None) -> list[dict]:
        return self.docs.find(
            KIND, guild_id, lambda d: d.get("active") and (where is None or where(d))
        )

    def due(self, guild_id: int, now: datetime) -> list[dict]:
        iso = _utc(now)
        return sorted(
            self.docs.find(KIND, guild_id, lambda d: d.get("active") and d["next_at"] <= iso),
            key=lambda d: d["next_at"],
        )

    def prune(self, guild_id: int, now: datetime) -> int:
        """Delete finished reminders older than PRUNE_AFTER."""
        cut = _utc(now - PRUNE_AFTER)
        old = self.docs.find(
            KIND, guild_id, lambda d: not d.get("active") and d.get("done_at", cut) < cut
        )
        for d in old:
            self.docs.delete(KIND, guild_id, d["id"])
        return len(old)


# ----------------------------------------------------------------------------- tools
@dataclass
class ReminderCtx:
    guild_id: int
    asker_id: int
    asker_name: str
    is_exec: bool
    lang: str
    book: ReminderBook
    now: Callable[[], datetime]  # aware, in the guild's zone
    resolve: Callable[[str], list]  # name / @mention -> candidate members (id, display_name)
    name_of: Callable[[int], str]
    channel_id: int | None = None  # where=here posts here (voice: the meeting's text channel)


def _err(ctx: ReminderCtx, e: ReminderError) -> dict:
    return {"ok": False, "error": e.code, "reason": t(f"rem_err_{e.code}", ctx.lang, **e.fmt)}


def _plan(at: str, every: str, weekdays: str, until: str, quiet: bool, now: datetime) -> dict:
    """Kind + first fire from the tool arguments. Raises ReminderError."""
    at, every, weekdays, until = (
        x.strip() for x in (at or "", every or "", weekdays or "", until or "")
    )
    if every and weekdays:
        raise ReminderError("both")
    until_d = parse_until(until) if until else None
    if until_d is not None and until_d < now.date():
        raise ReminderError("bad_until")
    plan: dict = {"until": until_d.isoformat() if until_d else None, "every_s": 0, "weekdays": []}
    plan.update(at_hhmm="", quiet=False)
    if weekdays:
        days = parse_weekdays(weekdays)
        if not at or _REL.match(at.lower()):
            raise ReminderError("weekly_needs_time")
        first_at = parse_at(at, now)  # only its HH:MM is used
        hhmm = f"{first_at:%H:%M}"
        first = weekly_slot(days, hhmm, now)
        plan.update(kind="weekly", weekdays=days, at_hhmm=hhmm)
    elif every:
        secs = parse_every(every)
        plan.update(kind="interval", every_s=secs, quiet=bool(quiet))
        first = parse_at(at, now) if at else interval_slot(now, secs, now, bool(quiet))
    elif at:
        first = parse_at(at, now)
        plan.update(kind="once", until=None)
    else:
        raise ReminderError("no_time")
    if first < now - PAST_GRACE:
        raise ReminderError("past")
    if first > now + MAX_AHEAD:
        raise ReminderError("too_far")
    if plan["until"] and first.date() > date.fromisoformat(plan["until"]):
        raise ReminderError("bad_until")
    plan["first"] = first
    return plan


def _target(ctx: ReminderCtx, for_user: str):
    """(user_id, display_name) the reminder is for. Only executives may pick someone else."""
    name = (for_user or "").strip()
    if name.lstrip("@").casefold() in _ME:
        return ctx.asker_id, ctx.asker_name
    if not ctx.is_exec:
        raise ReminderError("exec_only")
    found = ctx.resolve(name)
    if not found:
        raise ReminderError("person_not_found", person=name)
    if len(found) > 1:
        names = ", ".join(m.display_name for m in found[:6])
        raise ReminderError("ambiguous_person", person=name, names=names)
    return found[0].id, found[0].display_name


def reminder_tools(ctx: ReminderCtx) -> list:
    """`set_reminder`, `list_reminders`, `cancel_reminder` bound to one guild and one asker."""

    def set_reminder(
        message: str,
        at: str = "",
        every: str = "",
        weekdays: str = "",
        until: str = "",
        for_user: str = "",
        where: str = "me",
        quiet_hours: bool = True,
    ) -> dict:
        """Set a reminder that Salid sends later, once or repeating ("เตือนฉัน...", "remind me to
        ..."). Simple reminders need no confirmation question: set it, then confirm in ONE short
        sentence with the time (from "next").

        Args:
            message: what to say when it fires, up to 300 characters (e.g. "ออกกำลังกาย").
            at: local time. `YYYY-MM-DD HH:MM`, or `HH:MM` (today if still ahead, else tomorrow),
                or relative `+30m` / `+2h` / `+1d`. For daily/weekly reminders only its HH:MM is
                used. Empty with `every` = first fire after one interval.
            every: repeat interval `30m` / `2h` / `1d` (minimum 15m). Not together with weekdays.
            weekdays: for daily or weekly reminders (needs `at` = HH:MM): `daily`, `weekdays`
                (Mon-Fri), or a list like `mon,wed` / `จ,พฤ`.
            until: last day, `YYYY-MM-DD`, optional.
            for_user: someone else's name or @mention. Executives only; empty = the asker.
            where: `me` (private channel or DM, default) or `here` (this channel / the meeting
                chat, mentioning the person). Use `here` only when asked for it.
            quiet_hours: for `every` reminders, no pings between 22:00 and 08:00 (default true).
        """
        try:
            now = ctx.now()
            msg = clean_message(message)
            wh = (where or "me").strip().lower()
            if wh not in ("me", "here"):
                raise ReminderError("bad_where")
            if wh == "here" and ctx.channel_id is None:
                raise ReminderError("bad_where")
            uid, uname = _target(ctx, for_user)
            if uid != ctx.asker_id and SENSITIVE_DM_RE.search(msg):
                raise ReminderError("sensitive")
            plan = _plan(at, every, weekdays, until, quiet_hours, now)
            if len(ctx.book.active(ctx.guild_id, lambda d: d["user_id"] == uid)) >= MAX_ACTIVE:
                raise ReminderError("cap", n=MAX_ACTIVE)
        except ReminderError as e:
            return _err(ctx, e)
        first = plan.pop("first")
        rec = ctx.book.create(
            ctx.guild_id,
            {
                "user_id": uid,
                "set_by": ctx.asker_id,
                "set_by_name": ctx.asker_name,
                "message": msg,
                "kind": plan.pop("kind"),
                "next_at": _utc(first),
                **plan,
                "where": wh,
                "channel_id": ctx.channel_id if wh == "here" else None,
                "lang": ctx.lang,
                "created_at": _utc(now),
                "fired": 0,
                "fails": 0,
                "active": True,
            },
        )
        when = fmt_next(first, now, ctx.lang)
        sched = describe(rec, ctx.lang)
        summary = f"{rec['id']} · {msg} · {sched} · {when}"
        out = {"ok": True, "id": rec["id"], "next": when, "summary": summary}
        if uid != ctx.asker_id:
            out["for"] = uname
        if rec["kind"] == "interval" and rec["quiet"]:
            out["quiet_hours"] = f"{QUIET_START}:00-0{QUIET_END}:00"
        return out

    def _mine(d: dict) -> bool:
        return d["user_id"] == ctx.asker_id or (ctx.is_exec and d["set_by"] == ctx.asker_id)

    def list_reminders() -> dict:
        """The asker's active reminders (executives also see the ones they set for others)."""
        now = ctx.now()
        recs = ctx.book.active(ctx.guild_id, _mine)
        items = []
        for d in recs:
            item = {
                "id": d["id"],
                "text": d["message"],
                "schedule": describe(d, ctx.lang),
                "next": fmt_next(_from_iso(d["next_at"]), now, ctx.lang),
            }
            if d["user_id"] != ctx.asker_id:
                item["for"] = ctx.name_of(d["user_id"])
            items.append(item)
        return {"ok": True, "count": len(items), "reminders": items}

    def cancel_reminder(which: str) -> dict:
        """Cancel reminders: by id (`R-0012`), by a word from its text, or `all` for all of the
        asker's own. Use for "ยกเลิกเตือน..." / "cancel that reminder".

        Args:
            which: an id, a text fragment that matches exactly one reminder, or "all".
        """
        now = ctx.now()
        key = (which or "").strip()
        if key.casefold() in ("all", "ทั้งหมด", "ทุกอัน"):
            own = ctx.book.active(ctx.guild_id, lambda d: d["user_id"] == ctx.asker_id)
            for d in own:
                ctx.book.save({**d, "active": False, "done_at": _utc(now), "cancelled": True})
            return {"ok": True, "cancelled": [d["id"] for d in own]}
        pool = ctx.book.active(ctx.guild_id, _mine)
        rid = norm_id(key)
        if rid is not None:
            hits = [d for d in pool if d["id"] == rid]
        else:
            frag = key.casefold()
            hits = [d for d in pool if frag and frag in d["message"].casefold()]
        if not hits:
            return _err(ctx, ReminderError("not_found"))
        if len(hits) > 1:
            return {
                "ok": False,
                "error": "ambiguous",
                "reason": t("rem_err_ambiguous", ctx.lang),
                "candidates": [{"id": d["id"], "text": d["message"]} for d in hits[:10]],
            }
        d = hits[0]
        ctx.book.save({**d, "active": False, "done_at": _utc(now), "cancelled": True})
        return {"ok": True, "cancelled": [d["id"]], "text": d["message"]}

    return [set_reminder, list_reminders, cancel_reminder]


REMINDERS_PROMPT = (
    "\nเตือนความจำ (set_reminder / list_reminders / cancel_reminder): เมื่อมีคนขอให้เตือน (“เตือนฉัน...”, "
    "“ทุก 2 ชั่วโมงเตือนออกกำลังกาย”, “remind me to...”) ให้เรียก set_reminder ทันที ไม่ต้องถามยืนยันถ้าเป็นการเตือนทั่วไป "
    "แปลงเวลาที่พูดเป็นรูปแบบของเครื่องมือโดยใช้วันและเวลาปัจจุบันที่ให้ไว้: at='YYYY-MM-DD HH:MM' (เช่น พรุ่งนี้ 9 โมง), "
    "at='HH:MM' (วันนี้ถ้ายังไม่ถึง ไม่งั้นพรุ่งนี้), at='+30m' / '+2h' / '+1d' (อีก 30 นาที / 2 ชม. / 1 วัน). "
    "เตือนซ้ำ: every='2h' (ขั้นต่ำ 15 นาที) หรือรายวัน/รายสัปดาห์ at='08:00' + weekdays='daily' | 'weekdays' | 'mon,thu' "
    "(“ทุกวันจันทร์ 8 โมง” = at='08:00', weekdays='mon'); until='YYYY-MM-DD' ถ้าบอกวันสิ้นสุด. "
    "เตือนซ้ำแบบ every จะเงียบช่วง 22:00-08:00 โดยอัตโนมัติ ให้บอกข้อนี้ครั้งเดียวตอนยืนยัน (ปิดได้ด้วย quiet_hours=false ถ้าเขาขอ). "
    "ปกติเตือนทาง private channel/DM ของผู้ขอ (where='me'); ใช้ where='here' เฉพาะเมื่อเขาขอให้เตือนในห้องนี้/ในห้องประชุม. "
    "for_user ใช้ได้เฉพาะผู้บริหารที่ขอให้เตือนคนอื่น. ยืนยันด้วยประโยคสั้นๆ ประโยคเดียวพร้อมเวลาที่เตือน (ค่า next) "
    "ถ้าเครื่องมือปฏิเสธ ให้บอกเหตุผล (reason) ตรงๆ สั้นๆ. ถ้าถามว่ามีเตือนอะไรบ้างใช้ list_reminders; "
    "“ยกเลิกเตือน R-0012” หรือ “ไม่ต้องเตือนเรื่องออกกำลังกายแล้ว” ใช้ cancel_reminder (which = id หรือคำในข้อความ หรือ all)."
)
