"""'What did I miss?': a personal catch-up of what happened while someone was away.

Pure logic over the memory store, the task list and the meeting list; the Discord glue lives in
`salid.catchup_bot`. Scope is decided in code and never widened: messages and day summaries come
only from channels in `allowed` (the asker's readable channels from the memory tools); tasks are
the asker's own; meetings are the ones they were invited to; document alerts are the ones that
were already addressed to them (their targets were filtered by channel access when recorded).

`gather` returns plain data (also the agent tool's result); `render` is the deterministic text used
by the slash command and the morning brief (no model call, so it is fast and cannot invent).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from .i18n import t
from .memory_store import MAX_ROWS, MemoryMessage, MemoryStore
from .memory_summary import SummaryDoc

MAX_LOOKBACK_DAYS = 14  # how far back a catch-up reaches (keeps reads cheap)
ACTIVE_GAP = timedelta(hours=4)  # silence that ends "I was around": the default start point
DEFAULT_SINCE_HOURS = 24  # no gap found (always around): the last day
CAPS = {
    "mentions": 6,
    "tasks": 6,
    "decisions": 4,
    "missed": 4,
    "upcoming": 3,
    "channels": 4,
    "files": 4,
}
SNIP = 140

_WEEKDAYS = {
    "monday": 0, "mon": 0, "จันทร์": 0, "tuesday": 1, "tue": 1, "อังคาร": 1,
    "wednesday": 2, "wed": 2, "พุธ": 2, "thursday": 3, "thu": 3, "พฤหัส": 3, "พฤหัสบดี": 3,
    "friday": 4, "fri": 4, "ศุกร์": 4, "saturday": 5, "sat": 5, "เสาร์": 5,
    "sunday": 6, "sun": 6, "อาทิตย์": 6,
}  # fmt: skip
_NUM = re.compile(r"(\d{1,3})\s*(?:วัน|days?|d\b|สัปดาห์|อาทิตย์|weeks?|w\b)", re.I)
_WEEK_WORD = re.compile(r"สัปดาห์|อาทิตย์|weeks?", re.I)
_FILE = re.compile(r"\[[^\]\n]{1,24}:[^\]]*\]")  # "[image: x.png]", "[PDF: ...]" descriptors


def _cut(text: str, n: int = SNIP) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def start_of_day(d: date, zone) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=zone)


# ------------------------------------------------------------------ the starting point
def parse_since(text: str | None, now: datetime, zone) -> datetime | None:
    """A start time from "yesterday", "เมื่อวาน", "monday", "ช่วง 3 วัน", "2 weeks", an ISO date
    or datetime. None when the text says nothing usable (empty or not understood)."""
    s = (text or "").strip().casefold()
    if not s:
        return None
    today = now.astimezone(zone).date()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return start_of_day(date.fromisoformat(s), zone)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}.*", s):
            dt = datetime.fromisoformat(s.replace(" ", "T"))
            return dt if dt.tzinfo else dt.replace(tzinfo=zone)
    except ValueError:
        return None
    if re.search(r"เมื่อวานซืน|day before yesterday", s):
        return start_of_day(today - timedelta(days=2), zone)
    if re.search(r"เมื่อวาน|yesterday", s):
        return start_of_day(today - timedelta(days=1), zone)
    if re.search(r"วันนี้|today|เมื่อเช้า|this morning", s):
        return start_of_day(today, zone)
    m = _NUM.search(s)
    if m:
        n = int(m.group(1)) * (7 if _WEEK_WORD.search(m.group(0)) else 1)
        return start_of_day(today - timedelta(days=max(n, 1)), zone)
    if re.search(r"สัปดาห์ที่แล้ว|อาทิตย์ที่แล้ว|last week|a week", s):
        return start_of_day(today - timedelta(days=7), zone)
    for word, wd in sorted(_WEEKDAYS.items(), key=lambda kv: -len(kv[0])):
        if word in s:
            back = (today.weekday() - wd) % 7 or 7  # the most recent such day before today
            return start_of_day(today - timedelta(days=back), zone)
    return None


def last_active_since(times: list[datetime], now: datetime) -> tuple[datetime | None, bool]:
    """(start of the catch-up, found): the asker's last message before the most recent silence of
    at least ACTIVE_GAP. `times` are the asker's message times, any order. Just chatting with
    Salid therefore does not shrink the window to minutes."""
    if not times:
        return None, False
    ordered = [now, *sorted(times, reverse=True)]
    for newer, older in zip(ordered, ordered[1:], strict=False):
        if newer - older >= ACTIVE_GAP:
            return older, True
    return None, True  # always around


def working_days_away(since: datetime, now: datetime, zone, weekends: bool = False) -> int:
    """Whole days strictly between the day of `since` and today on which the person was away
    (Mon-Fri, or every day with `weekends`)."""
    a, b = since.astimezone(zone).date(), now.astimezone(zone).date()
    n, d = 0, a + timedelta(days=1)
    while d < b:
        if weekends or d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


# ------------------------------------------------------------------ gathering
@dataclass
class Sources:
    """Everything `gather` reads, bound by the bot to one guild and one person."""

    store: MemoryStore
    guild_id: int
    user_id: int
    name: str
    allowed: dict[int, str]  # readable channels: id -> name (never widened)
    zone: object
    visible: Callable[[MemoryMessage], bool] = lambda m: True
    tasks: Callable[[], list] = lambda: []  # the person's open tasks (Task objects)
    meetings: Callable[[str, str], list[dict]] = lambda a, b: []  # [a, b) ISO UTC
    alerts: Callable[[datetime], list[dict]] = lambda since: []
    retention_days: Callable[[], int] | None = None  # the guild's current memory retention
    meeting_link: Callable[[dict], str] = lambda m: m.get("link", "")


@dataclass
class CatchUp:
    since: datetime
    until: datetime
    guessed: bool = False  # since came from last activity (or the fallback), not from the person
    away_days: int = 0  # working days fully away
    mentions: list[dict] = field(default_factory=list)  # {time, channel, who, text, link}
    tasks: list[dict] = field(default_factory=list)  # {id, title, due, overdue, new, link}
    decisions: list[dict] = field(default_factory=list)  # {channel, text, link}
    missed: list[dict] = field(default_factory=list)  # meetings missed {time, title, minutes, link}
    upcoming: list[dict] = field(default_factory=list)  # meetings ahead {time, title, link}
    channels: list[dict] = field(default_factory=list)  # {channel, messages, text, link}
    files: list[dict] = field(default_factory=list)  # {channel, text, link}
    total_messages: int = 0

    def is_empty(self) -> bool:
        return not (
            self.mentions or self.tasks or self.decisions or self.missed or self.upcoming
            or self.channels or self.files
        )  # fmt: skip

    def as_data(self, zone) -> dict:
        out: dict = {
            "since": f"{self.since.astimezone(zone):%Y-%m-%d %H:%M}",
            "until": f"{self.until.astimezone(zone):%Y-%m-%d %H:%M}",
            "start_was_guessed_from_last_activity": self.guessed,
            "working_days_away": self.away_days,
            "messages_in_readable_channels": self.total_messages,
        }
        for key in ("mentions", "tasks", "decisions", "missed", "upcoming", "channels", "files"):
            rows = getattr(self, key)
            if rows:
                out[key] = rows
        if self.is_empty():
            out["note"] = "nothing new for this person in the channels they can read"
        else:
            out["how_to_answer"] = (
                "Short: the top items in this order: mentions, tasks and decisions, meetings, "
                "channel highlights, files; one line each with its link. Then offer to go into "
                "detail for any channel. Never add anything that is not here."
            )
        return out


def _link(guild_id: int, channel_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}"


def _mention_re(uid: int) -> re.Pattern:
    return re.compile(rf"<@!?{uid}>")


def gather(src: Sources, since: datetime, now: datetime, *, guessed: bool = False) -> CatchUp:
    """Collect the catch-up since `since` (clamped to MAX_LOOKBACK_DAYS). Blocking."""
    zone = src.zone
    floor = start_of_day(now.astimezone(zone).date() - timedelta(days=MAX_LOOKBACK_DAYS), zone)
    if src.retention_days is not None:  # never read what the purge would already have removed
        floor = max(
            floor,
            start_of_day(now.astimezone(zone).date() - timedelta(days=src.retention_days()), zone),
        )
    since = max(since, floor)
    out = CatchUp(since=since, until=now, guessed=guessed)
    out.away_days = working_days_away(since, now, zone)
    d0, d1 = since.astimezone(zone).date().isoformat(), now.astimezone(zone).date().isoformat()
    me = _mention_re(src.user_id)
    cid_name = src.allowed

    # raw messages (oldest first) from readable channels only
    rows = [
        m
        for m in src.store.messages(
            src.guild_id, list(src.allowed), d0, d1, limit=MAX_ROWS, newest_first=False
        )
        if m.created_at >= since and m.kind in ("chat", "voice") and src.visible(m)
    ]
    others = [m for m in rows if m.author_id != src.user_id]
    out.total_messages = len(others)
    for m in reversed(others):  # newest mentions first
        if not me.search(m.text):
            continue
        row = {
            "time": f"{m.created_at.astimezone(zone):%d/%m %H:%M}",
            "channel": cid_name.get(m.channel_id, ""),
            "who": m.author_name,
            "text": _cut(me.sub("@" + src.name, m.text)),
            "link": m.jump_url,
        }
        (out.files if _FILE.search(m.text) else out.mentions).append(row)
    out.mentions = out.mentions[: CAPS["mentions"]]
    out.files = out.files[: CAPS["files"]]

    # tasks: new since, or due soon / overdue
    today = now.astimezone(zone).date().isoformat()
    soon = (now.astimezone(zone).date() + timedelta(days=2)).isoformat()
    for tk in src.tasks():
        new = bool(tk.created_at) and tk.created_at >= _iso(since)
        late = bool(tk.due) and tk.due < today
        if new or late or (tk.due and tk.due <= soon):
            out.tasks.append(
                {
                    "id": tk.id,
                    "title": tk.title,
                    "due": tk.due or None,
                    "overdue": late,
                    "new": new,
                    "link": tk.source_link or None,
                }  # fmt: skip
            )
    out.tasks.sort(key=lambda r: (not r["overdue"], not r["new"], r["due"] or "9"))
    out.tasks = out.tasks[: CAPS["tasks"]]

    # day summaries of readable channels: decisions, items for this person, busiest channels
    mine = src.name.casefold()
    per_channel: dict[int, dict] = {}
    for s in src.store.summaries(src.guild_id, list(src.allowed), d0, d1):
        if s.message_count <= 0:
            continue
        try:
            doc = SummaryDoc.model_validate(s.summary or {})
        except ValueError:
            continue
        cname = cid_name.get(s.channel_id, "")
        link = _link(src.guild_id, s.channel_id)
        for x in doc.decisions:
            out.decisions.append({"channel": cname, "text": _cut(x), "link": link})
        for a in doc.action_items:
            if mine and mine in a.who.casefold():
                out.decisions.insert(
                    0,
                    {
                        "channel": cname,
                        "text": _cut(f"{a.what} ({a.due})" if a.due else a.what),
                        "link": link,
                        "assigned": True,
                    },
                )
        ch = per_channel.setdefault(
            s.channel_id, {"channel": cname, "messages": 0, "text": "", "link": link}
        )
        ch["messages"] += s.message_count
        if doc.overview:
            ch["text"] = _cut(doc.overview)  # the latest day's overview wins (dates ascend)
    out.decisions = out.decisions[: CAPS["decisions"]]
    busiest = sorted(per_channel.values(), key=lambda c: -c["messages"])
    out.channels = [c for c in busiest if c["text"]][: CAPS["channels"]]

    # meetings: missed (with minutes) and upcoming
    a, b = _iso(since), _iso(now)
    for m in src.meetings(a, b):
        invited = src.user_id in m.get("attendee_ids", [])
        if (
            invited
            and src.user_id not in m.get("attended", [])
            and m.get("status") in ("done", "started", "scheduled")
            and m.get("start_utc", "") < b
        ):
            start = datetime.fromisoformat(m["start_utc"]).astimezone(zone)
            out.missed.append(
                {
                    "time": f"{start:%d/%m %H:%M}",
                    "title": m.get("title", ""),
                    "minutes": m.get("summary_url") or None,
                }
            )
    out.missed = out.missed[-CAPS["missed"] :]
    for m in src.meetings(b, _iso(now + timedelta(days=3))):
        if src.user_id in m.get("attendee_ids", []) and m.get("status") in ("scheduled", "started"):
            start = datetime.fromisoformat(m["start_utc"]).astimezone(zone)
            out.upcoming.append(
                {
                    "time": f"{start:%d/%m %H:%M}",
                    "title": m.get("title", ""),
                    "link": src.meeting_link(m) or None,
                }  # fmt: skip
            )
    out.upcoming = out.upcoming[: CAPS["upcoming"]]

    # important documents already addressed to this person
    seen = {r["link"] for r in out.files}
    for d in src.alerts(since):
        link = d.get("jump_url", "")
        if link and link in seen:
            continue
        out.files.append(
            {"channel": d.get("channel_name", ""), "text": _cut(d.get("summary", "")), "link": link}
        )
    out.files = out.files[: CAPS["files"]]
    return out


def last_seen(src: Sources, now: datetime) -> tuple[datetime, str]:
    """(since, kind) from the person's activity in the readable channels. kind: "found" (their
    last message before the latest silence), "around" (active the whole window: the last day) or
    "unknown" (no messages of theirs at all: the last two days)."""
    zone = src.zone
    times: list[datetime] = []
    day = now.astimezone(zone).date()
    for w in range(0, MAX_LOOKBACK_DAYS, 7):  # newest week first; stop at the first hit
        end, start = day - timedelta(days=w), day - timedelta(days=w + 6)
        times = [
            m.created_at
            for m in src.store.messages(
                src.guild_id,
                list(src.allowed),
                start.isoformat(),
                end.isoformat(),
                limit=MAX_ROWS,
                newest_first=True,
            )
            if m.author_id == src.user_id and m.kind in ("chat", "voice")
        ]
        if times:
            break
    found, ok = last_active_since(times, now)
    if found is not None:
        return found, "found"
    if ok:
        return now - timedelta(hours=DEFAULT_SINCE_HOURS), "around"
    return start_of_day(day - timedelta(days=2), zone), "unknown"


def find_since(
    src: Sources, now: datetime, text: str | None = None
) -> tuple[datetime, bool, str | None]:
    """(since, guessed, error): an explicit start from `text`, else from the person's last
    activity in the readable channels. error is set when `text` was given but not understood."""
    if (text or "").strip():
        got = parse_since(text, now, src.zone)
        if got is None:
            return now - timedelta(hours=DEFAULT_SINCE_HOURS), True, "since not understood"
        return got, False, None
    return last_seen(src, now)[0], True, None


def catch_up(src: Sources, now: datetime, since_text: str | None = None) -> dict:
    """The agent tool's body (also the slash command's data): errors come back as data."""
    since, guessed, err = find_since(src, now, since_text)
    if err:
        return {"error": f"{err}: pass YYYY-MM-DD, or words like 'yesterday', 'monday', '3 days'"}
    return gather(src, since, now, guessed=guessed).as_data(src.zone)


# ------------------------------------------------------------------ rendering
def render(data: CatchUp, lang: str, zone, *, short: bool = False) -> str:
    """Deterministic catch-up text. `short`: the 2-3 line form for the morning brief."""
    since = f"{data.since.astimezone(zone):%d/%m %H:%M}"
    if data.is_empty():
        return t("cu_none", lang, since=since)
    lines = [t("cu_title", lang, since=since)]
    if short:
        bits = []
        if data.mentions:
            bits.append(t("cu_s_mentions", lang, n=len(data.mentions)))
        if data.tasks:
            bits.append(t("cu_s_tasks", lang, n=len(data.tasks)))
        if data.missed:
            bits.append(t("cu_s_missed", lang, n=len(data.missed)))
        if data.decisions:
            bits.append(t("cu_s_decisions", lang, n=len(data.decisions)))
        if bits:
            lines.append(" · ".join(bits))
        top = data.mentions[:1] or data.missed[:1] or data.channels[:1]
        for r in top:
            lines.append(_line(r, lang))
        lines.append(t("cu_ask_more", lang))
        return "\n".join(lines)

    def section(head: str, rows: list[str]) -> None:
        if rows:
            lines.extend(["", t(head, lang), *rows])

    section("cu_mentions", [_line(r, lang) for r in data.mentions])
    section(
        "cu_tasks",
        [
            t("cu_task_line", lang, id=r["id"], title=r["title"], due=r["due"] or "-",
              flag=t("cu_late" if r["overdue"] else "cu_new", lang) if (r["overdue"] or r["new"]) else "",
              link=r["link"] or "").strip()
            for r in data.tasks
        ],
    )  # fmt: skip
    section(
        "cu_decisions",
        [f"• {r['text']} ({r['channel']}) {r['link']}".strip() for r in data.decisions],
    )
    section(
        "cu_missed",
        [
            t("cu_meeting_line", lang, time=r["time"], title=r["title"],
              link=r["minutes"] or t("cu_no_minutes", lang)).strip()
            for r in data.missed
        ],
    )  # fmt: skip
    section(
        "cu_upcoming",
        [
            t("cu_meeting_line", lang, time=r["time"], title=r["title"], link=r["link"] or "").strip()
            for r in data.upcoming
        ],
    )  # fmt: skip
    section(
        "cu_channels",
        [
            t("cu_channel_line", lang, channel=r["channel"], n=r["messages"], text=r["text"],
              link=r["link"]).strip()
            for r in data.channels
        ],
    )  # fmt: skip
    section("cu_files", [f"• {r['text']} ({r['channel']}) {r['link']}".strip() for r in data.files])
    lines += ["", t("cu_ask_more", lang)]
    return "\n".join(lines)[:3800]


def _line(r: dict, lang: str) -> str:
    if "who" in r:
        return t("cu_mention_line", lang, who=r["who"], channel=r["channel"], text=r["text"],
                 link=r["link"]).strip()  # fmt: skip
    if "title" in r:
        return t("cu_meeting_line", lang, time=r["time"], title=r["title"],
                 link=r.get("minutes") or t("cu_no_minutes", lang)).strip()  # fmt: skip
    return t("cu_channel_line", lang, channel=r["channel"], n=r["messages"], text=r["text"],
             link=r["link"]).strip()  # fmt: skip


# ------------------------------------------------------------------ agent tool
def catchup_tools(src: Sources, now: Callable[[], datetime], calls: list | None = None) -> list:
    """The `catch_up` function the agent registers (bound to one guild and one person)."""
    log = calls if calls is not None else []

    def catch_up_tool(since: str | None = None) -> dict:
        """What the asker missed: a personal catch-up since they were last active, or since a
        time they name. Returns, limited to channels they can read: messages that mention them,
        their new or overdue tasks and decisions, meetings they missed (with minutes links) or
        have coming, highlights per channel and files meant for them, each with a link. Use for
        "what did I miss", "ช่วงที่ผมลาไปมีอะไรบ้าง", "สรุปให้หน่อยว่าเมื่อวานเกิดอะไรขึ้น". Reply
        short (top items only, plain sentences, at most one emoji) and offer to go into detail for
        any channel.

        Args:
            since: optional start: "yesterday", "monday", "3 days", "เมื่อวาน", "ช่วง 3 วัน" or
                YYYY-MM-DD. Leave empty to start from when the asker was last active.
        """
        log.append(("catch_up", {"since": since} if since else {}))
        try:
            return catch_up(src, now(), since)
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

    catch_up_tool.__name__ = "catch_up"  # the name the prompt and the voice tool list use
    return [catch_up_tool]
