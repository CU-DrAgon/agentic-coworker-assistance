"""Meeting attendance: who entered / left the voice room and when (pure logic, no Discord).

`MeetingManager` records `AttendanceEvent`s on the session; this module holds the dedupe rule for
reconnects, the chat lines ("🟢 23:12 คุณพีท เข้าห้องประชุม"), the per-person intervals and the
"attendees" block of the posted minutes / the minutes document. Everything is computed in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .i18n import t

REJOIN_S = 20.0  # leave + re-join within this long is a reconnect, not two events
BATCH_S = 3.0  # events this close together go into one chat message
KINDS = ("joined", "left")


@dataclass
class AttendanceEvent:
    user_id: int
    name: str
    kind: str  # "joined" | "left"
    at: datetime  # aware, in the guild's timezone
    mono: float = field(default=0.0, compare=False)  # monotonic time it was seen (dedupe, batching)


@dataclass(frozen=True)
class Presence:
    """One person's time in the room: (start, end) intervals, in order."""

    user_id: int
    name: str
    spans: tuple[tuple[datetime, datetime], ...]

    @property
    def minutes(self) -> int:
        return sum(minutes_between(a, b) for a, b in self.spans)


def minutes_between(a: datetime, b: datetime) -> int:
    return max(0, round((b - a).total_seconds() / 60))


def hm(at: datetime) -> str:
    return f"{at:%H:%M}"


def label(name: str, lang: str) -> str:
    """Thai gets the polite "คุณ" prefix (not doubled); English uses the name as is."""
    name = (name or "").strip()
    if lang == "en" or not name or name.startswith("คุณ"):
        return name
    return f"คุณ{name}"


def record(
    log: list[AttendanceEvent], ev: AttendanceEvent, rejoin_s: float = REJOIN_S
) -> tuple[bool, AttendanceEvent | None]:
    """Add `ev` to `log` unless it is redundant. Returns (added, retracted): `retracted` is the
    earlier "left" event removed because the person came straight back (a reconnect: neither the
    leave nor the join is kept). A join while already present / a leave while absent is ignored."""
    last = next((e for e in reversed(log) if e.user_id == ev.user_id), None)
    if ev.kind == "joined":
        if last is not None and last.kind == "joined":
            return False, None
        if last is not None and ev.mono - last.mono <= rejoin_s:
            log.remove(last)
            return False, last
    elif last is None or last.kind != "joined":
        return False, None
    log.append(ev)
    return True, None


def chat_lines(events: list[AttendanceEvent], lang: str) -> str:
    """One message for a batch: a 🟢 line for those who entered, a 🔴 line for those who left."""
    lines = []
    for kind, icon, key in (
        ("joined", "🟢", "attend_joined"),
        ("left", "🔴", "attend_left"),
    ):
        group = [e for e in events if e.kind == kind]
        if group:
            names = ", ".join(label(e.name, lang) for e in group)
            lines.append(t(key, lang, icon=icon, time=hm(group[0].at), names=names))
    return "\n".join(lines)


def presences(events: list[AttendanceEvent], end: datetime) -> list[Presence]:
    """Per person, in order of first arrival. Still in the room at the end -> until `end`."""
    order: list[int] = []
    names: dict[int, str] = {}
    open_at: dict[int, datetime] = {}
    spans: dict[int, list[tuple[datetime, datetime]]] = {}
    for e in sorted(events, key=lambda x: x.at):
        if e.user_id not in names:
            order.append(e.user_id)
            spans[e.user_id] = []
        names[e.user_id] = e.name or names.get(e.user_id, "")
        if e.kind == "joined":
            open_at.setdefault(e.user_id, e.at)
        elif e.user_id in open_at:
            spans[e.user_id].append((open_at.pop(e.user_id), e.at))
    for uid, start in open_at.items():
        spans[uid].append((start, max(end, start)))
    return [Presence(uid, names[uid], tuple(spans[uid])) for uid in order if spans[uid]]


def span_text(p: Presence, lang: str) -> str:
    return ", ".join(
        f"{hm(a)}–{hm(b)} ({minutes_text(minutes_between(a, b), lang)})" for a, b in p.spans
    )


def minutes_text(n: int, lang: str) -> str:
    return t("attend_minutes", lang, n=n) if n >= 1 else t("attend_lt_minute", lang)


def minutes_block(people: list[Presence], lang: str) -> str:
    """The attendees section of the posted minutes ("" when nobody was logged)."""
    if not people:
        return ""
    rows = [f"{label(p.name, lang)} {span_text(p, lang)}" for p in people]
    return "\n".join(
        [t("attend_header", lang), *rows, t("attend_total", lang, n=len(people)), "", ""]
    )
