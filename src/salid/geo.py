"""Time phrases and place-name resolution shared by the travel and weather tools.

Places follow the visibility rules of `places.py`: only places the asker may know are used (a
personal place only by its owner), and an address is only ever *shown* when everyone who hears the
answer may know it; otherwise the place can still be used for a computation but only its name is
said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .places import Place, PlacesCtx, matches, norm

HOME_WORDS = {"บ้านผม", "บ้านฉัน", "บ้านดิฉัน", "บ้านเรา", "บ้านหนู", "บ้าน", "home", "myhome", "myhouse"}
OFFICE_WORDS = {"ออฟฟิศ", "สำนักงาน", "ที่ทำงาน", "office", "hq", "headoffice", "myoffice"}
_TH_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_DAYS = {"วันนี้": 0, "today": 0, "พรุ่งนี้": 1, "tomorrow": 1, "มะรืนนี้": 2, "มะรืน": 2}


# ---------------------------------------------------------------------------- time phrases
def _clock(s: str) -> tuple[int, int] | None:
    """A time of day in Thai or English words/digits -> (hour, minute), or None."""
    m = re.search(r"(\d{1,2})[:.](\d{2})\s*(am|pm|น\.?)?", s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if m.group(3) == "pm" and h < 12:
            h += 12
        return (h, mi) if h < 24 and mi < 60 else None
    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", s)
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
        return h, 0
    half = 30 if "ครึ่ง" in s else 0
    if "เที่ยงคืน" in s:
        return 0, half
    if "เที่ยง" in s:
        return 12, half
    m = re.search(r"(\d{1,2})\s*ทุ่ม", s)
    if m:
        n = int(m.group(1))
        return ((n + 18) % 24, half) if 1 <= n <= 6 else None
    if "ทุ่ม" in s:
        return 19, half
    m = re.search(r"ตี\s*(\d{1,2})", s)
    if m:
        return int(m.group(1)) % 24, half
    if "บ่ายโมง" in s:
        return 13, half
    m = re.search(r"(?:บ่าย\s*(\d{1,2})|(\d{1,2})\s*โมง)", s)
    if m:
        n = int(m.group(1) or m.group(2))
        if m.group(1) or "บ่าย" in s or "เย็น" in s:
            return (n + 12 if n < 12 else n), half
        if 1 <= n <= 6:  # "4 โมง" alone: the afternoon reading
            return n + 12, half
        return n % 24, half
    m = re.fullmatch(r"\s*(\d{1,2})\s*", s)
    return (int(m.group(1)), 0) if m and int(m.group(1)) < 24 else None


def parse_when(text: str, now: datetime, zone) -> datetime | None:
    """ "now"/empty -> None. ISO 8601, "พรุ่งนี้ 8 โมง", "วันนี้ 17:30", "tomorrow 8am", "บ่าย 2",
    "20:00" -> an aware datetime in `zone`. A bare time already past today means tomorrow.
    ValueError when it cannot be understood."""
    s = (text or "").strip().casefold().translate(_TH_DIGITS)
    if s in ("", "now", "ตอนนี้", "เดี๋ยวนี้", "ทันที", "asap"):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("z", "+00:00"))
        return dt.replace(tzinfo=zone) if dt.tzinfo is None else dt.astimezone(zone)
    except ValueError:
        pass
    now = now.astimezone(zone)
    day = None
    for word, delta in _DAYS.items():
        if word in s:
            day = (now + timedelta(days=delta)).date()
            s = s.replace(word, " ")
            break
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    if m:
        day = datetime.fromisoformat(m.group(0)).date()
        s = s.replace(m.group(0), " ")
    hm = _clock(s)
    if hm is None:
        raise ValueError(f"cannot read the time {text!r}")
    base = datetime(*(day or now.date()).timetuple()[:3], hm[0], hm[1], tzinfo=zone)
    if day is None and base <= now:
        base += timedelta(days=1)
    return base


# ---------------------------------------------------------------------------- places
@dataclass
class Endpoint:
    label: str  # what may be said aloud
    address: str  # what is sent to Google
    reveal: bool  # the address may be said / put in a link for this audience
    place: Place | None = None


def usable_places(ctx: PlacesCtx) -> list[Place]:
    from .places import may_know

    out = []
    for p in ctx.repo.list(ctx.guild_id):
        if p.personal:
            if p.owner_id == ctx.asker_id:
                out.append(p)
        elif may_know(p, ctx.asker_id, ctx.ranks, ctx.viewers_of):
            out.append(p)
    return out


def find_place(ctx: PlacesCtx | None, query: str, *, kinds: tuple[str, ...] = ()) -> Place | None:
    """A remembered place the asker may use for `query` (home / office words included)."""
    if ctx is None or not (query or "").strip():
        return None
    pool = usable_places(ctx)
    q = norm(query)
    if q in HOME_WORDS:
        homes = [p for p in pool if p.personal and p.kind == "home"]
        if homes:
            return homes[0]
    if q in OFFICE_WORDS:
        co = [p for p in pool if not p.personal]
        co.sort(key=lambda p: (p.kind != "hq", norm(p.name) not in OFFICE_WORDS))
        if co and (co[0].kind == "hq" or any(w in norm(co[0].name) for w in OFFICE_WORDS)):
            return co[0]
    hits = [(matches(p, query), p) for p in pool]
    hits = [(s, p) for s, p in hits if s and (not kinds or p.kind in kinds)]
    if not hits:
        return None
    hits.sort(key=lambda x: (-x[0], x[1].owner_id != ctx.asker_id))
    return hits[0][1]


def resolve_endpoint(ctx: PlacesCtx | None, text: str) -> Endpoint:
    """A known place (visibility checked in code) or the user's own words as an address."""
    p = find_place(ctx, text)
    if p is None:
        return Endpoint(label=text.strip(), address=text.strip(), reveal=True)
    return Endpoint(label=p.name, address=p.address or p.name, reveal=ctx.can_see(p), place=p)
