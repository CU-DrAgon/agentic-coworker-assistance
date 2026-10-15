"""Where the meeting minutes go: spoken / typed "send the resolutions to #x" and the stored summary.

Pure logic only (regex parse, channel resolution, access rules); the Discord side lives in
`MeetingManager` (voice instruction) and `DeliverMixin.deliver_minutes` (chat request after the
meeting). The stored summary text of each finished meeting is a `MINUTES` doc (see coo_store).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

# "ส่งมติการประชุมไปห้อง #บัญชี" / "ส่งสรุปประชุมไปที่ห้องเปเปอร์นะครับ" / "send the minutes to #accounting"
_TH = re.compile(
    r"(?:ส่ง|โพสต์|โพส|ลง|ฝาก)\s*(?:มติ|สรุป|รายงาน|บันทึก)\s*(?:การ|ที่)?\s*(?:ประชุม)?\s*"
    r"(?:ทั้งหมด|เมื่อกี้|ตอนจบ|ด้วย)?\s*(?:ไป|ที่|ใน|ลง|ให้)?\s*(?:ที่)?\s*"
    r"(?P<mark>ห้อง|ช่อง|แชนแนล|แชแนล|#|channel\s+)?\s*(?P<name>[^\s,.!?;:]+(?:\s+[A-Za-z0-9_-]+)?)",
    re.I,
)
_EN = re.compile(
    r"(?:send|post|put|share|drop)\s+(?:me\s+)?(?:the\s+)?(?:meeting\s+)?"
    r"(?:minutes|summary|resolutions?|notes)\s*(?:of\s+the\s+meeting\s+)?(?:to|in|into|on)\s+"
    r"(?:the\s+)?(?P<mark>#|channel\s+)?(?P<name>[\w\-฀-๿]+)",
    re.I,
)
_TAIL = re.compile(r"(?:นะ|ครับ|ค่ะ|คะ|หน่อย|ด้วย|จ้า|น้า|เลย|ที|แล้ว)+$")
_NOT_CHANNELS = {"ทุกคน", "ผม", "ฉัน", "เรา", "everyone", "me", "us", "all", "ด้วย", "หน่อย"}


class MinutesRequest(NamedTuple):
    name: str  # the channel as spoken / typed
    explicit: bool  # said "ห้อง" / "#" / "channel": a miss deserves a reply


def parse_minutes_destination(text: str) -> MinutesRequest | None:
    """The channel named in a "send the minutes to <channel>" instruction, else None."""
    text = (text or "").strip()
    if not text or len(text) > 300:
        return None
    for rx in (_TH, _EN):
        m = rx.search(text)
        if m is None:
            continue
        name = _TAIL.sub("", m["name"].strip()).strip()
        if len(name) < 2 or name.casefold() in _NOT_CHANNELS:
            continue
        mark = (m["mark"] or "").strip()
        return MinutesRequest(name.lstrip("#"), bool(mark))
    return None


class Resolved(NamedTuple):
    status: str  # ok | not_found | ambiguous | not_allowed
    channel: Any = None
    candidates: tuple = ()


def resolve_channel(channels: Iterable, name: str, allowed: Callable[[Any], bool]) -> Resolved:
    """Find the destination among `channels`; only channels `allowed` accepts can be chosen.
    "not_allowed" means the name matches a channel that exists but may not be used."""
    from .deliver import match_channels  # deliver imports this module

    chans = list(channels)
    ok = [c for c in chans if allowed(c)]
    hits, _how = match_channels(ok, name)
    if len(hits) == 1:
        return Resolved("ok", hits[0])
    if len(hits) > 1:
        return Resolved("ambiguous", None, tuple(hits))
    if match_channels(chans, name)[0]:
        return Resolved("not_allowed")
    return Resolved("not_found")


def can_see_and_post(channel, who) -> bool:
    try:
        p = channel.permissions_for(who)
    except Exception:
        return False
    view = getattr(p, "view_channel", getattr(p, "read_messages", False))
    return bool(view and getattr(p, "send_messages", False))


def can_see(channel, who) -> bool:
    try:
        p = channel.permissions_for(who)
    except Exception:
        return False
    return bool(getattr(p, "view_channel", getattr(p, "read_messages", False)))


def latest_visible(records: list[dict], member_id: int, can_view_voice: Callable[[int], bool]):
    """The newest stored meeting the member attended or can see the channel of (None if none)."""
    ok = [
        r
        for r in records
        if member_id in (r.get("attendee_ids") or []) or can_view_voice(int(r["voice_channel_id"]))
    ]
    return max(ok, key=lambda r: r.get("ended_at", ""), default=None)
