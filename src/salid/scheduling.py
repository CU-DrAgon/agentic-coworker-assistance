"""Scheduling meetings: Discord Scheduled Events, reminders, no-show nudges and auto-join.

Layers:

* `CalendarBackend` is the only thing that talks to a calendar. `DiscordCalendar` (Guild Scheduled
  Events, needs the bot permissions Create Events + Manage Events, no OAuth) is the only
  implementation; a Google Calendar backend would implement the same four methods.
* `MeetingService` owns the meeting records (`salid_meetings`, see `secretary_store`): creating,
  cancelling, rescheduling, tracking events created by people directly in Discord, and the
  reminder loop (`tick`): reminder before the start and at the start (channel post + DM with a
  one-click link), no-show DMs for voice meetings, auto-join to take notes, attendance notes from
  DM replies, and the attendance line after the meeting.

Nothing here decides whether the user confirmed: the agent tools in `salid.secretary` gate every
side effect behind a confirmation in a *later* message before calling this module.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Protocol
from urllib.parse import urlparse

import discord

from .dm import send_dm
from .i18n import resolve_lang, t
from .secretary_store import SecretaryStore, new_id

logger = logging.getLogger(__name__)

THAI_DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
EN_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MAX_ATTENDEES = 25
MAX_DURATION_MIN = 12 * 60
START_REMINDER_GRACE = timedelta(minutes=15)  # a missed start reminder is not sent late
NUDGE_GRACE = timedelta(minutes=30)
FINISH_AFTER = timedelta(minutes=30)  # a meeting is "done" this long after its end
REPLY_WINDOW = timedelta(minutes=90)  # a DM reply this soon after our DM is an attendance note
MAX_REASON_CHARS = 300
VOICE_NAME_HINT = re.compile(r"ประชุม|meeting|meet|room|ห้องคุย|ห้องเสียง", re.I)

_LINK_HOSTS = (
    ("meet", ("meet.google.com",)),
    ("zoom", ("zoom.us", "zoom.com", "zoomgov.com")),
    ("teams", ("teams.microsoft.com", "teams.live.com", "teams.microsoft.us")),
)


def classify_link(url: str | None) -> str | None:
    """ "meet" / "zoom" / "teams" for those services, "url" for any other http(s) URL, else None."""
    s = (url or "").strip()
    if not s or " " in s:
        return None
    p = urlparse(s if "://" in s else "https://" + s)
    if p.scheme not in ("http", "https") or not p.hostname or "." not in p.hostname:
        return None
    host = p.hostname.lower()
    for kind, hosts in _LINK_HOSTS:
        if any(host == h or host.endswith("." + h) for h in hosts):
            return kind
    return "url"


def normalize_link(url: str | None) -> str:
    s = (url or "").strip()
    return s if "://" in s else ("https://" + s if s else "")


def fmt_when(dt: datetime, lang: str, tz=None) -> str:
    """ "จันทร์ 2026-10-19 14:00" in the guild's zone (weekday included so a wrong date is seen)."""
    if tz is not None:
        dt = dt.astimezone(tz)
    day = (THAI_DAYS if lang != "en" else EN_DAYS)[dt.weekday()]
    return f"{day} {dt:%Y-%m-%d %H:%M}"


def parse_local(text: str, zone) -> datetime:
    """ISO 8601 -> aware datetime; a naive value is read in `zone`. ValueError otherwise."""
    s = (text or "").strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=zone) if dt.tzinfo is None else dt


def parse_range(text: str, now: datetime) -> tuple[datetime, datetime]:
    """today | tomorrow | week | month | upcoming | YYYY-MM-DD | YYYY-MM-DD..YYYY-MM-DD, in now's
    zone, as [start, end)."""
    s = (text or "upcoming").strip().lower()
    d0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    one = timedelta(days=1)
    if s in ("today", "วันนี้"):
        return d0, d0 + one
    if s in ("tomorrow", "พรุ่งนี้"):
        return d0 + one, d0 + 2 * one
    if s in ("week", "this week", "สัปดาห์นี้", "7d"):
        return now, d0 + 7 * one
    if s in ("month", "this month", "เดือนนี้", "30d"):
        return now, d0 + 30 * one
    if s in ("upcoming", "next", ""):
        return now, d0 + 90 * one
    parts = s.split("..")
    a = date.fromisoformat(parts[0].strip())
    b = date.fromisoformat(parts[-1].strip())
    start = datetime(a.year, a.month, a.day, tzinfo=now.tzinfo)
    end = datetime(b.year, b.month, b.day, tzinfo=now.tzinfo) + one
    if end <= start:
        raise ValueError("end before start")
    return start, end


# ------------------------------------------------------------------ attendees
def _labels(x) -> list[str]:
    return [
        str(v).casefold()
        for v in (
            getattr(x, "display_name", None),
            getattr(x, "name", None),
            getattr(x, "global_name", None),
        )
        if v
    ]


def _match_name(members: list, key: str):
    """Exact name, else unique prefix, else unique substring; None when unknown or ambiguous."""
    tests = (
        lambda lbs: key in lbs,
        lambda lbs: any(lb.startswith(key) for lb in lbs),
        lambda lbs: any(key in lb for lb in lbs),
    )
    for test in tests:
        cands = [x for x in members if test(_labels(x))]
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            return None
    return None


def resolve_attendees(guild, names: list[str], exclude_bots: bool = True):
    """Names / mentions -> (members, unresolved names). A mention (<@id>) or numeric id matches by
    id; text matches display name / username / global name exactly (case-insensitive), then by
    a unique prefix, then by a unique substring. Ambiguous or unknown names are unresolved."""
    pool = getattr(guild, "members", None) or []
    pool = list(pool.values()) if isinstance(pool, dict) else list(pool)
    members = [m for m in pool if not (exclude_bots and m.bot)]
    found: dict[int, Any] = {}
    unresolved: list[str] = []
    for raw in names or []:
        s = (raw or "").strip()
        if not s:
            continue
        m = re.fullmatch(r"<@!?(\d+)>|@?(\d{15,25})", s)
        hit = None
        if m:
            uid = int(m.group(1) or m.group(2))
            hit = next((x for x in members if x.id == uid), None)
        else:
            hit = _match_name(members, s.lstrip("@").strip().casefold())
        if hit is None:
            unresolved.append(s)
        else:
            found[hit.id] = hit
    return list(found.values()), unresolved


def pick_voice_channel(guild, name: str | None):
    """The named voice channel, else the guild's one obvious meeting voice channel (the only
    voice channel, or the only one whose name looks like a meeting room), else None."""
    voices = list(getattr(guild, "voice_channels", []) or [])
    if name:
        want = name.strip().lstrip("#").casefold()
        exact = [c for c in voices if c.name.casefold() == want]
        part = exact or [c for c in voices if want and want in c.name.casefold()]
        return part[0] if len(part) == 1 or exact else None
    if len(voices) == 1:
        return voices[0]
    hinted = [c for c in voices if VOICE_NAME_HINT.search(c.name)]
    return hinted[0] if len(hinted) == 1 else None


def voice_link(guild_id: int, channel_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}"


def event_link(guild_id: int, event_id) -> str:
    return f"https://discord.com/events/{guild_id}/{event_id}"


def join_link(m: dict) -> str:
    """The one-click way in: the voice channel, else the meeting link, else the event page."""
    if m.get("voice_channel_id"):
        return voice_link(m["guild_id"], m["voice_channel_id"])
    return m.get("link") or (event_link(m["guild_id"], m["event_id"]) if m.get("event_id") else "")


# ------------------------------------------------------------------ calendar backend
class CalendarError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


@dataclass
class EventSpec:
    title: str
    start: datetime
    end: datetime
    description: str = ""
    voice_channel: Any = None
    location: str = "Discord"  # external events only (a meeting URL, or "Discord")


class CalendarBackend(Protocol):
    """A calendar Salid can write to. Only `DiscordCalendar` exists today."""

    async def create_event(self, guild, spec: EventSpec) -> tuple[str, str]:
        """-> (event id, url of the event)"""
        ...

    async def update_event(self, guild, event_id: str, start: datetime, end: datetime) -> None: ...
    async def delete_event(self, guild, event_id: str) -> None: ...
    async def subscribers(self, guild, event_id: str) -> list[int]: ...


class DiscordCalendar:
    """Guild Scheduled Events. A voice-channel event when `spec.voice_channel` is set, otherwise an
    external event whose location is the meeting link (or "Discord")."""

    @staticmethod
    def check_permissions(guild) -> None:
        perms = getattr(getattr(guild, "me", None), "guild_permissions", None)
        ok = perms is not None and getattr(perms, "manage_events", False)
        if not ok:
            raise CalendarError("missing_permission")

    async def create_event(self, guild, spec: EventSpec) -> tuple[str, str]:
        self.check_permissions(guild)
        kw: dict[str, Any] = dict(
            name=spec.title[:100],
            start_time=spec.start,
            end_time=spec.end,
            description=spec.description[:1000],
            privacy_level=discord.PrivacyLevel.guild_only,
        )
        if spec.voice_channel is not None:
            kw["channel"] = spec.voice_channel
        else:
            kw.update(entity_type=discord.EntityType.external, location=spec.location[:100])
        try:
            ev = await guild.create_scheduled_event(**kw)
        except discord.Forbidden as e:
            raise CalendarError("missing_permission") from e
        except discord.HTTPException as e:
            raise CalendarError("failed", str(getattr(e, "text", e))[:200]) from e
        return str(ev.id), getattr(ev, "url", "") or event_link(guild.id, ev.id)

    async def _event(self, guild, event_id: str):
        ev = guild.get_scheduled_event(int(event_id))
        if ev is None:
            try:
                ev = await guild.fetch_scheduled_event(int(event_id))
            except discord.NotFound as e:
                raise CalendarError("not_found") from e
        return ev

    async def update_event(self, guild, event_id: str, start: datetime, end: datetime) -> None:
        self.check_permissions(guild)
        ev = await self._event(guild, event_id)
        try:
            await ev.edit(start_time=start, end_time=end)
        except discord.Forbidden as e:
            raise CalendarError("missing_permission") from e
        except discord.HTTPException as e:
            raise CalendarError("failed", str(getattr(e, "text", e))[:200]) from e

    async def delete_event(self, guild, event_id: str) -> None:
        self.check_permissions(guild)
        try:
            ev = await self._event(guild, event_id)
        except CalendarError as e:
            if e.code == "not_found":
                return  # already gone
            raise
        try:
            await ev.delete()
        except discord.Forbidden as e:
            raise CalendarError("missing_permission") from e
        except discord.HTTPException as e:
            raise CalendarError("failed", str(getattr(e, "text", e))[:200]) from e

    async def subscribers(self, guild, event_id: str) -> list[int]:
        try:
            ev = await self._event(guild, event_id)
            return [u.id async for u in ev.users() if not getattr(u, "bot", False)]
        except (CalendarError, discord.HTTPException):
            return []


# ------------------------------------------------------------------ service
_REQUEST_RE = re.compile(
    r"[?？]|อะไร|ยังไง|อย่างไร|ไหม|มั้ย|หรือเปล่า|หรือไม่|กี่|เมื่อ(?:ไร|ไหร่)|ที่ไหน|ใคร|ทำไม|"
    r"เท่า(?:ไร|ไหร่)|บอก|ช่วย|ขอ(?!โทษ|ลา)|"
    r"\b(?:what|how|why|when|where|who|which|can(?!['’]t)|could|would|will you|please|pls|"
    r"help|show|tell|send)\b",
    re.I,
)


def looks_like_request(text: str) -> bool:
    """A question or a request to Salid (not a reason for being absent)."""
    return bool(_REQUEST_RE.search(text or ""))


class MeetingService:
    """Meeting records + the reminder loop. `clock()` returns an aware datetime (injectable);
    `get_guild(id)` is the bot's `get_guild`; `meetings` is the voice `MeetingManager`."""

    def __init__(
        self,
        store: SecretaryStore,
        calendar: CalendarBackend,
        settings,
        team,
        meetings,
        get_guild: Callable[[int], Any],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store, self.calendar, self.s = store, calendar, settings
        self.team, self.meetings, self.get_guild = team, meetings, get_guild
        self.clock = clock or (lambda: datetime.now(UTC))
        self._lock = asyncio.Lock()

    # ---- helpers
    def lang_for(self, guild_id: int, channel_id: int | None) -> str:
        setting = self.team.channel_language(guild_id, channel_id or 0)  # no channel: guild default
        return resolve_lang(setting, "")  # auto has no text to follow: Thai

    async def dm(self, guild, user_id: int, text: str) -> bool:
        return (await send_dm(guild.get_member(user_id), text)).ok

    async def post(self, guild, channel_id: int | None, text: str) -> None:
        ch = guild.get_channel(channel_id) if channel_id else None
        if ch is None:
            ch = getattr(guild, "system_channel", None)
        if ch is None:
            return
        try:
            await ch.send(text)
        except discord.HTTPException:
            logger.warning("could not post to channel %s", getattr(ch, "id", "?"))

    @staticmethod
    def mentions(m: dict) -> str:
        return " ".join(f"<@{i}>" for i in m.get("attendee_ids", []))

    def _names(self, guild, ids) -> list[str]:
        out = []
        for i in ids:
            mem = guild.get_member(i)
            out.append(getattr(mem, "display_name", None) or str(i))
        return out

    def display(self, m: dict, lang: str) -> dict:
        """The meeting as the model/user sees it."""
        zone = self.s.zone
        start = datetime.fromisoformat(m["start_utc"])
        end = datetime.fromisoformat(m["end_utc"])
        return {
            "id": m["id"],
            "event_id": m.get("event_id"),
            "title": m["title"],
            "when": f"{fmt_when(start, lang, zone)}-{end.astimezone(zone):%H:%M} ({self.s.tz})",
            "kind": m["kind"],
            "where": join_link(m),
            "attendees": m.get("attendee_names", []),
            "status": m["status"],
        }

    # ---- creation / changes
    async def create(
        self,
        guild,
        *,
        title: str,
        start: datetime,
        duration_minutes: int,
        attendees: list,
        voice_channel,
        link: str,
        description: str,
        organizer,
        text_channel_id: int | None,
        place: dict | None = None,
    ) -> dict:
        """`place`: a remembered place every invitee may know (salid.places.place_for_event). The
        address goes into the invitation DMs; into the Discord event (visible to the whole server)
        only when the whole server may know it. The record keeps only its id (never the address)."""
        end = start + timedelta(minutes=duration_minutes)
        link = normalize_link(link) if link else ""
        shown = bool(place and place.get("public"))
        loc = link or (place["address"][:100] if shown else "Discord")
        if shown and not description.endswith(place["maps_link"]):
            description = f"{description}\n{place['maps_link']}".strip()
        spec = EventSpec(title, start, end, description, voice_channel, loc)
        event_id, url = await self.calendar.create_event(guild, spec)
        tc = voice_channel.id if voice_channel is not None else text_channel_id
        m = {
            "id": new_id(),
            "guild_id": guild.id,
            "event_id": event_id,
            "event_url": url,
            "source": "salid",
            "title": title,
            "description": description,
            "start": start.isoformat(timespec="minutes"),
            "start_utc": start.astimezone(UTC).isoformat(timespec="seconds"),
            "end_utc": end.astimezone(UTC).isoformat(timespec="seconds"),
            "duration_minutes": duration_minutes,
            "tz": self.s.tz,
            "kind": "voice" if voice_channel is not None else "external",
            "voice_channel_id": voice_channel.id if voice_channel is not None else None,
            "text_channel_id": tc,
            "link": link,
            "link_type": classify_link(link),
            "place_id": place["id"]
            if place and place.get("id")
            else "",  # an id, never the address
            "organizer_id": organizer.id,
            "organizer_name": organizer.display_name,
            "attendee_ids": [a.id for a in attendees],
            "attendee_names": [a.display_name for a in attendees],
            "status": "scheduled",
            "rem_pre": False,
            "rem_start": False,
            "nudged": False,
            "joined": False,
            "attended": [],
            "dm_sent": {},
            "dm_failed": [],
            "attendance_notes": {},
            "summary": "",
            "summary_url": "",
            "created_at": self.clock().isoformat(timespec="seconds"),
        }
        self.store.put_meeting(m)
        lang = self.lang_for(guild.id, tc)
        when = fmt_when(start, lang, self.s.zone)
        invite = t(
            "sched_invite_dm",
            lang,
            title=title,
            when=when,
            guild=guild.name,
            by=organizer.display_name,
            link=join_link(m),
        )
        if place:
            invite += "\n" + t(
                "place_event_dm",
                lang,
                where=f"{place['name']}: {place['address']}",
                maps=place["maps_link"],
            )
        for a in attendees:
            if a.id == organizer.id:
                continue
            if not await self.dm(guild, a.id, invite):
                m["dm_failed"].append(a.display_name)
        self.store.put_meeting(m)
        return m

    def find(self, guild_id: int, ref: str) -> dict | None:
        ref = str(ref or "").strip()
        m = self.store.meeting_by_event(guild_id, ref) or self.store.get_meeting(ref)
        return m if m and m.get("guild_id") == guild_id else None

    async def cancel(self, guild, m: dict) -> None:
        if m.get("event_id"):
            await self.calendar.delete_event(guild, m["event_id"])
        m["status"] = "cancelled"
        self.store.put_meeting(m)
        lang = self.lang_for(guild.id, m.get("text_channel_id"))
        text = t("sched_cancelled_dm", lang, title=m["title"])
        for uid in m.get("attendee_ids", []):
            if uid != m.get("organizer_id"):
                await self.dm(guild, uid, text)

    async def reschedule(self, guild, m: dict, new_start: datetime) -> dict:
        end = new_start + timedelta(minutes=m.get("duration_minutes", 60))
        if m.get("event_id"):
            await self.calendar.update_event(guild, m["event_id"], new_start, end)
        m.update(
            start=new_start.isoformat(timespec="minutes"),
            start_utc=new_start.astimezone(UTC).isoformat(timespec="seconds"),
            end_utc=end.astimezone(UTC).isoformat(timespec="seconds"),
            rem_pre=False,
            rem_start=False,
            nudged=False,
            joined=False,
            status="scheduled",
        )
        self.store.put_meeting(m)
        lang = self.lang_for(guild.id, m.get("text_channel_id"))
        text = t(
            "sched_moved_dm",
            lang,
            title=m["title"],
            when=fmt_when(new_start, lang, self.s.zone),
            link=join_link(m),
        )
        for uid in m.get("attendee_ids", []):
            if uid != m.get("organizer_id"):
                await self.dm(guild, uid, text)
        return m

    def upcoming(self, guild_id: int, start: datetime, end: datetime) -> list[dict]:
        return self.store.list_meetings(
            guild_id,
            start.astimezone(UTC).isoformat(timespec="seconds"),
            end.astimezone(UTC).isoformat(timespec="seconds"),
            statuses=("scheduled", "started"),
        )

    # ---- events created or changed directly in Discord
    async def track_discord_event(self, event) -> dict | None:
        """A Guild Scheduled Event somebody made in Discord: keep a record so it gets the same
        reminders and no-show logic. Salid's own events are already tracked (matched by id)."""
        guild = event.guild
        if guild is None or not self.team.enabled(guild.id, "scheduling"):
            return None
        existing = self.store.meeting_by_event(guild.id, event.id)
        if existing is not None:
            return existing
        start = event.start_time
        end = event.end_time or (start + timedelta(hours=1))
        chan = getattr(event, "channel", None)
        loc = getattr(event, "location", None) or ""
        link = normalize_link(loc) if classify_link(loc) else ""
        creator = getattr(event, "creator", None)
        m = {
            "id": new_id(),
            "guild_id": guild.id,
            "event_id": str(event.id),
            "event_url": getattr(event, "url", "") or event_link(guild.id, event.id),
            "source": "discord",
            "title": event.name,
            "description": getattr(event, "description", "") or "",
            "start": start.astimezone(self.s.zone).isoformat(timespec="minutes"),
            "start_utc": start.astimezone(UTC).isoformat(timespec="seconds"),
            "end_utc": end.astimezone(UTC).isoformat(timespec="seconds"),
            "duration_minutes": max(1, int((end - start).total_seconds() // 60)),
            "tz": self.s.tz,
            "kind": "voice" if chan is not None else "external",
            "voice_channel_id": chan.id if chan is not None else None,
            "text_channel_id": chan.id if chan is not None else None,
            "link": link,
            "link_type": classify_link(link),
            "organizer_id": getattr(creator, "id", None),
            "organizer_name": getattr(creator, "display_name", "") or "",
            "attendee_ids": [],
            "attendee_names": [],
            "status": "scheduled",
            "rem_pre": False,
            "rem_start": False,
            "nudged": False,
            "joined": False,
            "attended": [],
            "dm_sent": {},
            "dm_failed": [],
            "attendance_notes": {},
            "summary": "",
            "summary_url": "",
            "created_at": self.clock().isoformat(timespec="seconds"),
        }
        self.store.put_meeting(m)
        return m

    async def update_discord_event(self, event) -> None:
        m = self.store.meeting_by_event(event.guild.id, event.id)
        if m is None:
            await self.track_discord_event(event)
            return
        start = event.start_time
        end = event.end_time or (start + timedelta(minutes=m.get("duration_minutes", 60)))
        s_utc = start.astimezone(UTC).isoformat(timespec="seconds")
        if s_utc != m["start_utc"]:
            m.update(rem_pre=False, rem_start=False, nudged=False, joined=False)
        m.update(
            title=event.name,
            start=start.astimezone(self.s.zone).isoformat(timespec="minutes"),
            start_utc=s_utc,
            end_utc=end.astimezone(UTC).isoformat(timespec="seconds"),
        )
        status = str(getattr(getattr(event, "status", None), "name", "")).lower()
        if status in ("cancelled", "canceled") and m["status"] != "done":
            m["status"] = "cancelled"
        elif status == "completed" and m["status"] in ("scheduled", "started"):
            m["status"] = "done"
        self.store.put_meeting(m)

    async def delete_discord_event(self, event) -> None:
        m = self.store.meeting_by_event(event.guild.id, event.id)
        if m is not None and m["status"] in ("scheduled", "started"):
            m["status"] = "cancelled"
            self.store.put_meeting(m)

    async def _refresh_attendees(self, guild, m: dict) -> None:
        """Members who marked themselves interested in the event join the attendee list."""
        if not m.get("event_id"):
            return
        ids = await self.calendar.subscribers(guild, m["event_id"])
        for uid in ids:
            if uid not in m["attendee_ids"]:
                m["attendee_ids"].append(uid)
                mem = guild.get_member(uid)
                m["attendee_names"].append(getattr(mem, "display_name", None) or str(uid))

    # ---- the loop
    async def tick(self) -> None:
        now = self.clock()
        for m in self.store.list_meetings(statuses=("scheduled", "started")):
            try:
                await self._tick_one(m, now)
            except Exception:
                logger.exception("meeting tick failed for %s", m.get("id"))

    def _session_of(self, meeting_id: str):
        """The voice session taking notes for this meeting (any body, any channel), if any."""
        return next(
            (
                s
                for s in self.meetings.sessions.values()
                if getattr(s, "meeting_id", None) == meeting_id
            ),
            None,
        )

    def meeting_now(self, guild_id: int, channel_id: int) -> dict | None:
        """The scheduled voice meeting happening now in this channel (not yet joined), if any."""
        if not self.team.enabled(guild_id, "scheduling"):
            return None
        now = self.clock()
        for m in self.store.list_meetings(statuses=("scheduled", "started")):
            if m["guild_id"] != guild_id or m.get("voice_channel_id") != channel_id:
                continue
            if m["kind"] != "voice" or m["joined"]:
                continue
            start = datetime.fromisoformat(m["start_utc"])
            if start <= now < start + NUDGE_GRACE and self._session_of(m["id"]) is None:
                return m
        return None

    async def auto_join_now(self, m: dict) -> bool:
        """Join a scheduled meeting right away (a person just entered its channel)."""
        guild = self.get_guild(m["guild_id"])
        if guild is None or not self.team.enabled(guild.id, "voice_meetings"):
            return False
        lang = self.lang_for(guild.id, m.get("text_channel_id"))
        joined = await self._auto_join(guild, m, lang)
        if joined:
            self.store.put_meeting(m)
        return joined

    def _humans_in_voice(self, m: dict, guild) -> list:
        ch = guild.get_channel(m["voice_channel_id"]) if m.get("voice_channel_id") else None
        return [x for x in (getattr(ch, "members", []) or []) if not x.bot]

    async def _tick_one(self, m: dict, now: datetime) -> None:
        guild = self.get_guild(m["guild_id"])
        if guild is None or not self.team.enabled(m["guild_id"], "scheduling"):
            return
        start = datetime.fromisoformat(m["start_utc"])
        end = datetime.fromisoformat(m["end_utc"])
        remind = timedelta(minutes=self.s.meeting_remind_minutes)
        noshow = timedelta(minutes=self.s.meeting_noshow_minutes)
        dirty = False
        lang = self.lang_for(guild.id, m.get("text_channel_id"))
        in_session = self._session_of(m["id"]) is not None

        if now >= end + FINISH_AFTER and not in_session:
            await self._finish(guild, m, lang)
            return
        if not m["rem_pre"] and now >= start - remind:
            m["rem_pre"] = dirty = True
            if now < start:
                await self._refresh_attendees(guild, m)
                mins = max(1, round((start - now).total_seconds() / 60))
                await self._remind(guild, m, "pre", lang, mins=mins)
        if not m["rem_start"] and now >= start:
            m["rem_start"] = dirty = True
            if now < start + START_REMINDER_GRACE:
                await self._refresh_attendees(guild, m)
                await self._remind(guild, m, "start", lang)
            if m["status"] == "scheduled":
                m["status"] = "started"
        if m["kind"] == "voice" and start <= now < end + FINISH_AFTER:
            humans = self._humans_in_voice(m, guild)
            for h in humans:
                if h.id not in m["attended"]:
                    m["attended"].append(h.id)
                    dirty = True
            if (
                humans
                and not m["joined"]
                and not in_session
                and now < start + NUDGE_GRACE
                and self.team.enabled(guild.id, "voice_meetings")
            ):
                dirty |= await self._auto_join(guild, m, lang)
        if not m["nudged"] and now >= start + noshow:
            m["nudged"] = dirty = True
            if now < start + NUDGE_GRACE and m["kind"] == "voice":
                await self._refresh_attendees(guild, m)
                await self._nudge(guild, m, lang)
        if dirty:
            self.store.put_meeting(m)

    def _stamp(self, m: dict, uid: int) -> None:
        m["dm_sent"][str(uid)] = self.clock().isoformat(timespec="seconds")

    async def _remind(self, guild, m: dict, which: str, lang: str, mins: int = 0) -> None:
        link = join_link(m)
        when = fmt_when(datetime.fromisoformat(m["start_utc"]), lang, self.s.zone)
        key = "sched_remind_pre" if which == "pre" else "sched_remind_start"
        body = t(key, lang, title=m["title"], mins=mins, when=when, link=link)
        if which == "start" and m["kind"] != "voice":
            body += "\n" + t("sched_reply_if_stuck", lang)
        mentions = self.mentions(m)
        await self.post(guild, m.get("text_channel_id"), f"{mentions} {body}".strip())
        for uid in m.get("attendee_ids", []):
            if await self.dm(guild, uid, body):
                self._stamp(m, uid)
            elif (n := self._names(guild, [uid])[0]) not in m["dm_failed"]:
                m["dm_failed"].append(n)

    async def _nudge(self, guild, m: dict, lang: str) -> None:
        present = {h.id for h in self._humans_in_voice(m, guild)}
        link = join_link(m)
        for uid in m.get("attendee_ids", []):
            if uid in present:
                continue
            text = t("sched_nudge_dm", lang, title=m["title"], link=link)
            if await self.dm(guild, uid, text):
                self._stamp(m, uid)

    async def _auto_join(self, guild, m: dict, lang: str) -> bool:
        ch = guild.get_channel(m["voice_channel_id"])
        if ch is None:
            return False
        starter = SimpleNamespace(
            id=0, guild=guild, voice=SimpleNamespace(channel=ch), display_name="Salid"
        )

        async def on_end(session) -> None:
            await self.on_voice_end(m["id"], session)

        # Salid already sits in this room (joined by itself or on command): take over as notes
        # for the meeting instead of joining twice.
        find = getattr(self.meetings, "session_in", None)
        there = find(guild.id, ch.id) if find is not None else None
        if there is not None and getattr(there, "meeting_id", None) is None:
            await self.meetings.adopt(there, m["id"], on_end)
            m["joined"] = True
            return True
        await self.meetings.join(
            starter,
            lang=lang,
            announce="meet_announce_scheduled",
            first_grace=self.s.meeting_empty_minutes * 60.0,
            on_end=on_end,
            meeting_id=m["id"],
            origin="scheduled",
        )
        if self._session_of(m["id"]) is not None:
            m["joined"] = True
            return True
        return False

    # ---- attendance
    def attendance_text(self, guild, m: dict, lang: str) -> str:
        attended = set(m.get("attended", []))
        notes = m.get("attendance_notes", {})
        names = {i: n for i, n in zip(m["attendee_ids"], m["attendee_names"], strict=False)}
        came = [names.get(i) or self._names(guild, [i])[0] for i in sorted(attended)]
        away = []
        for uid in m["attendee_ids"]:
            if uid in attended:
                continue
            why = (notes.get(str(uid)) or {}).get("note")
            nm = names.get(uid) or str(uid)
            away.append(f"{nm} ({why})" if why else nm)
        lines = []
        if came:
            lines.append(t("sched_attended", lang, names=", ".join(came)))
        if away:
            lines.append(t("sched_absent", lang, names=", ".join(away)))
        return "\n".join(lines)

    async def on_voice_end(self, meeting_id: str, session) -> None:
        """The auto-joined voice meeting ended: store the summary link and post attendance."""
        m = self.store.get_meeting(meeting_id)
        if m is None:
            return
        guild = self.get_guild(m["guild_id"])
        lang = self.lang_for(m["guild_id"], m.get("text_channel_id"))
        m["summary"] = (getattr(session, "summary_text", "") or "")[:20000]
        m["summary_url"] = getattr(session, "summary_url", "") or ""
        m["status"] = "done"
        m["attendance_posted"] = True
        self.store.put_meeting(m)
        if guild is not None:
            text = self.attendance_text(guild, m, lang)
            if m["summary_url"]:
                text = (text + "\n" if text else "") + t(
                    "sched_summary_link", lang, url=m["summary_url"]
                )
            if text:
                await self.post(guild, m.get("text_channel_id"), text)

    async def _finish(self, guild, m: dict, lang: str) -> None:
        m["status"] = "done"
        if not m.get("attendance_posted") and m.get("attendance_notes"):
            m["attendance_posted"] = True
            text = self.attendance_text(guild, m, lang)
            if text:
                await self.post(guild, m.get("text_channel_id"), text)
        self.store.put_meeting(m)

    async def handle_dm_reply(
        self, user_id: int, text: str, guild_id: int | None = None
    ) -> str | None:
        """A DM from an attendee soon after our reminder/nudge is the reason they cannot join:
        record it on the meeting and tell the organiser in the meeting's text channel. Returns the
        acknowledgement for the attendee, or None when the DM is not such a reply."""
        text = (text or "").strip()
        if not text or len(text) > 200 or "\n" in text or looks_like_request(text):
            return None
        now = self.clock()
        best = None
        for m in self.store.list_meetings(statuses=("scheduled", "started")):
            if guild_id is not None and m.get("guild_id") != guild_id:
                continue  # the DM is about another company
            sent = m.get("dm_sent", {}).get(str(user_id))
            if not sent or user_id not in m.get("attendee_ids", []):
                continue
            at = datetime.fromisoformat(sent)
            if now - at <= REPLY_WINDOW and (best is None or at > best[0]):
                best = (at, m)
        if best is None:
            return None
        m = best[1]
        guild = self.get_guild(m["guild_id"])
        if guild is None:
            return None
        mem = guild.get_member(user_id)
        name = getattr(mem, "display_name", None) or str(user_id)
        lang = self.lang_for(guild.id, m.get("text_channel_id"))
        reason = text[:MAX_REASON_CHARS]
        m["attendance_notes"][str(user_id)] = {
            "name": name,
            "note": reason,
            "at": now.isoformat(timespec="seconds"),
        }
        self.store.put_meeting(m)
        await self.post(
            guild,
            m.get("text_channel_id"),
            t("sched_absence_note", lang, name=name, title=m["title"], reason=reason),
        )
        return t("sched_absence_ack", lang, title=m["title"])


__all__ = [
    "CalendarBackend",
    "CalendarError",
    "DiscordCalendar",
    "EventSpec",
    "MeetingService",
    "classify_link",
    "fmt_when",
    "join_link",
    "normalize_link",
    "parse_local",
    "parse_range",
    "pick_voice_channel",
    "resolve_attendees",
    "voice_link",
]
