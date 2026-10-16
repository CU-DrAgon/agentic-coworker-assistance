"""Bot glue for find-a-time (a mixin of `SecretaryBot`); the rules are in `salid.freetime`.

Registers `find_free_time`, `set_availability` and `my_availability` for the `freetime` feature
flag. Busy sources: Salid's meetings, personal availability and Discord scheduled events (who
organises or is interested). Other sources plug in through `self.busy_sources` (an approvals module
feeds `LeaveSource` through `_leave_intervals`, a Google Calendar free/busy source adds
`GoogleCalendarSource(fetch)`); the Google one is a no-op until then.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, time, timedelta

import discord

from .freetime import (
    AvailabilityRepo,
    AvailabilitySource,
    Busy,
    CallbackSource,
    FreetimeContext,
    GoogleCalendarSource,
    LeaveSource,
    MeetingsSource,
    Person,
    freetime_tools,
)
from .mentions import match_members

logger = logging.getLogger(__name__)

MAX_EVENTS = 25
MAX_INTERESTED = 200


class FreetimeMixin:
    def _freetime_init(self) -> None:
        self.availability = AvailabilityRepo(self.docs)
        # Pluggable: replace or extend (LeaveSource(approved_leave), GoogleCalendarSource(fetch)).
        self.busy_sources = [
            MeetingsSource(self.secretary_store),
            AvailabilitySource(self.availability, self.s.zone),
            LeaveSource(self._leave_intervals),  # approved leave (salid.approvals)
            GoogleCalendarSource(None),
        ]

    def _leave_intervals(
        self, guild_id: int, user_id: int, start, end
    ) -> list[tuple[datetime, datetime]]:
        """Approved leave as whole local days: the approvals read API gives inclusive date
        pairs, the busy sources want datetimes. Blocking."""
        zone = self.s.zone
        lo = start.astimezone(zone).date() if start.tzinfo else start.date()
        hi = end.astimezone(zone).date() if end.tzinfo else end.date()
        out = []
        for first, last in self.approved_leave(guild_id, user_id, lo, hi):
            out.append(
                (
                    datetime.combine(first, time.min, zone),
                    datetime.combine(last + timedelta(days=1), time.min, zone),
                )
            )
        return out

    def _freetime_tools(self, guild, asker, lang: str) -> list:
        if not self.team.enabled(guild.id, "freetime"):
            return []
        loop = asyncio.get_running_loop()

        def run(coro):
            return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=60)

        def match(name: str) -> list[Person]:
            return [Person(m.id, m.display_name) for m in match_members(guild, name)]

        def by_id(uid: int):
            m = guild.get_member(uid) if hasattr(guild, "get_member") else None
            return Person(m.id, m.display_name) if m is not None and not m.bot else None

        def events(guild_id, user_ids, start, end):
            return run(self._event_busy(guild, set(user_ids), start, end))

        ctx = FreetimeContext(
            guild_id=guild.id,
            asker=Person(asker.id, asker.display_name),
            zone=self.s.zone,
            tz_name=self.s.tz,
            lang=lang,
            repo=self.availability,
            sources=[*self.busy_sources, CallbackSource(events)],
            now=self.s.now,
            match_people=match,
            person_by_id=by_id,
        )
        return freetime_tools(ctx)

    async def _event_busy(self, guild, wanted: set[int], start, end) -> dict[int, list[Busy]]:
        """Discord scheduled events in the range that these people organise or are interested in.
        Only the time range is returned: never the event's name."""
        out: dict[int, list[Busy]] = {}
        n = 0
        for ev in getattr(guild, "scheduled_events", ()):
            s = getattr(ev, "start_time", None)
            if s is None:
                continue
            e = getattr(ev, "end_time", None) or s + timedelta(hours=1)
            if not (s < end and start < e):
                continue
            n += 1
            if n > MAX_EVENTS:
                break
            status = str(getattr(getattr(ev, "status", None), "name", "")).lower()
            if status in ("cancelled", "canceled", "completed", "ended"):
                continue
            who = set()
            if getattr(ev, "creator_id", None) in wanted:
                who.add(ev.creator_id)
            with contextlib.suppress(discord.HTTPException, AttributeError, TypeError):
                async for u in ev.users(limit=MAX_INTERESTED):
                    if u.id in wanted:
                        who.add(u.id)
            for uid in who:
                out.setdefault(uid, []).append(Busy(s, e, "event"))
        return out
