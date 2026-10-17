"""Discord glue for the contests assistant (logic in `contests.py`).

`ContestsMixin` gives `SecretaryBot`: the agent tools (`_contest_tools`, appended to `extra_tools`
by `_agent_kwargs`, so voice has them too), the deadline reminders driven by the bot tick
(7 d / 3 d / 1 d / 3 h before each deadline, to the contest team members who can read the contest,
through `deliver_to_person`) and the "⏳" lines of the morning brief.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timedelta

from . import audience as aud
from . import contests as ct
from .freetime import FreetimeContext, FreetimeTools, Person, find_slots
from .i18n import t
from .pdf_report import find_fonts
from .web_read import read_web_call

logger = logging.getLogger(__name__)

REMIND_TICK_S = 60.0
MAX_PLAN_DAYS = 120


class ContestsMixin:
    def _contests_init(self) -> None:
        self.contests = ct.ContestRepo(self.docs, clock=lambda: self.s.now())
        self._contests_last = -1e9

    def _contests_on(self, guild_id: int) -> bool:
        return self.team.enabled(guild_id, "contests")

    # ------------------------------------------------------------- agent tools
    def _contest_tools(self, message, guild, asker, is_exec: bool, lang: str) -> list:
        """The contest tools for one request (chat or voice). [] when the flag is off; never
        raises."""
        try:
            if not self._contests_on(guild.id):
                return []
            ideas = self._ideas_ctx(message, guild, asker, is_exec, lang)
            ctx = ct.ContestCtx(
                guild_id=guild.id,
                guild_name=getattr(guild, "name", "") or "",
                asker_id=asker.id,
                asker_name=getattr(asker, "display_name", str(asker.id)),
                is_exec=is_exec,
                repo=self.contests,
                client=self.llm,
                model=self.s.gemini_model,
                lite_model=self.s.gate_model,
                lang=lang,
                now=self.s.now,
                zone=self.s.zone,
                tz_name=self.s.tz,
                outranked=ideas.outranked,
                private=ideas.private,
                resolve_person=ideas.resolve_person,
                free_days=lambda ids, a, b: self._contest_free_days(guild, ids, a, b, lang),
                idea_get=ideas.idea_get,
                read_web=lambda text, request: read_web_call(
                    self.llm, self.s.gemini_model, text, request
                ),
                fonts=lambda: find_fonts(self.s.pdf_font),
                room_id=getattr(getattr(message, "channel", None), "id", 0) or 0,
            )
            return ct.contest_tools(ctx)
        except Exception:
            logger.warning("contest tools unavailable", exc_info=True)
            return []

    def _contest_free_days(
        self, guild, user_ids: list[int], first: date, last: date, lang: str
    ) -> set[date] | None:
        """Days (first..last) on which at least one team member has an hour free, from the same
        busy sources as find_free_time. None when freetime is off or there is no team."""
        if not user_ids or not self.team.enabled(guild.id, "freetime"):
            return None
        last = min(last, first + timedelta(days=MAX_PLAN_DAYS))
        zone = self.s.zone
        people = []
        for uid in user_ids:
            m = guild.get_member(uid) if hasattr(guild, "get_member") else None
            if m is not None and not getattr(m, "bot", False):
                people.append(Person(m.id, m.display_name))
        if not people:
            return None
        ctx = FreetimeContext(
            guild_id=guild.id,
            asker=people[0],
            zone=zone,
            tz_name=self.s.tz,
            lang=lang,
            repo=self.availability,
            sources=list(self.busy_sources),
            now=self.s.now,
            match_people=lambda _n: [],
        )
        start = datetime.combine(first, time.min, zone)
        end = datetime.combine(last + timedelta(days=1), time.min, zone)
        busy = FreetimeTools(ctx)._busy(people, start - timedelta(days=1), end)
        work = {p.id: self.availability.work_days(guild.id, p.id) for p in people}
        free: set[date] = set()
        day = start
        while day < end:
            nxt = day + timedelta(days=1)
            if find_slots(
                people=people,
                busy=busy,
                work_days=work,
                start=day,
                end=nxt,
                now=start - timedelta(days=1),
                duration_min=60,
                limit=1,
            ):
                free.add(day.date())
            day = nxt
        return free

    # ------------------------------------------------------------- reminders (bot tick)
    async def _contests_due(self) -> int:
        """Deadline reminders (at most once a minute). Returns messages delivered."""
        if self._clock() - self._contests_last < REMIND_TICK_S or not self.guilds:
            return 0
        self._contests_last = self._clock()
        sent = 0
        for g in list(self.guilds):
            if not self._contests_on(g.id):
                continue
            try:
                now = self.s.now()
                due, changed = await asyncio.to_thread(self._contests_scan, g.id, now)
            except Exception:
                logger.exception("contest reminder scan failed")
                continue
            for r in due:
                for uid in r.user_ids:
                    member = g.get_member(uid)
                    if member is None:
                        continue
                    text = aud.render(
                        lambda lg, r=r, now=now: self._contest_reminder_text(r, now, lg),
                        self._deliver_langs(g.id, uid),
                    )
                    try:
                        sent += int((await self.deliver_to_person(g, member, text)).ok)
                    except Exception:
                        logger.warning("contest reminder delivery failed", exc_info=True)
        return sent

    def _contests_scan(self, guild_id: int, now):
        """(reminders due, contests changed): the sent-keys are saved BEFORE delivering, so a
        crash never repeats a reminder."""
        due, changed = ct.due_reminders(self.contests.all(guild_id), now)
        for c in changed:
            self.contests.put(guild_id, c)
        return due, changed

    def _contest_reminder_text(self, r: ct.Reminder, now, lang: str) -> str:
        zone = self.s.zone
        left = ct.left_text(r.at - now, lang)
        open_n = len(r.contest.open_items())
        key = "contest_remind" if open_n else "contest_remind_nolist"
        return t(
            key,
            lang,
            name=r.contest.name,
            label=r.deadline.label,
            left=left,
            when=ct.fmt_when(r.at, lang, zone),
            open=open_n,
        )

    # ------------------------------------------------------------- morning brief
    def _contest_brief_lines(self, guild, member, now) -> list[str]:
        """ "⏳ <contest> ส่งงานอีก N วัน (เหลือ X ข้อ)" lines for one member. Blocking."""
        if not self._contests_on(guild.id):
            return []
        try:
            lang = self._person_lang(guild.id, member.id)  # a line of the brief: company language
            rows = ct.brief_entries(self.contests.all(guild.id), member.id, now, self.s.zone)
            out = []
            for c, days, open_n in rows:
                key = (
                    ("contest_brief_today" if days == 0 else "contest_brief")
                    if open_n
                    else ("contest_brief_today_nolist" if days == 0 else "contest_brief_nolist")
                )
                out.append(t(key, lang, name=c.name, days=days, open=open_n))
            return out
        except Exception:
            logger.warning("contest brief lines failed", exc_info=True)
            return []
