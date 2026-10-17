"""Bot glue for personal briefings and notifications (a mixin of `SecretaryBot`).

* morning / evening brief DMs on a per-member schedule (replaces the old task follow-up DM);
* pending asks: a message that mentions a member and asks for something is tracked (regex only);
  answered when the member posts in the conversation or reacts; one gentle reminder after
  `ask_reminder_hours`;
* real-time alerts about important documents (PDF / Office / contract-like images), sent to the
  people they concern who can read the channel, limited per person and deduplicated;
* the `set_my_notifications` / `show_my_notifications` agent tools.

The rules are in `salid.briefings`; this module adapts them to Discord objects. DMs go through
`salid.dm.send_dm`: a member with closed DMs is skipped silently (and logged).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from . import audience as aud
from . import briefings as bf
from .channel_access import split_message
from .dm import send_dm
from .i18n import resolve_lang
from .mentions import match_members
from .profiles import profile_block
from .scheduling import join_link
from .tasks import due_for_followup
from .tone import tone_block

logger = logging.getLogger(__name__)

PREFS_TTL_S = 300.0
ASK_SNIPPET = 160


class BriefingMixin:
    def _briefing_init(self) -> None:
        self.prefs = bf.PrefsRepo(self.docs)
        self.asks = bf.AskBook(self.docs)
        self.alerts = bf.AlertLog(self.docs)
        self._prefs_cache: dict[int, tuple[float, dict[int, bf.NotifPrefs]]] = {}
        self._brief_done: set[tuple] = set()  # (guild, user, kind, day) already handled

    # ------------------------------------------------------------- helpers
    def _briefings_on(self, gid: int) -> bool:
        return self.team.enabled(gid, "briefings") and self.team.enabled(gid, "memory")

    def _guild_lang(self, gid: int) -> str:
        return resolve_lang(self.team.current(gid).language_setting)

    def _all_prefs(self, gid: int) -> dict[int, bf.NotifPrefs]:
        """Members' stored preferences, cached for a few minutes (blocking on a miss)."""
        hit = self._prefs_cache.get(gid)
        if hit is not None and self._clock() - hit[0] < PREFS_TTL_S:
            return hit[1]
        rows = self.prefs.all(gid)
        self._prefs_cache[gid] = (self._clock(), rows)
        return rows

    def _prefs_of(self, gid: int, uid: int) -> bf.NotifPrefs:
        return self._all_prefs(gid).get(uid) or bf.NotifPrefs()

    # ------------------------------------------------------------- timers
    async def _briefings_due(self, now: datetime) -> int:
        """Morning / evening briefs due now, once per member and day. Returns DMs sent."""
        sent = 0
        for guild in list(self.guilds):
            gid = guild.id
            try:
                await asyncio.to_thread(self.team.current, gid)
                if not self._briefings_on(gid):
                    continue
                defaults = dict(
                    zip(("morning", "evening"), self.team.brief_times(gid), strict=True)
                )
                prefs = await asyncio.to_thread(self._all_prefs, gid)
                day = now.date().isoformat()
                for member in self._members(guild):
                    p = prefs.get(member.id) or bf.NotifPrefs()
                    for kind in ("morning", "evening"):
                        k = (gid, member.id, kind, day)
                        if k in self._brief_done or not bf.brief_wanted(
                            kind, now, p, defaults[kind]
                        ):
                            continue
                        self._brief_done.add(k)
                        if len(self._brief_done) > 5000:  # yesterday's entries are not needed
                            self._brief_done = {x for x in self._brief_done if x[3] == day}
                        if await asyncio.to_thread(
                            bf.claim, self.docs, gid, f"brief:{kind}:{member.id}:{day}"
                        ):
                            sent += await self._send_brief(guild, member, kind, now)
            except Exception:
                logger.exception("briefs for guild %s failed", gid)
        return sent

    async def _send_brief(self, guild, member, kind: str, now: datetime) -> int:
        """Gather, compose and DM one brief. 0 when there is nothing to say or DMs are closed."""
        inp = await asyncio.to_thread(self._gather_brief, guild, member, kind, now)
        if inp.is_empty():
            return 0
        profile = self._profile_of(guild.id, member.id)  # the recipient's own, nobody else's
        langs = self._deliver_langs(guild.id, member.id)  # company language, then their own
        tone, note = self._tone_for(
            SimpleNamespace(guild=None, channel=None), guild, member, profile
        )  # a DM: the person's own tone, else the server default
        texts = [
            await asyncio.to_thread(
                bf.build_brief,
                self.llm,
                self.s.gemini_model,
                inp,
                lang,
                profile_block(profile),
                tone_block(tone, resolve_lang(lang), note),
            )
            for lang in langs
        ]
        text = "\n\n".join(dict.fromkeys(texts))
        if inp.catchup:  # deterministic and link-complete: appended, never rewritten by the model
            text = f"{text}\n\n{inp.catchup}"
        if inp.notes:  # leave-time and weather lines, built in code from the member's own places
            text = text + "\n\n" + "\n".join(inp.notes)
        if inp.markets:  # opt-in market line: live numbers, appended verbatim
            text = f"{text}\n\n{inp.markets}"
        if inp.contests:  # deadline lines of the contests the member works on
            text = text + "\n\n" + "\n".join(inp.contests)
        for chunk in split_message(text):
            res = await send_dm(member, chunk)
            if not res.ok:
                logger.info(
                    "brief skipped: guild=%s user=%s reason=%s", guild.id, member.id, res.reason
                )
                return 0
        logger.info("brief sent: guild=%s kind=%s", guild.id, kind)
        return 1

    # ------------------------------------------------------------- brief content
    def _meeting_rows(self, gid: int, uid: int, a: datetime, b: datetime, zone) -> list[dict]:
        rows = []
        for m in self.scheduler.upcoming(gid, a, b):
            if uid in m.get("attendee_ids", []):
                start = datetime.fromisoformat(m["start_utc"]).astimezone(zone)
                rows.append({"time": f"{start:%H:%M}", "title": m["title"], "link": join_link(m)})
        return rows

    def _gather_brief(self, guild, member, kind: str, now: datetime) -> bf.BriefInput:
        """The member's brief input. Blocking."""
        gid, uid, zone = guild.id, member.id, self.s.zone
        today = now.date()
        a, b = bf.day_bounds(today, zone)
        prefs = self._prefs_of(gid, uid)
        inp = bf.BriefInput(
            kind=kind, name=getattr(member, "display_name", str(uid)), day=today.isoformat()
        )
        now_utc = now.astimezone(UTC)
        inp.asks = [
            {
                "who": d.get("author_name", ""),
                "text": d.get("text", ""),
                "channel": d.get("channel_name", ""),
                "link": d.get("jump_url", ""),
            }
            for d in self.asks.open_for(gid, uid, now_utc)
        ]
        tasks = self.tasks.list(gid, status=None, assignee_id=uid)
        if kind == "morning":
            inp.meetings = self._meeting_rows(gid, uid, a, b, zone)
            inp.tasks_due = [
                {"id": x.id, "title": x.title, "due": x.due, "overdue": x.due < today.isoformat()}
                for x in due_for_followup([x for x in tasks if x.status == "open"], today).get(
                    uid, []
                )
            ]
            inp.approvals = self._approvals_waiting(gid, uid, now)
            edef = self.team.brief_times(gid)[1]
            h, m = (int(x) for x in (prefs.evening_time or edef).split(":"))
            since = datetime(today.year, today.month, today.day, h, m, tzinfo=zone) - timedelta(
                days=1
            )
            inp.docs = [
                {
                    "kind": d.get("kind", ""),
                    "summary": d.get("summary", ""),
                    "channel": d.get("channel_name", ""),
                    "link": d.get("jump_url", ""),
                }
                for d in self.alerts.since(gid, uid, since)
                if prefs.doc_alerts != "off" and (d.get("important") or prefs.doc_alerts == "all")
            ]
            inp.catchup = self._catchup_brief_text(guild, member, now, prefs.weekends)
            if self.s.maps_enabled and not inp.is_empty():  # leave-time + office weather lines
                mine = [
                    m for m in self.scheduler.upcoming(gid, a, b)
                    if uid in m.get("attendee_ids", []) and m.get("place_id")
                ]  # fmt: skip
                inp.notes = self._maps_brief_notes(
                    guild, member, mine, now, self._person_lang(gid, uid)
                )
            inp.markets = self._market_brief_line(guild, member, prefs, now)
            inp.contests = self._contest_brief_lines(guild, member, now)
            return inp
        iso = today.isoformat()
        stores = self.secretary_store.list_meetings(
            gid, bf._iso(a), bf._iso(b), statuses=("done", "started", "scheduled")
        )
        inp.attended = [
            {"title": m["title"], "summary": bf._cut(m.get("summary", ""), 300)}
            for m in stores
            if uid in m.get("attended", [])
        ]
        inp.new_tasks = [
            {"id": x.id, "title": x.title, "due": x.due}
            for x in tasks
            if bf.local_day(x.created_at, zone) == iso
        ]
        inp.done = [
            x.title for x in tasks if x.status == "done" and bf.local_day(x.done_at, zone) == iso
        ]
        inp.still_open = [
            {"id": x.id, "title": x.title, "due": x.due} for x in tasks if x.status == "open"
        ]
        inp.tomorrow = self._meeting_rows(
            gid, uid, a + timedelta(days=1), b + timedelta(days=1), zone
        )
        return inp

    # ------------------------------------------------------------- pending asks
    async def _asks_observe(self, message) -> None:
        """A message in a remembered channel: it may answer asks waiting for its author, and it
        may be a new ask for the members it mentions. Never raises."""
        try:
            guild, author = message.guild, message.author
            gid = guild.id
            if not self._briefings_on(gid):
                return
            now = self.s.now().astimezone(UTC)
            if not self.asks.loaded(gid):
                await asyncio.to_thread(self.asks.ensure, gid, now)
            loc = message.channel.id
            if self.asks.has_open(gid, loc, author.id):
                await asyncio.to_thread(
                    self.asks.answer_by_post, gid, loc, author.id, message.id, now
                )
            text = message.content or ""
            raw = list(getattr(message, "raw_mentions", ()) or ())
            if not raw or not bf.looks_like_ask(text):
                return
            members = {uid for uid in raw if (m := guild.get_member(uid)) is not None and not m.bot}
            targets = bf.ask_targets(raw, author.id, members)
            if not targets:
                return
            where = self._memory_channel(message.channel)
            channel_id = where[0] if where else loc
            created = getattr(message, "created_at", None) or now
            info = {
                "loc_id": loc,
                "channel_id": channel_id,
                "channel_name": getattr(message.channel, "name", ""),
                "author_id": author.id,
                "author_name": getattr(author, "display_name", str(author.id)),
                "text": bf._cut(self._strip_mentions(text, message), ASK_SNIPPET),
                "jump_url": getattr(message, "jump_url", ""),
                "created_at": bf._iso(created if created.tzinfo else created.replace(tzinfo=UTC)),
            }
            for uid in targets:
                await asyncio.to_thread(
                    self.asks.add, gid, message_id=message.id, target_id=uid, **info
                )
        except Exception:
            logger.warning("pending-ask tracking failed", exc_info=True)

    async def on_raw_reaction_add(self, payload) -> None:
        """A member reacting to a message that asked them something counts as an answer."""
        try:
            gid = getattr(payload, "guild_id", None)
            if gid is None or not self.asks.loaded(gid):
                return
            await asyncio.to_thread(
                self.asks.answer_by_reaction,
                gid,
                payload.message_id,
                payload.user_id,
                self.s.now().astimezone(UTC),
            )
        except Exception:
            logger.warning("ask reaction handling failed", exc_info=True)

    def _asks_forget(self, gid: int, message_ids: set[int]) -> None:
        """Messages were deleted: their asks are gone too (called from the delete events)."""
        if self.asks.loaded(gid):
            self.asks.drop_message(gid, message_ids, self.s.now().astimezone(UTC))

    async def _asks_due(self, now: datetime) -> int:
        """Send the one gentle reminder for asks still unanswered after `ask_reminder_hours`.
        Returns DMs sent."""
        sent = 0
        after = timedelta(hours=self.s.ask_reminder_hours)
        now_utc = now.astimezone(UTC)
        for guild in list(self.guilds):
            gid = guild.id
            try:
                await asyncio.to_thread(self.team.current, gid)
                if not self._briefings_on(gid) or not self.asks.loaded(gid):
                    continue
                if not bf.reminder_hour_ok(now):
                    continue
                prefs = await asyncio.to_thread(self._all_prefs, gid)
                for d in self.asks.due_reminders(gid, now_utc, after):
                    await asyncio.to_thread(self.asks.mark_reminded, gid, d["key"])  # once
                    member = guild.get_member(d["target_id"])
                    p = prefs.get(d["target_id"]) or bf.NotifPrefs()
                    channel = guild.get_channel(d.get("channel_id", 0))
                    if member is None or not p.ask_reminders:
                        continue
                    if channel is not None and not self._can_read(channel, member):
                        continue
                    text = aud.tt(
                        "ask_reminder", self._deliver_langs(gid, member.id), who=d.get("author_name", ""),
                        channel=f"#{d.get('channel_name', '')}", text=d.get("text", ""),
                        link=d.get("jump_url", ""),
                    )  # fmt: skip
                    res = await send_dm(member, text)
                    sent += int(res.ok)
                    if not res.ok:
                        logger.info(
                            "ask reminder skipped: user=%s reason=%s", member.id, res.reason
                        )
            except Exception:
                logger.exception("ask reminders for guild %s failed", gid)
        return sent

    # ------------------------------------------------------------- important-document alerts
    def _doc_alert_soon(self, message, idx: int, info: bf.DocInfo) -> None:
        if self._briefings_on(message.guild.id):
            self._spawn(self._doc_alert(message, idx, info))

    async def _doc_alert(self, message, idx: int, info: bf.DocInfo) -> int:
        """Classify the analysed document, find who it concerns, and DM them one short alert each.
        Returns DMs sent. Never raises."""
        try:
            return await self._doc_alert_inner(message, idx, info)
        except Exception:
            logger.warning("document alert failed", exc_info=True)
            return 0

    async def _doc_alert_inner(self, message, idx: int, info: bf.DocInfo) -> int:
        guild, uploader = message.guild, message.author
        gid = guild.id
        channel = message.channel
        parent = getattr(channel, "parent", None) if getattr(channel, "parent_id", None) else None
        viewers = await asyncio.to_thread(self._channel_viewers, guild, parent or channel)
        raw = list(getattr(message, "raw_mentions", ()) or ())
        mentioned = {u for u in raw if (m := guild.get_member(u)) is not None and not m.bot}
        ref = await self._referenced_message(message)
        replied = getattr(getattr(ref, "author", None), "id", 0) or 0
        if replied and getattr(ref.author, "bot", False):
            replied = 0
        addressed = set()
        for name in info.addressed:
            found = match_members(guild, name)
            if len(found) == 1:
                addressed.add(found[0].id)
        execs = {
            u for u in viewers if (m := guild.get_member(u)) is not None and self.is_exec_member(m)
        }
        important, targets = bf.alert_targets(
            info, threshold=self.s.doc_alert_min_amount, uploader_id=uploader.id,
            mentioned_ids=mentioned, replied_to_id=replied, addressed_ids=addressed,
            exec_ids=execs, viewers=viewers,
        )  # fmt: skip
        prefs = await asyncio.to_thread(self._all_prefs, gid)
        everything = {  # members who asked for "all": every document they can read
            u for u, p in prefs.items() if p.doc_alerts == "all" and u in viewers
        } - {uploader.id}
        eligible = []
        for uid in sorted(targets | everything):
            mode = (prefs.get(uid) or bf.NotifPrefs()).doc_alerts
            if mode == "off" or (mode == "important" and not important):
                continue
            member = guild.get_member(uid)
            if member is not None and not member.bot:
                eligible.append(uid)
        if not eligible:
            return 0
        where = self._memory_channel(channel)
        cname = getattr(parent or channel, "name", "")
        now = self.s.now()
        summary = bf._cut(info.summary or info.name, 200)
        kind = bf._cut(info.doc_type or bf.category(info.doc_type), 40)
        await asyncio.to_thread(
            self.alerts.record, gid, f"{message.id}:{idx}",
            {
                "targets": eligible, "important": important, "kind": kind, "summary": summary,
                "channel_id": where[0] if where else channel.id, "channel_name": cname,
                "jump_url": getattr(message, "jump_url", ""), "message_id": message.id,
                "at": bf._iso(now),
            },
        )  # fmt: skip
        sent = 0
        for uid in eligible:
            ok = await asyncio.to_thread(
                self.alerts.try_send, gid, uid, message.id, idx, now.date().isoformat(),
                self.s.doc_alerts_per_day,
            )  # fmt: skip
            if not ok:
                continue
            text = aud.tt(
                "alert_doc", self._deliver_langs(gid, uid), kind=kind, summary=summary,
                channel=f"#{cname}", link=getattr(message, "jump_url", ""),
            )  # fmt: skip
            res = await send_dm(guild.get_member(uid), text)
            sent += int(res.ok)
            if not res.ok:
                logger.info("doc alert skipped: user=%s reason=%s", uid, res.reason)
        return sent

    # ------------------------------------------------------------- agent tools
    def _notif_set(self, gid: int, uid: int, **changes) -> dict:
        p, err = self.prefs.update(gid, uid, **changes)
        if p is None:
            return {"error": err}
        self._prefs_cache.pop(gid, None)
        return {"status": "ok", "settings": self._notif_view(gid, p)}

    def _notif_show(self, gid: int, uid: int) -> dict:
        return {"settings": self._notif_view(gid, self.prefs.get(gid, uid))}

    def _notif_view(self, gid: int, p: bf.NotifPrefs) -> dict:
        m, e = self.team.brief_times(gid)
        view = bf.effective_view(p, m, e)
        view["briefings_enabled_for_server"] = self.team.enabled(gid, "briefings")
        return view
