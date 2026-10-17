"""Personal reminders, Discord side (logic in `reminders.py`).

`ReminderMixin` gives `SecretaryBot`: the agent tools (`_reminder_tools`, appended to `extra_tools`
by `_agent_kwargs`, so voice has them too) and the bot-tick delivery (`_reminders_due`, every 30 s).
A reminder reaches its person through `deliver_to_person` (private channel, else DM), or is posted
in the channel it was asked in (`where=here`, mentioning the person).
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import timedelta

import discord

from . import reminders as R
from .i18n import resolve_lang
from .mentions import match_members

logger = logging.getLogger(__name__)

REMINDER_TICK_S = 30.0
PRUNE_EVERY_S = 3600.0
MAX_FAILS = 3  # a ONCE reminder that cannot be delivered is retried this many times, then dropped
RETRY_AFTER = timedelta(minutes=2)
_RAW_MENTION = re.compile(r"<@[!&]?(\d+)>")


class ReminderMixin:
    def _reminders_init(self) -> None:
        """Call after `_coo_init` (needs `self.docs`)."""
        self.reminder_book = R.ReminderBook(self.docs)
        self._reminders_last = float("-inf")
        self._reminders_pruned = float("-inf")

    def _reminders_on(self, guild_id: int) -> bool:
        return self.team.enabled(guild_id, "reminders")

    # ------------------------------------------------------------- agent tools
    def _reminder_tools(self, message, guild, asker, is_exec: bool, lang: str) -> list:
        """The reminder tools for one request (chat or voice). [] when the flag is off; never
        raises. In a voice room `message.channel` is the meeting's text channel."""
        try:
            if not self._reminders_on(guild.id):
                return []

            def resolve(name: str) -> list:
                m = _RAW_MENTION.fullmatch(name.strip())
                if m:
                    who = guild.get_member(int(m.group(1)))
                    return [who] if who is not None and not getattr(who, "bot", False) else []
                return match_members(guild, name)

            def name_of(uid: int) -> str:
                who = guild.get_member(uid)
                return getattr(who, "display_name", str(uid)) if who else str(uid)

            ctx = R.ReminderCtx(
                guild_id=guild.id,
                asker_id=asker.id,
                asker_name=asker.display_name,
                is_exec=is_exec,
                lang=lang,
                book=self.reminder_book,
                now=self.s.now,
                resolve=resolve,
                name_of=name_of,
                channel_id=getattr(getattr(message, "channel", None), "id", None),
            )
            return R.reminder_tools(ctx)
        except Exception:
            logger.warning("reminder tools unavailable", exc_info=True)
            return []

    # ------------------------------------------------------------- tick
    async def _reminders_due(self, now=None) -> int:
        """Deliver the reminders whose time has come (at most every 30 s). Returns how many were
        sent. Failures are logged, never raised."""
        if not self.guilds or self._clock() - self._reminders_last < REMINDER_TICK_S:
            return 0
        self._reminders_last = self._clock()
        now = now or self.s.now()
        sent = 0
        for guild in list(self.guilds):
            if not self._reminders_on(guild.id):
                continue
            try:
                due = await asyncio.to_thread(self.reminder_book.due, guild.id, now)
            except Exception:
                logger.exception("reminder lookup failed")
                continue
            for rec in due:
                try:
                    sent += int(await self._reminder_fire(guild, rec, now))
                except Exception:
                    logger.exception("reminder %s failed", rec.get("id"))
        if self._clock() - self._reminders_pruned >= PRUNE_EVERY_S:
            self._reminders_pruned = self._clock()
            for guild in list(self.guilds):
                try:
                    await asyncio.to_thread(self.reminder_book.prune, guild.id, now)
                except Exception:
                    logger.exception("reminder prune failed")
        return sent

    async def _reminder_fire(self, guild, rec: dict, now) -> bool:
        book = self.reminder_book
        member = guild.get_member(rec["user_id"])
        if member is None:  # left the server
            await asyncio.to_thread(book.save, {**rec, "active": False, "done_at": R._utc(now)})
            return False
        due_at = R._from_iso(rec["next_at"])
        late = now - due_at
        if rec["kind"] != "once" and late > R.LATE_GRACE:
            # the bot was down: skip the missed slots, wait for the next future one (no burst)
            await asyncio.to_thread(book.save, R.advance(rec, now))
            return False
        setting = self._profile_of(guild.id, member.id).language or self._guild_lang(guild.id)
        lang = resolve_lang(setting, rec["message"])
        who = rec["set_by_name"] if rec["set_by"] != rec["user_id"] else ""
        after = R.advance(rec, now)
        text = R.fire_text(
            rec, lang, now,
            nxt=R._from_iso(after["next_at"]) if after["active"] else None,
            who=who,
            mention=member.mention if rec["where"] == "here" else "",
            late_for=due_at if rec["kind"] == "once" and late > R.LATE_GRACE else None,
        )  # fmt: skip
        ok = await self._reminder_send(guild, member, rec, text)
        if not ok:
            fails = rec.get("fails", 0) + 1
            if rec["kind"] == "once" and fails < MAX_FAILS:
                retry = {**rec, "fails": fails, "next_at": R._utc(now + RETRY_AFTER)}
                await asyncio.to_thread(book.save, retry)
                return False
            logger.warning("reminder %s could not be delivered", rec["id"])
        out = {**after, "fired": rec.get("fired", 0) + (1 if ok else 0), "fails": 0}
        await asyncio.to_thread(book.save, out)
        return ok

    async def _reminder_send(self, guild, member, rec: dict, text: str) -> bool:
        if rec["where"] == "here" and rec.get("channel_id"):
            getter = getattr(guild, "get_channel_or_thread", None) or guild.get_channel
            channel = getter(rec["channel_id"])
            if channel is not None:
                try:
                    await channel.send(
                        text, allowed_mentions=discord.AllowedMentions(users=[member])
                    )
                    return True
                except Exception:
                    logger.warning("reminder post failed", exc_info=True)
        res = await self.deliver_to_person(guild, member, text)
        return bool(res.ok)
