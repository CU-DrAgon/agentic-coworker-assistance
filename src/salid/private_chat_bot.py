"""Discord side of the private Salid channel (a mixin of `SecretaryBot`): create / delete it, route
proactive messages to it, clean up when the member leaves. See `salid.private_chat`."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime

import discord

from .dm import DMResult, send_dm
from .i18n import base_lang, resolve_lang, t
from .permissions import guild_missing, permission_message
from .private_chat import (
    BACKFILL_PACE_S,
    GUILD_CHANNEL_CAP,
    LEAVE_GRACE,
    PrivateChatRepo,
    channel_name,
    is_confirm,
    overwrites,
    pick_category,
    wants_delete,
    wants_private_channel,
)

logger = logging.getLogger(__name__)


class PrivateChatMixin:
    def _pc_init(self) -> None:
        self.private_chats = PrivateChatRepo(self.docs)
        self._pc_owners: dict[int, int | None] = {}  # channel id -> owner (None = not private)
        self._pc_delete_pending: set[tuple[int, int]] = set()
        self._pc_cap_told: set[int] = set()  # guilds whose admins were told about the cap
        self._pc_backfilling: set[int] = set()
        self._pc_pace = BACKFILL_PACE_S
        self._pc_sleep = asyncio.sleep  # injectable for tests

    def _pc_lang(self, gid: int, uid: int) -> str:
        lang = self._profile_of(gid, uid).language
        return resolve_lang(lang) if lang else self._guild_lang(gid)

    # ------------------------------------------------------------- lookup
    async def _pc_owner(self, gid: int, channel_id: int) -> int | None:
        """The member a private channel belongs to; None for every other channel."""
        if channel_id not in self._pc_owners:
            owner = await asyncio.to_thread(self.private_chats.owner_of, gid, channel_id)
            self._pc_owners[channel_id] = owner
        return self._pc_owners[channel_id]

    async def private_channel_for(self, guild, uid: int):
        """The member's private channel, or None (a manually deleted one is forgotten)."""
        cid = await asyncio.to_thread(self.private_chats.get, guild.id, uid)
        if cid is None:
            return None
        ch = guild.get_channel(cid)
        if ch is None:
            await asyncio.to_thread(self.private_chats.delete, guild.id, uid)
            self._pc_owners.pop(cid, None)
        return ch

    # ------------------------------------------------------------- create / delete
    async def ensure_private_channel(self, guild, member):
        """(channel, created, error text). One per member per guild; recreated when missing."""
        lang = self._pc_lang(guild.id, member.id)
        existing = await self.private_channel_for(guild, member.id)
        if existing is not None:
            return existing, False, ""
        if not self.team.enabled(guild.id, "private_channels"):
            return None, False, t("pc_off", lang)
        if len(getattr(guild, "channels", ())) >= GUILD_CHANNEL_CAP:
            await self._pc_cap_notice(guild)
            return None, False, t("pc_cap", lang)
        missing = guild_missing(guild, ("manage_channels",))
        if missing:
            return None, False, permission_message(missing, lang, self.application_id)
        base = t("pc_category", lang)
        category, cat_name = pick_category(getattr(guild, "categories", ()), base)
        try:
            if category is None:
                category = await guild.create_category(cat_name, reason="Salid private chats")
            nick = self._profile_of(guild.id, member.id).nickname or member.display_name
            ch = await guild.create_text_channel(
                channel_name(nick, lang),
                category=category,
                overwrites=overwrites(guild, member),
                topic=t("pc_topic", lang),
                reason=f"Salid private chat for {member.id}",
            )
        except discord.HTTPException:
            logger.warning("could not create a private channel in %s", guild.id, exc_info=True)
            missing = guild_missing(guild, ("manage_channels",))
            text = (
                permission_message(missing, lang, self.application_id)
                if missing
                else t("pc_failed", lang)
            )
            return None, False, text
        await asyncio.to_thread(self.private_chats.put, guild.id, member.id, ch.id)
        self._pc_owners[ch.id] = member.id
        try:
            await asyncio.to_thread(self.team.set_channel_mode, guild.id, ch.id, "chat")
            pref = self._profile_of(guild.id, member.id).language
            await asyncio.to_thread(self.team.set_channel_language, guild.id, ch.id, pref or "auto")
        except Exception:
            logger.warning("could not save the private channel settings", exc_info=True)
        with contextlib.suppress(discord.HTTPException):
            await ch.send(
                t("pc_welcome", lang, mention=member.mention),
                allowed_mentions=discord.AllowedMentions(users=[member]),
            )
        return ch, True, ""

    async def _private_request(self, message, guild, member) -> None:
        lang = self._pc_lang(guild.id, member.id)
        ch, created, err = await self.ensure_private_channel(guild, member)
        if ch is None:
            await message.reply(err, mention_author=False)
            return
        key = "pc_created" if created else "pc_exists"
        await message.reply(
            t(key, lang, channel=f"<#{ch.id}>", server=guild.name), mention_author=False
        )

    async def _private_delete(self, message, guild, member) -> bool:
        """ "ลบห้องส่วนตัว": ask, then delete on a "yes". True when the message was handled."""
        key = (guild.id, member.id)
        lang = self._pc_lang(*key)
        if wants_delete(message.content or ""):
            if await self.private_channel_for(guild, member.id) is None:
                await message.reply(t("pc_none", lang), mention_author=False)
            else:
                self._pc_delete_pending.add(key)
                await message.reply(t("pc_delete_ask", lang), mention_author=False)
            return True
        if key in self._pc_delete_pending:
            self._pc_delete_pending.discard(key)
            if is_confirm(message.content or ""):
                await self.remove_private_channel(guild, member.id)
                with contextlib.suppress(discord.HTTPException):
                    await message.reply(t("pc_deleted", lang), mention_author=False)
                return True
        return False

    async def remove_private_channel(self, guild, uid: int) -> None:
        cid = await asyncio.to_thread(self.private_chats.get, guild.id, uid)
        if cid is None:
            return
        await asyncio.to_thread(self.private_chats.delete, guild.id, uid)
        self._pc_owners.pop(cid, None)
        ch = guild.get_channel(cid)
        if ch is not None:
            with contextlib.suppress(discord.HTTPException):
                await ch.delete(reason="Salid private chat removed")

    async def _pc_rename_for_language(self, guild, lang: str) -> None:
        """The server switched language: rename the private categories and channels to match
        (paced for Discord's rate limits; every failure is only logged)."""
        try:
            rows = await asyncio.to_thread(self.private_chats.all, guild.id)
            lang = base_lang(lang)  # channel names exist in Thai and English only
            other = "en" if lang == "th" else "th"
            old_base, new_base = t("pc_category", other), t("pc_category", lang)
            seen: set[int] = set()
            for uid, cid in rows:
                ch = guild.get_channel(cid)
                if ch is None:
                    continue
                cat = getattr(ch, "category", None)
                if cat is not None and cat.id not in seen:
                    seen.add(cat.id)
                    if cat.name == old_base or cat.name.startswith(old_base + " "):
                        try:
                            await cat.edit(name=new_base + cat.name[len(old_base) :])
                            await self._pc_sleep(self._pc_pace)
                        except discord.HTTPException:
                            logger.warning("could not rename category %s", cat.id, exc_info=True)
                nick = self._profile_of(guild.id, uid).nickname
                if not nick:
                    member = guild.get_member(uid)
                    nick = member.display_name if member is not None else ""
                name = channel_name(nick, lang) if nick else ""
                if not name or ch.name == name:
                    continue
                try:
                    await ch.edit(name=name, topic=t("pc_topic", lang))
                    await self._pc_sleep(self._pc_pace)
                except discord.HTTPException:
                    logger.warning("could not rename channel %s", cid, exc_info=True)
        except Exception:
            logger.warning("private channel rename failed", exc_info=True)

    async def _pc_cap_notice(self, guild) -> None:
        """Once per run: the server is close to Discord's channel limit, so DMs are used."""
        if guild.id in self._pc_cap_told:
            return
        self._pc_cap_told.add(guild.id)
        logger.warning("guild %s is near the channel limit: private channels paused", guild.id)
        with contextlib.suppress(Exception):
            ch = await self._report_channel(guild, create=False)
            if ch is not None:
                await ch.send(t("pc_cap_admin", self._guild_lang(guild.id)))

    # ------------------------------------------------------------- every member gets one
    async def on_member_join(self, member) -> None:
        """A new human member: their private channel (a rejoin within 24 h keeps the old one)."""
        if getattr(member, "bot", False):
            return
        try:
            guild = member.guild
            await asyncio.to_thread(self.team.current, guild.id)
            if await asyncio.to_thread(self.private_chats.cancel_delete, guild.id, member.id):
                return  # came back in time: the same channel
            if self.team.enabled(guild.id, "private_channels"):
                await self.ensure_private_channel(guild, member)
        except Exception:
            logger.warning("could not set up a private channel", exc_info=True)

    async def on_member_remove(self, member) -> None:
        """The member left, was kicked or banned: the channel goes 24 hours later."""
        if getattr(member, "bot", False):
            return
        try:
            at = datetime.now(UTC) + LEAVE_GRACE
            await asyncio.to_thread(
                self.private_chats.schedule_delete, member.guild.id, member.id, at
            )
        except Exception:
            logger.warning("private channel clean-up could not be scheduled", exc_info=True)

    async def _pc_sweep(self) -> None:
        """Delete the channels whose member left more than 24 hours ago (called by the tick)."""
        now = datetime.now(UTC)
        for guild in list(self.guilds):
            uids = await asyncio.to_thread(self.private_chats.due_deletions, guild.id, now)
            for uid in uids:
                if guild.get_member(uid) is not None:  # back without us noticing
                    await asyncio.to_thread(self.private_chats.cancel_delete, guild.id, uid)
                    continue
                await self.remove_private_channel(guild, uid)

    def _pc_backfill_start(self, guild) -> None:
        """Make the missing channels for the current members in the background (idempotent)."""
        if guild.id in self._pc_backfilling:
            return
        self._pc_backfilling.add(guild.id)
        self._spawn(self._pc_backfill(guild))

    async def _pc_backfill(self, guild) -> None:
        try:
            await asyncio.to_thread(self.team.current, guild.id)
            if not self.team.enabled(guild.id, "private_channels"):
                return
            members = guild.members.values() if isinstance(guild.members, dict) else guild.members
            for member in list(members):
                if getattr(member, "bot", False) or not self.team.enabled(
                    guild.id, "private_channels"
                ):
                    continue
                ch, created, _ = await self.ensure_private_channel(guild, member)
                if ch is None and len(getattr(guild, "channels", ())) >= GUILD_CHANNEL_CAP:
                    return  # cap: the rest keep using DMs
                if created:
                    await self._pc_sleep(self._pc_pace)  # Discord rate limits channel creation
        except Exception:
            logger.warning("private channel backfill failed", exc_info=True)
        finally:
            self._pc_backfilling.discard(guild.id)

    async def _private_command(self, message, guild, member) -> bool:
        """ "ขอห้องส่วนตัว" / "ลบห้องส่วนตัว" typed to Salid (channel, DM or the private channel)."""
        text = message.content or ""
        if await self._private_delete(message, guild, member):
            return True
        if wants_private_channel(self._strip_mentions(text, message) if message.guild else text):
            await self._private_request(message, guild, member)
            return True
        return False

    async def handle_private_chat(self, interaction) -> None:
        """/ห้องส่วนตัว /private-chat: make (or show) the caller's private channel."""
        guild, member = interaction.guild, interaction.user
        await interaction.response.defer(ephemeral=True)
        lang = self._pc_lang(guild.id, member.id)
        ch, created, err = await self.ensure_private_channel(guild, member)
        text = err or t(
            "pc_created" if created else "pc_exists", lang, channel=f"<#{ch.id}>", server=guild.name
        )
        await interaction.followup.send(text, ephemeral=True)

    # ------------------------------------------------------------- proactive messages
    async def _private_route(self, member, text: str, files=(), view=None):
        """Post a proactive message in the person's private channel, mentioning them. None when
        there is no such channel, they prefer DMs, or it cannot be posted (the DM path follows)."""
        guild = getattr(member, "guild", None)
        if guild is None or not hasattr(self, "private_chats"):
            return None
        try:
            if self._profile_of(guild.id, member.id).dm_delivery:
                return None
            ch = await self.private_channel_for(guild, member.id)
            if ch is None:
                return None
            kw = {}
            if files:
                kw["files"] = list(files)
            if view is not None:
                kw["view"] = view
            sent = await ch.send(
                f"{member.mention} {text}",
                allowed_mentions=discord.AllowedMentions(
                    users=[member], roles=False, everyone=False
                ),
                **kw,
            )
        except Exception:
            logger.info("private channel post failed; falling back to DM", exc_info=True)
            return None
        return DMResult(True, message=sent)

    async def deliver_to_person(
        self, guild, member, content: str, files=None, view=None
    ) -> DMResult:
        """THE way to reach one person unprompted: their private Salid channel in `guild` when it
        exists (mentioning them), else, or when they prefer DMs, a DM. Never raises."""
        return await send_dm(member, content, files or (), view=view)
