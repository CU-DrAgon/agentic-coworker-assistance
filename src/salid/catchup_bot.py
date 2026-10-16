"""Discord glue of the catch-up ("what did I miss"): the agent tool, `/ตามทัน` `/catchup` and the
short section of the morning brief. The logic is in `salid.catchup`; scope comes from the memory
tools that the bot already computes for the asker (never widened here)."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import discord
from discord import app_commands

from . import catchup as cu
from .i18n import t
from .qa import MemoryTools
from .scheduling import join_link

logger = logging.getLogger(__name__)

CATCHUP_COMMANDS = ("ตามทัน", "catchup")


class CatchupMixin:
    def _catchup_on(self, gid: int) -> bool:
        return self.team.enabled(gid, "catchup") and self.team.enabled(gid, "memory")

    def _catchup_sources(self, guild, member, allowed: dict[int, str], visible=None) -> cu.Sources:
        """Everything the catch-up may read for one member; `allowed` = their readable channels."""
        gid, uid = guild.id, member.id
        kw = {"visible": visible} if visible is not None else {}
        return cu.Sources(
            store=self.memory_store,
            guild_id=gid,
            user_id=uid,
            name=getattr(member, "display_name", str(uid)),
            allowed=dict(allowed),
            zone=self.s.zone,
            tasks=lambda: self.tasks.list(gid, status="open", assignee_id=uid),
            meetings=lambda a, b: self.secretary_store.list_meetings(
                gid, a, b, statuses=("scheduled", "started", "done")
            ),
            alerts=lambda since: self.alerts.since(gid, uid, since),
            retention_days=lambda: self.team.fresh_memory_retention_days(gid, fail_closed=True),
            meeting_link=join_link,
            **kw,
        )

    def _catchup_tools(self, guild, member, memory) -> list:
        """The `catch_up` agent tool for this asker, or [] when the feature is off."""
        if not isinstance(memory, MemoryTools) or not self._catchup_on(guild.id):
            return []
        src = self._catchup_sources(guild, member, memory.allowed, memory._visible)
        return cu.catchup_tools(src, self.s.now)

    def _catchup_brief_text(self, guild, member, now, weekends: bool) -> str:
        """The short catch-up for the morning brief: "" unless the member was away for at least
        one working day. Blocking."""
        if not self._catchup_on(guild.id):
            return ""
        allowed = self._memory_scope(guild, member, None)
        src = self._catchup_sources(guild, member, allowed)
        since, kind = cu.last_seen(src, now)
        guessed = True
        if kind != "found" or cu.working_days_away(since, now, self.s.zone, weekends) < 1:
            return ""  # unknown activity is not "away"
        data = cu.gather(src, since, now, guessed=guessed)
        if data.is_empty():
            return ""
        lang = self._profile_of(guild.id, member.id).language or self._guild_lang(guild.id)
        return cu.render(data, lang, self.s.zone, short=True)

    # ------------------------------------------------------------- /ตามทัน /catchup
    def _register_catchup_commands(self) -> None:
        async def catchup_cb(interaction: discord.Interaction, since: str | None = None):
            await self.handle_catchup(interaction, since)

        catchup_cb = app_commands.guild_only()(catchup_cb)
        catchup_cb = app_commands.describe(since=t("cmd_catchup_since"))(catchup_cb)
        for name in CATCHUP_COMMANDS:
            self.tree.add_command(
                app_commands.Command(name=name, description=t("cmd_catchup"), callback=catchup_cb)
            )

    async def handle_catchup(self, interaction, since_text: str | None = None) -> None:
        """/ตามทัน: the caller's own catch-up, ephemeral, from channels they can read."""
        guild, user = interaction.guild, interaction.user
        lang = self._interaction_lang(interaction)
        await asyncio.to_thread(self.team.current, guild.id)
        if not self._catchup_on(guild.id):
            await interaction.response.send_message(t("cu_off", lang), ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        try:
            where = SimpleNamespace(guild=None, channel=interaction.channel)  # private reply
            memory = await self._memory_tools(where, guild, user)
            if memory is None:
                await interaction.followup.send(t("cu_off", lang), ephemeral=True)
                return
            src = self._catchup_sources(guild, user, memory.allowed, memory._visible)
            now = self.s.now()

            def work():
                since, guessed, err = cu.find_since(src, now, since_text)
                if err:
                    return None
                return cu.render(cu.gather(src, since, now, guessed=guessed), lang, self.s.zone)

            text = await asyncio.to_thread(work)
        except Exception:
            logger.exception("catch-up command failed")
            await interaction.followup.send(t("cu_failed", lang), ephemeral=True)
            return
        if text is None:
            await interaction.followup.send(t("cu_bad_since", lang), ephemeral=True)
            return
        for i in range(0, len(text), 1900):
            await interaction.followup.send(text[i : i + 1900], ephemeral=True)
