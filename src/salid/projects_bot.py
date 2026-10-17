"""PM dashboard projects, Discord side (logic in `projects.py`, web API in `pm_api.py`).

`ProjectsMixin` gives `SecretaryBot` the agent tools (`_project_tools`, appended to `extra_tools`
by `_agent_kwargs`, so chat and voice share them). Projects of a Discord server live under its
guild id, next to its tasks, so the web dashboard and Discord see the same ones. The asker is a
person ref built from the Discord member (plus the email of the org member linked to that Discord
id, when there is one); admin = a Salid executive or an org owner / admin. Nothing here is chosen
by the model.
"""

from __future__ import annotations

import asyncio
import logging
import re

import discord

from .mentions import match_members
from .orgs import OrgStore
from .projects import Person, ProjectRepo, ProjectsCtx, project_tools
from .projects_listen import ProjectListener
from .tasks import TaskRepo

logger = logging.getLogger(__name__)

_RAW_MENTION = re.compile(r"<@[!&]?(\d+)>")


class ProjectsMixin:
    def _projects_init(self) -> None:
        """Call after `_coo_init` (needs `self.docs`)."""
        self.project_repo = ProjectRepo(self.docs, clock=self.s.now)
        self.project_task_repo = TaskRepo(self.docs)
        # projects heard in normal chat; chained after the task tracker by `_on_summary_all` (coo.py)
        self.project_listener = ProjectListener(
            self.project_repo,
            self.project_task_repo,
            self.s,
            self.llm,
            people_of=lambda gid: self._people(self._guild(gid)) if self._guild(gid) else {},
            note=self._project_note,
            lang_of=lambda gid, cid: self.team.channel_language(gid, cid),
            enabled=lambda gid: self._projects_on(gid) and self.team.enabled(gid, "memory"),
            notes_on=lambda gid: self.team.enabled(gid, "project_notes"),
        )

    def _projects_on(self, guild_id: int) -> bool:
        return self.team.enabled(guild_id, "projects")

    def _project_note(self, guild_id: int, channel_id: int, text: str) -> None:
        """Say what was recorded in the room. Runs in the summary's worker thread, so the send is
        scheduled on the bot loop. Never in a member's private channel or while snoozed."""
        guild = self._guild(guild_id)
        channel = guild.get_channel(channel_id) if guild else None
        if channel is None or not hasattr(channel, "send"):
            return
        if self.private_chats.owner_of(guild_id, channel_id) is not None or self._is_snoozed(
            channel_id
        ):
            return
        loop = self.loop

        async def post() -> None:
            try:
                await channel.send(text, allowed_mentions=discord.AllowedMentions.none())
            except Exception:
                logger.warning("project note not sent", exc_info=True)

        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(post()))

    def _project_tools(self, message, guild, asker, is_exec: bool, lang: str) -> list:
        """The project tools for one request (chat or voice). [] when the flag is off; never
        raises."""
        try:
            if not self._projects_on(guild.id):
                return []
            org = OrgStore(self.docs).org_of_guild(guild.id)
            emails = {
                m.discord_user_id: m for m in (org.members if org else []) if m.discord_user_id
            }

            def person(member) -> Person:
                known = emails.get(str(member.id))
                return Person(
                    discord_user_id=str(member.id),
                    email=known.email if known else None,
                    name=getattr(member, "display_name", "") or str(member.id),
                )

            def resolve(name: str) -> list[Person]:
                m = _RAW_MENTION.fullmatch(name.strip())
                if m:
                    who = guild.get_member(int(m.group(1)))
                    return (
                        [person(who)] if who is not None and not getattr(who, "bot", False) else []
                    )
                return [person(x) for x in match_members(guild, name)]

            me = person(asker)
            role = org.role_of(me.email) if (org and me.email) else None
            ctx = ProjectsCtx(
                guild_id=guild.id,
                asker=me,
                is_admin=bool(is_exec or role in ("owner", "admin")),
                lang=lang,
                repo=self.project_repo,
                tasks=self.project_task_repo,
                now=self.s.now,
                resolve=resolve,
                workspace=f"discord-{guild.id}",
            )
            return project_tools(ctx)
        except Exception:
            logger.warning("project tools unavailable", exc_info=True)
            return []
